"""水饺策略 (Shuijiao Strategy) — QuantX Policy 架构实现

基于 TrendSupportProcessor 的核心指标：
- trend_line: 趋势振荡器 (55周期 stochastic → 三重平滑 → EMA)
- buy_signal: trend_line 上穿阈值 + close < midline
- ready_sell: trend_line > 89 + FILTER + close > midline
- 辅助指标: EMA12/26, BBI, RSI14, KDJ(9,3,3)
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from quantx.core.engine.types import Order, OrderAction
from quantx.core.strategy.base import (
    CompositeStrategy,
    ExecutionStrategy,
    OrderList,
    PolicyState,
    RebalanceStrategy,
    Signal,
    StockSelection,
    StockSelector,
    WeightAllocation,
)

logger = logging.getLogger(__name__)

# ============================================================
# 指标计算工具
# ============================================================

def _sma(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    """通达信 SMA(X, N, M)"""
    return series.ewm(alpha=m / n, adjust=False).mean()


def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def _llv(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=1).min()


def _hhv(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=1).max()


def _cross(a: pd.Series, b) -> pd.Series:
    """上穿判断"""
    if isinstance(b, pd.Series):
        return (a > b) & (a.shift(1) <= b.shift(1))
    return (a > b) & (a.shift(1) <= b)


def _filter(cond: pd.Series, n: int) -> pd.Series:
    """通达信 FILTER: 信号发生后 N 周期内不再触发"""
    res = pd.Series(False, index=cond.index)
    ignore_until = -1
    for i in range(len(cond)):
        if cond.iloc[i] and not pd.isna(cond.iloc[i]):
            if i > ignore_until:
                res.iloc[i] = True
                ignore_until = i + n
    return res


# ============================================================
# 选股策略
# ============================================================

class ShuijiaoSelector(StockSelector):
    """水饺选股策略

    1. 计算 trend_line 指标（55周期 stochastic → 三重平滑 → EMA）
    2. 生成 buy_signal = trend_line 上穿阈值 AND close < midline
    3. 二次筛选：RSI/KDJ/BBI 过滤
    4. 市场健康度判断（trend_line > 20 的股票占比）
    5. 综合打分排序
    """

    DEFAULT_PARAMS = {
        # 选股
        "market_health_entry": 0.22,
        "enable_market_health_entry": True,
        # RSI 过滤
        "enable_rsi_filter": True,
        "rsi_min": 38.0,
        "rsi_max": 85.0,
        # KDJ 过滤
        "enable_kdj_filter": True,
        "kdj_bias": 5.0,
        "kdj_j_max": 98.0,
        # BBI 过滤
        "enable_bbi_filter": True,
        "bbi_floor_ratio": 0.97,
        # 打分
        "score_trend_weight": 1.0,
        "score_ema_weight": 20.0,
        "score_rsi_center": 58.0,
        "score_rsi_penalty": 0.5,
        "score_kdj_center": 60.0,
        "score_kdj_penalty": 0.12,
        "enable_ema_score": True,
        "enable_rsi_score": True,
        "enable_kdj_score": True,
        # 趋势护栏
        "enable_trend_guard": False,
        "trend_guard_mode": "ema",
        "trend_guard_min_ema_spread": 0.0,
        # 放宽筛选
        "enable_relax_entry_filters": False,
        "relax_rsi_min": 28.0,
        "relax_rsi_max": 90.0,
        # 其他
        "trend_period": 55,
        "legacy_wrap_first_signal": False,
        "legacy_preserve_signal_order": False,
        "log_trades": True,
    }

    def __init__(self, **kwargs):
        self.params = dict(self.DEFAULT_PARAMS)
        self.params.update(kwargs)

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def prepare(self, context):
        runtime = getattr(context, "factor_runtime", None)
        if runtime is None:
            return
        existing = {"midline", "trend_line", "buy_signal", "ready_sell"}
        if existing <= set(runtime.values):
            if "market_health" not in runtime.values:
                runtime.compute_formulas({"market_health": "CSMean(trend_line > 20)"})
            return
        runtime.compute_formulas({
            "h1": "Maximum(high, Ref(close, 1))",
            "l1": "Minimum(low, Ref(close, 1))",
            "p1": "h1 - l1",
            "resistance": "l1 + p1 * 7 / 8",
            "support": "l1 + p1 * 0.5 / 8",
            "midline": "(support + resistance) / 2",
            "llv_low": "Min(low, ${trend_period})",
            "hhv_high": "Max(high, ${trend_period})",
            "rsv": "(close - llv_low) / (hhv_high - llv_low + 1e-10) * 100",
            "sma1": "SMA_TDX(rsv, 5, 1)",
            "sma2": "SMA_TDX(sma1, 3, 1)",
            "v11": "3 * sma1 - 2 * sma2",
            "trend_line": "EMA(v11, 3)",
            "trend_prev": "Ref(trend_line, 1)",
            "bb1": "And(trend_prev < 11, trend_prev > 6, Cross(trend_line, 11))",
            "bb2": "And(trend_prev < 6, trend_prev > 3, Cross(trend_line, 6))",
            "bb3": "And(trend_prev < 3, trend_prev > 1, Cross(trend_line, 3))",
            "bb4": "And(trend_prev < 1, trend_prev > 0, Cross(trend_line, 1))",
            "bb5": "And(trend_prev < 0, Cross(trend_line, 0))",
            "buy_signal": "And(Or(bb1, bb2, bb3, bb4, bb5), close < midline)",
            "ready_sell": "And(trend_line > 89, Filter(trend_line > 89, 15), close > midline)",
            "market_health": "CSMean(trend_line > 20)",
        }, params={"trend_period": self.params["trend_period"]})

    def act(self, state: PolicyState) -> StockSelection:
        p = self.params

        if state.context is None:
            return StockSelection(signals=[])

        date = state.date
        # 获取上一个交易日（T-1 信号）
        last_date = self._get_last_trade_date(state)
        if last_date is None:
            return StockSelection(signals=[])

        # 从预计算缓存中获取指标数据
        indicators = self._get_indicators(state, last_date)
        if indicators is None or indicators.empty:
            return StockSelection(signals=[])

        # 市场健康度
        market_health = self._calc_market_health(indicators)

        if p["enable_market_health_entry"] and market_health < p["market_health_entry"]:
            if p["log_trades"]:
                logger.info(f"[{date}] Market health {market_health:.2%} < {p['market_health_entry']}, skip")
            return StockSelection(signals=[])

        if "buy_signal" not in indicators.columns:
            return StockSelection(signals=[])

        select_df = indicators[indicators["buy_signal"] == 1]
        raw_df = select_df.copy()

        if select_df.empty:
            return StockSelection(signals=[])

        # RSI 过滤
        if p["enable_rsi_filter"] and "rsi14" in select_df.columns:
            select_df = select_df[
                (select_df["rsi14"] >= p["rsi_min"])
                & (select_df["rsi14"] <= p["rsi_max"])
            ]

        # KDJ 过滤
        if p["enable_kdj_filter"] and all(c in select_df.columns for c in ["kdj_k", "kdj_d", "kdj_j"]):
            select_df = select_df[
                (select_df["kdj_k"] >= select_df["kdj_d"] - p["kdj_bias"])
                & (select_df["kdj_j"] < p["kdj_j_max"])
            ]

        # BBI 过滤
        if p["enable_bbi_filter"] and "bbi" in select_df.columns:
            select_df = select_df[select_df["close"] >= select_df["bbi"] * p["bbi_floor_ratio"]]

        # 趋势护栏
        if p["enable_trend_guard"]:
            if p["trend_guard_mode"] == "ema" and "ema12" in select_df.columns and "ema26" in select_df.columns:
                spread = select_df["ema12"] / select_df["ema26"] - 1.0
                select_df = select_df[spread >= p["trend_guard_min_ema_spread"]]
            elif p["trend_guard_mode"] == "trend_line" and "trend_line" in select_df.columns:
                select_df = select_df[select_df["trend_line"] >= 20]

        # 放宽筛选
        if select_df.empty and p["enable_relax_entry_filters"]:
            select_df = raw_df.copy()
            if "rsi14" in select_df.columns:
                select_df = select_df[
                    (select_df["rsi14"] >= p["relax_rsi_min"])
                    & (select_df["rsi14"] <= p["relax_rsi_max"])
                ]

        if select_df.empty:
            return StockSelection(signals=[])

        # 打分
        score = pd.Series(0.0, index=select_df.index)
        if "trend_line" in select_df.columns:
            score += select_df["trend_line"] * p["score_trend_weight"]
        if p["enable_ema_score"] and "ema_diff" in select_df.columns:
            score += select_df["ema_diff"] * p["score_ema_weight"]
        if p["enable_rsi_score"] and "rsi14" in select_df.columns:
            score -= (select_df["rsi14"] - p["score_rsi_center"]).abs() * p["score_rsi_penalty"]
        if p["enable_kdj_score"] and "kdj_j" in select_df.columns:
            score -= (select_df["kdj_j"] - p["score_kdj_center"]).abs() * p["score_kdj_penalty"]

        select_df = select_df.assign(_score=score)
        if p.get("legacy_preserve_signal_order", False):
            select_df["_score"] = 0.0
        else:
            select_df = select_df.sort_values("_score", ascending=False)

        signals = []
        for idx in select_df.index:
            symbol = idx if isinstance(idx, str) else str(idx)
            signals.append(Signal(
                symbol=symbol,
                score=float(select_df.loc[idx, "_score"]),
                reason="shuijiao_buy",
                signal_date=last_date,
            ))

        if p["log_trades"]:
            logger.info(f"[{date}] Shuijiao select: {len(signals)} stocks, health={market_health:.2%}")

        return StockSelection(signals=signals)

    def _get_last_trade_date(self, state: PolicyState) -> Optional[str]:
        """获取上一个交易日"""
        if state.context is not None:
            try:
                dates = state.context.trade_dates
                idx = dates.index(state.date) if state.date in dates else -1
                if idx > 0:
                    return dates[idx - 1]
                if idx == 0 and self.params.get("legacy_wrap_first_signal", False) and dates:
                    return dates[-1]
            except (ValueError, AttributeError):
                pass
        return None

    def _get_indicators(self, state: PolicyState, date: str) -> Optional[pd.DataFrame]:
        """获取预计算指标（优先从缓存读，否则从 FactorRuntime 读取）"""
        # 从 context 缓存读取
        cache = getattr(state.context, '_shuijiao_indicators', None)
        if cache is not None:
            # 返回指定日期的横截面数据
            rows = []
            for sym, df in cache.items():
                try:
                    row = df.loc[pd.Timestamp(date)]
                    if isinstance(row, pd.Series):
                        row_dict = row.to_dict()
                        row_dict["instrument"] = sym
                        rows.append(row_dict)
                except (KeyError, IndexError):
                    continue
            if rows:
                result = pd.DataFrame(rows).set_index("instrument")
                return result

        runtime = getattr(state.context, "factor_runtime", None)
        if runtime is not None:
            required = ["close", "midline", "trend_line", "buy_signal", "ready_sell", "market_health"]
            missing = sorted(set(required) - set(runtime.values))
            if missing:
                raise RuntimeError(f"Shuijiao FactorRuntime missing required factors: {missing}")
            try:
                return runtime.get_cross_section(required, date)
            except KeyError as exc:
                raise RuntimeError(f"Shuijiao indicators are unavailable for date {date}") from exc

        raise RuntimeError("Shuijiao requires precomputed indicators or BacktestContext.factor_runtime")

    def _calc_market_health(self, day_data: pd.DataFrame) -> float:
        """用 trend_line > 20 占比衡量市场强弱"""
        if day_data.empty or "trend_line" not in day_data.columns:
            return 1.0
        total = len(day_data)
        if total == 0:
            return 1.0
        return float((day_data["trend_line"] > 20).sum()) / total


# ============================================================
# 调仓策略
# ============================================================

class ShuijiaoRebalance(RebalanceStrategy):
    """水饺调仓策略：等权分配 + 动态仓位管理"""

    DEFAULT_PARAMS = {
        "max_positions_base": 30,
        "max_positions_weak": 12,
        "max_positions_very_weak": 6,
        "market_health_weak": 0.22,
        "market_health_very_weak": 0.16,
        "enable_dynamic_positions": True,
        "enable_market_health_entry": True,
        "market_health_entry": 0.22,
        "cash_use_ratio": 0.99,
        "min_positions_floor": 0,
        "enable_hard_risk_off": True,
        "hard_risk_off_dd": -0.30,
        "log_trades": True,
    }

    def __init__(self, **kwargs):
        self.params = dict(self.DEFAULT_PARAMS)
        self.params.update(kwargs)

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def act(self, state: PolicyState, selection: StockSelection) -> WeightAllocation:
        p = self.params
        signals = selection.signals
        if not signals:
            return WeightAllocation(weights={})

        # 市场健康度
        market_health = self._calc_market_health(state)

        # 动态仓位
        max_positions = p["max_positions_base"]
        if p["enable_dynamic_positions"]:
            if market_health < p["market_health_very_weak"]:
                max_positions = p["max_positions_very_weak"]
            elif market_health < p["market_health_weak"]:
                max_positions = p["max_positions_weak"]

        # 组合熔断
        hard_risk_off = p["enable_hard_risk_off"] and self._calc_drawdown(state) < p["hard_risk_off_dd"]

        if hard_risk_off:
            return WeightAllocation(weights={})

        current_holdings = {
            sym for sym, pos in state.positions.items()
            if pos.quantity > 0
        }
        planned_sell_symbols = set(state.extra.get("planned_sell_symbols", set()))
        current_holdings_for_slots = current_holdings - planned_sell_symbols
        available_slots = max(0, max_positions - len(current_holdings_for_slots))
        if available_slots <= 0:
            return WeightAllocation(weights={})

        selected = [s for s in signals if s.symbol not in current_holdings][:available_slots]
        if not selected:
            return WeightAllocation(weights={})

        weight = 1.0 / len(selected)
        return WeightAllocation(weights={s.symbol: weight for s in selected})

    def _calc_market_health(self, state: PolicyState) -> float:
        """计算市场健康度"""
        runtime = getattr(state.context, "factor_runtime", None) if state.context is not None else None
        if runtime is not None and "market_health" in runtime.values:
            try:
                return float(runtime.get_matrix("market_health")[runtime.panel.date_index(state.date)])
            except Exception:
                pass
        if state.context is None or state.context.exchange.quote is None:
            return 1.0
        try:
            day_data = state.context.exchange.quote.loc[pd.Timestamp(state.date)]
            if "trend_line" not in day_data.columns:
                return 1.0
            total = len(day_data)
            if total == 0:
                return 1.0
            return float((day_data["trend_line"] > 20).sum()) / total
        except Exception:
            return 1.0

    def _calc_drawdown(self, state: PolicyState) -> float:
        """计算组合回撤"""
        if state.account is None:
            return 0.0
        if state.account.total_value <= 0:
            return 0.0
        # 简化：用累计收益率作为回撤估计
        return state.account.cumulative_return


# ============================================================
# 执行策略
# ============================================================

class ShuijiaoExecution(ExecutionStrategy):
    """水饺执行策略：止盈/止损/移动止盈/信号卖出/动量转弱"""

    DEFAULT_PARAMS = {
        "max_profit_pct": 0.22,
        "max_loss_pct": -0.09,
        "max_loss_days": 0,
        "trailing_stop_pct": 0.0,
        "trailing_activate_profit": 0.05,
        "enable_momentum_sell": True,
        "enable_hard_risk_off": True,
        "hard_risk_off_dd": -0.30,
        "cash_use_ratio": 0.99,
        "deal_price": "open",
        "legacy_full_cash_per_buy": False,
        "legacy_equal_cash_when_empty": False,
        "legacy_reuse_sell_cash_for_buys": False,
        "commission_rate": 0.0003,
        "min_commission": 5.0,
        "stamp_tax_rate": 0.0005,
        "transfer_fee_rate": 0.00002,
        "skip_buy_limit_up": False,
        "log_trades": True,
    }

    def __init__(self, **kwargs):
        self.params = dict(self.DEFAULT_PARAMS)
        self.params.update(kwargs)
        self._peak_prices: Dict[str, float] = {}

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def act(self, state: PolicyState, allocation: WeightAllocation) -> OrderList:
        p = self.params
        orders = []
        date = state.date

        # 先处理卖出
        for sym, pos, cur_price, pnl_pct, reason in self._iter_sell_decisions(state, mutate=True):
            if reason:
                orders.append(Order(
                    symbol=sym, action=OrderAction.SELL,
                    price=cur_price, quantity=pos.quantity,
                    date=date, reason=reason,
                ))
                if p["log_trades"]:
                    logger.info(f"[{reason}] SELL {sym} @ {cur_price:.2f}, PnL={pnl_pct:.1%}")

        # 买入
        current_holdings = set(
            sym for sym, pos in state.positions.items()
            if pos.quantity > 0 and sym not in [o.symbol for o in orders]
        )

        buyable = [
            (sym, w) for sym, w in allocation.weights.items()
            if sym not in current_holdings
        ]

        if buyable and state.account and state.account.total_value > 0:
            cash_base = state.account.cash
            if p.get("legacy_reuse_sell_cash_for_buys", False):
                cash_base += sum(
                    self._estimate_sell_proceeds(order.symbol, order.price, order.quantity)
                    for order in orders
                    if order.action == OrderAction.SELL
                )
            cash = cash_base * p["cash_use_ratio"]
            use_full_cash = p.get("legacy_full_cash_per_buy", False)
            if use_full_cash and p.get("legacy_equal_cash_when_empty", False) and not state.positions:
                use_full_cash = False
            cash_per_stock = cash if use_full_cash else cash / len(buyable)

            for sym, w in buyable:
                price = self._get_price(state, sym)
                if price is None or price <= 0:
                    continue
                if p.get("skip_buy_limit_up", False) and self._is_limit_up(state, sym):
                    continue
                qty = int(cash_per_stock / price)
                qty = (qty // 100) * 100
                if qty > 0:
                    orders.append(Order(
                        symbol=sym, action=OrderAction.BUY,
                        price=price, quantity=qty,
                        date=date, reason="shuijiao_buy",
                    ))

        return OrderList(orders=orders)

    def _estimate_sell_proceeds(self, symbol: str, price: float, quantity: int) -> float:
        amount = price * quantity
        commission = max(amount * self.params["commission_rate"], self.params["min_commission"])
        stamp_tax = amount * self.params["stamp_tax_rate"]
        transfer_fee = amount * self.params["transfer_fee_rate"] if symbol.startswith("SH") else 0.0
        return amount - commission - stamp_tax - transfer_fee

    def preview_sell_symbols(self, state: PolicyState) -> List[str]:
        """Return symbols likely to be sold today so rebalance can free slots first."""
        return [sym for sym, _, _, _, reason in self._iter_sell_decisions(state, mutate=False) if reason]

    def _iter_sell_decisions(self, state: PolicyState, mutate: bool = True):
        p = self.params
        for sym, pos in list(state.positions.items()):
            if pos.quantity <= 0:
                continue

            cur_price = self._get_price(state, sym)
            if cur_price is None or cur_price <= 0 or pos.avg_cost <= 0:
                continue

            pnl_pct = cur_price / pos.avg_cost - 1
            reason = None

            if self._check_ready_sell(state, sym):
                reason = "signal_sell"

            if reason is None and pnl_pct > p["max_profit_pct"]:
                reason = f"stop_profit_{pnl_pct:.1%}"

            if reason is None and pnl_pct < p["max_loss_pct"]:
                reason = f"stop_loss_{pnl_pct:.1%}"

            if reason is None and p["max_loss_days"] > 0 and pnl_pct <= 0 and pos.holding_days >= p["max_loss_days"]:
                reason = f"time_stop_{pos.holding_days}d"

            if reason is None and p["trailing_stop_pct"] > 0:
                peak = self._peak_prices.get(sym, cur_price)
                if cur_price > peak:
                    peak = cur_price
                    if mutate:
                        self._peak_prices[sym] = peak
                peak_gain = peak / pos.avg_cost - 1
                pullback = cur_price / peak - 1
                if peak_gain >= p["trailing_activate_profit"] and pullback <= -p["trailing_stop_pct"]:
                    reason = f"trailing_stop_{pullback:.1%}"

            if reason is None and p["enable_momentum_sell"] and pnl_pct <= 0:
                if self._check_momentum_weak(state, sym):
                    reason = "momentum_weak"

            yield sym, pos, cur_price, pnl_pct, reason

    def _get_price(self, state: PolicyState, symbol: str) -> Optional[float]:
        """获取策略用于估算订单数量和止盈止损的价格"""
        if state.context is None or state.context.exchange.quote is None:
            return None
        try:
            field = f"${self.params.get('deal_price', 'open').lstrip('$')}"
            price = state.context.exchange.quote.loc[(pd.Timestamp(state.date), symbol), field]
            return float(price) if not pd.isna(price) else None
        except Exception:
            return None

    def _is_limit_up(self, state: PolicyState, symbol: str) -> bool:
        if state.market_data is not None and not getattr(state.market_data, "empty", True):
            try:
                row = state.market_data.loc[symbol]
                if "limit_up" in row:
                    return bool(row["limit_up"] == 1)
            except Exception:
                pass
        if state.context is not None and getattr(state.context, "exchange", None) is not None:
            try:
                return bool(state.context.exchange.is_limit_up(symbol, state.date))
            except Exception:
                pass
        return False

    def _check_ready_sell(self, state: PolicyState, symbol: str) -> bool:
        """检查信号卖出"""
        if state.context is None:
            return False
        runtime = getattr(state.context, "factor_runtime", None)
        if runtime is not None and "ready_sell" in runtime.values:
            try:
                row = runtime.get_cross_section(["ready_sell"], state.date)
                if symbol in row.index:
                    return bool(row.loc[symbol, "ready_sell"])
            except Exception:
                pass
        try:
            day_data = state.context.exchange.quote.loc[pd.Timestamp(state.date)]
            if isinstance(day_data, pd.Series):
                return False
            if symbol in day_data.index and "ready_sell" in day_data.columns:
                return bool(day_data.loc[symbol, "ready_sell"])
        except Exception:
            pass
        return False

    def _check_momentum_weak(self, state: PolicyState, symbol: str) -> bool:
        """检查动量转弱：EMA12 < EMA26 且 KDJ_K < KDJ_D"""
        if state.context is None:
            return False
        try:
            day_data = state.context.exchange.quote.loc[pd.Timestamp(state.date)]
            if isinstance(day_data, pd.Series) or symbol not in day_data.index:
                return False
            row = day_data.loc[symbol]
            ema_bear = False
            kdj_bear = False
            if "ema12" in row and "ema26" in row:
                ema_bear = float(row["ema12"]) < float(row["ema26"])
            if "kdj_k" in row and "kdj_d" in row:
                kdj_bear = float(row["kdj_k"]) < float(row["kdj_d"])
            return ema_bear and kdj_bear
        except Exception:
            return False


# ============================================================
# 工厂函数
# ============================================================

def create_shuijiao_strategy(**kwargs) -> CompositeStrategy:
    """创建水饺策略"""
    return CompositeStrategy(
        selector=ShuijiaoSelector(**kwargs),
        rebalance=ShuijiaoRebalance(**kwargs),
        execution=ShuijiaoExecution(**kwargs),
    )
