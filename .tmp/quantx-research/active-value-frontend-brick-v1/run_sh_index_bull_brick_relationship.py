from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantx.core.factor_runtime import ops as factor_ops


ROOT = Path(".tmp/quantx-research/active-value-frontend-brick-v1")
BASE_SCRIPT = ROOT / "run_active_value_reversal_portfolio_research.py"
REGIME_PATH = Path("data/derived/market_regime/sh000001_daily.parquet")
ST_DAILY_PATH = Path("data/reference/security_state/st_daily.csv")
OUT_JSON = ROOT / "sh_index_bull_brick_relationship.json"
OUT_MD = ROOT / "sh_index_bull_brick_relationship.md"
OUT_CANDIDATES = ROOT / "sh_index_bull_brick_candidates.parquet"
OOS_START = pd.Timestamp("2025-01-02")
END = pd.Timestamp("2026-07-15")


def main() -> int:
    base = load_base_module()
    panel = base.load_panel(base.PANEL_PATH)
    panel = base.add_trade_state(panel)
    panel = add_stock_selection_state(panel)
    panel = attach_regime(panel, REGIME_PATH)
    panel = add_exact_limit_state(panel)
    panel = base.add_forward_returns(panel)
    panel = add_tail_entry_returns(panel, base)
    candidates = build_candidates(panel)
    candidates = apply_historical_st(candidates, ST_DAILY_PATH, REGIME_PATH)
    candidates.to_parquet(OUT_CANDIDATES, index=False)
    report = build_report(candidates, panel)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


def load_base_module():
    spec = importlib.util.spec_from_file_location("active_value_replay_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base research module: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def attach_regime(panel: pd.DataFrame, path: Path) -> pd.DataFrame:
    regime = pd.read_parquet(path)
    regime["date"] = pd.to_datetime(regime["date"])
    columns = [
        "date", "close", "ema20", "ema60", "kdj_j", "active_bull",
        "active_bull_kdj_low", "active_bull_age", "ema20_slope5", "close_to_ema60",
    ]
    missing = [column for column in columns if column not in regime]
    if missing:
        raise ValueError(f"Market regime data missing columns: {missing}")
    out = panel.merge(regime[columns].rename(columns={"date": "datetime"}), on="datetime", how="left")
    return out.rename(columns={
        "close": "sh_close",
        "ema20": "sh_ema20",
        "ema60": "sh_ema60",
        "kdj_j": "sh_kdj_j",
        "active_bull": "sh_active_bull",
        "active_bull_kdj_low": "sh_active_bull_kdj_low",
        "active_bull_age": "sh_active_bull_age",
        "ema20_slope5": "sh_ema20_slope5",
        "close_to_ema60": "sh_close_to_ema60",
    })


def add_stock_selection_state(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel
    stock_j = np.full(len(out), np.nan, dtype=np.float32)
    stock_j_prev_min3 = np.full(len(out), np.nan, dtype=np.float32)
    stock_white = np.full(len(out), np.nan, dtype=np.float32)
    stock_yellow = np.full(len(out), np.nan, dtype=np.float32)
    high_values = out["$high"].to_numpy(dtype=np.float32)
    low_values = out["$low"].to_numpy(dtype=np.float32)
    close_values = out["$close"].to_numpy(dtype=np.float32)
    for positions in out.groupby("instrument", sort=False).indices.values():
        loc = np.asarray(positions, dtype=np.int64)
        close = pd.Series(close_values[loc], dtype="float32")
        j = factor_ops.kdj_j(high_values[loc], low_values[loc], close_values[loc], 9)
        stock_j[loc] = j
        stock_j_prev_min3[loc] = pd.Series(j).shift(1).rolling(3, min_periods=1).min().to_numpy(dtype=np.float32)
        ema10 = close.ewm(span=10, adjust=False, min_periods=10).mean()
        stock_white[loc] = ema10.ewm(span=10, adjust=False, min_periods=10).mean().to_numpy(dtype=np.float32)
        yellow_parts = [close.rolling(window, min_periods=window).mean() for window in (14, 28, 57, 114)]
        stock_yellow[loc] = pd.concat(yellow_parts, axis=1).mean(axis=1, skipna=False).to_numpy(dtype=np.float32)
    out["stock_kdj_j"] = stock_j
    out["stock_kdj_j_prev_min3"] = stock_j_prev_min3
    out["stock_white"] = stock_white
    out["stock_yellow"] = stock_yellow
    grouped = out.groupby("instrument", sort=False)
    out["stock_kdj_j_delta"] = grouped["stock_kdj_j"].diff().astype("float32")
    out["stock_kdj_j_rank_pct"] = (
        out.groupby("datetime")["stock_kdj_j"].rank(pct=True).astype("float32")
    )
    out["stock_kdj_j_delta_rank_pct"] = (
        out.groupby("datetime")["stock_kdj_j_delta"].rank(pct=True).astype("float32")
    )
    out["stock_right_side"] = (out["$close"] > out["stock_yellow"]) & (out["stock_white"] > out["stock_yellow"])
    out["stock_b1_recent_j12"] = out["stock_kdj_j_prev_min3"] <= 12.0
    out["stock_b1_recent_jneg10"] = out["stock_kdj_j_prev_min3"] <= -10.0
    prev_open = grouped["$open"].shift(1)
    previous_body = (out["prev_close"] - prev_open).abs()
    current_body = out["$close"] - out["$open"]
    out["bull_body_cover_ratio"] = current_body.clip(lower=0).div(previous_body.where(previous_body > 1e-9))
    out["strong_red_body"] = (current_body > 0) & (out["bull_body_cover_ratio"] >= (2.0 / 3.0)) & (out["close_pos"] >= 0.70)
    return out


def add_exact_limit_state(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    chinext_20 = out["is_chinext"].fillna(False).astype(bool) & (out["datetime"] >= pd.Timestamp("2020-08-24"))
    high_limit_board = out["is_star"].fillna(False).astype(bool) | chinext_20
    out["limit_up_rate"] = np.where(high_limit_board, 0.20, 0.10)
    out["signal_limit_up"] = out["ret1"] >= (out["limit_up_rate"] - 0.005)
    one_price = (
        out["$high"].sub(out["$low"]).abs().le(1e-6)
        & out["$open"].sub(out["$close"]).abs().le(1e-6)
    )
    out["one_price_limit_up_exact"] = one_price & out["signal_limit_up"]
    grouped = out.groupby("instrument", sort=False)
    out["next_one_price_limit_up_exact"] = grouped["one_price_limit_up_exact"].shift(-1)
    out["next_basic_tradable"] = grouped["basic_tradable"].shift(-1)
    out["next_current_risk_name"] = grouped["is_current_risk_name"].shift(-1)
    out["next_st_like"] = grouped["st_like_limit_history"].shift(-1)
    return out


def add_tail_entry_returns(panel: pd.DataFrame, base) -> pd.DataFrame:
    out = panel
    grouped = out.groupby("instrument", sort=False)
    for horizon in (1, 2, 5, 10):
        exit_open = grouped["$open"].shift(-horizon)
        entry_cost = out["$close"] * (1 + base.BUY_SLIPPAGE) * (1 + base.COMMISSION)
        exit_proceeds = exit_open * (1 - base.SELL_SLIPPAGE) * (1 - base.COMMISSION - base.STAMP_TAX)
        out[f"tail_fwd{horizon}_net"] = (exit_proceeds / entry_cost - 1.0).astype("float32")
    for horizon in (1, 2):
        exit_open = grouped["$open"].shift(-(horizon + 1))
        entry_cost = out["next_open"] * (1 + base.BUY_SLIPPAGE) * (1 + base.COMMISSION)
        exit_proceeds = exit_open * (1 - base.SELL_SLIPPAGE) * (1 - base.COMMISSION - base.STAMP_TAX)
        out[f"open_fwd{horizon}_net"] = (exit_proceeds / entry_cost - 1.0).astype("float32")
    out["overnight_gap"] = (out["next_open"] / out["$close"] - 1.0).astype("float32")
    return out


def build_candidates(panel: pd.DataFrame) -> pd.DataFrame:
    reversal = (panel["frontend_delta_prev"] < 0) & (panel["frontend_delta"] > 0)
    strict_non_st = (
        (~panel["is_current_risk_name"].fillna(False).astype(bool))
        & (~panel["st_like_limit_history"].fillna(False).astype(bool))
    )
    signal = (
        panel["sh_active_bull"].fillna(False).astype(bool)
        & reversal
        & panel["basic_tradable"].fillna(False)
        & strict_non_st
        & (panel["listing_days"] >= 120)
        & (~panel["signal_limit_up"].fillna(False))
        & (panel["amount_rank_pct"] >= 0.35)
        & (panel["amplitude_pct"] < 0.22)
    )
    columns = [
        "instrument", "datetime", "frontend_brick", "frontend_delta", "frontend_delta_prev",
        "ret1", "ret3", "ret5", "vol_ratio", "amplitude_pct", "body_pct", "close_pos",
        "ma5_rel", "ma10_rel", "ma20_rel", "ma50_rel", "amount_rank_pct", "ret1_rank_pct",
        "ret3_rank_pct", "ret5_rank_pct", "brick_rank_pct", "delta_rank_pct",
        "vol_ratio_rank_pct", "amplitude_rank_pct", "is_mainboard", "is_chinext", "is_star",
        "listing_days", "sh_close", "sh_ema20", "sh_ema60", "sh_kdj_j", "sh_active_bull_age",
        "sh_active_bull_kdj_low", "sh_ema20_slope5", "sh_close_to_ema60", "fwd5_net", "fwd10_net",
        "tail_fwd1_net", "tail_fwd2_net", "tail_fwd5_net", "tail_fwd10_net",
        "open_fwd1_net", "open_fwd2_net", "overnight_gap", "stock_kdj_j", "stock_kdj_j_delta",
        "stock_kdj_j_prev_min3", "stock_b1_recent_j12", "stock_b1_recent_jneg10",
        "stock_kdj_j_rank_pct", "stock_kdj_j_delta_rank_pct", "signal_limit_up", "limit_up_rate",
        "stock_white", "stock_yellow", "stock_right_side", "bull_body_cover_ratio", "strong_red_body",
        "next_open", "next_volume", "next_amount", "next_one_price_limit_up_exact",
        "next_basic_tradable", "next_current_risk_name", "next_st_like",
    ]
    out = panel.loc[signal, columns].copy()
    out["reversal_value"] = out["frontend_delta"]
    out["stock_ret4"] = out["ret1"] >= 0.04
    out["active_start2"] = out["sh_active_bull_age"].between(1, 2)
    out["strong_not_limit"] = out["stock_ret4"] & (~out["signal_limit_up"].fillna(False))
    out["entry_tradable"] = (
        out["next_basic_tradable"].eq(True)
        & (~out["next_one_price_limit_up_exact"].eq(True))
        & (~out["next_current_risk_name"].eq(True))
        & (~out["next_st_like"].eq(True))
        & (out["next_open"] > 0)
        & (out["next_volume"] > 0)
        & (out["next_amount"] > 0)
    )
    return out[out["entry_tradable"]].sort_values(["datetime", "instrument"]).reset_index(drop=True)


def apply_historical_st(candidates: pd.DataFrame, path: Path, regime_path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing historical ST data: {path}")
    st = pd.read_csv(path, usecols=["ts_code", "trade_date"], dtype=str)
    st["datetime"] = pd.to_datetime(st["trade_date"])
    code_exchange = st["ts_code"].str.split(".", n=1, expand=True)
    st["instrument"] = code_exchange[1].str.upper() + code_exchange[0]
    st = st[["datetime", "instrument"]].drop_duplicates()

    regime_dates = pd.to_datetime(pd.read_parquet(regime_path, columns=["date"])["date"]).sort_values().unique()
    next_dates = {pd.Timestamp(left): pd.Timestamp(right) for left, right in zip(regime_dates[:-1], regime_dates[1:])}
    out = candidates.copy()
    out["next_datetime"] = out["datetime"].map(next_dates)
    out = out.merge(st.assign(historical_is_st=True), on=["datetime", "instrument"], how="left")
    next_st = st.rename(columns={"datetime": "next_datetime"}).assign(next_historical_is_st=True)
    out = out.merge(next_st, on=["next_datetime", "instrument"], how="left")
    out["historical_st_covered"] = out["datetime"] >= st["datetime"].min()
    out["historical_is_st"] = out["historical_is_st"].eq(True)
    out["next_historical_is_st"] = out["next_historical_is_st"].eq(True)
    return out[
        (~out["historical_is_st"]) & (~out["next_historical_is_st"])
    ].sort_values(["datetime", "instrument"]).reset_index(drop=True)


def build_report(candidates: pd.DataFrame, panel: pd.DataFrame | None) -> dict:
    oos = candidates[(candidates["datetime"] >= OOS_START) & (candidates["datetime"] <= END)].copy()
    if panel is None:
        all_bull_days = pd.read_parquet(REGIME_PATH, columns=["active_bull"])["active_bull"].fillna(False).sum()
    else:
        all_bull_days = panel.drop_duplicates("datetime")["sh_active_bull"].fillna(False).sum()
    report = {
        "definition": {
            "market_regime": "SH000001 close > EMA60 and EMA20 > EMA60",
            "brick_reversal": "frontend_delta_prev < 0 and frontend_delta > 0",
            "filters": "historical non-ST on signal/T+1, listing>=120, signal day not limit-up, T+1 not one-price limit-up",
        },
        "market": {
            "bull_days": int(all_bull_days),
            "regime_source": str(REGIME_PATH),
        },
        "all_candidates": describe(candidates),
        "oos_candidates": describe(oos),
        "oos_ret4": describe(oos[oos["stock_ret4"]]),
        "correlations": correlations(oos),
        "reversal_quantiles": quantile_table(oos, "reversal_value"),
        "ret1_quantiles": quantile_table(oos, "ret1"),
        "joint_reversal_ret1": joint_table(oos),
        "regime_age": regime_age_table(oos),
        "kdj_state": state_table(oos, "sh_active_bull_kdj_low"),
        "stock_kdj_j_quantiles": quantile_table(oos, "stock_kdj_j"),
        "stock_kdj_delta_quantiles": quantile_table(oos, "stock_kdj_j_delta"),
        "strategy_condition_topk": strategy_condition_topk(candidates),
        "yearly": yearly_table(candidates),
        "notes": [
            "Historical Tushare stock_st data is applied on both signal day and T+1 from 2016-08-31 onward.",
            "Rows before 2016-08-31 remain outside exact historical-ST coverage and cannot pass a production gate without another source.",
            "Returns use the already-audited T+1 open entry and T+6/T+11 open exit net-cost labels.",
        ],
    }
    return report


def strategy_condition_topk(frame: pd.DataFrame) -> list[dict]:
    conditions = {
        "bull_reversal": pd.Series(True, index=frame.index),
        "start2_reversal": frame["active_start2"],
        "start2_strong": frame["active_start2"] & frame["strong_not_limit"],
        "start2_strong_j_ge13": frame["active_start2"] & frame["strong_not_limit"] & (frame["stock_kdj_j"] >= 13),
        "start2_strong_j_ge50": frame["active_start2"] & frame["strong_not_limit"] & (frame["stock_kdj_j"] >= 50),
        "start2_strong_jdelta_ge3": frame["active_start2"] & frame["strong_not_limit"] & (frame["stock_kdj_j_delta"] >= 3),
        "start2_strong_j50_jdelta3": (
            frame["active_start2"] & frame["strong_not_limit"]
            & (frame["stock_kdj_j"] >= 50) & (frame["stock_kdj_j_delta"] >= 3)
        ),
        "start2_strong_j50_jdelta3_amount80": (
            frame["active_start2"] & frame["strong_not_limit"]
            & (frame["stock_kdj_j"] >= 50) & (frame["stock_kdj_j_delta"] >= 3)
            & (frame["amount_rank_pct"] >= 0.80)
        ),
        "start2_strong_j50_jdelta3_close70": (
            frame["active_start2"] & frame["strong_not_limit"]
            & (frame["stock_kdj_j"] >= 50) & (frame["stock_kdj_j_delta"] >= 3)
            & (frame["close_pos"] >= 0.70)
        ),
        "bull_strong": frame["strong_not_limit"],
        "bull_strong_right": frame["strong_not_limit"] & frame["stock_right_side"],
        "bull_strong_b1_j12": frame["strong_not_limit"] & frame["stock_b1_recent_j12"],
        "bull_strong_b1_j12_right": (
            frame["strong_not_limit"] & frame["stock_b1_recent_j12"] & frame["stock_right_side"]
        ),
        "bull_strong_b1_j12_right_jdelta10": (
            frame["strong_not_limit"] & frame["stock_b1_recent_j12"] & frame["stock_right_side"]
            & (frame["stock_kdj_j_delta"] >= 10)
        ),
        "bull_strong_b1_j12_right_jdelta10_vol1": (
            frame["strong_not_limit"] & frame["stock_b1_recent_j12"] & frame["stock_right_side"]
            & (frame["stock_kdj_j_delta"] >= 10) & (frame["vol_ratio"] >= 1.0)
        ),
        "bull_strong_b1_j12_right_strong_red": (
            frame["strong_not_limit"] & frame["stock_b1_recent_j12"] & frame["stock_right_side"]
            & frame["strong_red_body"]
        ),
    }
    rows = []
    for name, mask in conditions.items():
        sample = frame[mask.fillna(False)].copy()
        for sample_name, split in (
            ("train", sample[sample["datetime"] < OOS_START]),
            ("oos", sample[(sample["datetime"] >= OOS_START) & (sample["datetime"] <= END)]),
        ):
            for target in (
                "tail_fwd1_net", "tail_fwd2_net", "open_fwd1_net", "open_fwd2_net",
                "tail_fwd5_net", "tail_fwd10_net", "fwd5_net", "fwd10_net",
            ):
                rows.extend(topk_rows(split, condition=name, sample_name=sample_name, target=target))
    return rows


def topk_rows(
    frame: pd.DataFrame,
    *,
    condition: str,
    sample_name: str,
    target: str,
    topks: tuple[int, ...] = (1, 3, 5),
) -> list[dict]:
    data = frame.dropna(subset=["reversal_value", target]).sort_values(
        ["datetime", "reversal_value", "instrument"], ascending=[True, False, True]
    ).copy()
    if data.empty:
        return []
    data["daily_rank"] = data.groupby("datetime").cumcount() + 1
    pool = data.groupby("datetime")[target].mean()
    rows = []
    for topk in topks:
        selected = data[data["daily_rank"] <= topk]
        daily = selected.groupby("datetime")[target].mean()
        aligned = pd.concat({"selected": daily, "pool": pool}, axis=1).dropna()
        yearly = selected.groupby(selected["datetime"].dt.year)[target].mean()
        rows.append({
            "condition": condition,
            "sample": sample_name,
            "target": target,
            "topk": topk,
            "rows": int(len(selected)),
            "dates": int(selected["datetime"].nunique()),
            "mean": float(selected[target].mean()),
            "win": float((selected[target] > 0).mean()),
            "daily_win": float((daily > 0).mean()),
            "pool_excess": float((aligned["selected"] - aligned["pool"]).mean()),
            "yearly": {str(int(year)): float(value) for year, value in yearly.items()},
        })
    return rows


def describe(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "dates": int(frame["datetime"].nunique()) if len(frame) else 0,
        "fwd5": return_stats(frame, "fwd5_net"),
        "fwd10": return_stats(frame, "fwd10_net"),
    }


def return_stats(frame: pd.DataFrame, target: str) -> dict:
    values = pd.to_numeric(frame.get(target), errors="coerce").dropna()
    return {
        "rows": int(len(values)),
        "mean": float(values.mean()) if len(values) else None,
        "median": float(values.median()) if len(values) else None,
        "win": float((values > 0).mean()) if len(values) else None,
        "q10": float(values.quantile(0.10)) if len(values) else None,
        "q90": float(values.quantile(0.90)) if len(values) else None,
    }


def correlations(frame: pd.DataFrame) -> dict:
    columns = ["reversal_value", "frontend_brick", "ret1", "ret3", "ret5", "amount_rank_pct"]
    return {
        feature: {
            "pearson_fwd5": safe_corr(frame[feature], frame["fwd5_net"], method="pearson"),
            "spearman_fwd5": safe_corr(frame[feature], frame["fwd5_net"], method="spearman"),
            "pearson_fwd10": safe_corr(frame[feature], frame["fwd10_net"], method="pearson"),
            "spearman_fwd10": safe_corr(frame[feature], frame["fwd10_net"], method="spearman"),
        }
        for feature in columns
    }


def quantile_table(frame: pd.DataFrame, feature: str) -> list[dict]:
    data = frame.dropna(subset=[feature, "fwd5_net", "fwd10_net"]).copy()
    if data.empty or data[feature].nunique() < 5:
        return []
    data["bucket"] = pd.qcut(data[feature], 5, labels=False, duplicates="drop") + 1
    rows = []
    for bucket, group in data.groupby("bucket"):
        rows.append({
            "bucket": int(bucket),
            "rows": int(len(group)),
            "feature_mean": float(group[feature].mean()),
            "fwd5": return_stats(group, "fwd5_net"),
            "fwd10": return_stats(group, "fwd10_net"),
        })
    return rows


def joint_table(frame: pd.DataFrame) -> list[dict]:
    data = frame.dropna(subset=["reversal_value", "ret1", "fwd10_net"]).copy()
    if data.empty:
        return []
    data["reversal_group"] = pd.qcut(data["reversal_value"], 3, labels=["low", "mid", "high"], duplicates="drop")
    data["ret_group"] = pd.cut(data["ret1"], [-np.inf, 0.02, 0.04, np.inf], labels=["lt2", "2to4", "ge4"])
    return [
        {
            "reversal_group": str(reversal),
            "ret_group": str(ret_group),
            "rows": int(len(group)),
            "fwd10": return_stats(group, "fwd10_net"),
        }
        for (reversal, ret_group), group in data.groupby(["reversal_group", "ret_group"], observed=True)
    ]


def regime_age_table(frame: pd.DataFrame) -> list[dict]:
    data = frame.copy()
    data["age_group"] = pd.cut(data["sh_active_bull_age"], [0, 5, 20, 60, np.inf], labels=["1-5", "6-20", "21-60", "61+"])
    return [
        {"age_group": str(label), "rows": int(len(group)), "fwd10": return_stats(group, "fwd10_net")}
        for label, group in data.groupby("age_group", observed=True)
    ]


def state_table(frame: pd.DataFrame, column: str) -> list[dict]:
    return [
        {"state": bool(state), "rows": int(len(group)), "fwd10": return_stats(group, "fwd10_net")}
        for state, group in frame.groupby(frame[column].fillna(False).astype(bool))
    ]


def yearly_table(frame: pd.DataFrame) -> dict:
    return {
        str(year): describe(group)
        for year, group in frame.groupby(pd.to_datetime(frame["datetime"]).dt.year)
    }


def safe_corr(left, right, *, method: str) -> float | None:
    data = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3 or data["left"].nunique() < 2 or data["right"].nunique() < 2:
        return None
    return float(data["left"].corr(data["right"], method=method))


def render_markdown(report: dict) -> str:
    lines = [
        "# SH000001 Bull-Regime Brick Reversal Relationship",
        "",
        f"- Market regime: {report['definition']['market_regime']}",
        f"- Brick reversal: {report['definition']['brick_reversal']}",
        f"- Filters: {report['definition']['filters']}",
        f"- Bull days: {report['market']['bull_days']}",
        "",
        "## Candidate Summary",
        "",
        f"- All: `{json.dumps(report['all_candidates'], ensure_ascii=False)}`",
        f"- OOS: `{json.dumps(report['oos_candidates'], ensure_ascii=False)}`",
        f"- OOS ret1>=4%: `{json.dumps(report['oos_ret4'], ensure_ascii=False)}`",
        "",
        "## Correlations",
        "",
    ]
    for feature, payload in report["correlations"].items():
        lines.append(f"- {feature}: `{json.dumps(payload, ensure_ascii=False)}`")
    for title, key in [
        ("Reversal Quantiles", "reversal_quantiles"),
        ("Ret1 Quantiles", "ret1_quantiles"),
        ("Joint Reversal x Ret1", "joint_reversal_ret1"),
        ("Regime Age", "regime_age"),
        ("KDJ State", "kdj_state"),
        ("Stock KDJ J Quantiles", "stock_kdj_j_quantiles"),
        ("Stock KDJ J Delta Quantiles", "stock_kdj_delta_quantiles"),
        ("Strategy Condition TopK", "strategy_condition_topk"),
    ]:
        lines.extend(["", f"## {title}", ""])
        for row in report[key]:
            lines.append(f"- `{json.dumps(row, ensure_ascii=False)}`")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in report["notes"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
