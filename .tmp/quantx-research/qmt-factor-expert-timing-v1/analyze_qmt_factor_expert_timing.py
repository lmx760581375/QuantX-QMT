from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantx.core.data.qlib_reader import QlibBinReader
from quantx.tools.run_research import _build_universe


TOPKS = (10, 15, 20, 30)
EXPERTS = (
    "low_vol20",
    "quiet_trend",
    "trend60",
    "trend20_accel",
    "pullback20",
    "near_high20",
    "breakout20",
    "reversal5",
    "volume_surge",
    "amount_leader",
    "vwap_reclaim",
    "close_strength",
    "range_expansion",
    "liquidity_dispersion_leader",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    parser.add_argument("--raw-dir", default="data/raw/qmt/stocks")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--load-start", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--history-window", type=int, default=24)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    panel = build_panel(Path(args.provider), Path(args.raw_dir), args.start, args.end, args.load_start, args.horizon, args.sample_step)
    detail = evaluate_experts(panel, args.history_window)
    result = summarize(detail)
    result["config"] = {
        "provider": args.provider,
        "raw_dir": args.raw_dir,
        "start": args.start,
        "end": args.end,
        "load_start": args.load_start,
        "horizon": args.horizon,
        "sample_step": args.sample_step,
        "history_window": args.history_window,
        "panel_rows": int(len(panel)),
        "sessions": int(panel["session"].nunique()) if not panel.empty else 0,
        "experts": list(EXPERTS),
        "causality": "Expert scores use QMT/qlib daily OHLCV/VWAP/raw amount/pct_chg/is_st through completed T only. Labels use T+1 open to T+6 open. Causal selectors use only expert session results whose exit_date <= current session.",
        "scope": "QMT retrievable daily market data only; no news, announcements, LHB, ETF flow, northbound, financing, or event/message data.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def build_panel(provider: Path, raw_dir: Path, start: str, end: str, load_start: str, horizon: int, sample_step: int) -> pd.DataFrame:
    universe = _build_universe(provider, {"universe": "all_mainboard"})
    symbols = tuple(universe.all_symbols)
    reader = QlibBinReader(provider)
    calendar = reader.calendar(load_start, end)
    quotes = reader.features(symbols, ["$open", "$high", "$low", "$close", "$volume", "$vwap"], load_start, end)
    frames = {field: quotes[field].unstack("instrument").reindex(index=calendar, columns=symbols).astype(float) for field in quotes.columns}
    raw = load_raw_fields(raw_dir, symbols, calendar)
    scores = build_expert_scores(frames, raw)
    dates = [pd.Timestamp(day).strftime("%Y-%m-%d") for day in calendar]
    valid_sessions = [day for day in dates if start <= day <= end]
    sampled = set(valid_sessions[::sample_step])
    date_index = {day: idx for idx, day in enumerate(dates)}
    symbols_arr = np.asarray(symbols, dtype=object)
    open_values = frames["$open"].to_numpy(dtype=float, copy=False)
    rows: list[pd.DataFrame] = []
    for session in valid_sessions:
        if session not in sampled:
            continue
        idx = date_index[session]
        entry_idx = idx + 1
        exit_idx = entry_idx + horizon
        if idx < 130 or exit_idx >= len(open_values):
            continue
        raw5 = open_values[exit_idx] / open_values[entry_idx] - 1.0
        finite = np.isfinite(raw5)
        if finite.sum() < 500:
            continue
        bench = float(np.nanmean(raw5[finite]))
        hist = scores["history"].iloc[idx].to_numpy(dtype=float, copy=False)
        not_st = scores["not_st"].iloc[idx].to_numpy(dtype=float, copy=False) > 0.5
        tradable = finite & (hist >= 130) & not_st
        if tradable.sum() < 500:
            continue
        block: dict[str, Any] = {
            "session": np.full(int(tradable.sum()), session, dtype=object),
            "exit_date": np.full(int(tradable.sum()), dates[exit_idx], dtype=object),
            "year": np.full(int(tradable.sum()), int(session[:4]), dtype=int),
            "instrument": symbols_arr[tradable],
            "label5": raw5[tradable] - bench,
            "raw5": raw5[tradable],
        }
        for name in EXPERTS:
            block[name] = scores[name].iloc[idx].to_numpy(dtype=float, copy=False)[tradable]
        frame = pd.DataFrame(block)
        frame[list(EXPERTS)] = frame[list(EXPERTS)].replace([np.inf, -np.inf], np.nan).fillna(0.5)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_raw_fields(raw_dir: Path, symbols: tuple[str, ...], calendar: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    amount = pd.DataFrame(index=calendar, columns=symbols, dtype=float)
    pct_chg = pd.DataFrame(index=calendar, columns=symbols, dtype=float)
    is_st = pd.DataFrame(index=calendar, columns=symbols, dtype=float)
    for symbol in symbols:
        path = raw_dir / f"{symbol}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=lambda c: c in {"date", "amount", "pct_chg", "is_st"})
        if df.empty or "date" not in df:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").reindex(calendar)
        if "amount" in df:
            amount[symbol] = pd.to_numeric(df["amount"], errors="coerce")
        if "pct_chg" in df:
            pct_chg[symbol] = pd.to_numeric(df["pct_chg"], errors="coerce")
        if "is_st" in df:
            is_st[symbol] = pd.to_numeric(df["is_st"], errors="coerce")
    return {"amount": amount, "pct_chg": pct_chg, "is_st": is_st.fillna(0.0)}


def build_expert_scores(frames: dict[str, pd.DataFrame], raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    open_ = frames["$open"]
    high = frames["$high"]
    low = frames["$low"]
    close = frames["$close"]
    volume = frames["$volume"]
    vwap = frames["$vwap"]
    amount = raw["amount"].where(raw["amount"] > 0, vwap * volume)
    is_st = raw["is_st"]
    ret1 = close / close.shift(1) - 1.0
    ret5 = close / close.shift(5) - 1.0
    ret20 = close / close.shift(20) - 1.0
    ret60 = close / close.shift(60) - 1.0
    vol20 = ret1.rolling(20, min_periods=10).std()
    ma20 = close.rolling(20, min_periods=10).mean()
    high20 = high.rolling(20, min_periods=10).max()
    low20 = low.rolling(20, min_periods=10).min()
    range_pct = high / low - 1.0
    close_loc = (close - low) / (high - low)
    vwap_dist = close / vwap - 1.0
    amt20 = amount.rolling(20, min_periods=10).mean()
    amt_ratio20 = amount / amt20
    amount_rank = np.log1p(amount).rank(axis=1, pct=True)
    market_amount_disp = amount_rank.std(axis=1)
    market_disp = ret20.std(axis=1)
    mkt_amt_disp_frame = repeat_market(market_amount_disp, close)
    mkt_disp_frame = repeat_market(market_disp, close)
    out = {
        "low_vol20": (-vol20).rank(axis=1, pct=True),
        "quiet_trend": (ret20 / (vol20 + 1e-12)).replace([np.inf, -np.inf], np.nan).rank(axis=1, pct=True),
        "trend60": ret60.rank(axis=1, pct=True),
        "trend20_accel": (ret20 - ret60).rank(axis=1, pct=True),
        "pullback20": ((close / high20 - 1.0) + 0.35 * ret60).rank(axis=1, pct=True),
        "near_high20": (close / high20 - 1.0).rank(axis=1, pct=True),
        "breakout20": (close / high20.shift(1) - 1.0).rank(axis=1, pct=True),
        "reversal5": (-ret5).rank(axis=1, pct=True),
        "volume_surge": amt_ratio20.rank(axis=1, pct=True),
        "amount_leader": amount_rank,
        "vwap_reclaim": vwap_dist.rank(axis=1, pct=True),
        "close_strength": close_loc.rank(axis=1, pct=True),
        "range_expansion": range_pct.rank(axis=1, pct=True),
        "liquidity_dispersion_leader": (amount_rank * mkt_amt_disp_frame + ret20.rank(axis=1, pct=True) * mkt_disp_frame).rank(axis=1, pct=True),
        "history": pd.DataFrame(np.cumsum(np.isfinite(open_.to_numpy()) & (open_.to_numpy() > 0), axis=0), index=open_.index, columns=open_.columns),
        "not_st": 1.0 - is_st,
    }
    return out


def repeat_market(series: pd.Series, template: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.repeat(series.to_numpy(dtype=float)[:, None], len(template.columns), axis=1), index=template.index, columns=template.columns)


def evaluate_experts(panel: pd.DataFrame, history_window: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    sessions = sorted(panel["session"].unique())
    for session in sessions:
        group = panel[panel["session"] == session]
        available = [row for row in history if row["exit_date"] <= session]
        selected_names = causal_selector_names(available, history_window)
        variants: list[tuple[str, str | None]] = [(name, name) for name in EXPERTS]
        variants.extend((name, None) for name in selected_names)
        session_results: dict[tuple[str, int], dict[str, Any]] = {}
        for variant, score_col in variants:
            if score_col is None:
                score = selector_score(group, variant, available, history_window)
                ranked = group.assign(_score=score).sort_values(["_score", "instrument"], ascending=[False, True])
            else:
                ranked = group.sort_values([score_col, "instrument"], ascending=[False, True])
            for topk in TOPKS:
                if len(ranked) < topk:
                    continue
                selected = ranked.head(topk)
                row = {
                    "session": session,
                    "exit_date": str(selected["exit_date"].iloc[0]),
                    "year": int(str(session)[:4]),
                    "variant": variant,
                    "topk": int(topk),
                    "mean_label5": float(selected["label5"].mean()),
                    "mean_raw5": float(selected["raw5"].mean()),
                    "positive_label5": bool(selected["label5"].mean() > 0),
                    "count": int(len(selected)),
                }
                rows.append(row)
                session_results[(variant, topk)] = row
        # Store only fixed expert outcomes for future causal selection.
        for expert in EXPERTS:
            for topk in TOPKS:
                row = session_results.get((expert, topk))
                if row is not None:
                    history.append(row)
    return pd.DataFrame(rows)


def causal_selector_names(available: list[dict[str, Any]], history_window: int) -> list[str]:
    if len(available) < len(EXPERTS) * 6:
        return ["selector_prior_uniform"]
    return [
        "selector_expanding_best",
        "selector_roll_best",
        "selector_expanding_softmax",
        "selector_roll_softmax",
        "selector_positive_blend",
    ]


def selector_score(group: pd.DataFrame, variant: str, available: list[dict[str, Any]], history_window: int) -> pd.Series:
    topk = 20
    if variant == "selector_prior_uniform" or not available:
        weights = {expert: 1.0 / len(EXPERTS) for expert in EXPERTS}
    else:
        hist = pd.DataFrame(available)
        hist = hist[hist["topk"] == topk]
        if variant in {"selector_roll_best", "selector_roll_softmax"}:
            recent_sessions = sorted(hist["session"].unique())[-history_window:]
            hist = hist[hist["session"].isin(recent_sessions)]
        means = hist.groupby("variant")["mean_label5"].mean().reindex(EXPERTS).fillna(0.0)
        if variant in {"selector_expanding_best", "selector_roll_best"}:
            best = str(means.idxmax())
            weights = {expert: 1.0 if expert == best else 0.0 for expert in EXPERTS}
        elif variant in {"selector_expanding_softmax", "selector_roll_softmax"}:
            values = np.exp(np.clip(means.to_numpy(dtype=float) * 80.0, -6, 6))
            values = values / values.sum()
            weights = {expert: float(value) for expert, value in zip(EXPERTS, values)}
        elif variant == "selector_positive_blend":
            values = np.maximum(means.to_numpy(dtype=float), 0.0) + 0.0005
            values = values / values.sum()
            weights = {expert: float(value) for expert, value in zip(EXPERTS, values)}
        else:
            raise ValueError(variant)
    score = pd.Series(0.0, index=group.index)
    for expert, weight in weights.items():
        score = score + float(weight) * group[expert].astype(float)
    return score


def summarize(detail: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"daily_row_count": int(len(detail)), "results": {}}
    if detail.empty:
        return result
    for (variant, topk), group in detail.groupby(["variant", "topk"]):
        key = f"{variant}::top{int(topk)}"
        result["results"][key] = {
            "daily_count": int(len(group)),
            "mean_daily_label5": float(group["mean_label5"].mean()),
            "mean_daily_raw5": float(group["mean_raw5"].mean()),
            "positive_label5_ratio": float(group["positive_label5"].mean()),
            "avg_selected_count": float(group["count"].mean()),
            "by_year": {
                str(int(year)): {
                    "mean_daily_label5": float(yg["mean_label5"].mean()),
                    "mean_daily_raw5": float(yg["mean_raw5"].mean()),
                    "positive_label5_ratio": float(yg["positive_label5"].mean()),
                }
                for year, yg in group.groupby("year")
            },
        }
    return result


def print_summary(result: dict[str, Any], checksum: str) -> None:
    rows = []
    for key, value in result.get("results", {}).items():
        if key.endswith("::top20"):
            rows.append((float(value["mean_daily_label5"]), float(value["mean_daily_raw5"]), key, float(value["positive_label5_ratio"]), value["by_year"]))
    rows.sort(key=lambda item: item[0], reverse=True)
    print(json.dumps({"checksum": checksum, "daily_row_count": result.get("daily_row_count"), "top20_results": rows[:40]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
