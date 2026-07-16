"""Analyze Round-F f3 weak-to-strong buy conversion with active-value states."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
F3_SIGNAL = ROOT / "raw12_enhancement_round_f/signals/f3_good_top12_else_top8.parquet"
F3_ARTIFACT = ROOT / "raw12_enhancement_round_f/artifacts/f3_good_top12_else_top8"
ACTIVE_VALUE = Path("data/derived/active_value/daily.parquet")
OUT = ROOT / "f3_active_value_conversion"


ACTIVE_COLUMNS = [
    "active_core_amount_ret1",
    "active_core_amount_ret2",
    "active_core_amount_above_ma10",
    "active_core_amount_strong_up_day",
    "active_tradable_amount_ret1",
    "active_tradable_amount_above_ma10",
    "right_side_ratio",
    "right_side_core_ratio",
    "right_side_amount_ratio",
    "right_side_core_amount_ratio",
    "right_side_ratio_ret1",
    "right_side_core_ratio_ret1",
    "right_side_amount_ratio_ret1",
    "breadth_bull",
    "breadth_bull_start",
    "breadth_bull_age",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates()
    active = load_active_value()
    signals = pd.read_parquet(F3_SIGNAL)
    selected = build_selected(candidates, signals, active)
    buys = load_buys()
    closed = load_closed_positions()
    frame = attach_trade_outcomes(selected, buys, closed)

    frame.to_parquet(OUT / "f3_selected_with_active_value.parquet", index=False)
    summarize_buy_conversion(frame).to_csv(OUT / "buy_conversion_by_state.csv", index=False)
    summarize_active_bins(frame).to_csv(OUT / "buy_outcome_by_active_bins.csv", index=False)
    summarize_label_bins(frame).to_csv(OUT / "candidate_label_by_active_bins.csv", index=False)
    write_report(frame)


def load_candidates() -> pd.DataFrame:
    path = ROOT / "run_market_state_ranking_round_c.py"
    spec = importlib.util.spec_from_file_location("round_c_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = module.load_enriched_candidates(ROOT).copy()
    frame["signal_time"] = pd.to_datetime(frame["signal_time"]).dt.strftime("%Y-%m-%d")
    frame["entry_session"] = pd.to_datetime(frame["entry_session"]).dt.strftime("%Y-%m-%d")
    frame["instrument"] = frame["instrument"].astype(str)
    frame["base_score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame["base_rank"] = frame.groupby("signal_time")["base_score"].rank(method="first", ascending=False)
    return frame


def load_active_value() -> pd.DataFrame:
    frame = pd.read_parquet(ACTIVE_VALUE).copy()
    frame["signal_time"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    keep = ["signal_time", *[col for col in ACTIVE_COLUMNS if col in frame.columns]]
    return frame[keep]


def build_selected(candidates: pd.DataFrame, signals: pd.DataFrame, active: pd.DataFrame) -> pd.DataFrame:
    signals = signals.copy()
    signals["signal_time"] = pd.to_datetime(signals["signal_time"]).dt.strftime("%Y-%m-%d")
    signals["instrument"] = signals["instrument"].astype(str)
    selected = signals[["signal_time", "instrument"]].drop_duplicates().merge(
        candidates,
        on=["signal_time", "instrument"],
        how="left",
        suffixes=("", "_candidate"),
    )
    selected = selected.merge(active, on="signal_time", how="left")
    selected["state3"] = selected["regime"].astype(str) + "|" + selected["amount_state"].astype(str) + "|" + selected["breadth_state"].astype(str)
    return selected


def load_buys() -> pd.DataFrame:
    trades = pd.DataFrame(json.loads((F3_ARTIFACT / "trades.json").read_text(encoding="utf-8")))
    if trades.empty:
        return pd.DataFrame(columns=["entry_session", "instrument", "buy_value", "buy_quantity"])
    buys = trades[trades["action"].astype(str).str.upper().eq("BUY")].copy()
    buys["entry_session"] = pd.to_datetime(buys["date"]).dt.strftime("%Y-%m-%d")
    buys["instrument"] = buys["symbol"].astype(str)
    buys["buy_value"] = pd.to_numeric(buys["trade_value"], errors="coerce")
    buys["buy_quantity"] = pd.to_numeric(buys["quantity"], errors="coerce")
    return buys.groupby(["entry_session", "instrument"], as_index=False).agg(
        buy_value=("buy_value", "sum"),
        buy_quantity=("buy_quantity", "sum"),
        buy_count=("instrument", "size"),
    )


def load_closed_positions() -> pd.DataFrame:
    closed = pd.DataFrame(json.loads((F3_ARTIFACT / "closed_positions.json").read_text(encoding="utf-8")))
    if closed.empty:
        return pd.DataFrame(columns=["entry_session", "instrument", "net_pnl", "closed_return"])
    closed["entry_session"] = pd.to_datetime(closed["entry_date"]).dt.strftime("%Y-%m-%d")
    closed["instrument"] = closed["symbol"].astype(str)
    closed["net_pnl"] = pd.to_numeric(closed["net_pnl"], errors="coerce")
    closed["return"] = pd.to_numeric(closed["return"], errors="coerce")
    closed["entry_value"] = pd.to_numeric(closed["quantity"], errors="coerce") * pd.to_numeric(closed["entry_price"], errors="coerce")
    grouped = closed.groupby(["entry_session", "instrument"], as_index=False).agg(
        net_pnl=("net_pnl", "sum"),
        entry_value=("entry_value", "sum"),
        closed_legs=("instrument", "size"),
    )
    grouped["closed_return"] = grouped["net_pnl"] / grouped["entry_value"].replace(0, np.nan)
    return grouped


def attach_trade_outcomes(selected: pd.DataFrame, buys: pd.DataFrame, closed: pd.DataFrame) -> pd.DataFrame:
    frame = selected.merge(buys, on=["entry_session", "instrument"], how="left")
    frame = frame.merge(closed, on=["entry_session", "instrument"], how="left")
    frame["was_bought"] = frame["buy_value"].fillna(0).gt(0)
    for col in ["ret_3d", "ret_5d", "ret_7d", "closed_return", "net_pnl", "buy_value"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def summarize_buy_conversion(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys in (["state3"], ["regime", "amount_state", "breadth_state"], ["year"], ["year", "state3"]):
        grouped = frame.groupby(keys, dropna=False)
        for key, group in grouped:
            bought = group[group["was_bought"]]
            rows.append({
                "group": "+".join(keys),
                "key": key if isinstance(key, str) else "|".join(map(str, key if isinstance(key, tuple) else (key,))),
                "selected_rows": int(len(group)),
                "selected_days": int(group["signal_time"].nunique()),
                "buy_rows": int(len(bought)),
                "buy_rate": float(group["was_bought"].mean()) if len(group) else np.nan,
                "buy_value": float(bought["buy_value"].sum()) if "buy_value" in bought else 0.0,
                "net_pnl": float(bought["net_pnl"].sum()) if "net_pnl" in bought else 0.0,
                "closed_return_mean": float(bought["closed_return"].mean()) if len(bought) else np.nan,
                "label_ret5_mean_all": float(group["ret_5d"].mean()) if "ret_5d" in group else np.nan,
                "label_ret5_mean_bought": float(bought["ret_5d"].mean()) if len(bought) else np.nan,
            })
    return pd.DataFrame(rows).sort_values(["group", "net_pnl"], ascending=[True, False])


def summarize_active_bins(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bought = frame[frame["was_bought"]].copy()
    for col in ACTIVE_COLUMNS:
        if col not in bought.columns:
            continue
        rows.extend(summarize_numeric_or_bool(bought, col, outcome="closed_return"))
    return pd.DataFrame(rows)


def summarize_label_bins(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ACTIVE_COLUMNS:
        if col not in frame.columns:
            continue
        rows.extend(summarize_numeric_or_bool(frame, col, outcome="ret_5d"))
    return pd.DataFrame(rows)


def summarize_numeric_or_bool(frame: pd.DataFrame, col: str, *, outcome: str) -> list[dict[str, Any]]:
    if frame.empty or outcome not in frame.columns:
        return []
    values = frame[col]
    if values.dropna().isin([True, False, 0, 1]).all() and values.notna().any():
        labels = values.fillna(False).astype(bool).map({True: "true", False: "false"})
    else:
        numeric = pd.to_numeric(values, errors="coerce")
        try:
            labels = pd.qcut(numeric.rank(method="first"), 4, labels=["q1_low", "q2", "q3", "q4_high"])
        except ValueError:
            return []
    rows = []
    for label, group in frame.groupby(labels, dropna=False):
        out = pd.to_numeric(group[outcome], errors="coerce")
        rows.append({
            "feature": col,
            "bucket": str(label),
            "outcome": outcome,
            "rows": int(len(group)),
            "days": int(group["signal_time"].nunique()) if "signal_time" in group else 0,
            "mean": float(out.mean()) if out.notna().any() else np.nan,
            "median": float(out.median()) if out.notna().any() else np.nan,
            "win_rate": float((out > 0).mean()) if out.notna().any() else np.nan,
            "net_pnl": float(group["net_pnl"].sum()) if "net_pnl" in group else np.nan,
            "buy_value": float(group["buy_value"].sum()) if "buy_value" in group else np.nan,
        })
    return rows


def write_report(frame: pd.DataFrame) -> None:
    bought = frame[frame["was_bought"]]
    top_state = summarize_buy_conversion(frame)
    active_bins = summarize_active_bins(frame)
    label_bins = summarize_label_bins(frame)
    lines = [
        "# F3 Active-Value Conversion",
        "",
        f"- selected rows: {len(frame)}",
        f"- selected days: {frame['signal_time'].nunique()}",
        f"- bought rows: {len(bought)}",
        f"- buy rate: {frame['was_bought'].mean():.4f}",
        f"- bought net pnl: {bought['net_pnl'].sum():.2f}",
        "",
        "## Top State PnL",
        "",
    ]
    state_rows = top_state[top_state["group"].eq("state3")].sort_values("net_pnl", ascending=False).head(12)
    for _, row in state_rows.iterrows():
        lines.append(f"- {row['key']}: pnl={row['net_pnl']:.2f}, buy_rows={row['buy_rows']}, ret={row['closed_return_mean']:.4f}")
    lines.extend(["", "## Worst State PnL", ""])
    for _, row in top_state[top_state["group"].eq("state3")].sort_values("net_pnl").head(12).iterrows():
        lines.append(f"- {row['key']}: pnl={row['net_pnl']:.2f}, buy_rows={row['buy_rows']}, ret={row['closed_return_mean']:.4f}")
    lines.extend(["", "## Best Active Bins For Bought Closed Return", ""])
    best_active = active_bins[active_bins["rows"].ge(80)].sort_values(["mean", "net_pnl"], ascending=[False, False]).head(16)
    for _, row in best_active.iterrows():
        lines.append(f"- {row['feature']} {row['bucket']}: mean={row['mean']:.4f}, win={row['win_rate']:.4f}, pnl={row['net_pnl']:.2f}, rows={row['rows']}")
    lines.extend(["", "## Best Active Bins For Candidate Ret5", ""])
    best_label = label_bins[label_bins["rows"].ge(200)].sort_values(["mean", "win_rate"], ascending=[False, False]).head(16)
    for _, row in best_label.iterrows():
        lines.append(f"- {row['feature']} {row['bucket']}: ret5={row['mean']:.4f}, win={row['win_rate']:.4f}, rows={row['rows']}")
    (OUT / "F3_ACTIVE_VALUE_CONVERSION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
