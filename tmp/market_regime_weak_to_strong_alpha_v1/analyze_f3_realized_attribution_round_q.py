"""Realized attribution for the current best weak-to-strong f3 run.

This script does not run a new backtest. It maps f3 realized closed lots back to
their point-in-time entry signal context, then summarizes which rank/state/time
slices actually contributed or dragged account PnL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
F3_ROOT = ROOT / "raw12_enhancement_round_f"
F3_ARTIFACT = F3_ROOT / "artifacts/f3_good_top12_else_top8"
F3_SIGNAL = F3_ROOT / "signals/f3_good_top12_else_top8.parquet"
OUT_DIR = ROOT / "realized_attribution_round_q"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = build_entry_lots()
    summary = summarize_all(entries)
    entries.to_csv(OUT_DIR / "f3_entry_lot_attribution.csv", index=False)
    for name, frame in summary.items():
        frame.to_csv(OUT_DIR / f"{name}.csv", index=False)
    write_report(entries, summary, OUT_DIR / "F3_REALIZED_ATTRIBUTION_ROUND_Q.md")
    print(f"[round_q_attr] lots={len(entries)} output={OUT_DIR}", flush=True)


def build_entry_lots() -> pd.DataFrame:
    closed = pd.DataFrame(read_json(F3_ARTIFACT / "closed_positions.json"))
    if closed.empty:
        raise RuntimeError(f"No closed positions found under {F3_ARTIFACT}")
    closed["entry_date"] = pd.to_datetime(closed["entry_date"])
    closed["exit_date"] = pd.to_datetime(closed["exit_date"])
    for col in ("quantity", "entry_price", "exit_price", "holding_days", "return", "net_pnl"):
        closed[col] = pd.to_numeric(closed[col], errors="coerce")
    closed["entry_value"] = closed["quantity"] * closed["entry_price"]
    closed["exit_value"] = closed["quantity"] * closed["exit_price"]

    lots = (
        closed.groupby(["symbol", "entry_date"], as_index=False)
        .agg(
            exit_date=("exit_date", "max"),
            first_exit_date=("exit_date", "min"),
            quantity=("quantity", "sum"),
            entry_value=("entry_value", "sum"),
            exit_value=("exit_value", "sum"),
            net_pnl=("net_pnl", "sum"),
            chunk_count=("symbol", "size"),
            max_holding_days=("holding_days", "max"),
            min_holding_days=("holding_days", "min"),
        )
        .sort_values(["entry_date", "symbol"])
    )
    lots["realized_return"] = lots["net_pnl"] / lots["entry_value"].replace(0.0, np.nan)
    lots["year"] = lots["entry_date"].dt.year
    lots["entry_month"] = lots["entry_date"].dt.to_period("M").astype(str)

    signals = load_signal_context()
    merged = lots.merge(signals, on=["symbol", "entry_date"], how="left", validate="many_to_one")
    merged["rank_bucket"] = pd.cut(
        pd.to_numeric(merged["base_rank"], errors="coerce"),
        bins=[0, 1, 3, 5, 8, 12],
        labels=["r1", "r2_3", "r4_5", "r6_8", "r9_12"],
        include_lowest=True,
    ).astype("object").fillna("unmapped")
    merged["f3_good"] = merged["regime"].eq("bull") | merged["amount_state"].eq("hi") | merged["breadth_state"].eq("hi")
    merged["state_key"] = (
        merged["regime"].fillna("missing")
        + "|"
        + merged["amount_state"].fillna("missing")
        + "|"
        + merged["breadth_state"].fillna("missing")
    )
    merged["hold_bucket"] = pd.cut(
        pd.to_numeric(merged["max_holding_days"], errors="coerce"),
        bins=[0, 5, 10, 20, 35, 9999],
        labels=["d1_5", "d6_10", "d11_20", "d21_35", "d36p"],
        include_lowest=True,
    ).astype("object").fillna("unknown")
    return merged


def load_signal_context() -> pd.DataFrame:
    signals = pd.read_parquet(F3_SIGNAL).copy()
    signals["signal_time"] = pd.to_datetime(signals["signal_time"])
    signals = signals.rename(columns={"instrument": "symbol"})
    calendar = load_signal_calendar()
    signals = signals.merge(calendar, on="signal_time", how="left", validate="many_to_one")
    if signals["entry_date"].isna().any():
        missing = int(signals["entry_date"].isna().sum())
        raise RuntimeError(f"Missing entry_date for {missing} f3 signals")
    keep_cols = ["symbol", "entry_date", "signal_time", "score", "raw_rank", "base_rank", "regime", "amount_state", "breadth_state"]
    signals = signals[keep_cols].copy()
    signals["base_rank"] = pd.to_numeric(signals["base_rank"], errors="coerce")
    signals["raw_rank"] = pd.to_numeric(signals["raw_rank"], errors="coerce")
    return signals.sort_values(["entry_date", "symbol", "score"], ascending=[True, True, False]).drop_duplicates(["symbol", "entry_date"], keep="first")


def load_signal_calendar() -> pd.DataFrame:
    rows = pd.DataFrame(read_json(F3_ARTIFACT / "daily_selection_candidates.json"))
    dates = pd.Series(pd.to_datetime(rows["date"].dropna().unique())).sort_values().reset_index(drop=True)
    calendar = pd.DataFrame({"signal_time": dates, "entry_date": dates.shift(-1)})
    return calendar.dropna(subset=["entry_date"])


def summarize_all(entries: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "by_rank_bucket": summarize(entries, ["rank_bucket"]),
        "by_base_rank": summarize(entries, ["base_rank"]),
        "by_f3_good": summarize(entries, ["f3_good"]),
        "by_state_key": summarize(entries, ["state_key"]),
        "by_year": summarize(entries, ["year"]),
        "by_year_rank_bucket": summarize(entries, ["year", "rank_bucket"]),
        "by_year_f3_good": summarize(entries, ["year", "f3_good"]),
        "by_state_rank_bucket": summarize(entries, ["state_key", "rank_bucket"]),
        "by_hold_bucket": summarize(entries, ["hold_bucket"]),
        "negative_slices": negative_slices(entries),
    }


def summarize(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(keys, dropna=False)
    out = grouped.agg(
        lots=("symbol", "size"),
        symbols=("symbol", "nunique"),
        entry_value=("entry_value", "sum"),
        net_pnl=("net_pnl", "sum"),
        avg_return=("realized_return", "mean"),
        median_return=("realized_return", "median"),
        win_rate=("realized_return", lambda x: float((x > 0).mean())),
        avg_hold_days=("max_holding_days", "mean"),
        max_return=("realized_return", "max"),
        min_return=("realized_return", "min"),
    ).reset_index()
    total_pnl = float(frame["net_pnl"].sum())
    total_entry = float(frame["entry_value"].sum())
    out["pnl_share"] = out["net_pnl"] / total_pnl if abs(total_pnl) > 1e-12 else np.nan
    out["entry_value_share"] = out["entry_value"] / total_entry if total_entry > 0 else np.nan
    out["pnl_per_100k"] = out["net_pnl"] / out["entry_value"].replace(0.0, np.nan) * 100000.0
    return out.sort_values(["net_pnl", "avg_return"], ascending=[True, True])


def negative_slices(entries: pd.DataFrame) -> pd.DataFrame:
    frames = []
    specs = [
        ("state_key", ["state_key"]),
        ("rank_bucket", ["rank_bucket"]),
        ("state_rank", ["state_key", "rank_bucket"]),
        ("year_rank", ["year", "rank_bucket"]),
        ("year_state", ["year", "state_key"]),
        ("f3_rank", ["f3_good", "rank_bucket"]),
    ]
    for name, keys in specs:
        item = summarize(entries, keys)
        item.insert(0, "slice_type", name)
        frames.append(item)
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out[(out["lots"] >= 8) & (out["net_pnl"] < 0)].copy()
    return out.sort_values(["net_pnl", "lots"], ascending=[True, False])


def write_report(entries: pd.DataFrame, summary: dict[str, pd.DataFrame], path: Path) -> None:
    total_pnl = float(entries["net_pnl"].sum())
    total_entry = float(entries["entry_value"].sum())
    unmapped = int(entries["signal_time"].isna().sum())
    lines = [
        "# F3 Realized Attribution Round Q",
        "",
        "口径：把 `closed_positions` 先按 `symbol + entry_date` 聚合成原始入场批次，再映射回 `signal_time + lag=1` 的 f3 信号上下文。",
        "",
        f"- entry lots: {len(entries)}",
        f"- unmapped lots: {unmapped}",
        f"- total realized pnl: {total_pnl:.2f}",
        f"- total entry value: {total_entry:.2f}",
        f"- realized pnl per 100k entry: {total_pnl / total_entry * 100000.0:.2f}",
        "",
        "## Rank Buckets",
        "",
        table(summary["by_rank_bucket"], ["rank_bucket", "lots", "net_pnl", "avg_return", "win_rate", "pnl_per_100k"]),
        "",
        "## F3 Good vs Bad",
        "",
        table(summary["by_f3_good"], ["f3_good", "lots", "net_pnl", "avg_return", "win_rate", "pnl_per_100k"]),
        "",
        "## Worst State Slices",
        "",
        table(summary["by_state_key"].head(12), ["state_key", "lots", "net_pnl", "avg_return", "win_rate", "pnl_per_100k"]),
        "",
        "## Negative Candidate Slices",
        "",
        table(summary["negative_slices"].head(20), ["slice_type", "state_key", "year", "rank_bucket", "f3_good", "lots", "net_pnl", "avg_return", "win_rate", "pnl_per_100k"]),
        "",
        "## Files",
        "",
        "- `f3_entry_lot_attribution.csv`",
        "- `by_rank_bucket.csv`",
        "- `by_state_key.csv`",
        "- `by_year_rank_bucket.csv`",
        "- `negative_slices.csv`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "(empty)"
    cols = [col for col in columns if col in frame.columns]
    out = frame[cols].copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return out.to_markdown(index=False)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


if __name__ == "__main__":
    main()
