from __future__ import annotations

import argparse
import json
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime

import numpy as np
import pandas as pd


ROOT = Path(".tmp/quantx-research/active-value-frontend-brick-v1")
SCORES_PATH = ROOT / "walk_forward_active_brick_scores.parquet"
OUT_JSON = ROOT / "active_brick_liquidity_filter_diagnostics.json"
OUT_MD = ROOT / "active_brick_liquidity_filter_diagnostics.md"
TOPKS = (1, 3, 5, 10)
SCORE_COLUMNS = {
    "ml": "ml_score",
    "ml_consensus": "ml_consensus_score",
    "ml_base": "ml_base_score",
    "ml_base_consensus": "ml_base_consensus_score",
    "momentum": "momentum_score",
    "rule": "rule_score",
    "random": "random_score",
}
FILTER_SPECS = [
    {"name": "all", "rules": []},
    {"name": "turnover_000_040", "rules": [("turnover_rate_rank_pct", 0.00, 0.40)]},
    {"name": "turnover_020_060", "rules": [("turnover_rate_rank_pct", 0.20, 0.60)]},
    {"name": "turnover_040_080", "rules": [("turnover_rate_rank_pct", 0.40, 0.80)]},
    {"name": "turnover_060_100", "rules": [("turnover_rate_rank_pct", 0.60, 1.00)]},
    {"name": "turnover_080_100", "rules": [("turnover_rate_rank_pct", 0.80, 1.00)]},
    {"name": "active_proxy_000_040", "rules": [("active_free_mv_proxy_rank_pct", 0.00, 0.40)]},
    {"name": "active_proxy_020_060", "rules": [("active_free_mv_proxy_rank_pct", 0.20, 0.60)]},
    {"name": "active_proxy_040_080", "rules": [("active_free_mv_proxy_rank_pct", 0.40, 0.80)]},
    {"name": "active_proxy_060_100", "rules": [("active_free_mv_proxy_rank_pct", 0.60, 1.00)]},
    {"name": "active_proxy_080_100", "rules": [("active_free_mv_proxy_rank_pct", 0.80, 1.00)]},
    {"name": "mv_020_080", "rules": [("free_float_mv_rank_pct", 0.20, 0.80)]},
    {"name": "mv_030_085", "rules": [("free_float_mv_rank_pct", 0.30, 0.85)]},
    {"name": "turnover_020_060_mv_020_080", "rules": [("turnover_rate_rank_pct", 0.20, 0.60), ("free_float_mv_rank_pct", 0.20, 0.80)]},
    {"name": "turnover_040_080_mv_020_080", "rules": [("turnover_rate_rank_pct", 0.40, 0.80), ("free_float_mv_rank_pct", 0.20, 0.80)]},
    {"name": "turnover_020_060_active_proxy_020_080", "rules": [("turnover_rate_rank_pct", 0.20, 0.60), ("active_free_mv_proxy_rank_pct", 0.20, 0.80)]},
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "Analyze active brick liquidity filters.")
    parser.add_argument("--scores-path", default=str(SCORES_PATH))
    parser.add_argument("--json-output", default=str(OUT_JSON))
    parser.add_argument("--markdown-output", default=str(OUT_MD))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = analyze(Path(args.scores_path))
    json_output = Path(args.json_output)
    markdown_output = Path(args.markdown_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    markdown_output.write_text(render_markdown(report), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        best = report["best_rows"][:5]
        print(f"ok=True rows={report['rows']} dates={report['dates']} output={json_output}")
        for row in best:
            print(
                f"{row['filter']} {row['method']} top{row['topk']} "
                f"mean={row['fwd10_mean']:.6f} win={row['fwd10_win']:.4f} dates={row['dates']}"
            )
    return 0


def analyze(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing scores parquet: {path}")
    frame = pd.read_parquet(path)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    for column in set(SCORE_COLUMNS.values()) | {"fwd10_net", "fwd5_net"}:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    rows = []
    for spec in FILTER_SPECS:
        filtered = apply_filter(frame, spec["rules"])
        for method, score_col in SCORE_COLUMNS.items():
            if score_col not in filtered:
                continue
            ranked = eligible_by_method(filtered, method, score_col)
            for topk in TOPKS:
                selected = ranked[ranked["daily_rank"] <= topk]
                rows.append(summarize(selected, filter_name=spec["name"], method=method, topk=topk))
    best_rows = sorted(
        [row for row in rows if row["dates"] >= 50 and row["rows"] >= 50],
        key=lambda row: (row["fwd10_mean"] if row["fwd10_mean"] is not None else -np.inf, row["dates"]),
        reverse=True,
    )[:20]
    return {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "scores_path": str(path),
        "rows": int(len(frame)),
        "dates": int(frame["datetime"].nunique()),
        "filters": FILTER_SPECS,
        "topk": rows,
        "best_rows": best_rows,
        "notes": [
            "All returns are score-level label summaries, not account-level replay.",
            "Filters use same-day cross-sectional percentile columns from Qlib turnover/capital features.",
        ],
    }


def apply_filter(frame: pd.DataFrame, rules: list[tuple[str, float, float]]) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for column, low, high in rules:
        if column not in frame:
            return frame.iloc[0:0].copy()
        values = pd.to_numeric(frame[column], errors="coerce")
        mask &= values.ge(low) & values.le(high)
    return frame[mask].copy()


def eligible_by_method(frame: pd.DataFrame, method: str, score_col: str) -> pd.DataFrame:
    data = frame.dropna(subset=[score_col, "datetime", "instrument", "fwd10_net"]).copy()
    if method in {"ml", "ml_base"}:
        data = data[data[score_col] > 0]
    data = data.sort_values(["datetime", score_col, "instrument"], ascending=[True, False, True])
    data["daily_rank"] = data.groupby("datetime").cumcount() + 1
    return data


def summarize(frame: pd.DataFrame, *, filter_name: str, method: str, topk: int) -> dict:
    if frame.empty:
        return {"filter": filter_name, "method": method, "topk": topk, "rows": 0, "dates": 0, "fwd5_mean": None, "fwd10_mean": None, "fwd10_win": None}
    return {
        "filter": filter_name,
        "method": method,
        "topk": int(topk),
        "rows": int(len(frame)),
        "dates": int(frame["datetime"].nunique()),
        "fwd5_mean": float(frame["fwd5_net"].mean()) if "fwd5_net" in frame else None,
        "fwd10_mean": float(frame["fwd10_net"].mean()),
        "fwd10_win": float((frame["fwd10_net"] > 0).mean()),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Active Brick Liquidity Filter Diagnostics",
        "",
        f"- Scores: `{report['scores_path']}`",
        f"- Rows/dates: {report['rows']} / {report['dates']}",
        "",
        "## Best Rows",
        "",
    ]
    for row in report["best_rows"]:
        lines.append(
            f"- `{row['filter']}` `{row['method']}` top{row['topk']}: "
            f"fwd10_mean={row['fwd10_mean']:.4%}, win={row['fwd10_win']:.2%}, "
            f"rows={row['rows']}, dates={row['dates']}"
        )
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in report["notes"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
