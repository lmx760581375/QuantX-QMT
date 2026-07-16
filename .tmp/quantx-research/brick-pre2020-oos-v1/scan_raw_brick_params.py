from __future__ import annotations

import importlib.util
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantx.core.data.qlib_reader import QlibBinReader


ROOT = Path(".tmp/quantx-research/brick-pre2020-oos-v1")
ALIGN_SCRIPT = ROOT / "analyze_raw_brick_alignment.py"
PROVIDER = Path("data/qlib_data_fixed")
OUT_JSON = ROOT / "raw_brick_param_scan.json"
OUT_CSV = ROOT / "raw_brick_param_scan.csv"


def main() -> int:
    align = load_align_module()
    official = align.load_official_rows()
    official = official[official["official_brick"].notna()].copy()
    symbols = sorted(s for s in official["instrument"].dropna().unique() if not str(s).startswith("BJ"))
    start = "2025-01-01"
    end = str(official["date"].max())
    reader = QlibBinReader(PROVIDER)
    quote = reader.features(symbols, ["$high", "$low", "$close"], start, end)
    quote = align.normalize_quote(quote)
    frame = quote.reset_index().sort_values(["instrument", "datetime"]).copy()
    frame["date"] = pd.to_datetime(frame["datetime"]).dt.strftime("%Y-%m-%d")
    frame["row_pos"] = np.arange(len(frame), dtype=np.int64)

    groups = []
    for symbol, idx in frame.groupby("instrument", sort=False).groups.items():
        pos = frame.loc[idx, "row_pos"].to_numpy(dtype=np.int64)
        groups.append({
            "symbol": symbol,
            "pos": pos,
            "high": frame.loc[idx, "$high"].to_numpy(dtype="float32"),
            "low": frame.loc[idx, "$low"].to_numpy(dtype="float32"),
            "close": frame.loc[idx, "$close"].to_numpy(dtype="float32"),
        })
    key_frame = frame[["date", "instrument", "row_pos"]]
    official_pos = official.merge(key_frame, on=["date", "instrument"], how="inner")
    official_pos = official_pos.sort_values("row_pos").reset_index(drop=True)
    sample_pos = official_pos["row_pos"].to_numpy(dtype=np.int64)
    official_brick = official_pos["official_brick"].to_numpy(dtype="float64")
    official_prev = official_pos["official_prev_brick"].to_numpy(dtype="float64")
    official_delta = official_pos["official_brick_delta_1"].to_numpy(dtype="float64")

    grid = {
        "n": [6, 8, 10, 13],
        "m1": [2, 3, 4],
        "m2": [10, 12, 14],
        "m3": [10, 12, 14],
        "t": [6, 8, 10],
        "shift1": [88, 92, 96],
        "shift2": [108, 114, 120],
    }
    rows = []
    keys = list(grid)
    total = int(np.prod([len(grid[k]) for k in keys]))
    for i, values in enumerate(product(*[grid[k] for k in keys]), start=1):
        params = dict(zip(keys, values))
        raw_all = np.full(len(frame), np.nan, dtype="float32")
        prev_all = np.full(len(frame), np.nan, dtype="float32")
        delta_all = np.full(len(frame), np.nan, dtype="float32")
        for group in groups:
            raw = align.brick_chart_raw(
                group["high"],
                group["low"],
                group["close"],
                int(params["n"]),
                int(params["m1"]),
                int(params["m2"]),
                int(params["m3"]),
                float(params["t"]),
                float(params["shift1"]),
                float(params["shift2"]),
                1,
                1,
                1,
            )
            pos = group["pos"]
            raw_all[pos] = raw
            prev = np.full_like(raw, np.nan, dtype="float32")
            prev[1:] = raw[:-1]
            prev_all[pos] = prev
            delta_all[pos] = raw - prev
        raw_sample = raw_all[sample_pos].astype("float64")
        prev_sample = prev_all[sample_pos].astype("float64")
        delta_sample = delta_all[sample_pos].astype("float64")
        valid = np.isfinite(raw_sample) & np.isfinite(prev_sample) & np.isfinite(delta_sample)
        if int(valid.sum()) < 50:
            continue
        metrics = score_arrays(
            official_brick[valid],
            official_prev[valid],
            official_delta[valid],
            raw_sample[valid],
            prev_sample[valid],
            delta_sample[valid],
        )
        rows.append({**params, **metrics, "rows": int(valid.sum())})
        if i % 250 == 0:
            print(json.dumps({"progress": i, "total": total, "best_score": max(r["score"] for r in rows)}, ensure_ascii=False), flush=True)
    result = pd.DataFrame(rows).sort_values(["score", "delta_corr", "brick_corr"], ascending=False)
    result.to_csv(OUT_CSV, index=False)
    report = {
        "grid": grid,
        "rows": int(len(result)),
        "top50": result.head(50).to_dict("records"),
        "baseline_8_3_12_12_8_92_114": result[
            (result["n"] == 8)
            & (result["m1"] == 3)
            & (result["m2"] == 12)
            & (result["m3"] == 12)
            & (result["t"] == 8)
            & (result["shift1"] == 92)
            & (result["shift2"] == 114)
        ].head(1).to_dict("records"),
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT_JSON),
        "csv": str(OUT_CSV),
        "rows": int(len(result)),
        "top10": result.head(10).to_dict("records"),
    }, ensure_ascii=False, indent=2, default=str))
    return 0


def load_align_module():
    spec = importlib.util.spec_from_file_location("raw_brick_alignment_module", ALIGN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {ALIGN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def score_arrays(
    official_brick: np.ndarray,
    official_prev: np.ndarray,
    official_delta: np.ndarray,
    raw_brick: np.ndarray,
    raw_prev: np.ndarray,
    raw_delta: np.ndarray,
) -> dict[str, float]:
    brick_corr = corr_array(official_brick, raw_brick)
    prev_corr = corr_array(official_prev, raw_prev)
    delta_corr = corr_array(official_delta, raw_delta)
    brick_mae = mae_scaled_array(official_brick, raw_brick)
    delta_mae = mae_scaled_array(official_delta, raw_delta)
    sign_match = float(((official_delta > 0) == (raw_delta > 0)).mean())
    score = 0.35 * brick_corr + 0.25 * prev_corr + 0.30 * delta_corr + 0.10 * sign_match - 0.05 * delta_mae
    return {
        "score": float(score),
        "brick_corr": float(brick_corr),
        "prev_corr": float(prev_corr),
        "delta_corr": float(delta_corr),
        "brick_mae_scaled": float(brick_mae),
        "delta_mae_scaled": float(delta_mae),
        "delta_sign_match": float(sign_match),
    }


def corr_array(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3:
        return 0.0
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def mae_scaled_array(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if not mask.any():
        return 1.0
    denom = float(np.median(np.abs(left[mask]))) or 1.0
    return float(np.median(np.abs(left[mask] - right[mask])) / denom)


if __name__ == "__main__":
    raise SystemExit(main())
