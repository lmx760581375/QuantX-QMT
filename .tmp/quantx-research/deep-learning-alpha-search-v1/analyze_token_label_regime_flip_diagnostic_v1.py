from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
OUT = ROOT / "token_label_regime_flip_diagnostic_v1_summary.json"

VALID_START = "2025-01-01"
VALID_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"
TOP_Q = 0.80
BOTTOM_Q = 0.20

FEATURE_NAMES = [
    "ret1",
    "open_gap",
    "intraday",
    "day_range",
    "close_vwap",
    "log_vol_ratio",
    "ret5",
    "ret20",
    "rank_ret1",
    "rank_ret20",
    "rank_amount",
]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RT = load_module("right_tail_token_ranker_v1_for_diag", ROOT / "train_cross_sectional_right_tail_token_ranker_v1.py")
POS = RT.POS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_arrays(split: dict[str, Any]) -> dict[str, np.ndarray]:
    x = split["x"]
    last = x[:, -1, :]
    mean5 = x[:, -5:, :].mean(axis=1)
    mean20 = x[:, -20:, :].mean(axis=1)
    return {"last": last, "mean5": mean5, "mean20": mean20}


def group_indices(split: dict[str, Any]) -> dict[str, dict[str, list[int]]]:
    by_session: dict[str, list[int]] = {}
    for i, session in enumerate(split["sessions"]):
        by_session.setdefault(session, []).append(i)
    out = {"top": {}, "bottom": {}, "middle": {}, "all": {}}
    y = split["y_rank"]
    for session, indices in by_session.items():
        values = y[indices]
        hi = float(np.quantile(values, TOP_Q))
        lo = float(np.quantile(values, BOTTOM_Q))
        top = [i for i in indices if y[i] >= hi]
        bottom = [i for i in indices if y[i] <= lo]
        middle = [i for i in indices if lo < y[i] < hi]
        out["top"][session] = top
        out["bottom"][session] = bottom
        out["middle"][session] = middle
        out["all"][session] = indices
    return out


def feature_stats(split: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    arrays = split_arrays(split)
    y = split["y_rank"]
    stats: dict[str, Any] = {
        "count": len(indices),
        "label_mean": float(np.mean(y[indices])) if indices else 0.0,
        "label_median": float(np.median(y[indices])) if indices else 0.0,
        "label_std": float(np.std(y[indices])) if indices else 0.0,
    }
    for view, values in arrays.items():
        if indices:
            means = values[indices].mean(axis=0)
        else:
            means = np.zeros(len(FEATURE_NAMES), dtype=float)
        stats[f"{view}_features"] = {name: float(value) for name, value in zip(FEATURE_NAMES, means)}
    return stats


def aggregate_group(split: dict[str, Any], grouped: dict[str, dict[str, list[int]]], group: str) -> dict[str, Any]:
    all_indices = [i for indices in grouped[group].values() for i in indices]
    per_session_label = [float(np.mean(split["y_rank"][indices])) for indices in grouped[group].values() if indices]
    stats = feature_stats(split, all_indices)
    stats["session_label_mean"] = float(np.mean(per_session_label)) if per_session_label else 0.0
    stats["session_label_std"] = float(np.std(per_session_label)) if per_session_label else 0.0
    return stats


def diff_features(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {key: float(a.get(key, 0.0) - b.get(key, 0.0)) for key in FEATURE_NAMES}


def top_bottom_contrast(summary: dict[str, Any], view: str) -> dict[str, float]:
    return diff_features(summary["top"][f"{view}_features"], summary["bottom"][f"{view}_features"])


def summarize_split(split: dict[str, Any]) -> dict[str, Any]:
    grouped = group_indices(split)
    summary = {group: aggregate_group(split, grouped, group) for group in ("top", "bottom", "middle", "all")}
    summary["contrasts"] = {view: top_bottom_contrast(summary, view) for view in ("last", "mean5", "mean20")}
    summary["sessions"] = len(grouped["all"])
    summary["samples"] = len(split["sessions"])
    return summary


def largest_abs(values: dict[str, float], n: int = 8) -> list[dict[str, Any]]:
    return [{"feature": key, "value": float(value)} for key, value in sorted(values.items(), key=lambda item: -abs(item[1]))[:n]]


def compare(valid: dict[str, Any], forward: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for view in ("last", "mean5", "mean20"):
        valid_contrast = valid["contrasts"][view]
        forward_contrast = forward["contrasts"][view]
        contrast_delta = diff_features(forward_contrast, valid_contrast)
        top_delta = diff_features(forward["top"][f"{view}_features"], valid["top"][f"{view}_features"])
        sign_flips = []
        for name in FEATURE_NAMES:
            if valid_contrast[name] * forward_contrast[name] < 0:
                sign_flips.append({"feature": name, "valid": valid_contrast[name], "forward": forward_contrast[name]})
        out[view] = {
            "largest_top_feature_shift_forward_minus_valid": largest_abs(top_delta),
            "largest_contrast_shift_forward_minus_valid": largest_abs(contrast_delta),
            "contrast_sign_flips": sign_flips,
        }
    return out


def main() -> None:
    started = time.perf_counter()
    market = POS.load_market()
    feats = RT.build_feature_tensor(market)
    amount20_arr = RT.amount20(market)
    valid = RT.build_split_samples(market, feats, amount20_arr, VALID_START, VALID_END)
    forward = RT.build_split_samples(market, feats, amount20_arr, FWD_START, FWD_END)
    valid_summary = summarize_split(valid)
    forward_summary = summarize_split(forward)
    rt_summary_path = ROOT / "cross_sectional_right_tail_token_ranker_v1_summary.json"
    rt_summary = json.loads(rt_summary_path.read_text()) if rt_summary_path.exists() else {}
    out = {
        "experiment": "token_label_regime_flip_diagnostic_v1",
        "method": "compare_valid_2025_vs_forward_2026_true_tail_token_distributions_no_training",
        "params": {"valid": [VALID_START, VALID_END], "forward": [FWD_START, FWD_END], "top_q": TOP_Q, "bottom_q": BOTTOM_Q, "feature_names": FEATURE_NAMES},
        "inputs_sha256": {"script": sha256(Path(__file__)), "right_tail_script": sha256(ROOT / "train_cross_sectional_right_tail_token_ranker_v1.py"), "right_tail_summary": sha256(rt_summary_path) if rt_summary_path.exists() else None},
        "valid": valid_summary,
        "forward": forward_summary,
        "comparison": compare(valid_summary, forward_summary),
        "right_tail_model_forward": rt_summary.get("results", {}).get("forward", {}),
        "right_tail_model_valid": rt_summary.get("results", {}).get("valid", {}),
        "verdict": "tail_feature_regime_shift_confirmed_needs_regime_conditioned_label" if forward_summary["top"]["label_mean"] > 0 and rt_summary.get("results", {}).get("forward", {}).get("rank_metrics", {}).get("mean_top10_raw_label", 0.0) < 0 else "tail_shift_diagnostic_inconclusive",
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": out["verdict"], "valid_top_label": valid_summary["top"]["label_mean"], "forward_top_label": forward_summary["top"]["label_mean"], "model_forward_top10_label": rt_summary.get("results", {}).get("forward", {}).get("rank_metrics", {}).get("mean_top10_raw_label"), "comparison_last": out["comparison"]["last"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
