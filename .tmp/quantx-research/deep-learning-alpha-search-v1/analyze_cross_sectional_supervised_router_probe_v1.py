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
OUT = ROOT / "cross_sectional_supervised_router_probe_v1_summary.json"

TRAIN_START = "2021-01-01"
TRAIN_END = "2024-12-31"
VALID_START = "2025-01-01"
VALID_END = "2025-12-31"
DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"

ALPHAS = (0.1, 1.0, 10.0, 100.0)
THRESHOLDS = (None, 0.0, 0.001, 0.002)
LOOKBACKS = (4, 12, 24)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R36 = load_module("regime_router_v1", ROOT / "analyze_cross_sectional_regime_router_oracle_scan_v1.py")
POS = R36.POS
BASE_VARIANTS = R36.BASE_VARIANTS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_sessions(sessions: list[str], start: str, end: str) -> list[str]:
    return [session for session in sessions if start <= session <= end]


def feature_names() -> list[str]:
    names = [
        "median_ret5",
        "median_ret20",
        "median_ret60",
        "breadth20",
        "breadth5",
        "median_vol20",
        "dispersion20",
        "median_vol_ratio20",
    ]
    for variant in BASE_VARIANTS:
        for lookback in LOOKBACKS:
            names.extend([f"{variant}_mean_{lookback}", f"{variant}_std_{lookback}", f"{variant}_hit_{lookback}"])
        names.extend([f"{variant}_last1", f"{variant}_last3_mean"])
    return names


NAMES = feature_names()


def session_features(session: str, ordered_sessions: list[str], tables: dict[str, Any]) -> list[float]:
    idx = ordered_sessions.index(session)
    feats = [float(tables["regimes"][session][name]) for name in NAMES[:8]]
    for variant in BASE_VARIANTS:
        returns = tables["period_returns"][variant]
        for lookback in LOOKBACKS:
            hist = [returns[day] for day in ordered_sessions[max(0, idx - lookback):idx]]
            if hist:
                feats.extend([float(np.mean(hist)), float(np.std(hist)), float(np.mean(np.asarray(hist) > 0))])
            else:
                feats.extend([0.0, 0.0, 0.0])
        last1 = returns[ordered_sessions[idx - 1]] if idx >= 1 else 0.0
        last3_hist = [returns[day] for day in ordered_sessions[max(0, idx - 3):idx]]
        feats.extend([float(last1), float(np.mean(last3_hist)) if last3_hist else 0.0])
    return feats


def build_xy(sessions: list[str], ordered_sessions: list[str], tables: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    x = np.asarray([session_features(session, ordered_sessions, tables) for session in sessions], dtype=float)
    y = {variant: np.asarray([tables["period_returns"][variant][session] for session in sessions], dtype=float) for variant in BASE_VARIANTS}
    return x, y


def standardize(train_x: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(train_x, axis=0)
    sigma = np.nanstd(train_x, axis=0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1.0)
    return (np.nan_to_num((x - mu) / sigma), mu, sigma)


def apply_standardize(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return np.nan_to_num((x - mu) / sigma)


def fit_ridge(train_x: np.ndarray, train_y: dict[str, np.ndarray], alpha: float) -> dict[str, np.ndarray]:
    xb = np.column_stack([np.ones(len(train_x)), train_x])
    eye = np.eye(xb.shape[1])
    eye[0, 0] = 0.0
    models = {}
    for variant, y in train_y.items():
        models[variant] = np.linalg.solve(xb.T @ xb + alpha * eye, xb.T @ y)
    return models


def predict(models: dict[str, np.ndarray], x: np.ndarray) -> dict[str, np.ndarray]:
    xb = np.column_stack([np.ones(len(x)), x])
    return {variant: xb @ beta for variant, beta in models.items()}


def prediction_quality(preds: dict[str, np.ndarray], y: dict[str, np.ndarray]) -> dict[str, Any]:
    out = {}
    for variant in BASE_VARIANTS:
        p = preds[variant]
        t = y[variant]
        corr = float(np.corrcoef(p, t)[0, 1]) if len(p) > 2 and np.std(p) > 0 and np.std(t) > 0 else 0.0
        sign_acc = float(np.mean((p > 0) == (t > 0))) if len(p) else 0.0
        mse = float(np.mean((p - t) ** 2)) if len(p) else 0.0
        out[variant] = {"corr": corr, "sign_acc": sign_acc, "mse": mse}
    return out


def decisions_from_preds(sessions: list[str], preds: dict[str, np.ndarray], threshold: float | None) -> dict[str, str]:
    decisions = {}
    for i, session in enumerate(sessions):
        best = max(BASE_VARIANTS, key=lambda variant: preds[variant][i])
        if threshold is not None and preds[best][i] <= threshold:
            decisions[session] = "cash"
        else:
            decisions[session] = best
    return decisions


def eval_decisions(decisions: dict[str, str], tables: dict[str, Any], market: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    sessions = split_sessions(tables["sessions"], start, end)
    split_decisions = {session: decisions[session] for session in sessions if session in decisions}
    selections = R36.decisions_to_selections(split_decisions, tables["selections"])
    return {"summary": R36.replay_router(selections, market, start, end), "decisions": R36.summarize_decisions(split_decisions)}


def train_and_score(alpha: float, threshold: float | None, train_x: np.ndarray, train_y: dict[str, np.ndarray], all_x: np.ndarray, all_sessions: list[str], tables: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    x_train_std, mu, sigma = standardize(train_x, train_x)
    models = fit_ridge(x_train_std, train_y, alpha)
    all_x_std = apply_standardize(all_x, mu, sigma)
    preds = predict(models, all_x_std)
    decisions = decisions_from_preds(all_sessions, preds, threshold)
    splits = {
        "train": eval_decisions(decisions, tables, market, TRAIN_START, TRAIN_END),
        "valid": eval_decisions(decisions, tables, market, VALID_START, VALID_END),
        "dev": eval_decisions(decisions, tables, market, DEV_START, DEV_END),
        "forward": eval_decisions(decisions, tables, market, FWD_START, FWD_END),
    }
    return {"alpha": alpha, "threshold": threshold, "decisions": decisions, "splits": splits, "preds": preds, "models": {k: v.tolist() for k, v in models.items()}}


def main() -> None:
    started = time.perf_counter()
    market = POS.load_market()
    tables = R36.build_tables(market)
    all_sessions = tables["sessions"]
    train_sessions = split_sessions(all_sessions, TRAIN_START, TRAIN_END)
    valid_sessions = split_sessions(all_sessions, VALID_START, VALID_END)
    forward_sessions = split_sessions(all_sessions, FWD_START, FWD_END)
    all_x, all_y = build_xy(all_sessions, all_sessions, tables)
    train_x, train_y = build_xy(train_sessions, all_sessions, tables)
    valid_x, valid_y = build_xy(valid_sessions, all_sessions, tables)
    forward_x, forward_y = build_xy(forward_sessions, all_sessions, tables)
    configs = []
    for alpha in ALPHAS:
        for threshold in THRESHOLDS:
            scored = train_and_score(alpha, threshold, train_x, train_y, all_x, all_sessions, tables, market)
            configs.append(scored)
    best_valid = max(configs, key=lambda item: item["splits"]["valid"]["summary"]["final_multiple"])
    best_train = max(configs, key=lambda item: item["splits"]["train"]["summary"]["final_multiple"])
    def compact_config(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "alpha": item["alpha"],
            "threshold": item["threshold"],
            "splits": item["splits"],
        }
    x_train_std, mu, sigma = standardize(train_x, train_x)
    models = fit_ridge(x_train_std, train_y, best_valid["alpha"])
    qualities = {
        "train": prediction_quality(predict(models, apply_standardize(train_x, mu, sigma)), train_y),
        "valid": prediction_quality(predict(models, apply_standardize(valid_x, mu, sigma)), valid_y),
        "forward": prediction_quality(predict(models, apply_standardize(forward_x, mu, sigma)), forward_y),
    }
    verdict = "supervised_router_probe_not_learnable"
    fwd = best_valid["splits"]["forward"]["summary"]
    dev = best_valid["splits"]["dev"]["summary"]
    if dev["final_multiple"] >= 20.0 and dev["all_years_positive"] and fwd["final_multiple"] > 1.2 and fwd["avg_position_count"] > 5 and fwd["remove_best_period_multiples"].get("remove_best_3", 0.0) > 1.0:
        verdict = "supervised_router_probe_candidate_needs_deep_modeling"
    out = {
        "experiment": "cross_sectional_supervised_router_probe_v1",
        "method": "ridge_predict_next_5d_base_strategy_return_from_past_only_regime_and_strategy_history",
        "params": {"train": [TRAIN_START, TRAIN_END], "valid": [VALID_START, VALID_END], "dev": [DEV_START, DEV_END], "forward": [FWD_START, FWD_END], "alphas": ALPHAS, "thresholds": THRESHOLDS, "lookbacks": LOOKBACKS, "base_variants": BASE_VARIANTS, "feature_count": len(NAMES)},
        "inputs_sha256": {"script": sha256(Path(__file__)), "router_script": sha256(ROOT / "analyze_cross_sectional_regime_router_oracle_scan_v1.py")},
        "sessions": {"train": len(train_sessions), "valid": len(valid_sessions), "forward": len(forward_sessions), "all": len(all_sessions)},
        "feature_names": NAMES,
        "best_valid_config": compact_config(best_valid),
        "best_train_config": compact_config(best_train),
        "prediction_quality_best_valid_alpha": qualities,
        "config_leaderboard": [compact_config(item) for item in sorted(configs, key=lambda item: item["splits"]["valid"]["summary"]["final_multiple"], reverse=True)[:8]],
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "best_valid": compact_config(best_valid), "best_train": compact_config(best_train), "prediction_quality": qualities}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
