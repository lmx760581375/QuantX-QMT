from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
OUT = ROOT / "candidate_return_opportunity_gate_replay_v1_summary.json"

TRAIN_START = "2021-01-01"
TRAIN_END = "2024-12-31"
VALID_START = "2025-01-01"
VALID_END = "2025-12-31"
DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"

POOL_SIZE = 500
TOP_K_REPLAY = 10
TOP_K_LABEL = 20
INTERVAL = 5
SEED = 20260714
EPOCHS = 160
LR = 7e-4
WEIGHT_DECAY = 1e-3
HIDDEN = 32
GATE_QUANTILES = [0.50, 0.60, 0.70, 0.80]
CANDIDATES = [
    "avoid_extreme_trend_low_range",
    "mid_trend_not_extreme",
    "mid_trend_volume_not_extreme",
    "mid_trend_low_crowding",
]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OP55 = load_module("opportunity_first_world_model_for_candidate_gate", ROOT / "train_cross_sectional_opportunity_first_world_model_v1.py")
CA52 = OP55.CA52
RT = OP55.RT
POS = OP55.POS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else 0.0


def rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    return OP55.rank_ic(a, b)


def build_rows(market: dict[str, Any], amount20_arr: np.ndarray, start: str, end: str) -> list[dict[str, Any]]:
    rows = []
    for session in CA52.CVR.sessions(market, start, end, INTERVAL):
        idx = market["date_index"][session]
        cols = np.asarray(RT.pool_for_session(market, amount20_arr, idx)[:POOL_SIZE], dtype=int)
        if len(cols) < TOP_K_LABEL:
            continue
        labels = []
        kept_cols = []
        for col in cols:
            y = CA52.future_return(market, idx, int(col), INTERVAL)
            if y is None:
                continue
            labels.append(y)
            kept_cols.append(int(col))
        if len(labels) < TOP_K_LABEL:
            continue
        kept = np.asarray(kept_cols, dtype=int)
        labels_arr = np.asarray(labels, dtype=float)
        scores = CA52.candidate_scores(CA52.feature_frame(market, idx, kept))
        selections_top10: dict[str, list[str]] = {}
        target_top20: dict[str, float] = {}
        target_top10: dict[str, float] = {}
        for name in CANDIDATES:
            order = sorted(range(len(labels_arr)), key=lambda j: (-float(scores[name][j]), str(market["symbols"][kept[j]])))
            selections_top10[name] = [market["symbols"][kept[j]] for j in order[:TOP_K_REPLAY]]
            target_top20[name] = safe_mean(labels_arr[order[:TOP_K_LABEL]])
            target_top10[name] = safe_mean(labels_arr[order[:TOP_K_REPLAY]])
        rows.append({
            "date": session,
            "year": session[:4],
            "x_dict": OP55.feature_vector(market, idx, kept, scores),
            "target_top20": target_top20,
            "target_top10": target_top10,
            "selections_top10": selections_top10,
        })
    return rows


def rows_to_arrays(rows: list[dict[str, Any]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[row["x_dict"].get(name, 0.0) for name in feature_names] for row in rows], dtype=np.float32)
    y = np.asarray([[row["target_top20"][name] for name in CANDIDATES] for row in rows], dtype=np.float32)
    return x, y


def standardize(train_x: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mu = np.nanmean(train_x, axis=0)
    sd = np.nanstd(train_x, axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return tuple(np.nan_to_num((x - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32) for x in (train_x, *others))


class CandidateReturnMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_model(train_x: np.ndarray, train_y: np.ndarray, valid_x: np.ndarray, valid_y: np.ndarray) -> dict[str, Any]:
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    y_mu = train_y.mean(axis=0)
    y_sd = train_y.std(axis=0)
    y_sd = np.where(y_sd > 1e-8, y_sd, 1.0)
    train_yt = ((train_y - y_mu) / y_sd).astype(np.float32)
    model = CandidateReturnMLP(train_x.shape[1], train_y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.SmoothL1Loss()
    tx = torch.from_numpy(train_x).to(device)
    ty = torch.from_numpy(train_yt).to(device)
    vx = torch.from_numpy(valid_x).to(device)
    best_state = None
    best_score = -1e18
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(tx)
        loss = loss_fn(pred, ty)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            valid_pred = model(vx).detach().cpu().numpy() * y_sd + y_mu
        ics = [rank_ic(valid_pred[:, i], valid_y[:, i]) for i in range(len(CANDIDATES))]
        score = float(np.mean(ics) + 0.25 * np.max(ics))
        history.append({"epoch": epoch, "loss": float(loss.detach().cpu()), "valid_mean_candidate_ic": float(np.mean(ics)), "valid_max_candidate_ic": float(np.max(ics)), "valid_score": score})
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return {"model": model, "device": device, "y_mu": y_mu, "y_sd": y_sd, "history": history, "best_epoch": max(history, key=lambda row: row["valid_score"])["epoch"], "best_valid_score": best_score}


def predict(model_info: dict[str, Any], x: np.ndarray) -> np.ndarray:
    model = model_info["model"]
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x).to(model_info["device"])).detach().cpu().numpy()
    return pred * model_info["y_sd"] + model_info["y_mu"]


def selections_for_candidate(rows: list[dict[str, Any]], candidate: str) -> dict[str, list[str]]:
    return {row["date"]: row["selections_top10"][candidate] for row in rows}


def selections_for_gate(rows: list[dict[str, Any]], pred: np.ndarray, candidate: str, threshold: float, allow_switch: bool) -> dict[str, list[str]]:
    cidx = CANDIDATES.index(candidate)
    out = {}
    for i, row in enumerate(rows):
        if allow_switch:
            best_idx = int(np.argmax(pred[i]))
            if float(pred[i, best_idx]) >= threshold:
                out[row["date"]] = row["selections_top10"][CANDIDATES[best_idx]]
        elif float(pred[i, cidx]) >= threshold:
            out[row["date"]] = row["selections_top10"][candidate]
    return out


def replay_summary(selections: dict[str, list[str]], market: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    return POS.replay(selections, market, start, end)["summary"]


def evaluate_gate(rows_by_split: dict[str, list[dict[str, Any]]], preds: dict[str, np.ndarray], market: dict[str, Any], candidate: str, threshold: float, allow_switch: bool) -> dict[str, Any]:
    ranges = {"train": (TRAIN_START, TRAIN_END), "valid": (VALID_START, VALID_END), "dev": (DEV_START, DEV_END), "forward": (FWD_START, FWD_END)}
    out = {}
    for split, rows in rows_by_split.items():
        selections = selections_for_gate(rows, preds[split], candidate, threshold, allow_switch)
        start, end = ranges[split]
        out[split] = {"sessions_enabled": len(selections), "summary": replay_summary(selections, market, start, end)}
    return out


def choose_gates(valid_rows: list[dict[str, Any]], valid_pred: np.ndarray, market: dict[str, Any]) -> list[dict[str, Any]]:
    configs = []
    for candidate in CANDIDATES:
        cidx = CANDIDATES.index(candidate)
        values = valid_pred[:, cidx]
        for q in GATE_QUANTILES:
            threshold = float(np.quantile(values, q))
            selections = selections_for_gate(valid_rows, valid_pred, candidate, threshold, allow_switch=False)
            summary = replay_summary(selections, market, VALID_START, VALID_END)
            configs.append({"candidate": candidate, "allow_switch": False, "quantile": q, "threshold": threshold, "valid_sessions_enabled": len(selections), "valid_summary": summary})
    best_values = np.max(valid_pred, axis=1)
    for q in GATE_QUANTILES:
        threshold = float(np.quantile(best_values, q))
        selections = selections_for_gate(valid_rows, valid_pred, CANDIDATES[0], threshold, allow_switch=True)
        summary = replay_summary(selections, market, VALID_START, VALID_END)
        configs.append({"candidate": "dynamic_best_predicted_candidate", "allow_switch": True, "quantile": q, "threshold": threshold, "valid_sessions_enabled": len(selections), "valid_summary": summary})
    return sorted(configs, key=lambda row: (row["valid_summary"].get("all_years_positive", False), row["valid_summary"].get("final_multiple", 0.0), row["valid_sessions_enabled"]), reverse=True)


def target_metrics(rows: list[dict[str, Any]], pred: np.ndarray) -> dict[str, Any]:
    out = {}
    for i, candidate in enumerate(CANDIDATES):
        actual20 = np.asarray([row["target_top20"][candidate] for row in rows], dtype=float)
        actual10 = np.asarray([row["target_top10"][candidate] for row in rows], dtype=float)
        out[candidate] = {"top20_rank_ic": rank_ic(pred[:, i], actual20), "top10_rank_ic": rank_ic(pred[:, i], actual10), "top20_mean": safe_mean(actual20), "top10_mean": safe_mean(actual10)}
    return out


def main() -> None:
    started = time.perf_counter()
    set_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    market = POS.load_market()
    amount20_arr = RT.amount20(market)
    rows_by_split = {
        "train": build_rows(market, amount20_arr, TRAIN_START, TRAIN_END),
        "valid": build_rows(market, amount20_arr, VALID_START, VALID_END),
        "dev": build_rows(market, amount20_arr, DEV_START, DEV_END),
        "forward": build_rows(market, amount20_arr, FWD_START, FWD_END),
    }
    feature_names = sorted(rows_by_split["train"][0]["x_dict"].keys())
    arrays = {split: rows_to_arrays(rows, feature_names) for split, rows in rows_by_split.items()}
    train_x, train_y = arrays["train"]
    valid_x, valid_y = arrays["valid"]
    dev_x, _ = arrays["dev"]
    fwd_x, _ = arrays["forward"]
    train_xs, valid_xs, dev_xs, fwd_xs = standardize(train_x, valid_x, dev_x, fwd_x)
    model_info = train_model(train_xs, train_y, valid_xs, valid_y)
    preds = {"train": predict(model_info, train_xs), "valid": predict(model_info, valid_xs), "dev": predict(model_info, dev_xs), "forward": predict(model_info, fwd_xs)}
    base_replays = {}
    for candidate in CANDIDATES:
        base_replays[candidate] = {
            split: replay_summary(selections_for_candidate(rows, candidate), market, *({"train": (TRAIN_START, TRAIN_END), "valid": (VALID_START, VALID_END), "dev": (DEV_START, DEV_END), "forward": (FWD_START, FWD_END)}[split]))
            for split, rows in rows_by_split.items()
        }
    gate_configs = choose_gates(rows_by_split["valid"], preds["valid"], market)
    evaluated = []
    for config in gate_configs[:10]:
        result = evaluate_gate(rows_by_split, preds, market, CANDIDATES[0] if config["allow_switch"] else config["candidate"], config["threshold"], bool(config["allow_switch"]))
        evaluated.append({k: v for k, v in config.items() if k != "valid_summary"} | {"results": result})
    leaderboard = sorted(evaluated, key=lambda row: (row["results"]["dev"]["summary"].get("all_years_positive", False), row["results"]["dev"]["summary"].get("final_multiple", 0.0), row["results"]["forward"]["summary"].get("final_multiple", 0.0)), reverse=True)
    best = leaderboard[0] if leaderboard else {}
    verdict = "candidate_return_opportunity_gate_not_enough"
    if best:
        dev_s = best["results"]["dev"]["summary"]
        fwd_s = best["results"]["forward"]["summary"]
        if dev_s.get("all_years_positive") and dev_s.get("final_multiple", 0.0) >= 5.0 and fwd_s.get("final_multiple", 0.0) > 1.2 and fwd_s.get("avg_position_count", 0.0) > 5 and fwd_s.get("remove_best_period_multiples", {}).get("remove_best_3", 0.0) > 1.0:
            verdict = "candidate_return_opportunity_gate_candidate_needs_strict_walkforward"
    out = {
        "experiment": "candidate_return_opportunity_gate_replay_v1",
        "method": "session_mlp_directly_predict_candidate_top20_return_select_valid_gate_then_replay_top10",
        "params": {
            "train": [TRAIN_START, TRAIN_END],
            "valid": [VALID_START, VALID_END],
            "dev": [DEV_START, DEV_END],
            "forward": [FWD_START, FWD_END],
            "pool_size": POOL_SIZE,
            "top_k_replay": TOP_K_REPLAY,
            "top_k_label": TOP_K_LABEL,
            "interval": INTERVAL,
            "epochs": EPOCHS,
            "hidden": HIDDEN,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "gate_quantiles": GATE_QUANTILES,
            "candidates": CANDIDATES,
            "feature_count": len(feature_names),
            "device": str(model_info["device"]),
            "seed": SEED,
        },
        "inputs_sha256": {
            "script": sha256(Path(__file__)),
            "round55_script": sha256(ROOT / "train_cross_sectional_opportunity_first_world_model_v1.py"),
            "round55_summary": sha256(ROOT / "cross_sectional_opportunity_first_world_model_v1_summary.json"),
        },
        "sample_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "training_history": model_info["history"],
        "best_epoch": model_info["best_epoch"],
        "best_valid_score": model_info["best_valid_score"],
        "target_metrics": {split: target_metrics(rows_by_split[split], preds[split]) for split in rows_by_split},
        "base_replays": base_replays,
        "gate_configs_by_valid": gate_configs,
        "leaderboard": leaderboard,
        "best": best,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(OUT),
        "sha256": sha256(OUT),
        "verdict": verdict,
        "best_epoch": model_info["best_epoch"],
        "best": best,
        "target_metrics_forward": out["target_metrics"]["forward"],
        "sample_counts": out["sample_counts"],
        "elapsed_seconds": out["elapsed_seconds"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
