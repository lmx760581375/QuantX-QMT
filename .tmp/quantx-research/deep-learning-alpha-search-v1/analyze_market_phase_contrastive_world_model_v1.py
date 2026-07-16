from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
OUT = ROOT / "market_phase_contrastive_world_model_v1_summary.json"

TRAIN_START = "2021-01-01"
TRAIN_END = "2024-12-31"
VALID_START = "2025-01-01"
VALID_END = "2025-12-31"
DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"

KMEANS_K = 5
PCA_DIM = 6
KNN_K = 12
SEED = 20260714
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


OP55 = load_module("opportunity_first_world_model_for_phase", ROOT / "train_cross_sectional_opportunity_first_world_model_v1.py")
CA52 = OP55.CA52
RT = OP55.RT
POS = OP55.POS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else 0.0


def rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    return OP55.rank_ic(a, b)


def build_rows(market: dict[str, Any], amount20_arr: np.ndarray, start: str, end: str) -> list[dict[str, Any]]:
    rows = OP55.build_rows(market, amount20_arr, start, end)
    for row in rows:
        candidate_values = [row["candidate_top20"][name] for name in CANDIDATES]
        row["candidate_best_name"] = CANDIDATES[int(np.argmax(candidate_values))]
        row["candidate_best_top20"] = float(np.max(candidate_values))
    return rows


def rows_to_x(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    return np.asarray([[row["x_dict"].get(name, 0.0) for name in feature_names] for row in rows], dtype=np.float64)


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return mu, sd


def standardize_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return np.nan_to_num((x - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)


def pca_fit(x: np.ndarray, dim: int) -> np.ndarray:
    _, _, vt = np.linalg.svd(x - np.mean(x, axis=0, keepdims=True), full_matrices=False)
    return vt[:dim].T


def kmeans_fit(z: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = z.shape[0]
    centroids = z[rng.choice(n, size=k, replace=False)].copy()
    for _ in range(80):
        dist = ((z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(dist, axis=1)
        new_centroids = centroids.copy()
        for c in range(k):
            part = z[labels == c]
            if len(part):
                new_centroids[c] = np.mean(part, axis=0)
        if np.max(np.abs(new_centroids - centroids)) < 1e-7:
            break
        centroids = new_centroids
    return centroids


def assign_clusters(z: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    dist = ((z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(dist, axis=1)


def row_values(row: dict[str, Any]) -> dict[str, float]:
    out = {
        "pool_mean": row["labels"]["pool_mean"],
        "oracle_top20": row["labels"]["oracle_top20"],
        "oracle_top50": row["labels"]["oracle_top50"],
        "candidate_best_top20": row["candidate_best_top20"],
    }
    for name in CANDIDATES:
        out[f"candidate_{name}"] = row["candidate_top20"][name]
    return out


def aggregate(rows: list[dict[str, Any]], indices: list[int]) -> dict[str, Any]:
    if not indices:
        return {"sessions": 0}
    keys = list(row_values(rows[indices[0]]).keys())
    out: dict[str, Any] = {"sessions": len(indices)}
    for key in keys:
        out[key] = safe_mean([row_values(rows[i])[key] for i in indices])
    out["year_counts"] = {}
    out["best_candidate_counts"] = {}
    for i in indices:
        out["year_counts"][rows[i]["year"]] = out["year_counts"].get(rows[i]["year"], 0) + 1
        name = rows[i]["candidate_best_name"]
        out["best_candidate_counts"][name] = out["best_candidate_counts"].get(name, 0) + 1
    return out


def cluster_report(rows: list[dict[str, Any]], labels: np.ndarray) -> dict[str, Any]:
    return {str(c): aggregate(rows, np.flatnonzero(labels == c).tolist()) for c in sorted(set(labels.tolist()))}


def knn_predict(lib_z: np.ndarray, lib_rows: list[dict[str, Any]], query_z: np.ndarray, k: int) -> tuple[np.ndarray, list[dict[str, int]]]:
    target_names = ["pool_mean", "candidate_best_top20", "oracle_top20", "oracle_top50"] + [f"candidate_{name}" for name in CANDIDATES]
    lib_y = np.asarray([[row_values(row)[name] for name in target_names] for row in lib_rows], dtype=float)
    pred = []
    year_counts = []
    for z in query_z:
        dist = np.sum((lib_z - z[None, :]) ** 2, axis=1)
        idx = np.argsort(dist, kind="mergesort")[:k]
        pred.append(np.mean(lib_y[idx], axis=0))
        counts: dict[str, int] = {}
        for i in idx:
            year = lib_rows[int(i)]["year"]
            counts[year] = counts.get(year, 0) + 1
        year_counts.append(counts)
    return np.asarray(pred, dtype=float), year_counts


def prediction_metrics(rows: list[dict[str, Any]], pred: np.ndarray) -> dict[str, Any]:
    target_names = ["pool_mean", "candidate_best_top20", "oracle_top20", "oracle_top50"] + [f"candidate_{name}" for name in CANDIDATES]
    actual = np.asarray([[row_values(row)[name] for name in target_names] for row in rows], dtype=float)
    return {name: {"rank_ic": rank_ic(pred[:, i], actual[:, i]), "actual_mean": safe_mean(actual[:, i]), "pred_mean": safe_mean(pred[:, i])} for i, name in enumerate(target_names)}


def nearest_year_share(year_counts: list[dict[str, int]]) -> dict[str, float]:
    total = sum(sum(item.values()) for item in year_counts)
    out: dict[str, float] = {}
    for item in year_counts:
        for year, count in item.items():
            out[year] = out.get(year, 0.0) + count
    return {year: value / total for year, value in sorted(out.items())} if total else {}


def phase_similarity_by_split(z_by_split: dict[str, np.ndarray]) -> dict[str, Any]:
    means = {name: np.mean(z, axis=0) for name, z in z_by_split.items() if len(z)}
    out = {}
    for a, za in means.items():
        out[a] = {}
        for b, zb in means.items():
            out[a][b] = float(np.linalg.norm(za - zb))
    return out


def main() -> None:
    started = time.perf_counter()
    market = POS.load_market()
    amount20_arr = RT.amount20(market)
    rows_by_split = {
        "train": build_rows(market, amount20_arr, TRAIN_START, TRAIN_END),
        "valid": build_rows(market, amount20_arr, VALID_START, VALID_END),
        "dev": build_rows(market, amount20_arr, DEV_START, DEV_END),
        "forward": build_rows(market, amount20_arr, FWD_START, FWD_END),
    }
    feature_names = sorted(rows_by_split["train"][0]["x_dict"].keys())
    x_by_split = {name: rows_to_x(rows, feature_names) for name, rows in rows_by_split.items()}
    mu, sd = standardize_fit(x_by_split["train"])
    xs_by_split = {name: standardize_apply(x, mu, sd) for name, x in x_by_split.items()}
    pca = pca_fit(xs_by_split["train"], PCA_DIM)
    z_by_split = {name: xs @ pca for name, xs in xs_by_split.items()}
    centroids = kmeans_fit(z_by_split["train"], KMEANS_K, SEED)
    labels_by_split = {name: assign_clusters(z, centroids) for name, z in z_by_split.items()}
    train_pred, train_years = knn_predict(z_by_split["train"], rows_by_split["train"], z_by_split["train"], KNN_K)
    valid_pred, valid_years = knn_predict(z_by_split["train"], rows_by_split["train"], z_by_split["valid"], KNN_K)
    dev_pred, dev_years = knn_predict(z_by_split["train"], rows_by_split["train"], z_by_split["dev"], KNN_K)
    fwd_pred_train, fwd_years_train = knn_predict(z_by_split["train"], rows_by_split["train"], z_by_split["forward"], KNN_K)
    lib_rows_train_valid = rows_by_split["train"] + rows_by_split["valid"]
    lib_z_train_valid = np.vstack([z_by_split["train"], z_by_split["valid"]])
    fwd_pred_train_valid, fwd_years_train_valid = knn_predict(lib_z_train_valid, lib_rows_train_valid, z_by_split["forward"], KNN_K)
    reports = {
        "clusters": {name: cluster_report(rows, labels_by_split[name]) for name, rows in rows_by_split.items()},
        "knn_train_library_metrics": {
            "train": prediction_metrics(rows_by_split["train"], train_pred),
            "valid": prediction_metrics(rows_by_split["valid"], valid_pred),
            "dev": prediction_metrics(rows_by_split["dev"], dev_pred),
            "forward": prediction_metrics(rows_by_split["forward"], fwd_pred_train),
        },
        "knn_forward_train_valid_library_metrics": prediction_metrics(rows_by_split["forward"], fwd_pred_train_valid),
        "nearest_year_share_train_library": {
            "valid": nearest_year_share(valid_years),
            "forward": nearest_year_share(fwd_years_train),
        },
        "nearest_year_share_train_valid_library_forward": nearest_year_share(fwd_years_train_valid),
        "phase_distances": phase_similarity_by_split(z_by_split),
    }
    # Which historical cluster would imply the 2026 trend/volume candidate should be on?
    fwd_cluster = reports["clusters"]["forward"]
    train_cluster = reports["clusters"]["train"]
    cluster_transfer = {}
    for cluster, fwd_stats in fwd_cluster.items():
        train_stats = train_cluster.get(cluster, {"sessions": 0})
        cluster_transfer[cluster] = {
            "forward_sessions": fwd_stats.get("sessions", 0),
            "forward_mid_trend_volume": fwd_stats.get("candidate_mid_trend_volume_not_extreme", 0.0),
            "train_sessions": train_stats.get("sessions", 0),
            "train_mid_trend_volume": train_stats.get("candidate_mid_trend_volume_not_extreme", 0.0),
            "direction_agrees": (fwd_stats.get("candidate_mid_trend_volume_not_extreme", 0.0) > 0) == (train_stats.get("candidate_mid_trend_volume_not_extreme", 0.0) > 0) if train_stats.get("sessions", 0) else False,
        }
    verdict = "market_phase_contrastive_not_enough"
    fwd_ic = reports["knn_forward_train_valid_library_metrics"]["candidate_mid_trend_volume_not_extreme"]["rank_ic"]
    fwd_year_share = reports["nearest_year_share_train_valid_library_forward"]
    if fwd_ic > 0.20 and fwd_year_share.get("2025", 0.0) + fwd_year_share.get("2024", 0.0) > 0.55:
        verdict = "market_phase_contrastive_candidate_needs_phase_conditioned_ranker"
    out = {
        "experiment": "market_phase_contrastive_world_model_v1",
        "method": "unsupervised_pca_kmeans_and_knn_phase_embedding_on_session_world_state",
        "params": {
            "train": [TRAIN_START, TRAIN_END],
            "valid": [VALID_START, VALID_END],
            "dev": [DEV_START, DEV_END],
            "forward": [FWD_START, FWD_END],
            "pca_dim": PCA_DIM,
            "kmeans_k": KMEANS_K,
            "knn_k": KNN_K,
            "feature_count": len(feature_names),
            "candidates": CANDIDATES,
            "seed": SEED,
        },
        "inputs_sha256": {
            "script": sha256(Path(__file__)),
            "round55_script": sha256(ROOT / "train_cross_sectional_opportunity_first_world_model_v1.py"),
            "round56_summary": sha256(ROOT / "candidate_return_opportunity_gate_replay_v1_summary.json"),
        },
        "sample_counts": {name: len(rows) for name, rows in rows_by_split.items()},
        "feature_names": feature_names,
        "reports": reports,
        "cluster_transfer_mid_trend_volume": cluster_transfer,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(OUT),
        "sha256": sha256(OUT),
        "verdict": verdict,
        "sample_counts": out["sample_counts"],
        "forward_knn_train_valid_metrics": reports["knn_forward_train_valid_library_metrics"],
        "forward_nearest_year_share_train_valid": reports["nearest_year_share_train_valid_library_forward"],
        "cluster_transfer_mid_trend_volume": cluster_transfer,
        "phase_distances": reports["phase_distances"],
        "elapsed_seconds": out["elapsed_seconds"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
