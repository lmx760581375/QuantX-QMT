from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
ROOT = REPO_ROOT / ".tmp/quantx-research/deep-learning-alpha-search-v1"
OUT = ROOT / "group_diffusion_world_model_v1_summary.json"
INDUSTRY_CSV = REPO_ROOT / "data/meta/snapshots/industry_membership.csv"
SECTOR_CSV = REPO_ROOT / "data/meta/snapshots/sector_membership.csv"

TRAIN_START = "2021-01-01"
TRAIN_END = "2024-12-31"
VALID_START = "2025-01-01"
VALID_END = "2025-12-31"
DEV_START = "2021-01-01"
DEV_END = "2025-12-31"
FWD_START = "2026-01-01"
FWD_END = "2026-07-10"

INTERVAL = 5
POOL_SIZE = 500
DOMAIN_SIZE = 60
TOP_KS = [10, 20]
SEED = 20260714
EPOCHS = 12
BATCH_SIZE = 1024
LR = 8e-4
WEIGHT_DECAY = 2e-4


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CA52 = load_module("crowded_trend_for_group_diffusion", ROOT / "analyze_crowded_trend_exhaustion_avoidance_v1.py")
RT = CA52.RT
POS = CA52.POS


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


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=np.isfinite(a) & np.isfinite(b) & (b != 0))


def finite_rank(values: np.ndarray) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    fill = float(np.nanmedian(clean[np.isfinite(clean)])) if np.isfinite(clean).any() else 0.0
    order = np.argsort(np.where(np.isfinite(clean), clean, fill), kind="mergesort")
    ranks = np.empty(len(clean), dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(clean), endpoint=True) if len(clean) > 1 else 0.5
    return ranks


def rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    x = finite_rank(a)
    y = finite_rank(b)
    if len(x) < 4 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def symbol_keys(symbol: str) -> set[str]:
    raw = str(symbol).strip().upper()
    digits = "".join(ch for ch in raw if ch.isdigit())
    keys = {raw}
    if len(digits) == 6:
        suffix = "SH" if digits.startswith("6") else "SZ"
        keys.update({digits, f"{suffix}{digits}", f"{digits}.{suffix}"})
    return keys


def build_symbol_lookup(symbols: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, symbol in enumerate(symbols):
        for key in symbol_keys(symbol):
            out[key] = i
    return out


def load_groups(symbols: list[str]) -> dict[str, Any]:
    lookup = build_symbol_lookup(symbols)
    industry = np.full(len(symbols), -1, dtype=np.int32)
    industry_frame = pd.read_csv(INDUSTRY_CSV, dtype=str)
    industry_names = sorted(industry_frame["industry_code"].dropna().astype(str).unique())
    industry_index = {name: i for i, name in enumerate(industry_names)}
    for _, row in industry_frame.iterrows():
        code = row.get("industry_code")
        symbol = row.get("symbol") or row.get("source_code")
        if pd.isna(code) or pd.isna(symbol):
            continue
        for key in symbol_keys(str(symbol)):
            if key in lookup:
                industry[lookup[key]] = industry_index[str(code)]

    sector_frame = pd.read_csv(SECTOR_CSV, dtype=str)
    if "sector_type" in sector_frame.columns:
        sector_frame = sector_frame[sector_frame["sector_type"].fillna("concept") == "concept"]
    concept_names = sorted(sector_frame["sector_code"].dropna().astype(str).unique())
    concept_index = {name: i for i, name in enumerate(concept_names)}
    concept = np.zeros((len(symbols), len(concept_names)), dtype=bool)
    for _, row in sector_frame.iterrows():
        code = row.get("sector_code")
        symbol = row.get("symbol") or row.get("source_code")
        if pd.isna(code) or pd.isna(symbol):
            continue
        group_idx = concept_index.get(str(code))
        if group_idx is None:
            continue
        for key in symbol_keys(str(symbol)):
            if key in lookup:
                concept[lookup[key], group_idx] = True
    return {
        "industry": industry,
        "concept": concept,
        "industry_count": len(industry_names),
        "concept_count": len(concept_names),
        "industry_coverage": float(np.mean(industry >= 0)),
        "concept_coverage": float(np.mean(concept.any(axis=1))) if concept.shape[1] else 0.0,
        "mean_concepts_per_symbol": float(concept.sum(axis=1).mean()) if concept.shape[1] else 0.0,
    }


def window(arr: np.ndarray, end_idx: int, cols: np.ndarray, length: int) -> np.ndarray:
    start = max(0, end_idx - length + 1)
    return arr[start:end_idx + 1, :][:, cols].astype(float)


def market_paths(market: dict[str, Any], idx: int, cols: np.ndarray) -> dict[str, np.ndarray]:
    arrays = market["arrays"]
    close = arrays["close"].astype(float)
    volume = arrays["volume"].astype(float)
    c0 = close[idx, cols]
    c5 = close[idx - 5, cols] if idx >= 5 else np.full(len(cols), np.nan)
    c20 = close[idx - 20, cols] if idx >= 20 else np.full(len(cols), np.nan)
    c60 = close[idx - 60, cols] if idx >= 60 else np.full(len(cols), np.nan)
    vol5 = np.nanmean(window(volume, idx, cols, 5), axis=0)
    vol20 = np.nanmean(window(volume, idx, cols, 20), axis=0)
    return {
        "ret5": safe_div(c0, c5) - 1.0,
        "ret20": safe_div(c0, c20) - 1.0,
        "ret60": safe_div(c0, c60) - 1.0,
        "vol5_vs_20": np.log(np.where(safe_div(vol5, vol20) > 0, safe_div(vol5, vol20), np.nan)),
    }


def group_stats_for_one(member_mask: np.ndarray, pool_pos: np.ndarray, paths: dict[str, np.ndarray], stock_pos: int) -> list[float]:
    local = member_mask[pool_pos]
    if int(local.sum()) < 3:
        return [0.0] * 13
    ret5 = paths["ret5"][local]
    ret20 = paths["ret20"][local]
    ret60 = paths["ret60"][local]
    vol = paths["vol5_vs_20"][local]
    stock_ret5 = paths["ret5"][stock_pos]
    stock_ret20 = paths["ret20"][stock_pos]
    stock_ret60 = paths["ret60"][stock_pos]
    return [
        float(local.mean()),
        safe_mean(ret5),
        safe_mean(ret20),
        safe_mean(ret60),
        float(np.nanmean(ret5 > 0.0)) if len(ret5) else 0.0,
        float(np.nanmean(ret20 > 0.0)) if len(ret20) else 0.0,
        float(np.nanstd(ret20)) if len(ret20) else 0.0,
        safe_mean(vol),
        float(stock_ret5 - safe_mean(ret5)) if np.isfinite(stock_ret5) else 0.0,
        float(stock_ret20 - safe_mean(ret20)) if np.isfinite(stock_ret20) else 0.0,
        float(stock_ret60 - safe_mean(ret60)) if np.isfinite(stock_ret60) else 0.0,
        float(finite_rank(np.r_[ret20, stock_ret20])[-1]) if np.isfinite(stock_ret20) else 0.5,
        float(finite_rank(np.r_[ret60, stock_ret60])[-1]) if np.isfinite(stock_ret60) else 0.5,
    ]


GROUP_FEATURE_NAMES = [
    "industry_size_share", "industry_ret5", "industry_ret20", "industry_ret60", "industry_breadth5", "industry_breadth20", "industry_disp20", "industry_vol5_vs_20",
    "industry_rel_ret5", "industry_rel_ret20", "industry_rel_ret60", "industry_rank_ret20", "industry_rank_ret60",
    "concept_size_share_max", "concept_ret5_max", "concept_ret20_max", "concept_ret60_max", "concept_breadth5_max", "concept_breadth20_max", "concept_disp20_max", "concept_vol5_vs_20_max",
    "concept_rel_ret5_max", "concept_rel_ret20_max", "concept_rel_ret60_max", "concept_rank_ret20_max", "concept_rank_ret60_max",
    "concept_size_share_mean", "concept_ret20_mean", "concept_breadth20_mean", "concept_rel_ret20_mean", "concept_rank_ret20_mean", "concept_count_norm",
]


def group_feature_matrix(market: dict[str, Any], idx: int, pool_cols: np.ndarray, domain_cols: np.ndarray, groups: dict[str, Any]) -> np.ndarray:
    paths = market_paths(market, idx, pool_cols)
    pool_pos_by_col = {int(col): pos for pos, col in enumerate(pool_cols)}
    industry = groups["industry"]
    concept = groups["concept"]
    rows = []
    for col in domain_cols:
        stock_pos = pool_pos_by_col[int(col)]
        ind = int(industry[int(col)])
        if ind >= 0:
            ind_stats = group_stats_for_one(industry == ind, pool_cols, paths, stock_pos)
        else:
            ind_stats = [0.0] * 13
        concept_ids = np.flatnonzero(concept[int(col)]) if concept.shape[1] else np.asarray([], dtype=int)
        c_stats = []
        for cid in concept_ids[:24]:
            c_stats.append(group_stats_for_one(concept[:, cid], pool_cols, paths, stock_pos))
        if c_stats:
            carr = np.asarray(c_stats, dtype=float)
            cmax = np.nanmax(carr, axis=0)
            cmean = np.nanmean(carr, axis=0)
            concept_stats = list(cmax) + [float(cmean[0]), float(cmean[2]), float(cmean[5]), float(cmean[9]), float(cmean[11]), float(min(len(c_stats), 24) / 24.0)]
        else:
            concept_stats = [0.0] * 19
        rows.append(ind_stats + concept_stats)
    return np.nan_to_num(np.asarray(rows, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def session_table(market: dict[str, Any], amount20_arr: np.ndarray, groups: dict[str, Any], session: str) -> dict[str, Any] | None:
    idx = market["date_index"][session]
    pool_cols = np.asarray(RT.pool_for_session(market, amount20_arr, idx)[:POOL_SIZE], dtype=int)
    labels = []
    kept_cols = []
    for col in pool_cols:
        y = CA52.future_return(market, idx, int(col), INTERVAL)
        if y is not None:
            labels.append(float(y))
            kept_cols.append(int(col))
    if len(labels) < DOMAIN_SIZE:
        return None
    kept = np.asarray(kept_cols, dtype=int)
    labels_arr = np.asarray(labels, dtype=float)
    frame = CA52.feature_frame(market, idx, kept)
    scores = CA52.candidate_scores(frame)
    base_score = scores["mid_trend_volume_not_extreme"]
    order = np.asarray(sorted(range(len(labels_arr)), key=lambda j: (-float(base_score[j]), str(market["symbols"][kept[j]])))[:DOMAIN_SIZE], dtype=int)
    domain_cols = kept[order]
    feature_names = sorted(frame.keys())
    stock_x = np.vstack([finite_rank(np.asarray(frame[name], dtype=float))[order] for name in feature_names]).T
    group_x = group_feature_matrix(market, idx, kept, domain_cols, groups)
    y = labels_arr[order].astype(np.float32)
    target = finite_rank(y).astype(np.float32)
    return {
        "date": session,
        "year": session[:4],
        "stock_x": np.nan_to_num(stock_x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        "group_x": group_x,
        "y": y,
        "target": target,
        "base_order": np.arange(len(y)),
        "feature_names": feature_names,
    }


def build_split(market: dict[str, Any], amount20_arr: np.ndarray, groups: dict[str, Any], start: str, end: str) -> list[dict[str, Any]]:
    rows = []
    for session in CA52.CVR.sessions(market, start, end, INTERVAL):
        item = session_table(market, amount20_arr, groups, session)
        if item is not None:
            rows.append(item)
    return rows


def stack_rows(sessions: list[dict[str, Any]], use_group: bool) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for item in sessions:
        x = item["stock_x"] if not use_group else np.hstack([item["stock_x"], item["group_x"]])
        xs.append(x)
        ys.append(item["target"])
    return np.vstack(xs).astype(np.float32), np.concatenate(ys).astype(np.float32)


def standard_params(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0).astype(np.float32)
    return mu.astype(np.float32), sd


class RankMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 96), nn.GELU(), nn.Dropout(0.12), nn.Linear(96, 48), nn.GELU(), nn.Linear(48, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_model(train_x: np.ndarray, train_y: np.ndarray, valid_x: np.ndarray, valid_y: np.ndarray) -> dict[str, Any]:
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = RankMLP(train_x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=BATCH_SIZE, shuffle=True)
    valid_tensor = torch.from_numpy(valid_x).to(device)
    history = []
    best_state = None
    best_ic = -9.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            pred = model(valid_tensor).detach().cpu().numpy()
        ic = rank_ic(pred, valid_y)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_row_rank_ic": ic})
        if ic > best_ic:
            best_ic = ic
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return {"model": model, "device": device, "history": history, "best_epoch": max(history, key=lambda r: r["valid_row_rank_ic"])["epoch"], "best_valid_row_rank_ic": best_ic}


def score_item(item: dict[str, Any], model_info: dict[str, Any], mu: np.ndarray, sd: np.ndarray, use_group: bool) -> np.ndarray:
    x = item["stock_x"] if not use_group else np.hstack([item["stock_x"], item["group_x"]])
    xs = ((x.astype(np.float32) - mu) / sd).astype(np.float32)
    model = model_info["model"]
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(xs).to(model_info["device"])).detach().cpu().numpy()


def evaluate(sessions: list[dict[str, Any]], model_info: dict[str, Any], mu: np.ndarray, sd: np.ndarray, use_group: bool) -> dict[str, Any]:
    model_top = {k: [] for k in TOP_KS}
    base_top = {k: [] for k in TOP_KS}
    ics = []
    by_year: dict[str, list[float]] = {}
    for item in sessions:
        score = score_item(item, model_info, mu, sd, use_group)
        y = item["y"]
        order = np.argsort(-score, kind="mergesort")
        ics.append(rank_ic(score, y))
        for k in TOP_KS:
            value = safe_mean(y[order[:k]])
            model_top[k].append(value)
            base_top[k].append(safe_mean(y[item["base_order"][:k]]))
            if k == 20:
                by_year.setdefault(item["year"], []).append(value)
    return {
        "sessions": len(sessions),
        "model_top": {f"top{k}": safe_mean(v) for k, v in model_top.items()},
        "base_top": {f"top{k}": safe_mean(v) for k, v in base_top.items()},
        "mean_rank_ic": safe_mean(ics),
        "by_year_model_top20": {year: safe_mean(vals) for year, vals in sorted(by_year.items())},
    }


def run_variant(name: str, splits: dict[str, list[dict[str, Any]]], use_group: bool) -> dict[str, Any]:
    train_x, train_y = stack_rows(splits["train"], use_group)
    valid_x, valid_y = stack_rows(splits["valid"], use_group)
    mu, sd = standard_params(train_x)
    train_xs = ((train_x - mu) / sd).astype(np.float32)
    valid_xs = ((valid_x - mu) / sd).astype(np.float32)
    model_info = train_model(train_xs, train_y, valid_xs, valid_y)
    results = {split: evaluate(items, model_info, mu, sd, use_group) for split, items in splits.items()}
    return {
        "variant": name,
        "use_group": use_group,
        "feature_dim": int(train_x.shape[1]),
        "row_counts": {"train": int(len(train_y)), "valid": int(len(valid_y))},
        "training_history": model_info["history"],
        "best_epoch": model_info["best_epoch"],
        "best_valid_row_rank_ic": model_info["best_valid_row_rank_ic"],
        "results": results,
    }


def main() -> None:
    started = time.perf_counter()
    set_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    market = POS.load_market()
    groups = load_groups(list(market["symbols"]))
    amount20_arr = RT.amount20(market)
    splits = {
        "train": build_split(market, amount20_arr, groups, TRAIN_START, TRAIN_END),
        "valid": build_split(market, amount20_arr, groups, VALID_START, VALID_END),
        "dev": build_split(market, amount20_arr, groups, DEV_START, DEV_END),
        "forward": build_split(market, amount20_arr, groups, FWD_START, FWD_END),
    }
    variants = {
        "stock_only": run_variant("stock_only", splits, False),
        "stock_plus_group": run_variant("stock_plus_group", splits, True),
    }
    g = variants["stock_plus_group"]["results"]
    s = variants["stock_only"]["results"]
    verdict = "group_diffusion_world_model_not_enough"
    dev_years = g["dev"]["by_year_model_top20"]
    if (
        g["forward"]["model_top"]["top20"] > g["forward"]["base_top"]["top20"] + 0.003
        and g["dev"]["model_top"]["top20"] > s["dev"]["model_top"]["top20"]
        and min(dev_years.values()) > 0.0
    ):
        verdict = "group_diffusion_world_model_candidate_needs_replay"
    out = {
        "experiment": "group_diffusion_world_model_v1",
        "method": "tiny_rank_mlp_with_stock_features_plus_static_industry_concept_diffusion_features",
        "caveat": "industry/concept membership is a 2026 static snapshot, so positive results would require PIT metadata validation before strategy use",
        "params": {"train": [TRAIN_START, TRAIN_END], "valid": [VALID_START, VALID_END], "dev": [DEV_START, DEV_END], "forward": [FWD_START, FWD_END], "interval": INTERVAL, "pool_size": POOL_SIZE, "domain_size": DOMAIN_SIZE, "top_ks": TOP_KS, "epochs": EPOCHS, "seed": SEED},
        "inputs_sha256": {"script": sha256(Path(__file__)), "round59_summary": sha256(ROOT / "cross_sectional_pairwise_spread_world_model_v1_summary.json"), "industry_csv": sha256(INDUSTRY_CSV), "sector_csv": sha256(SECTOR_CSV)},
        "group_coverage": {key: value for key, value in groups.items() if key not in {"industry", "concept"}},
        "sample_counts": {name: len(items) for name, items in splits.items()},
        "stock_feature_names": splits["train"][0]["feature_names"] if splits["train"] else [],
        "group_feature_names": GROUP_FEATURE_NAMES,
        "variants": variants,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "sha256": sha256(OUT), "verdict": verdict, "group_coverage": out["group_coverage"], "sample_counts": out["sample_counts"], "variants": {k: {"best_epoch": v["best_epoch"], "best_valid_row_rank_ic": v["best_valid_row_rank_ic"], "results": v["results"]} for k, v in variants.items()}, "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
