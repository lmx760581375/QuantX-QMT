from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FINALIZE_PATH = REPO_ROOT / ".tmp/quantx-research/path-sequence-ranker-v2/finalize_candidate_report.py"

RANDOM_SEED = 20260715
GRID_0932 = {"base": 0.30, "path_ev": 0.45, "path_spread": 0.05, "path_topq": 0.05, "exec_cls": 0.15}
GRID_0878 = {"base": 0.30, "path_ev": 0.30, "path_spread": 0.20, "path_topq": 0.05, "exec_cls": 0.15}
DEFAULT_BLEND_ALPHAS = "0,0.01,0.02,0.03,0.05,0.1,0.15,0.2,0.3,0.4"
DEFAULT_TOPKS = "20"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FINALIZE = load_module("path_sequence_v2_finalize_for_dl", FINALIZE_PATH)
SEARCH = FINALIZE.SEARCH
V1 = FINALIZE.V1


class PairwiseMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev, dim))
            if i == 0:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-cache", default=".tmp/quantx-research/path-sequence-ranker-v2/scored_panel_v2_with_aux.parquet")
    parser.add_argument("--provider", default="data/qlib_data_fixed")
    parser.add_argument("--raw-dir", default="data/raw/qmt/stocks")
    parser.add_argument("--load-start", default="2020-01-01")
    parser.add_argument("--end", default="2026-07-10")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--pairs-per-session", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dims", default="128,64")
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--alphas", default=DEFAULT_BLEND_ALPHAS)
    parser.add_argument("--topks", default=DEFAULT_TOPKS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=".tmp/quantx-research/path-sequence-dl-v1/pairwise_mlp_ranker_report.json")
    return parser.parse_args()


def main() -> int:
    set_seeds(RANDOM_SEED)
    args = parse_args()
    device = choose_device(args.device)
    args.hidden_dims_list = parse_int_list(args.hidden_dims)
    args.alpha_list = parse_float_list(args.alphas)
    args.topk_list = parse_int_list(args.topks)
    scored = pd.read_parquet(args.prepared_cache).replace([np.inf, -np.inf], np.nan).fillna(0.5)
    features = feature_columns(scored)
    scored, folds = walk_forward_pairwise(scored, features, args, device)
    scored = add_fixed_scores(scored)
    market_args = argparse.Namespace(provider=args.provider, raw_dir=args.raw_dir, load_start=args.load_start, end=args.end)
    market = V1.EXP124.load_market(market_args)
    replay_variants.alpha_list = args.alpha_list
    replay_variants.topk_list = args.topk_list
    accounts = replay_variants(scored, market, args.horizon)
    result = {
        "status": "pairwise_mlp_ranker_complete",
        "experiment": "path_sequence_dl_v1_pairwise_mlp",
        "device": str(device),
        "features": features,
        "folds": folds,
        "account_results": accounts,
        "best_by_dev": best_rows(accounts, "dev_final_2022_2025"),
        "best_by_2026_with_dev_gate": best_2026_with_dev_gate(accounts),
        "references": {
            "v2_grid_0932_dev_final_2022_2025": 9.100280862807718,
            "v2_grid_0932_2026": 0.2167798572108801,
            "v2_grid_0932_period_sharpe_proxy": 1.7032439401289254,
            "documented_exp40_dev_final_2022_2025": 8.7749,
            "documented_exp40_2026_return": 0.1918,
        },
        "config": {
            "prepared_cache": args.prepared_cache,
            "prepared_cache_sha256": sha256(Path(args.prepared_cache)),
            "epochs": args.epochs,
            "pairs_per_session": args.pairs_per_session,
            "batch_size": args.batch_size,
            "hidden_dims": args.hidden_dims_list,
            "dropout": args.dropout,
            "alphas": args.alpha_list,
            "topks": args.topk_list,
            "positive_label": "session exec_label5_open top 15pct",
            "negative_label": "session exec_label5_open bottom 35pct",
            "causality": "Pairwise MLP is trained walk-forward by year on prior years only. 2022 has neutral DL score due to unavailable prior scored history. 2026 is forward evaluation and is not used to select blend alpha.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = sha256(output)
    print(checksum)
    for row in result["best_by_dev"][:20]:
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return 0


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # CPU is more deterministic and fast enough for this tabular pairwise probe.
    return torch.device("cpu")


def feature_columns(scored: pd.DataFrame) -> list[str]:
    excluded = {
        "session", "year", "instrument", "raw5_open", "label5_open", "exec_label5_open",
        "target_quintile", "target_exec_top20", "entry_ok", "exit_ok", "aux_available",
    }
    preferred = [
        "part_base", "part_path_ev", "part_path_spread", "part_path_topq", "part_exec_cls",
        "score_blend_topq_25", "score_exec_reg", "score_exec_cls",
        "score_rank_consensus", "score_stable_exec_cls_15",
    ]
    cols = [c for c in preferred if c in scored.columns]
    for c in V1.FEATURES:
        if c in scored.columns and c not in excluded and c not in cols:
            cols.append(c)
    return cols


def walk_forward_pairwise(scored: pd.DataFrame, features: list[str], args: argparse.Namespace, device: torch.device) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for year in (2022, 2023, 2024, 2025, 2026):
        train = scored[scored["year"] < year].copy()
        test = scored[scored["year"] == year].copy()
        if test.empty:
            continue
        if train["session"].nunique() < 80:
            out = test.copy()
            out["score_dl_pairwise"] = 0.0
            out["score_dl_pairwise_pct"] = 0.5
            rows.append(out)
            folds.append({"year": year, "train_rows": int(len(train)), "test_rows": int(len(test)), "train_sessions": int(train["session"].nunique()), "dl_available": False})
            print(json.dumps({"dl_fold": folds[-1]}, ensure_ascii=False), flush=True)
            continue
        x_train, y_train, scaler = prepare_matrix(train, features)
        model, train_summary = train_pairwise_model(train, x_train, y_train, args, device, year)
        x_test = apply_scaler(test, features, scaler)
        with torch.no_grad():
            pred = predict_in_batches(model, x_test, device)
        out = test.copy()
        out["score_dl_pairwise"] = pred
        out["score_dl_pairwise_pct"] = session_rank_pct(out, "score_dl_pairwise")
        rows.append(out)
        fold = {"year": year, "train_rows": int(len(train)), "test_rows": int(len(test)), "train_sessions": int(train["session"].nunique()), "test_sessions": int(test["session"].nunique()), "dl_available": True, **train_summary}
        folds.append(fold)
        print(json.dumps({"dl_fold": fold}, ensure_ascii=False), flush=True)
    return pd.concat(rows, ignore_index=True), folds


def prepare_matrix(frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    x = frame[features].to_numpy(dtype=np.float32, copy=True)
    y = frame["exec_label5_open"].to_numpy(dtype=np.float32, copy=True)
    mean = np.nanmean(x, axis=0).astype(np.float32)
    std = np.nanstd(x, axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    x = np.clip((x - mean) / std, -6.0, 6.0).astype(np.float32)
    return x, y, {"mean": mean, "std": std}


def apply_scaler(frame: pd.DataFrame, features: list[str], scaler: dict[str, np.ndarray]) -> np.ndarray:
    x = frame[features].to_numpy(dtype=np.float32, copy=True)
    return np.clip((x - scaler["mean"]) / scaler["std"], -6.0, 6.0).astype(np.float32)


def train_pairwise_model(frame: pd.DataFrame, x: np.ndarray, y: np.ndarray, args: argparse.Namespace, device: torch.device, year: int) -> tuple[PairwiseMLP, dict[str, Any]]:
    model = PairwiseMLP(x.shape[1], args.hidden_dims_list, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    pairs = make_pairs(frame, y, args.pairs_per_session, RANDOM_SEED + year)
    if len(pairs) == 0:
        raise RuntimeError(f"no pairs for {year}")
    pos_idx = pairs[:, 0]
    neg_idx = pairs[:, 1]
    losses: list[float] = []
    model.train()
    for epoch in range(args.epochs):
        order = np.random.permutation(len(pairs))
        epoch_losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            xp = torch.from_numpy(x[pos_idx[batch]]).to(device)
            xn = torch.from_numpy(x[neg_idx[batch]]).to(device)
            sp = model(xp)
            sn = model(xn)
            loss = torch.nn.functional.softplus(-(sp - sn)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    return model.eval(), {"pair_count": int(len(pairs)), "loss_first": losses[0], "loss_last": losses[-1], "param_count": count_params(model)}


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def make_pairs(frame: pd.DataFrame, y: np.ndarray, pairs_per_session: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = frame.reset_index(drop=True)
    pairs: list[tuple[int, int]] = []
    for _, idx in frame.groupby("session", sort=False).indices.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        vals = y[idx_arr]
        if len(idx_arr) < 20:
            continue
        ranks = pd.Series(vals).rank(pct=True, method="first").to_numpy(dtype=np.float32)
        pos = idx_arr[ranks >= 0.85]
        neg = idx_arr[ranks <= 0.35]
        if len(pos) == 0 or len(neg) == 0:
            continue
        count = min(pairs_per_session, len(pos) * len(neg))
        pos_pick = rng.choice(pos, size=count, replace=True)
        neg_pick = rng.choice(neg, size=count, replace=True)
        pairs.extend(zip(pos_pick.tolist(), neg_pick.tolist()))
    rng.shuffle(pairs)
    return np.asarray(pairs, dtype=np.int64)


def predict_in_batches(model: PairwiseMLP, x: np.ndarray, device: torch.device, batch_size: int = 8192) -> np.ndarray:
    preds = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start:start + batch_size]).to(device)
        preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds).astype(np.float64)


def session_rank_pct(frame: pd.DataFrame, col: str) -> pd.Series:
    ranked = frame.groupby("session")[col].rank(pct=True, method="first")
    nunique = frame.groupby("session")[col].transform("nunique")
    return ranked.where(nunique > 1, 0.5)


def add_fixed_scores(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["score_grid_0932"] = weighted_part_score(out, GRID_0932)
    out["score_grid_0878"] = weighted_part_score(out, GRID_0878)
    return out


def weighted_part_score(frame: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    score = np.zeros(len(frame), dtype=np.float64)
    for part, weight in weights.items():
        if weight:
            score += weight * SEARCH.one_dim(frame, f"part_{part}")
    return score


def replay_variants(scored: pd.DataFrame, market: dict[str, Any], horizon: int) -> list[dict[str, Any]]:
    variants: dict[str, np.ndarray] = {}
    for base_name in ("grid_0932", "grid_0878"):
        base = scored[f"score_{base_name}"].to_numpy(dtype=np.float64)
        dl = scored["score_dl_pairwise_pct"].to_numpy(dtype=np.float64)
        for alpha in getattr(replay_variants, "alpha_list"):
            variants[f"{base_name}_dl_alpha_{alpha:.2f}"] = (1.0 - alpha) * base + alpha * dl
    variants["dl_pairwise_only"] = scored["score_dl_pairwise_pct"].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for name, score in variants.items():
        for topk in getattr(replay_variants, "topk_list"):
            rows.append(evaluate_score_variant(name, score, scored, market, horizon, topk))
    rows.sort(key=lambda r: (r["all_years_positive"], r["dev_final_2022_2025"], r["annual_returns"].get("2026", -999.0), r["period_sharpe_proxy"]), reverse=True)
    return rows


def evaluate_score_variant(name: str, score: np.ndarray, scored: pd.DataFrame, market: dict[str, Any], horizon: int, topk: int) -> dict[str, Any]:
    due = V1.due_sessions(scored)
    work = scored[scored["session"].isin(due)].copy()
    work["variant_score"] = score[work.index.to_numpy()]
    selections = {
        session: list(group.sort_values(["variant_score", "instrument"], ascending=[False, True]).head(topk)["instrument"].astype(str))
        for session, group in work.groupby("session", sort=True)
    }
    account = FINALIZE.replay_with_curve(selections, market, horizon)
    annual = account["annual_returns"]
    dev_final = float(np.prod([1.0 + annual.get(str(y), 0.0) for y in (2022, 2023, 2024, 2025)]))
    full_final = float(dev_final * (1.0 + annual.get("2026", 0.0)))
    return {
        "name": name,
        "topk": int(topk),
        "dev_final_2022_2025": dev_final,
        "full_final_2022_2026": full_final,
        **account,
        "improves_grid_0932": bool(
            dev_final > 9.100280862807718
            and annual.get("2026", -999.0) > 0.2167798572108801
            and account["period_sharpe_proxy"] > 1.7032439401289254
            and account["avg_selected_count"] >= 5.0
            and account["avg_selected_count"] <= 10.0
            and account["max_drawdown_period"] >= -0.3262
        ),
    }


def best_rows(accounts: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    rows = sorted(accounts, key=lambda r: (r["all_years_positive"], r[key], r["period_sharpe_proxy"]), reverse=True)
    return [summary_row(r) for r in rows[:20]]


def best_2026_with_dev_gate(accounts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [r for r in accounts if r["dev_final_2022_2025"] >= 8.7749 and r["all_years_positive"]]
    rows.sort(key=lambda r: (r["annual_returns"].get("2026", -999.0), r["period_sharpe_proxy"]), reverse=True)
    return [summary_row(r) for r in rows[:20]]


def summary_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": row["name"],
        "topk": row.get("topk"),
        "dev_final_2022_2025": row["dev_final_2022_2025"],
        "return_2026": row["annual_returns"].get("2026"),
        "full_final_2022_2026": row["full_final_2022_2026"],
        "max_drawdown_period": row["max_drawdown_period"],
        "avg_selected_count": row["avg_selected_count"],
        "avg_hold_days": row["avg_hold_days"],
        "period_sharpe_proxy": row["period_sharpe_proxy"],
        "annual_sharpe_proxy": row["annual_sharpe_proxy"],
        "improves_grid_0932": row["improves_grid_0932"],
    }


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
