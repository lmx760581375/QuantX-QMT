from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
EXP129_PATH = REPO_ROOT / ".tmp/quantx-research/qmt-supply-demand-rebalance-v1/analyze_qmt_supply_demand_rebalance.py"
RANDOM_SEED = 20260714
TOPKS = (10, 15, 20, 30)
PAIR_POS_Q = 0.92
PAIR_NEG_Q = 0.25
MAX_POS_PER_SESSION = 36
MAX_NEG_PER_SESSION = 72
MAX_DUEL_POOL = 320
MAX_DUEL_OPPONENTS = 96


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP129 = load_module("exp129_for_session_contrastive_ranker", EXP129_PATH)
BASE_FEATURES = list(EXP129.FEATURES)
CONTRAST_FEATURES = [f"d_{name}" for name in BASE_FEATURES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    parser.add_argument("--raw-dir", default="data/raw/qmt/stocks")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--load-start", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--min-train-rows", type=int, default=100000)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    args = parse_args()
    market = EXP129.EXP124.load_market(args)
    panel = EXP129.build_panel(market, args)
    panel = add_targets(panel)
    scored, folds, importances = walk_forward(panel, args.min_train_rows)
    detail = evaluate_label(scored)
    accounts = replay_accounts(scored, market, args.horizon)
    stress = stress_diagnostics(accounts)
    result = {
        "status": classify_status(accounts, stress),
        "experiment": "qmt_session_contrastive_ranker_v1",
        "source_experiment": "qmt_supply_demand_rebalance_v1",
        "panel_rows": int(len(panel)),
        "scored_rows": int(len(scored)),
        "folds": folds,
        "label_results": summarize_detail(detail),
        "account_results": accounts,
        "leaderboard": build_leaderboard(accounts, detail, stress),
        "stress_diagnostics": stress,
        "feature_importance": summarize_importance(importances),
        "references": {
            "exp40_formal_rebalance5_dev_multiple": 8.7749,
            "exp40_formal_rebalance5_2026_return": 0.1918,
            "exp156_best_top20_multiple": 1.5205,
            "exp156_best_top20_2026": -0.0737,
        },
        "config": {
            "provider": args.provider,
            "raw_dir": args.raw_dir,
            "start": args.start,
            "end": args.end,
            "load_start": args.load_start,
            "horizon": args.horizon,
            "sample_step": args.sample_step,
            "topks": TOPKS,
            "features": BASE_FEATURES,
            "pair_pos_quantile": PAIR_POS_Q,
            "pair_neg_quantile": PAIR_NEG_Q,
            "max_pos_per_session": MAX_POS_PER_SESSION,
            "max_neg_per_session": MAX_NEG_PER_SESSION,
            "max_duel_pool": MAX_DUEL_POOL,
            "max_duel_opponents": MAX_DUEL_OPPONENTS,
            "causality": "Pairwise contrastive samples are built only from training sessions in years strictly before each test year. Test scoring uses T-visible stock features plus train-only winner/loser anchors, or within-test-session feature duels without any test labels. Labels and replay use scheduled T+1 open entry and T+6 open target exit only for historical training and OOS evaluation.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP plus raw amount/is_st only; no news, announcements, LHB, ETF, northbound, financing, static concept tables, or other message data.",
            "interpretation_guardrail": "This experiment changes the loss shape to same-session positive/negative contrast. It does not shrink TopK, hard-filter tails, use test-year labels for anchors, or tune by 2026 outcomes.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def add_targets(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.replace([np.inf, -np.inf], np.nan).fillna(0.5).copy()
    pct = out.groupby("session")["exec_label5_open"].rank(pct=True, method="first")
    out["right_tail_target"] = (pct >= 0.95).astype(int)
    out["top_quintile_target"] = (pct >= 0.80).astype(int)
    out["left_tail_target"] = (pct <= 0.20).astype(int)
    out["rank_pct_target"] = pct.astype(float)
    return out


def walk_forward(panel: pd.DataFrame, min_train_rows: int) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, float]]]:
    rows: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    importances: list[dict[str, float]] = []
    work = panel.replace([np.inf, -np.inf], np.nan).fillna(0.5).copy()
    for year in sorted(int(y) for y in work["year"].unique()):
        train = work[work["year"] < year].copy()
        test = work[work["year"] == year].copy()
        if len(train) < min_train_rows or test.empty:
            continue
        rng = np.random.default_rng(RANDOM_SEED + year)
        pair_x, pair_y, pair_meta = build_pair_training_frame(train, rng)
        contrast = make_contrast_classifier(year)
        reg = make_regressor(year, 0, 260)
        tail_cls = make_classifier(year, 101, 220)
        quintile_cls = make_classifier(year, 211, 220)
        contrast.fit(pair_x[CONTRAST_FEATURES], pair_y)
        reg.fit(train[BASE_FEATURES], train["exec_label5_open"])
        tail_cls.fit(train[BASE_FEATURES], train["right_tail_target"])
        quintile_cls.fit(train[BASE_FEATURES], train["top_quintile_target"])

        anchors = build_train_anchors(train)
        out = score_test(test, contrast, reg, tail_cls, quintile_cls, anchors, year)
        rows.append(out)
        folds.append({
            "year": int(year),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "sessions": int(test["session"].nunique()),
            "pair_rows": int(len(pair_x)),
            "pair_sessions": int(pair_meta["sessions"]),
            "pair_positive_rate": float(np.mean(pair_y)),
            "mean_pair_gap": float(pair_meta["mean_gap"]),
            "anchor_winner_rows": int(anchors["winner_count"]),
            "anchor_loser_rows": int(anchors["loser_count"]),
        })
        imp = {f"contrast::{name}": float(value) for name, value in zip(CONTRAST_FEATURES, contrast.feature_importances_)}
        imp.update({f"reg::{name}": float(value) for name, value in zip(BASE_FEATURES, reg.feature_importances_)})
        imp.update({f"right_tail::{name}": float(value) for name, value in zip(BASE_FEATURES, tail_cls.feature_importances_)})
        imp.update({f"top_quintile::{name}": float(value) for name, value in zip(BASE_FEATURES, quintile_cls.feature_importances_)})
        importances.append(imp)
        print(json.dumps(folds[-1], ensure_ascii=False), flush=True)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), folds, importances


def build_pair_training_frame(train: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    gaps: list[float] = []
    sessions = 0
    for _, group in train.groupby("session", sort=True):
        labels = group["exec_label5_open"].to_numpy(dtype=float)
        if len(group) < 200 or not np.isfinite(labels).any():
            continue
        hi = np.nanquantile(labels, PAIR_POS_Q)
        lo = np.nanquantile(labels, PAIR_NEG_Q)
        pos = np.flatnonzero(labels >= hi)
        neg = np.flatnonzero(labels <= lo)
        if len(pos) == 0 or len(neg) == 0:
            continue
        pos = sample_indices(pos, MAX_POS_PER_SESSION, rng)
        neg = sample_indices(neg, MAX_NEG_PER_SESSION, rng)
        x = group[BASE_FEATURES].to_numpy(dtype=np.float32, copy=False)
        pairs_pos = np.repeat(pos, len(neg))
        pairs_neg = np.tile(neg, len(pos))
        diff = x[pairs_pos] - x[pairs_neg]
        rev = -diff
        xs.append(np.vstack([diff, rev]).astype(np.float32, copy=False))
        ys.append(np.concatenate([np.ones(len(diff), dtype=np.int8), np.zeros(len(rev), dtype=np.int8)]))
        gaps.append(float(np.mean(labels[pairs_pos] - labels[pairs_neg])))
        sessions += 1
    if not xs:
        raise RuntimeError("empty pair training frame")
    pair_x = pd.DataFrame(np.vstack(xs), columns=CONTRAST_FEATURES)
    pair_y = np.concatenate(ys)
    return pair_x.replace([np.inf, -np.inf], np.nan).fillna(0.0), pair_y, {"sessions": sessions, "mean_gap": float(np.mean(gaps))}


def sample_indices(values: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if len(values) <= limit:
        return values
    return np.sort(rng.choice(values, size=limit, replace=False))


def build_train_anchors(train: pd.DataFrame) -> dict[str, Any]:
    winner_rows: list[pd.DataFrame] = []
    loser_rows: list[pd.DataFrame] = []
    mid_rows: list[pd.DataFrame] = []
    for _, group in train.groupby("session", sort=True):
        pct = group["rank_pct_target"].to_numpy(dtype=float)
        winner_rows.append(group.loc[pct >= 0.95, BASE_FEATURES])
        loser_rows.append(group.loc[pct <= 0.20, BASE_FEATURES])
        mid_rows.append(group.loc[(pct >= 0.45) & (pct <= 0.55), BASE_FEATURES])
    winner = pd.concat(winner_rows, ignore_index=True) if winner_rows else train[BASE_FEATURES].head(0)
    loser = pd.concat(loser_rows, ignore_index=True) if loser_rows else train[BASE_FEATURES].head(0)
    mid = pd.concat(mid_rows, ignore_index=True) if mid_rows else train[BASE_FEATURES].head(0)
    return {
        "winner": winner.mean(axis=0).to_numpy(dtype=np.float32),
        "loser": loser.mean(axis=0).to_numpy(dtype=np.float32),
        "mid": mid.mean(axis=0).to_numpy(dtype=np.float32),
        "winner_count": int(len(winner)),
        "loser_count": int(len(loser)),
        "mid_count": int(len(mid)),
    }


def score_test(
    test: pd.DataFrame,
    contrast: LGBMClassifier,
    reg: LGBMRegressor,
    tail_cls: LGBMClassifier,
    quintile_cls: LGBMClassifier,
    anchors: dict[str, Any],
    year: int,
) -> pd.DataFrame:
    out = test.copy()
    x = out[BASE_FEATURES].to_numpy(dtype=np.float32, copy=False)
    loser = anchors["loser"]
    winner = anchors["winner"]
    mid = anchors["mid"]
    out["score_anchor_loser"] = predict_diff(contrast, x - loser[None, :])
    out["score_anchor_mid"] = predict_diff(contrast, x - mid[None, :])
    out["score_anchor_margin"] = out["score_anchor_loser"] - predict_diff(contrast, winner[None, :] - x)
    out["score_return_reg"] = reg.predict(out[BASE_FEATURES])
    out["score_right_tail"] = tail_cls.predict_proba(out[BASE_FEATURES])[:, 1]
    out["score_top_quintile"] = quintile_cls.predict_proba(out[BASE_FEATURES])[:, 1]
    out["score_contrast_blend"] = blend_ranks(
        out,
        [("score_anchor_loser", 0.30), ("score_anchor_margin", 0.28), ("score_anchor_mid", 0.18), ("score_return_reg", 0.14), ("score_top_quintile", 0.10)],
    )
    out["score_duel_pool"] = np.nan
    out["score_duel_blend"] = np.nan
    for session, idx in out.groupby("session", sort=True).groups.items():
        group = out.loc[idx]
        pre_rank = group["score_contrast_blend"].rank(pct=True, method="first").to_numpy(dtype=float)
        pool_local = np.flatnonzero(pre_rank >= max(0.0, 1.0 - MAX_DUEL_POOL / max(len(group), 1)))
        if len(pool_local) == 0:
            continue
        session_scores = duel_scores_for_pool(group.iloc[pool_local], contrast, year + int(str(session).replace("-", "")[-4:]))
        out.loc[group.index[pool_local], "score_duel_pool"] = session_scores
    out["score_duel_pool"] = out["score_duel_pool"].fillna(out["score_anchor_margin"])
    out["score_duel_blend"] = blend_ranks(
        out,
        [("score_duel_pool", 0.38), ("score_anchor_margin", 0.24), ("score_anchor_loser", 0.16), ("score_return_reg", 0.12), ("score_right_tail", 0.10)],
    )
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.5)


def predict_diff(model: LGBMClassifier, diff: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(diff, columns=CONTRAST_FEATURES).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return model.predict_proba(frame)[:, 1]


def duel_scores_for_pool(group: pd.DataFrame, contrast: LGBMClassifier, seed: int) -> np.ndarray:
    x = group[BASE_FEATURES].to_numpy(dtype=np.float32, copy=False)
    n = len(group)
    if n <= 1:
        return np.ones(n, dtype=float) * 0.5
    rng = np.random.default_rng(RANDOM_SEED + seed)
    left: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    for i in range(n):
        opponents = np.delete(np.arange(n), i)
        opponents = sample_indices(opponents, min(MAX_DUEL_OPPONENTS, len(opponents)), rng)
        left.append(x[i][None, :] - x[opponents])
        owners.append(np.full(len(opponents), i, dtype=np.int32))
    diff = np.vstack(left).astype(np.float32, copy=False)
    owner = np.concatenate(owners)
    probs = predict_diff(contrast, diff)
    scores = np.bincount(owner, weights=probs, minlength=n)
    counts = np.bincount(owner, minlength=n)
    return scores / np.maximum(counts, 1.0)


def make_contrast_classifier(year: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=320, learning_rate=0.035, num_leaves=35, max_depth=6,
        min_child_samples=420, subsample=0.88, colsample_bytree=0.90,
        reg_alpha=1.0, reg_lambda=8.0, random_state=RANDOM_SEED + year,
        n_jobs=6, verbosity=-1,
    )


def make_regressor(year: int, offset: int, estimators: int) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=estimators, learning_rate=0.032, num_leaves=35, max_depth=6,
        min_child_samples=520, subsample=0.85, colsample_bytree=0.88,
        reg_alpha=1.0, reg_lambda=8.0, random_state=RANDOM_SEED + year + offset,
        n_jobs=6, verbosity=-1,
    )


def make_classifier(year: int, offset: int, estimators: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=estimators, learning_rate=0.034, num_leaves=31, max_depth=5,
        min_child_samples=500, subsample=0.86, colsample_bytree=0.88,
        reg_alpha=1.0, reg_lambda=8.0, random_state=RANDOM_SEED + year + offset,
        n_jobs=6, verbosity=-1,
    )


def blend_ranks(frame: pd.DataFrame, parts: list[tuple[str, float]]) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    for col, weight in parts:
        rank = frame.groupby("session")[col].rank(pct=True, method="first")
        if weight >= 0:
            score = score + weight * rank
        else:
            score = score + abs(weight) * (1.0 - rank)
    return score


def variants() -> list[tuple[str, str]]:
    return [
        ("anchor_loser", "score_anchor_loser"),
        ("anchor_mid", "score_anchor_mid"),
        ("anchor_margin", "score_anchor_margin"),
        ("duel_pool", "score_duel_pool"),
        ("contrast_blend", "score_contrast_blend"),
        ("duel_blend", "score_duel_blend"),
        ("return_reg", "score_return_reg"),
        ("right_tail", "score_right_tail"),
        ("top_quintile", "score_top_quintile"),
    ]


def evaluate_label(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session, group in scored.groupby("session", sort=True):
        for variant, score_col in variants():
            ranked = group.sort_values([score_col, "instrument"], ascending=[False, True])
            for topk in TOPKS:
                selected = ranked.head(topk)
                rows.append({
                    "session": str(session),
                    "year": int(str(session)[:4]),
                    "variant": variant,
                    "topk": int(topk),
                    "mean_label5_open": float(selected["label5_open"].mean()),
                    "mean_exec_label5_open": float(selected["exec_label5_open"].mean()),
                    "mean_raw5_open": float(selected["raw5_open"].mean()),
                    "entry_ok": float(selected["entry_ok"].mean()),
                    "exit_ok": float(selected["exit_ok"].mean()),
                    "count": int(len(selected)),
                })
    return pd.DataFrame(rows)


def summarize_detail(detail: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for (variant, topk), group in detail.groupby(["variant", "topk"]):
        out[f"{variant}::top{topk}"] = {
            "mean_label5_open": float(group["mean_label5_open"].mean()),
            "mean_exec_label5_open": float(group["mean_exec_label5_open"].mean()),
            "mean_raw5_open": float(group["mean_raw5_open"].mean()),
            "entry_ok": float(group["entry_ok"].mean()),
            "exit_ok": float(group["exit_ok"].mean()),
            "positive_session_ratio": float((group["mean_exec_label5_open"] > 0).mean()),
            "by_year": {
                str(int(year)): {
                    "mean_exec_label5_open": float(yg["mean_exec_label5_open"].mean()),
                    "mean_raw5_open": float(yg["mean_raw5_open"].mean()),
                    "entry_ok": float(yg["entry_ok"].mean()),
                    "positive_session_ratio": float((yg["mean_exec_label5_open"] > 0).mean()),
                }
                for year, yg in group.groupby("year")
            },
        }
    return out


def replay_accounts(scored: pd.DataFrame, market: dict[str, Any], horizon: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    grouped = {str(session): group for session, group in scored.groupby("session", sort=True)}
    for variant, score_col in variants():
        for topk in TOPKS:
            selections = {
                session: list(group.sort_values([score_col, "instrument"], ascending=[False, True]).head(topk)["instrument"].astype(str))
                for session, group in grouped.items()
            }
            account = EXP129.replay_open_selection(selections, market, horizon)
            account["period_curve"] = account_period_curve(selections, market, horizon)
            out[f"{variant}::top{topk}"] = account
    return out


def account_period_curve(selections: dict[str, list[str]], market: dict[str, Any], horizon: int) -> list[dict[str, Any]]:
    symbol_index = {s: i for i, s in enumerate(market["symbols"])}
    dates = market["dates"]
    arr = market["arrays"]
    rows: list[dict[str, Any]] = []
    nav = 1.0
    for session, candidates in sorted(selections.items()):
        idx = market["date_index"][session]
        entry_idx = idx + 1
        exit_idx = entry_idx + horizon
        if exit_idx >= len(dates):
            continue
        rets: list[float] = []
        holds: list[int] = []
        for sym in candidates:
            col = symbol_index.get(sym)
            if col is None:
                continue
            entry = float(arr["open"][entry_idx, col])
            preclose = float(arr["close"][idx, col])
            vol = float(arr["volume"][entry_idx, col])
            if not np.isfinite(entry) or entry <= 0 or not np.isfinite(preclose) or preclose <= 0 or not np.isfinite(vol) or vol <= 0 or entry / preclose - 1.0 >= 0.095:
                continue
            sell_idx = exit_idx
            while sell_idx < min(len(dates), exit_idx + 6):
                sell_open = float(arr["open"][sell_idx, col])
                sell_preclose = float(arr["close"][sell_idx - 1, col])
                sell_vol = float(arr["volume"][sell_idx, col])
                if np.isfinite(sell_open) and sell_open > 0 and np.isfinite(sell_preclose) and sell_preclose > 0 and np.isfinite(sell_vol) and sell_vol > 0 and sell_open / sell_preclose - 1.0 > -0.095:
                    break
                sell_idx += 1
            if sell_idx >= min(len(dates), exit_idx + 6):
                continue
            exit_ = float(arr["open"][sell_idx, col])
            rets.append(exit_ / entry - 1.0 - 0.00154)
            holds.append(sell_idx - entry_idx)
        period_return = float(np.mean(rets)) if rets else 0.0
        nav *= 1.0 + period_return
        rows.append({
            "session": str(session),
            "year": int(str(session)[:4]),
            "period_return": period_return,
            "selected_count": int(len(rets)),
            "avg_hold_days": float(np.mean(holds)) if holds else 0.0,
            "nav": float(nav),
        })
    return rows


def stress_diagnostics(accounts: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, account in accounts.items():
        curve = account.get("period_curve", [])
        returns = np.asarray([row["period_return"] for row in curve], dtype=float)
        if returns.size == 0:
            continue
        rows: dict[str, Any] = {}
        for n in (1, 3, 5):
            if returns.size <= n:
                rows[f"remove_best_{n}_final_multiple"] = 1.0
                continue
            keep = np.ones(returns.size, dtype=bool)
            keep[np.argsort(returns)[-n:]] = False
            rows[f"remove_best_{n}_final_multiple"] = float(np.prod(1.0 + returns[keep]))
        rows["best_period_return"] = float(np.max(returns))
        rows["worst_period_return"] = float(np.min(returns))
        rows["positive_period_ratio"] = float(np.mean(returns > 0))
        out[key] = rows
    return out


def build_leaderboard(accounts: dict[str, Any], detail: pd.DataFrame, stress: dict[str, Any]) -> list[dict[str, Any]]:
    label_map = summarize_detail(detail)
    rows: list[dict[str, Any]] = []
    for key, account in accounts.items():
        annual = account.get("annual_returns", {})
        label = label_map.get(key, {})
        stress_row = stress.get(key, {})
        rows.append({
            "key": key,
            "final_multiple": float(account.get("final_multiple", 1.0)),
            "ret2026": float(annual.get("2026", np.nan)) if annual else np.nan,
            "min_annual_return": float(min(annual.values())) if annual else 0.0,
            "all_years_positive": bool(account.get("all_years_positive", False)),
            "max_drawdown_period": float(account.get("max_drawdown_period", 0.0)),
            "avg_selected_count": float(account.get("avg_selected_count", 0.0)),
            "avg_hold_days": float(account.get("avg_hold_days", 0.0)),
            "mean_exec_label5_open": float(label.get("mean_exec_label5_open", 0.0)),
            "mean_raw5_open": float(label.get("mean_raw5_open", 0.0)),
            "entry_ok": float(label.get("entry_ok", 0.0)),
            "remove_best_3_final_multiple": float(stress_row.get("remove_best_3_final_multiple", np.nan)),
            "positive_period_ratio": float(stress_row.get("positive_period_ratio", np.nan)),
        })
    rows.sort(key=lambda r: (r["all_years_positive"], r["final_multiple"], r["ret2026"]), reverse=True)
    return rows


def summarize_importance(importances: list[dict[str, float]]) -> list[dict[str, Any]]:
    bucket: dict[str, list[float]] = {}
    for imp in importances:
        for name, value in imp.items():
            bucket.setdefault(name, []).append(float(value))
    rows = [(name, float(np.mean(values))) for name, values in bucket.items()]
    rows.sort(key=lambda item: item[1], reverse=True)
    return [{"feature": name, "mean_importance": value} for name, value in rows[:100]]


def classify_status(accounts: dict[str, Any], stress: dict[str, Any]) -> str:
    for key, account in accounts.items():
        annual = account.get("annual_returns", {})
        robust_after_best3 = stress.get(key, {}).get("remove_best_3_final_multiple", 0.0)
        if (
            account.get("final_multiple", 0.0) >= 20.0
            and annual.get("2026", -1.0) > 0.0
            and account.get("all_years_positive", False)
            and account.get("avg_selected_count", 0.0) > 5.0
            and 4.5 <= account.get("avg_hold_days", 0.0) <= 11.0
            and robust_after_best3 >= 10.0
        ):
            return "candidate_needs_deeper_audit_before_merge"
    return "rejected_before_formal_candidate"


def print_summary(result: dict[str, Any], checksum: str) -> None:
    print(f"sha256:{checksum}")
    print(f"status={result['status']}")
    for row in result["leaderboard"][:18]:
        print(
            f"{row['key']}: final={row['final_multiple']:.2f}x 2026={row['ret2026']:+.4f} "
            f"min_year={row['min_annual_return']:+.4f} avg_count={row['avg_selected_count']:.2f} "
            f"hold={row['avg_hold_days']:.2f} exec={row['mean_exec_label5_open']:+.6f} "
            f"rm_best3={row['remove_best_3_final_multiple']:.2f}x"
        )


if __name__ == "__main__":
    raise SystemExit(main())
