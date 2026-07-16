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
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path("/Users/mingxiaoli/Documents/QuantX-QMT")
EXP130_PATH = REPO_ROOT / ".tmp/quantx-research/qmt-style-flow-rotation-v1/analyze_qmt_style_flow_rotation.py"
EXP132_PATH = REPO_ROOT / ".tmp/quantx-research/qmt-regime-conditioned-path-family-soil-v1/analyze_qmt_regime_conditioned_path_family_soil.py"
TOPKS = (10, 20)
RANDOM_DRAWS = 200


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP130 = load_module(EXP130_PATH, "exp130")
EXP132 = load_module(EXP132_PATH, "exp132")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    parser.add_argument("--raw-dir", default="data/raw/qmt/stocks")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--load-start", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    market = EXP130.EXP124.load_market(args)
    panel = EXP130.build_panel(market, args)
    EXP132.add_stock_scores(panel)
    session = build_session_surface(panel)
    models = walk_forward_opportunity_models(session)
    accounts = summarize_accounts(session)
    result = {
        "status": "diagnostic_complete",
        "experiment": "qmt_cross_sectional_payoff_surface_diagnostic_v1",
        "panel_rows": int(len(panel)),
        "session_rows": int(len(session)),
        "payoff_surface": summarize_surface(session),
        "opportunity_models": models,
        "account_proxies": accounts,
        "leaderboard": build_leaderboard(accounts),
        "config": {
            "source_experiments": [str(EXP130_PATH.relative_to(REPO_ROOT)), str(EXP132_PATH.relative_to(REPO_ROOT))],
            "provider": args.provider,
            "raw_dir": args.raw_dir,
            "start": args.start,
            "end": args.end,
            "load_start": args.load_start,
            "horizon": args.horizon,
            "sample_step": args.sample_step,
            "topks": TOPKS,
            "random_draws_per_session": RANDOM_DRAWS,
            "families": list(EXP132.stock_score_specs().keys()),
            "causality": "Session features use completed daily bars through signal date T only. Oracle payoff, best-family payoff, and random payoff use future returns only as diagnostics. Walk-forward opportunity models train only on sessions from years strictly before the test year.",
            "scope": "QMT/qlib retrievable daily OHLCV/VWAP and raw amount/is_st only; no news, announcements, LHB, ETF, northbound, financing, static industry/concept table, or message/event data.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print_summary(result, checksum)
    return 0


def build_session_surface(panel: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260714)
    rows: list[dict[str, Any]] = []
    families = list(EXP132.stock_score_specs().keys())
    visible_cols = [
        "mkt_disp20", "mkt_amount_disp20", "mkt_breadth20", "mkt_ret20_median", "mkt_near_high20",
        "ret20_rank", "ret60_rank", "amount_rank", "amt_ratio20_rank", "vol20_low_rank", "range20_low_rank",
        "near_high20_rank", "close_strength_rank", "upper_wick_low_rank", "price_rank",
    ]
    for session, group in panel.groupby("session", sort=True):
        g = group.copy()
        bench = (g["raw5_open"] - g["label5_open"]).to_numpy(dtype=float)
        g["exec_raw_proxy"] = g["exec_label5_open"].to_numpy(dtype=float) + bench
        row: dict[str, Any] = {"session": str(session), "year": int(str(session)[:4]), "universe_count": int(len(g))}
        for col in ["mkt_disp20", "mkt_amount_disp20", "mkt_breadth20", "mkt_ret20_median", "mkt_near_high20"]:
            row[col] = float(g[col].iloc[0])
        for col in visible_cols[5:]:
            values = g[col].to_numpy(dtype=float)
            row[f"xs_{col}_mean"] = float(np.nanmean(values))
            row[f"xs_{col}_std"] = float(np.nanstd(values))
            row[f"xs_{col}_p90"] = float(np.nanquantile(values, 0.90))
        for topk in TOPKS:
            oracle = g.sort_values(["exec_label5_open", "instrument"], ascending=[False, True]).head(topk)
            row[f"oracle_top{topk}_exec_excess"] = float(oracle["exec_label5_open"].mean())
            row[f"oracle_top{topk}_raw_proxy"] = float(oracle["exec_raw_proxy"].mean())
            row[f"oracle_top{topk}_entry_ok"] = float(oracle["entry_ok"].mean())
            random_values = random_topk_means(g["exec_raw_proxy"].to_numpy(dtype=float), topk, rng)
            row[f"random_top{topk}_raw_mean"] = float(np.mean(random_values))
            row[f"random_top{topk}_raw_p95"] = float(np.quantile(random_values, 0.95))
            best_family_value = -np.inf
            best_family = ""
            for family in families:
                selected = g.sort_values([f"score_{family}", "instrument"], ascending=[False, True]).head(topk)
                raw_mean = float(selected["exec_raw_proxy"].mean())
                exec_mean = float(selected["exec_label5_open"].mean())
                row[f"family_{family}_top{topk}_raw_proxy"] = raw_mean
                row[f"family_{family}_top{topk}_exec_excess"] = exec_mean
                row[f"family_{family}_top{topk}_score_mean"] = float(selected[f"score_{family}"].mean())
                row[f"family_{family}_top{topk}_entry_ok"] = float(selected["entry_ok"].mean())
                if raw_mean > best_family_value:
                    best_family_value = raw_mean
                    best_family = family
            row[f"best_family_top{topk}"] = best_family
            row[f"best_family_top{topk}_raw_proxy"] = float(best_family_value)
            row[f"oracle_minus_best_family_top{topk}"] = float(row[f"oracle_top{topk}_raw_proxy"] - best_family_value)
        rows.append(row)
    return pd.DataFrame(rows)


def random_topk_means(values: np.ndarray, topk: int, rng: np.random.Generator) -> np.ndarray:
    valid = values[np.isfinite(values)]
    if len(valid) <= topk:
        return np.asarray([float(np.nanmean(valid))])
    draws = []
    for _ in range(RANDOM_DRAWS):
        idx = rng.choice(len(valid), size=topk, replace=False)
        draws.append(float(valid[idx].mean()))
    return np.asarray(draws, dtype=float)


def walk_forward_opportunity_models(session: pd.DataFrame) -> dict[str, Any]:
    feature_cols = [c for c in session.columns if c.startswith("mkt_") or c.startswith("xs_") or c.endswith("_score_mean")]
    out: dict[str, Any] = {"features": feature_cols[:80], "folds": []}
    for topk in TOPKS:
        target_col = f"oracle_top{topk}_raw_proxy"
        threshold = float(session[target_col].quantile(0.67))
        pred_rows = []
        for year in sorted(int(y) for y in session["year"].unique()):
            train = session[session["year"] < year]
            test = session[session["year"] == year]
            if len(train) < 40 or test.empty:
                continue
            y_train = train[target_col].to_numpy(dtype=float)
            y_cls = (y_train >= np.quantile(y_train, 0.67)).astype(int)
            reg = LGBMRegressor(n_estimators=80, learning_rate=0.04, num_leaves=7, min_child_samples=8, reg_alpha=1.0, reg_lambda=5.0, random_state=year, n_jobs=4, verbosity=-1)
            cls = LGBMClassifier(n_estimators=80, learning_rate=0.04, num_leaves=7, min_child_samples=8, reg_alpha=1.0, reg_lambda=5.0, random_state=year + 17, n_jobs=4, verbosity=-1)
            reg.fit(train[feature_cols], y_train)
            cls.fit(train[feature_cols], y_cls)
            pred_reg = reg.predict(test[feature_cols])
            pred_prob = cls.predict_proba(test[feature_cols])[:, 1]
            for i, (_, row) in enumerate(test.iterrows()):
                pred_rows.append({
                    "year": int(year),
                    "session": row["session"],
                    "actual": float(row[target_col]),
                    "is_high_global": int(float(row[target_col]) >= threshold),
                    "pred_reg": float(pred_reg[i]),
                    "pred_prob": float(pred_prob[i]),
                })
        preds = pd.DataFrame(pred_rows)
        if preds.empty:
            continue
        summary = {
            "topk": int(topk),
            "global_high_threshold": threshold,
            "rank_corr_reg": safe_corr(preds["actual"], preds["pred_reg"]),
            "rank_corr_prob": safe_corr(preds["actual"], preds["pred_prob"]),
            "auc_global_high": safe_auc(preds["is_high_global"], preds["pred_prob"]),
            "by_year": {},
        }
        for year, group in preds.groupby("year"):
            summary["by_year"][str(int(year))] = {
                "sessions": int(len(group)),
                "actual_mean": float(group["actual"].mean()),
                "rank_corr_reg": safe_corr(group["actual"], group["pred_reg"]),
                "rank_corr_prob": safe_corr(group["actual"], group["pred_prob"]),
                "auc_global_high": safe_auc(group["is_high_global"], group["pred_prob"]),
            }
        out[f"top{topk}"] = summary
    return out


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3 or a.nunique() <= 1 or b.nunique() <= 1:
        return float("nan")
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def safe_auc(y: pd.Series, p: pd.Series) -> float:
    if len(set(int(v) for v in y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def summarize_accounts(session: pd.DataFrame) -> dict[str, Any]:
    accounts: dict[str, Any] = {}
    for topk in TOPKS:
        accounts[f"oracle_top{topk}"] = account_from_period_returns(session, f"oracle_top{topk}_raw_proxy")
        accounts[f"best_family_oracle_top{topk}"] = account_from_period_returns(session, f"best_family_top{topk}_raw_proxy")
        accounts[f"random_mean_top{topk}"] = account_from_period_returns(session, f"random_top{topk}_raw_mean")
        accounts[f"random_p95_top{topk}"] = account_from_period_returns(session, f"random_top{topk}_raw_p95")
        for family in EXP132.stock_score_specs():
            accounts[f"family_{family}_top{topk}"] = account_from_period_returns(session, f"family_{family}_top{topk}_raw_proxy")
    return accounts


def account_from_period_returns(session: pd.DataFrame, col: str) -> dict[str, Any]:
    vals = session[col].fillna(0.0).to_numpy(dtype=float)
    nav = np.cumprod(1.0 + vals)
    peak = np.maximum.accumulate(nav) if len(nav) else np.asarray([1.0])
    annual = {str(int(y)): float(np.prod(1.0 + g[col].fillna(0.0).to_numpy(dtype=float)) - 1.0) for y, g in session.groupby("year")}
    return {
        "final_multiple": float(nav[-1]) if len(nav) else 1.0,
        "total_return": float(nav[-1] - 1.0) if len(nav) else 0.0,
        "max_drawdown_period": float(np.min(nav / peak - 1.0)) if len(nav) else 0.0,
        "annual_returns": annual,
        "all_years_positive": bool(all(v > 0 for v in annual.values())),
        "mean_period_return": float(np.mean(vals)) if len(vals) else 0.0,
        "median_period_return": float(np.median(vals)) if len(vals) else 0.0,
        "positive_period_ratio": float(np.mean(vals > 0.0)) if len(vals) else 0.0,
    }


def summarize_surface(session: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for topk in TOPKS:
        out[f"top{topk}"] = {
            "oracle_raw_mean": float(session[f"oracle_top{topk}_raw_proxy"].mean()),
            "best_family_raw_mean": float(session[f"best_family_top{topk}_raw_proxy"].mean()),
            "random_raw_mean": float(session[f"random_top{topk}_raw_mean"].mean()),
            "oracle_minus_best_family_mean": float(session[f"oracle_minus_best_family_top{topk}"].mean()),
            "oracle_positive_ratio": float((session[f"oracle_top{topk}_raw_proxy"] > 0).mean()),
            "best_family_positive_ratio": float((session[f"best_family_top{topk}_raw_proxy"] > 0).mean()),
            "by_year": {
                str(int(year)): {
                    "oracle_raw_mean": float(group[f"oracle_top{topk}_raw_proxy"].mean()),
                    "best_family_raw_mean": float(group[f"best_family_top{topk}_raw_proxy"].mean()),
                    "random_raw_mean": float(group[f"random_top{topk}_raw_mean"].mean()),
                    "oracle_minus_best_family_mean": float(group[f"oracle_minus_best_family_top{topk}"].mean()),
                }
                for year, group in session.groupby("year")
            },
        }
    return out


def build_leaderboard(accounts: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, data in accounts.items():
        annual = data["annual_returns"]
        rows.append({
            "name": name,
            "final_multiple": float(data["final_multiple"]),
            "ret2026": float(annual.get("2026", 0.0)),
            "min_annual_return": float(min(annual.values())) if annual else 0.0,
            "all_years_positive": bool(data["all_years_positive"]),
            "max_drawdown_period": float(data["max_drawdown_period"]),
            "mean_period_return": float(data["mean_period_return"]),
            "positive_period_ratio": float(data["positive_period_ratio"]),
        })
    rows.sort(key=lambda r: (r["all_years_positive"], r["final_multiple"], r["ret2026"]), reverse=True)
    return rows[:50]


def print_summary(result: dict[str, Any], checksum: str) -> None:
    print(f"sha256:{checksum}")
    print("surface", json.dumps(result["payoff_surface"], ensure_ascii=False)[:1200])
    for row in result["leaderboard"][:20]:
        print(f"{row['name']}: final={row['final_multiple']:.2f}x 2026={row['ret2026']:+.4f} min_year={row['min_annual_return']:+.4f} mean={row['mean_period_return']:+.4f}")
    for key, value in result["opportunity_models"].items():
        if key.startswith("top"):
            print(f"model {key}: corr_reg={value['rank_corr_reg']:+.4f} corr_prob={value['rank_corr_prob']:+.4f} auc={value['auc_global_high']:+.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
