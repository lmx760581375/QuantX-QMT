"""Round S f3-pool walk-forward ranking tests.

This round keeps the best Round F f3 candidate pool, max_positions=12,
lag=1, next-day close execution, cash_equal sizing, and original sell rules.
It only changes the order of the already-selected f3 candidates with
walk-forward LightGBM predictions trained on buyable non-ST labeled rows.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("tmp/market_regime_weak_to_strong_alpha_v1")
OUTPUT_NAME = "raw12_f3_rank_ml_round_s"
ROUND_F_SCRIPT = ROOT / "run_raw12_enhancement_round_f.py"
ROUND_C_SCRIPT = ROOT / "run_market_state_ranking_round_c.py"
ROUND_F_BEST = ROOT / "raw12_enhancement_round_f/artifacts/f3_good_top12_else_top8"
RAW12_BASE = ROOT / "capacity_extension_round_a_current/artifacts/raw_top12_pos12"


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "s0_f3_reference", "description": "Exact Round F f3 reference.", "mode": "f3_reference", "topk": 12, "max_positions": 12},
    {"variant": "s1_f3_ml5_full_order", "description": "f3 pool ranked by OOS 5d LightGBM prediction.", "mode": "ml_full", "horizon": 5, "topk": 12, "max_positions": 12},
    {"variant": "s2_f3_ml3_full_order", "description": "f3 pool ranked by OOS 3d LightGBM prediction.", "mode": "ml_full", "horizon": 3, "topk": 12, "max_positions": 12},
    {"variant": "s3_f3_ml7_full_order", "description": "f3 pool ranked by OOS 7d LightGBM prediction.", "mode": "ml_full", "horizon": 7, "topk": 12, "max_positions": 12},
    {"variant": "s4_f3_ml357_ensemble", "description": "f3 pool ranked by OOS 3/5/7d ensemble prediction.", "mode": "ml_ensemble", "topk": 12, "max_positions": 12},
    {"variant": "s5_f3_ml5_good_state_only", "description": "use OOS 5d ML order only in f3 good states; otherwise original order.", "mode": "ml_good_only", "horizon": 5, "topk": 12, "max_positions": 12},
    {"variant": "s6_f3_ml5_bad_state_only", "description": "use OOS 5d ML order only outside f3 good states; otherwise original order.", "mode": "ml_bad_only", "horizon": 5, "topk": 12, "max_positions": 12},
)


def main() -> None:
    round_f = load_module(ROUND_F_SCRIPT, "round_f_harness")
    patch_round_f(round_f)
    round_f.main()
    normalize_outputs(ROOT / OUTPUT_NAME)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_round_f(round_f) -> None:
    round_f.OUTPUT_NAME = OUTPUT_NAME
    round_f.VARIANTS = VARIANTS
    round_f.build_signals = build_signals


def build_signals(candidates: pd.DataFrame, signals_dir: Path, runs_dir: Path) -> list[dict[str, Any]]:
    data = prepare_data(candidates)
    data = add_walk_forward_predictions(data, runs_dir)
    rows: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    for spec in VARIANTS:
        selected = data[data["f3_selected"]].copy()
        selected["score"] = variant_score(selected, spec)
        selected = selected.dropna(subset=["signal_time", "instrument", "score"])
        selected = selected.sort_values(["signal_time", "score", "instrument"], ascending=[True, False, True])
        path = signals_dir / f"{spec['variant']}.parquet"
        columns = ["signal_time", "instrument", "score", "raw_rank", "base_rank", "regime", "amount_state", "breadth_state"]
        selected[columns].to_parquet(path, index=False)
        manifest = signal_manifest(spec, path, selected)
        rows.append(manifest)
        stats.append({**manifest, **signal_stats(selected), **rank_change_stats(selected)})
    pd.DataFrame(stats).to_csv(runs_dir / "round_s_signal_stats.csv", index=False)
    return rows


def prepare_data(candidates: pd.DataFrame) -> pd.DataFrame:
    data = candidates.copy()
    data["base_score"] = pd.to_numeric(data["score"], errors="coerce")
    data["base_rank"] = data.groupby("signal_time")["base_score"].rank(method="first", ascending=False)
    rank = pd.to_numeric(data["base_rank"], errors="coerce")
    good = good_state_mask(data)
    data["f3_selected"] = (rank <= np.where(good, 12, 8)) & (rank <= 12)
    data["f3_good_state"] = good
    data["year"] = pd.to_datetime(data["signal_time"]).dt.year.astype(int)
    for column in ("is_buyable_nonst", "label_available_3d", "label_available_5d", "label_available_7d"):
        if column not in data:
            data[column] = False
    return data


def good_state_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["regime"].eq("bull") | frame["amount_state"].eq("hi") | frame["breadth_state"].eq("hi")


def add_walk_forward_predictions(data: pd.DataFrame, runs_dir: Path) -> pd.DataFrame:
    try:
        import lightgbm as lgb
    except Exception as exc:
        raise RuntimeError(f"LightGBM is required for Round S: {exc}") from exc

    round_c = load_module(ROUND_C_SCRIPT, "round_c_features")
    features = [column for column in round_c.ML_FEATURES]
    out = data.copy()
    for feature in features:
        if feature not in out:
            out[feature] = np.nan
        out[feature] = pd.to_numeric(out[feature], errors="coerce")
    for horizon in (3, 5, 7):
        out[f"ml_pred_{horizon}d"] = np.nan

    diag_rows: list[dict[str, Any]] = []
    train_base = out["f3_selected"] & out["is_buyable_nonst"].fillna(False).astype(bool)
    years = sorted(int(year) for year in out["year"].dropna().unique() if int(year) >= 2021)
    for horizon in (3, 5, 7):
        label = f"ret_{horizon}d"
        label_ok = f"label_available_{horizon}d"
        out[label] = pd.to_numeric(out.get(label), errors="coerce")
        train_mask = train_base & out[label_ok].fillna(False).astype(bool) & out[label].notna()
        for year in years:
            train = out[train_mask & (out["year"] < year)].copy()
            test = out[out["year"] == year].copy()
            if len(train) < 500 or test.empty:
                continue
            train_x = clean_features(train, features)
            train_y = train[label].astype(float)
            test_x = clean_features(test, features)
            model = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=260,
                learning_rate=0.025,
                num_leaves=15,
                min_child_samples=30,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.05,
                reg_lambda=0.80,
                random_state=20260716 + year * 10 + horizon,
                n_jobs=1,
                verbosity=-1,
            )
            model.fit(train_x, train_y)
            pred = model.predict(test_x)
            out.loc[test.index, f"ml_pred_{horizon}d"] = pred
            valid = test.copy()
            valid[f"ml_pred_{horizon}d"] = pred
            valid = valid[valid["f3_selected"] & valid["is_buyable_nonst"].fillna(False).astype(bool) & valid[label].notna()]
            diag_rows.append(
                {
                    "horizon": horizon,
                    "year": int(year),
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "valid_f3_buyable_rows": int(len(valid)),
                    "spearman_ic": safe_corr(valid[f"ml_pred_{horizon}d"], valid[label], "spearman"),
                    "pearson_ic": safe_corr(valid[f"ml_pred_{horizon}d"], valid[label], "pearson"),
                }
            )
    out["ml_pred_357d"] = out[["ml_pred_3d", "ml_pred_5d", "ml_pred_7d"]].mean(axis=1, skipna=True)
    pd.DataFrame(diag_rows).to_csv(runs_dir / "round_s_ml_oos_diagnostics.csv", index=False)
    return out


def clean_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return frame[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def variant_score(selected: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    mode = str(spec["mode"])
    if mode == "f3_reference":
        return pd.to_numeric(selected["base_score"], errors="coerce")
    if mode == "ml_ensemble":
        return ml_rank_score(selected, "ml_pred_357d").fillna(pd.to_numeric(selected["base_score"], errors="coerce"))
    if mode in {"ml_full", "ml_good_only", "ml_bad_only"}:
        pred_col = f"ml_pred_{int(spec['horizon'])}d"
        ml_score = ml_rank_score(selected, pred_col)
        base_score = pd.to_numeric(selected["base_score"], errors="coerce")
        if mode == "ml_full":
            return ml_score.fillna(base_score)
        use_ml = selected["f3_good_state"] if mode == "ml_good_only" else ~selected["f3_good_state"]
        return ml_score.where(use_ml, base_score).fillna(base_score)
    raise ValueError(f"Unknown Round S mode: {mode}")


def ml_rank_score(selected: pd.DataFrame, pred_col: str) -> pd.Series:
    pred = pd.to_numeric(selected[pred_col], errors="coerce")
    base_rank = pd.to_numeric(selected["base_rank"], errors="coerce").fillna(99.0)
    ml_rank = pred.groupby(selected["signal_time"]).rank(method="first", ascending=False)
    return (100000.0 - ml_rank).where(pred.notna()) - base_rank * 1e-4


def signal_manifest(spec: dict[str, Any], path: Path, signals: pd.DataFrame) -> dict[str, Any]:
    counts = signals.groupby("signal_time", sort=True).size() if not signals.empty else pd.Series(dtype=int)
    return {
        "variant": str(spec["variant"]),
        "description": str(spec.get("description", "")),
        "path": str(path),
        "mode": str(spec["mode"]),
        "topk": int(spec["topk"]),
        "max_positions": int(spec["max_positions"]),
        "sizing": "cash_equal",
        "sell_profile": None,
        "bad_condition": None,
        "bad_topn": None,
        "signal_rows": int(len(signals)),
        "signal_days": int(counts.size),
        "days_with_at_least_topk": int((counts >= int(spec["topk"])).sum()) if not counts.empty else 0,
        "bear_breadth_lo_signal_days": int((signals["regime"].eq("bear") & signals["breadth_state"].eq("lo")).groupby(signals["signal_time"]).any().sum()) if not signals.empty else 0,
        "first_signal": str(signals["signal_time"].min()) if not signals.empty else None,
        "last_signal": str(signals["signal_time"].max()) if not signals.empty else None,
    }


def signal_stats(signals: pd.DataFrame) -> dict[str, Any]:
    if signals.empty:
        return {"avg_daily_candidates": np.nan, "median_daily_candidates": np.nan, "max_daily_candidates": np.nan}
    counts = signals.groupby("signal_time").size()
    return {
        "avg_daily_candidates": float(counts.mean()),
        "median_daily_candidates": float(counts.median()),
        "max_daily_candidates": int(counts.max()),
    }


def rank_change_stats(signals: pd.DataFrame) -> dict[str, Any]:
    if signals.empty:
        return {"avg_abs_rank_change": np.nan, "top1_changed_ratio": np.nan}
    frame = signals.copy()
    frame["new_rank"] = frame.groupby("signal_time")["score"].rank(method="first", ascending=False)
    frame["base_rank"] = pd.to_numeric(frame["base_rank"], errors="coerce")
    top1 = frame.loc[frame["new_rank"].eq(1), ["signal_time", "base_rank"]]
    return {
        "avg_abs_rank_change": float((frame["new_rank"] - frame["base_rank"]).abs().mean()),
        "top1_changed_ratio": float((top1["base_rank"] != 1).mean()) if not top1.empty else np.nan,
    }


def safe_corr(left: pd.Series, right: pd.Series, method: str) -> float:
    frame = pd.DataFrame({"left": pd.to_numeric(left, errors="coerce"), "right": pd.to_numeric(right, errors="coerce")}).dropna()
    if len(frame) < 3 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return float("nan")
    return float(frame["left"].corr(frame["right"], method=method))


def normalize_outputs(output_root: Path) -> None:
    runs_dir = output_root / "runs"
    reports_dir = output_root / "reports"
    rename_pairs = (
        (runs_dir / "round_f_metrics_full.csv", runs_dir / "round_s_metrics_full.csv"),
        (runs_dir / "round_f_yearly_nav.csv", runs_dir / "round_s_yearly_nav.csv"),
        (runs_dir / "round_f_entry_weight_buckets.csv", runs_dir / "round_s_entry_weight_buckets.csv"),
        (runs_dir / "round_f_comparison.csv", runs_dir / "round_s_comparison.csv"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F_PRE.md", reports_dir / "RAW12_F3_RANK_ML_ROUND_S_PRE.md"),
        (reports_dir / "RAW12_ENHANCEMENT_ROUND_F.md", reports_dir / "RAW12_F3_RANK_ML_ROUND_S.md"),
    )
    for src, dst in rename_pairs:
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)
    augment_comparison(output_root)
    write_report(output_root)


def augment_comparison(output_root: Path) -> None:
    path = output_root / "runs/round_s_comparison.csv"
    if not path.exists():
        return
    comparison = pd.read_csv(path)
    refs = {"raw12": read_json(RAW12_BASE / "summary.json"), "f3": read_json(ROUND_F_BEST / "summary.json")}
    for name, ref in refs.items():
        comparison[f"vs_{name}_total_return_delta"] = comparison["total_return"] - float(ref.get("total_return", np.nan))
        comparison[f"vs_{name}_mdd_delta"] = comparison["max_drawdown"] - float(ref.get("max_drawdown", np.nan))
        comparison[f"vs_{name}_sharpe_delta"] = comparison["sharpe"] - float(ref.get("sharpe", np.nan))
    comparison = comparison.sort_values(["sharpe", "total_return"], ascending=[False, False])
    comparison.to_csv(path, index=False)


def write_report(output_root: Path) -> None:
    comparison_path = output_root / "runs/round_s_comparison.csv"
    signal_path = output_root / "runs/round_s_signal_stats.csv"
    diag_path = output_root / "runs/round_s_ml_oos_diagnostics.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    diag = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()
    signal_rows = signals.set_index("variant")["signal_rows"].to_dict() if not signals.empty and "variant" in signals else {}
    rank_change = signals.set_index("variant")["avg_abs_rank_change"].to_dict() if not signals.empty and "avg_abs_rank_change" in signals else {}
    lines = [
        "# Raw12 F3 Rank ML Round S",
        "",
        "范围：保持 f3 候选池、`max_positions=12`、`cash_equal`、`selector.lag=1`、第二天 `close` 买入和原卖出规则；只测试池内排序。",
        "",
        "| variant | ret | mdd | sharpe | avg_pos | capital | rows | avg_rank_move | vs_f3_ret | vs_f3_mdd | vs_f3_sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison.iterrows():
        variant = row.get("variant")
        lines.append(
            f"| {variant} | {safe_float(row.get('total_return')):.4f} | {safe_float(row.get('max_drawdown')):.4f} | "
            f"{safe_float(row.get('sharpe')):.4f} | {safe_float(row.get('full_avg_position_count')):.4f} | "
            f"{safe_float(row.get('avg_capital_utilization')):.4f} | {int(signal_rows.get(variant, 0))} | "
            f"{safe_float(rank_change.get(variant)):.4f} | {safe_float(row.get('vs_f3_total_return_delta')):.4f} | "
            f"{safe_float(row.get('vs_f3_mdd_delta')):.4f} | {safe_float(row.get('vs_f3_sharpe_delta')):.4f} |"
        )
    if not diag.empty:
        lines.extend(["", "## OOS IC", "", "| horizon | avg_ic | avg_pearson | years |", "| ---: | ---: | ---: | ---: |"])
        for horizon, group in diag.groupby("horizon", sort=True):
            lines.append(
                f"| {int(horizon)} | {safe_float(group['spearman_ic'].mean()):.4f} | "
                f"{safe_float(group['pearson_ic'].mean()):.4f} | {int(group['year'].nunique())} |"
            )
    lines.extend(["", "## Artifacts", "", "- `runs/round_s_comparison.csv`", "- `runs/round_s_signal_stats.csv`", "- `runs/round_s_ml_oos_diagnostics.csv`"])
    (output_root / "reports/RAW12_F3_RANK_ML_ROUND_S.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


if __name__ == "__main__":
    main()
