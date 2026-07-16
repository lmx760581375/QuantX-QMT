"""Market-regime weak-to-strong alpha mining experiment.

All outputs stay under ./tmp. The script intentionally does not modify QuantX core
modules while the direction is still exploratory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantx.core.data.qlib_reader import QlibBinReader
from quantx.core.factor_runtime import FactorRuntime, MarketPanel


MAINBOARD_PREFIXES = ("SH600", "SH601", "SH603", "SH605", "SZ000", "SZ001", "SZ002", "SZ003")
ID_COLUMNS = {
    "signal_time",
    "instrument",
    "layer",
    "regime",
    "entry_session",
    "raw_candidate_count",
    "candidate_rank",
    "tradable_rank",
}
DIAGNOSTIC_COLUMNS = {
    "is_signal_member",
    "is_entry_member",
    "is_st_signal",
    "is_st_entry",
    "entry_has_bar",
    "entry_zero_volume",
    "entry_zero_amount",
    "entry_one_price_limit_up",
    "is_tradable",
}


@dataclass(frozen=True)
class UniverseRecord:
    symbol: str
    start: pd.Timestamp
    end: pd.Timestamp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "analyze", "model", "all"])
    parser.add_argument("--config", default="tmp/market_regime_weak_to_strong_alpha_v1/configs/experiment.yaml")
    parser.add_argument("--start", help="Override data.start for smoke runs")
    parser.add_argument("--end", help="Override data.end for smoke runs")
    parser.add_argument("--max-rank", type=int, help="Override candidate.max_rank_per_layer_day")
    parser.add_argument("--first-year", type=int, help="Override validation.first_prediction_year")
    parser.add_argument("--last-year", type=int, help="Override validation.last_prediction_year")
    parser.add_argument("--validation-sessions", type=int, help="Override validation.validation_sessions")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.start:
        cfg["data"]["start"] = str(args.start)
    if args.end:
        cfg["data"]["end"] = str(args.end)
    if args.max_rank:
        cfg["candidate"]["max_rank_per_layer_day"] = int(args.max_rank)
    if args.first_year:
        cfg["validation"]["first_prediction_year"] = int(args.first_year)
    if args.last_year:
        cfg["validation"]["last_prediction_year"] = int(args.last_year)
    if args.validation_sessions:
        cfg["validation"]["validation_sessions"] = int(args.validation_sessions)

    output_root = Path(cfg["output_root"])
    for subdir in ("data", "runs", "reports"):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    if args.command in {"build", "all"}:
        build_dataset(cfg)
    if args.command in {"analyze", "all"}:
        analyze_dataset(cfg)
    if args.command in {"model", "all"}:
        run_models(cfg)


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Experiment config must be a YAML mapping")
    return cfg


def build_dataset(cfg: dict[str, Any]) -> None:
    output_root = Path(cfg["output_root"])
    data_cfg = cfg["data"]
    start = pd.Timestamp(data_cfg["start"])
    end = pd.Timestamp(data_cfg["end"])
    reader = QlibBinReader(data_cfg["provider_uri"])
    calendar = reader.calendar(None, end.strftime("%Y-%m-%d"))
    start_pos = int(calendar.searchsorted(start, side="left"))
    load_pos = max(0, start_pos - int(data_cfg.get("lookback_sessions", 320)))
    load_start = calendar[load_pos]
    records = load_universe_records(Path(data_cfg["provider_uri"]) / "instruments" / "all.txt")
    symbols = sorted({r.symbol for r in records if r.end >= load_start and r.start <= end})
    print(f"[build] symbols={len(symbols)} load_start={load_start.date()} start={start.date()} end={end.date()}", flush=True)

    quote = reader.features(
        symbols,
        ["$open", "$high", "$low", "$close", "$volume", "$amount", "$change", "$vwap"],
        load_start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )
    panel = MarketPanel.from_frame(quote)
    print(f"[build] panel_dates={len(panel.dates)} panel_instruments={len(panel.instruments)}", flush=True)

    regime = build_regime_frame(cfg, panel.dates)
    st_by_date = load_st_sets(data_cfg["st_daily_path"])
    rows: list[dict[str, Any]] = []
    max_rank = int(cfg["candidate"].get("max_rank_per_layer_day", 200))
    horizons = [int(x) for x in cfg["labels"]["horizons"]]
    max_horizon = max(horizons)
    member = build_membership_lookup(records)
    layer_files = cfg["candidate_layers"]

    for layer, yaml_path in layer_files.items():
        print(f"[build] compute layer={layer}", flush=True)
        strategy_cfg = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
        formulas = {**strategy_cfg.get("factors", {}), **strategy_cfg.get("signals", {})}
        runtime = FactorRuntime(panel)
        values = runtime.compute_formulas(formulas)
        where_name = str(strategy_cfg["selector"]["where"])
        score_name = str(strategy_cfg["selector"]["score"])
        where = np.asarray(values[where_name], dtype=bool)
        score = np.asarray(values[score_name], dtype=float)
        feature_names = sorted(name for name, value in runtime.values.items() if _is_panel_matrix(value, panel))
        for date_idx, session in enumerate(panel.dates):
            if session < start or session > end:
                continue
            mask = where[date_idx] & np.isfinite(score[date_idx])
            candidate_indices = np.flatnonzero(mask)
            raw_count = int(len(candidate_indices))
            if raw_count == 0:
                continue
            order = candidate_indices[np.argsort(-score[date_idx, candidate_indices], kind="mergesort")]
            order = order[:max_rank]
            signal_key = session.strftime("%Y-%m-%d")
            for rank, instrument_idx in enumerate(order, start=1):
                instrument = str(panel.instruments[instrument_idx])
                row = base_candidate_row(
                    cfg=cfg,
                    panel=panel,
                    regime=regime,
                    member=member,
                    st_by_date=st_by_date,
                    layer=layer,
                    signal_idx=date_idx,
                    instrument_idx=instrument_idx,
                    instrument=instrument,
                    raw_count=raw_count,
                    rank=rank,
                    score_value=score[date_idx, instrument_idx],
                    horizons=horizons,
                    max_horizon=max_horizon,
                )
                for feature in feature_names:
                    value = runtime.values[feature][date_idx, instrument_idx]
                    row[feature] = _to_scalar(value)
                rows.append(row)
            if len(rows) and len(rows) % 100000 == 0:
                print(f"[build] rows={len(rows)} latest={layer}:{signal_key}", flush=True)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No candidates were generated")
    frame = assign_tradable_rank(frame)
    out_path = output_root / "data" / "candidate_panel.parquet"
    frame.to_parquet(out_path, index=False)
    manifest = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "start": str(start.date()),
        "end": str(end.date()),
        "max_rank_per_layer_day": max_rank,
        "horizons": horizons,
        "layers": list(layer_files),
        "sha256": sha256_file(out_path),
    }
    (output_root / "data" / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[build] wrote {out_path} rows={len(frame)} sha256={manifest['sha256']}", flush=True)


def base_candidate_row(
    *,
    cfg: dict[str, Any],
    panel: MarketPanel,
    regime: pd.DataFrame,
    member: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    st_by_date: dict[str, set[str]],
    layer: str,
    signal_idx: int,
    instrument_idx: int,
    instrument: str,
    raw_count: int,
    rank: int,
    score_value: float,
    horizons: list[int],
    max_horizon: int,
) -> dict[str, Any]:
    entry_idx = signal_idx + int(cfg["labels"].get("entry_lag_sessions", 1))
    signal_time = panel.dates[signal_idx]
    entry_session = panel.dates[entry_idx] if entry_idx < len(panel.dates) else pd.NaT
    signal_key = signal_time.strftime("%Y-%m-%d")
    entry_key = None if pd.isna(entry_session) else entry_session.strftime("%Y-%m-%d")
    row = {
        "signal_time": signal_key,
        "instrument": instrument,
        "layer": layer,
        "layer_score": float(score_value),
        "candidate_rank": int(rank),
        "raw_candidate_count": int(raw_count),
        "entry_session": entry_key,
    }
    regime_row = regime.loc[signal_time] if signal_time in regime.index else None
    if regime_row is None:
        row.update({"regime": "invalid", "amount_slope": np.nan, "breadth_gap": np.nan})
    else:
        row.update(
            {
                "regime": str(regime_row["regime"]),
                "amount_slope": float(regime_row["amount_slope"]),
                "breadth_gap": float(regime_row["breadth_gap"]),
            }
        )
    signal_member = is_member(member, instrument, signal_time)
    entry_member = False if pd.isna(entry_session) else is_member(member, instrument, entry_session)
    row["is_signal_member"] = bool(signal_member)
    row["is_entry_member"] = bool(entry_member)
    row["is_st_signal"] = instrument in st_by_date.get(signal_key, set())
    row["is_st_entry"] = False if entry_key is None else instrument in st_by_date.get(entry_key, set())

    opens = panel.get("open")
    highs = panel.get("high")
    lows = panel.get("low")
    closes = panel.get("close")
    volumes = panel.get("volume")
    amounts = panel.get("amount") if panel.has("amount") else np.full_like(opens, np.nan)
    preclose = closes[signal_idx, instrument_idx]
    if entry_idx < len(panel.dates):
        entry_open = opens[entry_idx, instrument_idx]
        entry_high = highs[entry_idx, instrument_idx]
        entry_low = lows[entry_idx, instrument_idx]
        entry_close = closes[entry_idx, instrument_idx]
        entry_volume = volumes[entry_idx, instrument_idx]
        entry_amount = amounts[entry_idx, instrument_idx]
    else:
        entry_open = entry_high = entry_low = entry_close = entry_volume = entry_amount = np.nan
    entry_has_bar = all(np.isfinite(x) and x > 0 for x in (entry_open, entry_high, entry_low, entry_close))
    one_price = (
        entry_has_bar
        and np.isfinite(preclose)
        and preclose > 0
        and abs(entry_open - entry_high) <= 1e-6
        and abs(entry_open - entry_low) <= 1e-6
        and abs(entry_open - entry_close) <= 1e-6
        and entry_open / preclose - 1.0 >= 0.095
    )
    zero_volume = not (np.isfinite(entry_volume) and entry_volume > 0)
    zero_amount = not (np.isfinite(entry_amount) and entry_amount > 0)
    row.update(
        {
            "entry_open": _to_scalar(entry_open),
            "entry_preclose": _to_scalar(preclose),
            "entry_gap": _safe_ratio(entry_open, preclose) - 1.0,
            "entry_has_bar": bool(entry_has_bar),
            "entry_zero_volume": bool(zero_volume),
            "entry_zero_amount": bool(zero_amount),
            "entry_one_price_limit_up": bool(one_price),
        }
    )
    row["is_tradable"] = bool(
        entry_has_bar
        and signal_member
        and entry_member
        and not row["is_st_signal"]
        and not row["is_st_entry"]
        and not zero_volume
        and not zero_amount
        and not one_price
    )
    for horizon in horizons:
        exit_idx = entry_idx + horizon
        if entry_idx < len(panel.dates) and exit_idx < len(panel.dates):
            exit_open = opens[exit_idx, instrument_idx]
            ret = _safe_ratio(exit_open, entry_open) - 1.0
            row[f"ret_{horizon}d"] = ret
            row[f"up_{horizon}d"] = bool(np.isfinite(ret) and ret > 0)
            row[f"label_available_{horizon}d"] = bool(np.isfinite(ret))
        else:
            row[f"ret_{horizon}d"] = np.nan
            row[f"up_{horizon}d"] = False
            row[f"label_available_{horizon}d"] = False
    return row


def assign_tradable_rank(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["signal_time", "layer", "candidate_rank", "instrument"]).copy()
    frame["tradable_rank"] = np.nan
    mask = frame["is_tradable"].astype(bool)
    frame.loc[mask, "tradable_rank"] = (
        frame.loc[mask].groupby(["signal_time", "layer"], sort=False).cumcount() + 1
    )
    return frame


def analyze_dataset(cfg: dict[str, Any]) -> None:
    output_root = Path(cfg["output_root"])
    frame = pd.read_parquet(output_root / "data" / "candidate_panel.parquet")
    horizons = [int(x) for x in cfg["labels"]["horizons"]]
    topn_values = [int(x) for x in cfg["candidate"]["topn"]]
    rows = []
    random_rows = []
    for horizon in horizons:
        label_col = f"ret_{horizon}d"
        avail_col = f"label_available_{horizon}d"
        valid = frame[frame["is_tradable"] & frame[avail_col] & frame[label_col].notna()].copy()
        for keys, group in valid.groupby(["layer", "regime"], sort=True):
            layer, regime = keys
            for topn in topn_values:
                selected = group[group["tradable_rank"] <= topn]
                rows.append(metric_row(selected, horizon, topn, layer, regime, "original_score"))
                random_rows.append(random_metric_row(group, horizon, topn, layer, regime))
        for keys, group in valid.groupby(["layer", "regime", valid["signal_time"].str[:4]], sort=True):
            layer, regime, year = keys
            for topn in topn_values:
                selected = group[group["tradable_rank"] <= topn]
                row = metric_row(selected, horizon, topn, layer, regime, "original_score_year")
                row["year"] = str(year)
                rows.append(row)
    stats = pd.DataFrame(rows)
    random_stats = pd.DataFrame(random_rows)
    stats.to_csv(output_root / "runs" / "baseline_stats.csv", index=False)
    random_stats.to_csv(output_root / "runs" / "random_matched_stats.csv", index=False)
    diagnostics = build_diagnostics(frame, horizons)
    (output_root / "runs" / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[analyze] wrote baseline_stats rows={len(stats)} random_rows={len(random_stats)}", flush=True)


def run_models(cfg: dict[str, Any]) -> None:
    from lightgbm import LGBMRegressor

    output_root = Path(cfg["output_root"])
    frame = pd.read_parquet(output_root / "data" / "candidate_panel.parquet")
    horizon = int(cfg["model"].get("target_horizon", 5))
    label_col = f"ret_{horizon}d"
    avail_col = f"label_available_{horizon}d"
    data = frame[frame["is_tradable"] & frame[avail_col] & frame[label_col].notna()].copy()
    data["signal_time"] = pd.to_datetime(data["signal_time"])
    data["regime_code"] = data["regime"].map({"bear": -1, "range": 0, "bull": 1}).fillna(0).astype(float)
    data["layer_code"] = data["layer"].astype("category").cat.codes.astype(float)
    feature_columns = choose_feature_columns(data, horizon)
    params = dict(cfg["model"]["lightgbm"])
    years = range(int(cfg["validation"]["first_prediction_year"]), int(cfg["validation"]["last_prediction_year"]) + 1)
    seeds = [int(x) for x in cfg["validation"]["seeds"]]
    topn_values = [int(x) for x in cfg["model"]["topn"]]
    summary_rows = []
    importance_rows = []
    prediction_parts = []
    min_train_rows = int(cfg["model"].get("min_train_rows", 500))

    for variant in ("global", "global_with_regime", "regime_expert"):
        for year in years:
            pred_mask = data["signal_time"].dt.year == year
            if not pred_mask.any():
                continue
            pred_start = data.loc[pred_mask, "signal_time"].min()
            train_cutoff = cutoff_before_validation(data, pred_start, cfg)
            if pd.isna(train_cutoff):
                print(f"[model] variant={variant} year={year} skipped: no prior training sessions", flush=True)
                continue
            base_train = data[data["signal_time"] <= train_cutoff]
            prediction = data[pred_mask].copy()
            if base_train.empty or len(base_train) < min_train_rows:
                continue
            for seed in seeds:
                if variant == "regime_expert":
                    scored_parts = []
                    for regime_name, pred_group in prediction.groupby("regime", sort=True):
                        train_group = base_train[base_train["regime"] == regime_name]
                        if len(train_group) < min_train_rows:
                            continue
                        model, medians = fit_lgbm(train_group, feature_columns, label_col, params, seed)
                        scored = score_frame(pred_group, model, feature_columns, medians)
                        scored_parts.append(scored)
                        importance_rows.extend(feature_importance_rows(model, feature_columns, variant, year, seed, regime_name))
                    if not scored_parts:
                        continue
                    scored_prediction = pd.concat(scored_parts, ignore_index=True)
                else:
                    cols = feature_columns if variant == "global_with_regime" else tuple(
                        col for col in feature_columns if col not in {"regime_code"}
                    )
                    model, medians = fit_lgbm(base_train, cols, label_col, params, seed)
                    scored_prediction = score_frame(prediction, model, cols, medians)
                    importance_rows.extend(feature_importance_rows(model, cols, variant, year, seed, "all"))
                scored_prediction["variant"] = variant
                scored_prediction["seed"] = seed
                scored_prediction["prediction_year"] = year
                prediction_parts.append(
                    scored_prediction[
                        [
                            "signal_time",
                            "instrument",
                            "layer",
                            "regime",
                            "prediction_year",
                            "variant",
                            "seed",
                            label_col,
                            "model_score",
                            "layer_score",
                            "tradable_rank",
                        ]
                    ]
                )
                summary_rows.extend(evaluate_scored(scored_prediction, label_col, variant, year, seed, topn_values))
            print(f"[model] variant={variant} year={year} done", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "runs" / "model_summary.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(output_root / "runs" / "feature_importance.csv", index=False)
    if prediction_parts:
        pd.concat(prediction_parts, ignore_index=True).to_parquet(output_root / "runs" / "model_predictions.parquet", index=False)
    print(f"[model] wrote summary rows={len(summary)}", flush=True)


def choose_feature_columns(frame: pd.DataFrame, horizon: int) -> tuple[str, ...]:
    excluded = set(ID_COLUMNS) | set(DIAGNOSTIC_COLUMNS)
    excluded.update({col for col in frame.columns if col.startswith("ret_") or col.startswith("up_")})
    excluded.update({col for col in frame.columns if col.startswith("label_available_")})
    excluded.update({col for col in frame.columns if col.startswith("entry_")})
    candidates = []
    for col in frame.columns:
        if col in excluded:
            continue
        if col == f"ret_{horizon}d":
            continue
        if pd.api.types.is_numeric_dtype(frame[col]) or pd.api.types.is_bool_dtype(frame[col]):
            candidates.append(col)
    return tuple(dict.fromkeys(["layer_score", "candidate_rank", "raw_candidate_count", "regime_code", "layer_code", *candidates]))


def fit_lgbm(frame: pd.DataFrame, feature_columns: tuple[str, ...], label_col: str, params: dict[str, Any], seed: int):
    from lightgbm import LGBMRegressor

    values = frame.loc[:, feature_columns].to_numpy(dtype=float, copy=True)
    labels = pd.to_numeric(frame[label_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(labels)
    medians = np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    missing = ~np.isfinite(values)
    if missing.any():
        values[missing] = np.take(medians, np.where(missing)[1])
    model = LGBMRegressor(random_state=seed, verbosity=-1, **params)
    model.fit(values[valid], labels[valid])
    return model, medians


def score_frame(frame: pd.DataFrame, model: Any, feature_columns: tuple[str, ...], medians: np.ndarray) -> pd.DataFrame:
    values = frame.loc[:, feature_columns].to_numpy(dtype=float, copy=True)
    missing = ~np.isfinite(values)
    if missing.any():
        values[missing] = np.take(medians, np.where(missing)[1])
    out = frame.copy()
    out["model_score"] = model.predict(values)
    return out


def cutoff_before_validation(data: pd.DataFrame, pred_start: pd.Timestamp, cfg: dict[str, Any]) -> pd.Timestamp:
    sessions = pd.DatetimeIndex(sorted(data.loc[data["signal_time"] < pred_start, "signal_time"].unique()))
    if len(sessions) == 0:
        return pd.NaT
    reserve = int(cfg["validation"].get("validation_sessions", 252)) + int(cfg["validation"].get("embargo_sessions", 0))
    if len(sessions) <= reserve:
        return sessions[-1]
    return sessions[-reserve - 1]


def evaluate_scored(frame: pd.DataFrame, label_col: str, variant: str, year: int, seed: int, topn_values: list[int]) -> list[dict[str, Any]]:
    rows = []
    for (layer, regime), group in frame.groupby(["layer", "regime"], sort=True):
        for topn in topn_values:
            selected = group.sort_values(["signal_time", "model_score"], ascending=[True, False]).groupby(
                "signal_time", sort=False
            ).head(topn)
            row = metric_row(selected, int(label_col.split("_")[1].rstrip("d")), topn, layer, regime, variant)
            row.update({"year": year, "seed": seed})
            rows.append(row)
    return rows


def feature_importance_rows(model: Any, features: tuple[str, ...], variant: str, year: int, seed: int, regime: str) -> list[dict[str, Any]]:
    gain = model.booster_.feature_importance(importance_type="gain")
    split = model.booster_.feature_importance(importance_type="split")
    return [
        {
            "variant": variant,
            "year": year,
            "seed": seed,
            "regime": regime,
            "feature": feature,
            "gain": float(g),
            "split": int(s),
        }
        for feature, g, s in zip(features, gain, split)
    ]


def metric_row(frame: pd.DataFrame, horizon: int, topn: int, layer: str, regime: str, method: str) -> dict[str, Any]:
    label_col = f"ret_{horizon}d"
    values = pd.to_numeric(frame[label_col], errors="coerce").dropna()
    daily = frame.dropna(subset=[label_col]).groupby("signal_time")[label_col].mean() if not frame.empty else pd.Series(dtype=float)
    return {
        "method": method,
        "layer": layer,
        "regime": regime,
        "horizon": horizon,
        "topn": topn,
        "sample_count": int(len(values)),
        "session_count": int(daily.size),
        "mean_return": mean_or_none(values),
        "median_return": quantile_or_none(values, 0.5),
        "win_rate": mean_or_none(values > 0),
        "q10": quantile_or_none(values, 0.1),
        "q90": quantile_or_none(values, 0.9),
        "daily_mean_return": mean_or_none(daily),
        "positive_day_rate": mean_or_none(daily > 0),
        "avg_daily_count": mean_or_none(frame.groupby("signal_time").size()) if not frame.empty else None,
    }


def random_metric_row(group: pd.DataFrame, horizon: int, topn: int, layer: str, regime: str) -> dict[str, Any]:
    rng = np.random.default_rng(20260715 + horizon * 1000 + topn)
    pieces = []
    for _, daily in group.groupby("signal_time", sort=True):
        n = min(topn, len(daily))
        if n <= 0:
            continue
        pieces.append(daily.iloc[rng.choice(len(daily), size=n, replace=False)])
    sampled = pd.concat(pieces, ignore_index=True) if pieces else group.iloc[0:0]
    return metric_row(sampled, horizon, topn, layer, regime, "random_matched")


def build_diagnostics(frame: pd.DataFrame, horizons: list[int]) -> dict[str, Any]:
    out = {
        "rows": int(len(frame)),
        "layers": frame.groupby("layer").size().astype(int).to_dict(),
        "regimes": frame.groupby("regime").size().astype(int).to_dict(),
        "tradable_ratio": float(frame["is_tradable"].mean()),
        "hard_filter_rates": {col: float(frame[col].mean()) for col in sorted(DIAGNOSTIC_COLUMNS & set(frame.columns))},
    }
    for horizon in horizons:
        avail = f"label_available_{horizon}d"
        out[f"label_available_{horizon}d"] = float(frame[avail].mean())
    return out


def build_regime_frame(cfg: dict[str, Any], dates: pd.DatetimeIndex) -> pd.DataFrame:
    active = pd.read_parquet(cfg["data"]["active_value_path"]).copy()
    active["date"] = pd.to_datetime(active["date"])
    active = active.sort_values("date").set_index("date")
    reg_cfg = cfg["regime"]
    fast = active["active_core_amount"].rolling(int(reg_cfg["amount_fast_window"]), min_periods=1).mean()
    slow = active["active_core_amount"].rolling(int(reg_cfg["amount_slow_window"]), min_periods=1).mean()
    active["amount_slope"] = fast / slow - 1.0
    if {"right_side_core_ratio_ema20", "right_side_core_ratio_ema60"}.issubset(active.columns):
        active["breadth_gap"] = active["right_side_core_ratio_ema20"] - active["right_side_core_ratio_ema60"]
    else:
        active["breadth_gap"] = active["right_side_core_ratio"].ewm(span=20, adjust=False).mean() - active[
            "right_side_core_ratio"
        ].ewm(span=60, adjust=False).mean()
    complete = active.get("is_complete_day", pd.Series(True, index=active.index)).astype(bool)
    bull = (active["amount_slope"] > float(reg_cfg["bull_amount_slope_gt"])) & (
        active["breadth_gap"] > float(reg_cfg["bull_breadth_gap_gt"])
    )
    bear = (active["amount_slope"] < float(reg_cfg["bear_amount_slope_lt"])) & (
        active["breadth_gap"] < float(reg_cfg["bear_breadth_gap_lt"])
    )
    active["regime"] = np.where(~complete, "invalid", np.where(bull, "bull", np.where(bear, "bear", "range")))
    return active.reindex(dates)[["regime", "amount_slope", "breadth_gap"]]


def load_universe_records(path: Path) -> list[UniverseRecord]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        symbol = parts[0]
        if symbol.startswith(MAINBOARD_PREFIXES):
            records.append(UniverseRecord(symbol, pd.Timestamp(parts[1]), pd.Timestamp(parts[2])))
    if not records:
        raise ValueError("No mainboard universe records found")
    return records


def build_membership_lookup(records: list[UniverseRecord]) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    return {record.symbol: (record.start, record.end) for record in records}


def is_member(member: dict[str, tuple[pd.Timestamp, pd.Timestamp]], symbol: str, session: pd.Timestamp) -> bool:
    bounds = member.get(symbol)
    return bool(bounds and bounds[0] <= session <= bounds[1])


def load_st_sets(path: str | Path) -> dict[str, set[str]]:
    st = pd.read_csv(path, dtype={"ts_code": str, "trade_date": str})
    st["trade_date"] = pd.to_datetime(st["trade_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    st["instrument"] = st["ts_code"].map(tushare_to_qlib_symbol)
    return {date: set(group["instrument"].dropna()) for date, group in st.groupby("trade_date", sort=False)}


def tushare_to_qlib_symbol(ts_code: str) -> str | None:
    if not isinstance(ts_code, str) or "." not in ts_code:
        return None
    code, exchange = ts_code.split(".", 1)
    return f"{exchange}{code}"


def _is_panel_matrix(value: Any, panel: MarketPanel) -> bool:
    arr = np.asarray(value)
    return arr.shape == (len(panel.dates), len(panel.instruments)) and arr.dtype.kind in "biufc"


def _to_scalar(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else np.nan
    return value


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not (np.isfinite(numerator) and np.isfinite(denominator)) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def mean_or_none(values: Any) -> float | None:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def quantile_or_none(values: Any, q: float) -> float | None:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.quantile(q))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
