from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from quantx.tools.export_active_brick_ml_scores import export_score_asset


ROOT = Path(".tmp/quantx-research/active-value-frontend-brick-v1")
CANDIDATE_PATH = ROOT / "sh_index_bull_brick_candidates.parquet"
ACTIVE_PATH = Path("data/derived/active_value/daily.parquet")
AMV_PATH = Path("data/derived/0amv/daily.csv")
PANEL_PATH = ROOT / "panel_all_a_20130101_20260715.parquet"
MARKET_STRUCTURE_PATH = ROOT / "daily_market_structure.parquet"
MARKET_STRUCTURE_META_PATH = ROOT / "daily_market_structure.meta.json"
OUT_SCORES = ROOT / "walk_forward_active_brick_scores.parquet"
OUT_JSON = ROOT / "walk_forward_active_brick_ml.json"
OUT_MD = ROOT / "walk_forward_active_brick_ml.md"
OOS_START = pd.Timestamp("2025-01-02")
END = pd.Timestamp("2026-07-15")
TOPKS = (1, 3, 5, 10)

BASE_FEATURES = [
    "reversal_value", "frontend_brick", "frontend_delta_prev",
    "ret1", "ret3", "ret5", "vol_ratio", "amplitude_pct", "body_pct", "close_pos",
    "ma5_rel", "ma10_rel", "ma20_rel", "ma50_rel",
    "amount_rank_pct", "ret1_rank_pct", "ret3_rank_pct", "ret5_rank_pct",
    "brick_rank_pct", "delta_rank_pct", "vol_ratio_rank_pct", "amplitude_rank_pct",
    "stock_kdj_j", "stock_kdj_j_delta", "stock_kdj_j_prev_min3",
    "stock_kdj_j_rank_pct", "stock_kdj_j_delta_rank_pct",
    "bull_body_cover_ratio", "sh_active_bull_age", "sh_kdj_j", "sh_ema20_slope5",
    "sh_close_to_ema60", "amv_shock", "amv_shock_threshold", "amv_wave_age",
    "right_side_core_ratio", "right_side_core_amount_ratio",
    "right_side_core_kdj_low_ratio", "breadth_bull_age",
]

ENHANCEMENT_FEATURES = [
    "turnover_rate", "turnover_rate_circulating", "turnover_rate_free_float",
    "turnover_rate_rank_pct", "turnover_rate_circulating_rank_pct",
    "free_float_mv", "circ_mv", "free_float_mv_rank_pct", "circ_mv_rank_pct",
    "active_free_mv_proxy", "active_free_mv_proxy_rank_pct",
    "log_free_float_mv", "log_active_free_mv_proxy",
]

STRUCTURE_FEATURES = [
    "market_amount_top10_share", "market_amount_top20_share", "market_amount_top50_share",
    "market_advancer_ratio", "market_strong4_ratio", "market_strong4_amount_share",
    "market_brick_reversal_ratio", "market_brick_reversal_amount_share",
    "candidate_count", "candidate_amount_top3_share", "candidate_amount_top5_share",
    "candidate_amount_share", "candidate_count_change5", "market_amount_top20_share_change5",
]
FEATURES = BASE_FEATURES + ENHANCEMENT_FEATURES + STRUCTURE_FEATURES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "Train active-value brick walk-forward ML scores.")
    parser.add_argument("--candidate-path", default=str(CANDIDATE_PATH))
    parser.add_argument("--active-path", default=str(ACTIVE_PATH))
    parser.add_argument("--0amv-path", dest="amv_path", default=str(AMV_PATH))
    parser.add_argument("--panel-path", default=str(PANEL_PATH))
    parser.add_argument("--market-structure-path", default=str(MARKET_STRUCTURE_PATH))
    parser.add_argument("--market-structure-meta-path", default=str(MARKET_STRUCTURE_META_PATH))
    parser.add_argument("--scores-output", default=str(OUT_SCORES))
    parser.add_argument("--json-output", default=str(OUT_JSON))
    parser.add_argument("--markdown-output", default=str(OUT_MD))
    parser.add_argument("--oos-start", default=str(OOS_START.date()))
    parser.add_argument("--end", default=str(END.date()))
    parser.add_argument("--topks", default=",".join(map(str, TOPKS)))
    parser.add_argument("--export-score-output")
    parser.add_argument("--export-metadata-output")
    parser.add_argument("--export-min-rows", type=int, default=1)
    parser.add_argument("--export-min-signal-days", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    configure_paths(args)
    candidates = load_candidates()
    scored, folds = walk_forward_score(candidates)
    scored.to_parquet(OUT_SCORES, index=False)
    report = build_report(scored, folds)
    report["outputs"] = {
        "scores": str(OUT_SCORES),
        "json": str(OUT_JSON),
        "markdown": str(OUT_MD),
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_markdown(report), encoding="utf-8")
    if args.export_score_output:
        metadata_output = Path(args.export_metadata_output) if args.export_metadata_output else Path(args.export_score_output).with_name("metadata.json")
        report["export"] = export_score_asset(
            source=OUT_SCORES,
            output=Path(args.export_score_output),
            metadata_output=metadata_output,
            start=str(OOS_START.date()),
            end=str(END.date()),
            min_rows=args.export_min_rows,
            min_signal_days=args.export_min_signal_days,
        )
        OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(
            f"ok=True scored_rows={report['rows_scored']} scored_dates={report['dates_scored']} "
            f"scores={OUT_SCORES} report={OUT_JSON}"
        )
    return 0


def configure_paths(args: argparse.Namespace) -> None:
    global CANDIDATE_PATH, ACTIVE_PATH, AMV_PATH, PANEL_PATH
    global MARKET_STRUCTURE_PATH, MARKET_STRUCTURE_META_PATH
    global OUT_SCORES, OUT_JSON, OUT_MD, OOS_START, END, TOPKS

    CANDIDATE_PATH = Path(args.candidate_path)
    ACTIVE_PATH = Path(args.active_path)
    AMV_PATH = Path(args.amv_path)
    PANEL_PATH = Path(args.panel_path)
    MARKET_STRUCTURE_PATH = Path(args.market_structure_path)
    MARKET_STRUCTURE_META_PATH = Path(args.market_structure_meta_path)
    OUT_SCORES = Path(args.scores_output)
    OUT_JSON = Path(args.json_output)
    OUT_MD = Path(args.markdown_output)
    OOS_START = pd.Timestamp(args.oos_start)
    END = pd.Timestamp(args.end)
    TOPKS = tuple(int(part) for part in str(args.topks).split(",") if part.strip())
    if not TOPKS:
        raise ValueError("--topks must contain at least one integer")
    for path in (OUT_SCORES, OUT_JSON, OUT_MD, MARKET_STRUCTURE_PATH, MARKET_STRUCTURE_META_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)


def load_candidates() -> pd.DataFrame:
    candidates = pd.read_parquet(CANDIDATE_PATH)
    candidates["datetime"] = pd.to_datetime(candidates["datetime"])
    market = build_market_state()
    market_structure, candidate_amount = load_market_structure(candidates)
    data = candidates.merge(market, on="datetime", how="left")
    data = data.merge(market_structure, on="datetime", how="left")
    data = data.merge(candidate_amount, on=["datetime", "instrument"], how="left")
    data["reversal_value"] = data["frontend_delta"]
    data["rule_score"] = (
        0.35 * data["ret1_rank_pct"]
        + 0.25 * data["delta_rank_pct"]
        + 0.20 * data["brick_rank_pct"]
        + 0.20 * data["amount_rank_pct"]
    )
    data["momentum_score"] = 0.60 * data["ret1_rank_pct"] + 0.40 * data["ret3_rank_pct"]
    data["eligible"] = (
        data["amv_quantile_wave"].fillna(False)
        & data["strong_not_limit"].fillna(False)
        & data["stock_right_side"].fillna(False)
        & data["historical_st_covered"].fillna(False)
    )
    data = data[data["eligible"] & data["fwd10_net"].notna()].copy()
    data = add_candidate_structure(data)
    data["label_rank"] = data.groupby("datetime")["fwd10_net"].rank(pct=True, method="average")
    calendar = pd.DatetimeIndex(market["datetime"].sort_values().unique())
    positions = calendar.searchsorted(pd.DatetimeIndex(data["datetime"]), side="left")
    end_positions = np.minimum(positions + 11, len(calendar) - 1)
    data["label_end"] = calendar[end_positions]
    for feature in FEATURES:
        if feature not in data.columns:
            data[feature] = np.nan
        data[feature] = pd.to_numeric(data[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return data.sort_values(["datetime", "instrument"]).reset_index(drop=True)


def load_market_structure(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = {"size": PANEL_PATH.stat().st_size, "mtime_ns": PANEL_PATH.stat().st_mtime_ns}
    if MARKET_STRUCTURE_PATH.exists() and MARKET_STRUCTURE_META_PATH.exists():
        metadata = json.loads(MARKET_STRUCTURE_META_PATH.read_text(encoding="utf-8"))
        if metadata.get("source") == source:
            cached = pd.read_parquet(MARKET_STRUCTURE_PATH)
            cached["datetime"] = pd.to_datetime(cached["datetime"])
            amount = cached[cached["instrument"].notna()][["datetime", "instrument", "candidate_amount"]]
            market = cached[cached["instrument"].isna()].drop(columns=["instrument", "candidate_amount"])
            return market.reset_index(drop=True), amount.reset_index(drop=True)

    keys = candidates[["datetime", "instrument"]].drop_duplicates()
    key_index = pd.MultiIndex.from_frame(keys)
    aggregate_parts = []
    candidate_parts = []
    market_top_parts = []
    parquet = pq.ParquetFile(PANEL_PATH)
    columns = ["instrument", "datetime", "$amount", "ret1", "frontend_delta", "frontend_delta_prev"]
    for batch in parquet.iter_batches(columns=columns, batch_size=1_048_576):
        frame = batch.to_pandas()
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame["$amount"] = pd.to_numeric(frame["$amount"], errors="coerce").clip(lower=0)
        frame = frame[frame["$amount"].notna()].copy()
        frame["advancer"] = frame["ret1"] > 0
        frame["strong4"] = frame["ret1"] >= 0.04
        frame["brick_reversal"] = (frame["frontend_delta_prev"] < 0) & (frame["frontend_delta"] > 0)
        for flag in ("strong4", "brick_reversal"):
            frame[f"{flag}_amount"] = frame["$amount"].where(frame[flag], 0.0)
        grouped = frame.groupby("datetime", sort=False)
        part = grouped.agg(
            market_amount_total=("$amount", "sum"),
            market_stock_count=("instrument", "size"),
            market_advancer_count=("advancer", "sum"),
            market_strong4_count=("strong4", "sum"),
            market_strong4_amount=("strong4_amount", "sum"),
            market_brick_reversal_count=("brick_reversal", "sum"),
            market_brick_reversal_amount=("brick_reversal_amount", "sum"),
        ).reset_index()
        top = grouped["$amount"].nlargest(50).reset_index(level=0)
        market_top_parts.append(top)
        top["amount_rank"] = top.groupby("datetime").cumcount() + 1
        for k in (10, 20, 50):
            sums = top[top["amount_rank"] <= k].groupby("datetime")["$amount"].sum()
            part = part.merge(sums.rename(f"market_amount_top{k}"), on="datetime", how="left")
        aggregate_parts.append(part)

        batch_index = pd.MultiIndex.from_frame(frame[["datetime", "instrument"]])
        matched = frame.loc[batch_index.isin(key_index), ["datetime", "instrument", "$amount"]]
        if not matched.empty:
            candidate_parts.append(matched.rename(columns={"$amount": "candidate_amount"}))

    partial = pd.concat(aggregate_parts, ignore_index=True)
    sum_columns = [column for column in partial.columns if column != "datetime"]
    market = partial.groupby("datetime", as_index=False)[sum_columns].sum()
    # Each batch contributes its local top 50. Their union contains the global top 50.
    # A date can span batches. The union of each batch's local top 50 always
    # contains the exact global top 50, so rank that small union once.
    top_candidates = pd.concat(market_top_parts, ignore_index=True)
    top_candidates["amount_rank"] = top_candidates.groupby("datetime")["$amount"].rank(method="first", ascending=False)
    exact = market[["datetime"]].copy()
    for k in (10, 20, 50):
        sums = top_candidates[top_candidates["amount_rank"] <= k].groupby("datetime")["$amount"].sum()
        exact = exact.merge(sums.rename(f"market_amount_top{k}"), on="datetime", how="left")
    market = market.drop(columns=["market_amount_top10", "market_amount_top20", "market_amount_top50"])
    market = market.merge(exact, on="datetime", how="left")
    market = finalize_market_structure(market)
    amount = pd.concat(candidate_parts, ignore_index=True).drop_duplicates(["datetime", "instrument"], keep="last")

    cache = pd.concat(
        [
            market.assign(instrument=None, candidate_amount=np.nan),
            amount.assign(**{column: np.nan for column in market.columns if column != "datetime"}),
        ],
        ignore_index=True,
        sort=False,
    )
    cache.to_parquet(MARKET_STRUCTURE_PATH, index=False)
    MARKET_STRUCTURE_META_PATH.write_text(json.dumps({"source": source}, indent=2) + "\n", encoding="utf-8")
    return market, amount


def finalize_market_structure(market: pd.DataFrame) -> pd.DataFrame:
    total = market["market_amount_total"].replace(0, np.nan)
    count = market["market_stock_count"].replace(0, np.nan)
    for k in (10, 20, 50):
        market[f"market_amount_top{k}_share"] = market[f"market_amount_top{k}"] / total
    market["market_advancer_ratio"] = market["market_advancer_count"] / count
    market["market_strong4_ratio"] = market["market_strong4_count"] / count
    market["market_strong4_amount_share"] = market["market_strong4_amount"] / total
    market["market_brick_reversal_ratio"] = market["market_brick_reversal_count"] / count
    market["market_brick_reversal_amount_share"] = market["market_brick_reversal_amount"] / total
    market = market.sort_values("datetime")
    market["market_amount_top20_share_change5"] = market["market_amount_top20_share"].diff(5)
    return market


def add_candidate_structure(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby("datetime", sort=False)
    data["candidate_count"] = grouped["instrument"].transform("size")
    data["candidate_amount_share"] = data["candidate_amount"] / data["market_amount_total"].replace(0, np.nan)
    ranked = data.sort_values(["datetime", "candidate_amount"], ascending=[True, False]).copy()
    ranked["candidate_amount_order"] = ranked.groupby("datetime").cumcount() + 1
    daily = data[["datetime", "candidate_count"]].drop_duplicates("datetime").set_index("datetime")
    for k in (3, 5):
        top = ranked[ranked["candidate_amount_order"] <= k].groupby("datetime")["candidate_amount"].sum()
        daily[f"candidate_amount_top{k}_share"] = top / grouped["candidate_amount"].sum()
    daily = daily.sort_index()
    daily["candidate_count_change5"] = daily["candidate_count"].diff(5)
    return data.merge(daily.drop(columns="candidate_count").reset_index(), on="datetime", how="left")


def build_market_state() -> pd.DataFrame:
    active = pd.read_parquet(ACTIVE_PATH).rename(columns={"date": "datetime"})
    active["datetime"] = pd.to_datetime(active["datetime"])
    amv = pd.read_csv(AMV_PATH)
    amv = amv[amv["symbol"].eq("0AMV_ALL")].sort_values("date").copy()
    amv["datetime"] = pd.to_datetime(amv["date"])
    amv["amv_ret1"] = amv["close"].pct_change(fill_method=None)
    amv["amv_ret2"] = amv["close"].pct_change(2, fill_method=None)
    amv["amv_shock"] = amv[["amv_ret1", "amv_ret2"]].max(axis=1)
    amv["amv_shock_threshold"] = (
        amv["amv_shock"].shift(1).rolling(252, min_periods=126).quantile(0.85)
    )
    amv["amv_quantile_start"] = (
        (amv["amv_shock"] > 0)
        & (amv["amv_shock"] >= amv["amv_shock_threshold"])
    )
    amv["amv_ma10"] = amv["close"].rolling(10, min_periods=5).mean()
    wave, age = quantile_wave(amv)
    amv["amv_quantile_wave"] = wave
    amv["amv_wave_age"] = age
    columns = [
        "datetime", "amv_ret1", "amv_ret2", "amv_shock", "amv_shock_threshold",
        "amv_quantile_start", "amv_quantile_wave", "amv_wave_age",
    ]
    breadth = [
        "datetime", "right_side_core_ratio", "right_side_core_amount_ratio",
        "right_side_core_kdj_low_ratio", "breadth_bull", "breadth_bull_age",
    ]
    return active[breadth].merge(amv[columns], on="datetime", how="left")


def quantile_wave(amv: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    states = []
    ages = []
    active = False
    age = 0
    for row in amv.itertuples():
        if bool(row.amv_quantile_start):
            active = True
            age = 1
        elif active and pd.notna(row.amv_ma10) and row.close < row.amv_ma10:
            active = False
            age = 0
        elif active:
            age += 1
        states.append(active)
        ages.append(age)
    return pd.Series(states, index=amv.index), pd.Series(ages, index=amv.index, dtype="int32")


def walk_forward_score(data: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    scored_parts = []
    folds = []
    prediction_months = pd.period_range(OOS_START, END, freq="M")
    for period in prediction_months:
        month_start = max(OOS_START, period.start_time)
        month_end = min(END, period.end_time)
        train = data[data["label_end"] < month_start].copy()
        predict = data[data["datetime"].between(month_start, month_end)].copy()
        if len(train) < 5000 or predict.empty:
            continue
        recency_days = (month_start - train["datetime"]).dt.days.clip(lower=0)
        recency_weights = np.exp(-recency_days / (365.25 * 4.0)).clip(lower=0.15)
        daily_counts = train.groupby("datetime")["instrument"].transform("size").clip(lower=1)
        return_weights = recency_weights / daily_counts
        return_weights *= len(return_weights) / return_weights.sum()
        return_model = build_model()
        rank_model = build_model()
        base_return_model = build_model()
        base_rank_model = build_model()
        return_model.fit(
            train[FEATURES],
            train["fwd10_net"],
            histgradientboostingregressor__sample_weight=return_weights,
        )
        rank_model.fit(
            train[FEATURES],
            train["label_rank"],
            histgradientboostingregressor__sample_weight=recency_weights,
        )
        base_return_model.fit(
            train[BASE_FEATURES],
            train["fwd10_net"],
            histgradientboostingregressor__sample_weight=return_weights,
        )
        base_rank_model.fit(
            train[BASE_FEATURES],
            train["label_rank"],
            histgradientboostingregressor__sample_weight=recency_weights,
        )
        predict["ml_score"] = return_model.predict(predict[FEATURES])
        predict["ml_rank_score"] = rank_model.predict(predict[FEATURES])
        predict["ml_base_score"] = base_return_model.predict(predict[BASE_FEATURES])
        predict["ml_base_rank_score"] = base_rank_model.predict(predict[BASE_FEATURES])
        predict["ml_consensus_score"] = predict["ml_rank_score"].where(predict["ml_score"] > 0)
        predict["ml_base_consensus_score"] = predict["ml_base_rank_score"].where(predict["ml_base_score"] > 0)
        predict["random_score"] = deterministic_random_score(predict)
        scored_parts.append(predict)
        folds.append({
            "month": str(period),
            "train_rows": int(len(train)),
            "train_start": str(train["datetime"].min().date()),
            "max_label_end": str(train["label_end"].max().date()),
            "predict_rows": int(len(predict)),
            "predict_dates": int(predict["datetime"].nunique()),
            "positive_score_rows": int((predict["ml_score"] > 0).sum()),
            "positive_score_dates": int(predict.loc[predict["ml_score"] > 0, "datetime"].nunique()),
            "base_positive_score_rows": int((predict["ml_base_score"] > 0).sum()),
            "base_positive_score_dates": int(predict.loc[predict["ml_base_score"] > 0, "datetime"].nunique()),
        })
    if not scored_parts:
        raise ValueError("No walk-forward folds could be trained")
    return pd.concat(scored_parts, ignore_index=True).sort_values(["datetime", "instrument"]), folds


def build_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=160,
            learning_rate=0.045,
            max_leaf_nodes=23,
            min_samples_leaf=80,
            l2_regularization=0.10,
            random_state=20260715,
        ),
    )


def deterministic_random_score(frame: pd.DataFrame) -> pd.Series:
    hashed = pd.util.hash_pandas_object(
        frame[["datetime", "instrument"]].astype(str), index=False
    ).astype("uint64")
    return (hashed / np.float64(np.iinfo(np.uint64).max)).astype(float)


def build_report(scored: pd.DataFrame, folds: list[dict]) -> dict:
    methods = {
        "ml": "ml_score",
        "ml_rank": "ml_rank_score",
        "ml_base": "ml_base_score",
        "ml_base_rank": "ml_base_rank_score",
        "ml_consensus": "ml_consensus_score",
        "ml_base_consensus": "ml_base_consensus_score",
        "rule": "rule_score",
        "reversal": "reversal_value",
        "momentum": "momentum_score",
        "random": "random_score",
    }
    rows = []
    for method, score in methods.items():
        eligible = scored.dropna(subset=[score]).copy()
        if method in {"ml", "ml_base"}:
            eligible = eligible[eligible[score] > 0]
        ranked = eligible.sort_values(["datetime", score, "instrument"], ascending=[True, False, True]).copy()
        ranked["daily_rank"] = ranked.groupby("datetime").cumcount() + 1
        for topk in TOPKS:
            selected = ranked[ranked["daily_rank"] <= topk]
            rows.append(summarize_selection(selected, method=method, topk=topk))
    paired = paired_bootstrap_table(scored, methods)
    return {
        "definition": {
            "market_gate": "0AMV_ALL positive 1d/2d shock >= prior 252d q85; exit below MA10",
            "stock_gate": "SH active bull + brick reversal + ret1>=4% but below limit + stock right-side + historical non-ST",
            "label": "T+1 open to T+11 open net return, daily cross-sectional percentile",
            "return_model_target": "net fwd10 return; inverse candidate-count and recency weighted",
            "cash_gate": "direct-return ML selects only predictions > 0; missing positive picks leave slots in cash",
            "purge": "train label_end < prediction month first day",
            "features": FEATURES,
            "base_features": BASE_FEATURES,
            "enhancement_features": ENHANCEMENT_FEATURES,
            "structure_features": STRUCTURE_FEATURES,
        },
        "rows_scored": int(len(scored)),
        "dates_scored": int(scored["datetime"].nunique()),
        "folds": folds,
        "topk": rows,
        "paired_bootstrap": paired,
        "notes": [
            "The rolling q85 threshold is an explicit high-quality proxy calibration, not the proprietary Compass 0AMV formula.",
            "TopK label summaries are not an account-level backtest because overlapping holdings and capital constraints are not replayed here.",
        ],
    }


def paired_bootstrap_table(scored: pd.DataFrame, methods: dict[str, str]) -> list[dict]:
    daily_by_method: dict[tuple[str, int], pd.Series] = {}
    for method, score in methods.items():
        eligible = scored.dropna(subset=[score]).copy()
        if method in {"ml", "ml_base"}:
            eligible = eligible[eligible[score] > 0]
        ranked = eligible.sort_values(["datetime", score, "instrument"], ascending=[True, False, True]).copy()
        ranked["daily_rank"] = ranked.groupby("datetime").cumcount() + 1
        for topk in TOPKS:
            selected = ranked[ranked["daily_rank"] <= topk]
            daily_by_method[(method, topk)] = selected.groupby("datetime")["fwd10_net"].mean()
    rows = []
    for topk in TOPKS:
        ml = daily_by_method[("ml", topk)]
        for control in ("ml_base", "ml_rank", "ml_base_rank", "rule", "reversal", "momentum", "random"):
            pair = pd.concat({"ml": ml, "control": daily_by_method[(control, topk)]}, axis=1).dropna()
            difference = pair["ml"] - pair["control"]
            low, high, probability_positive = moving_block_bootstrap(difference)
            rows.append({
                "topk": topk,
                "control": control,
                "dates": int(len(difference)),
                "mean_daily_label_excess": float(difference.mean()),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "bootstrap_probability_positive": probability_positive,
            })
    return rows


def moving_block_bootstrap(values: pd.Series, *, block_size: int = 10, samples: int = 2000) -> tuple[float, float, float]:
    data = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(20260715)
    starts = np.arange(len(data))
    blocks_needed = int(np.ceil(len(data) / block_size))
    means = np.empty(samples, dtype=float)
    offsets = np.arange(block_size)
    for index in range(samples):
        selected_starts = rng.choice(starts, size=blocks_needed, replace=True)
        positions = (selected_starts[:, None] + offsets[None, :]) % len(data)
        means[index] = data[positions.ravel()[:len(data)]].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), float((means > 0).mean())


def summarize_selection(frame: pd.DataFrame, *, method: str, topk: int) -> dict:
    return {
        "method": method,
        "topk": topk,
        "rows": int(len(frame)),
        "dates": int(frame["datetime"].nunique()),
        "fwd5_mean": float(frame["fwd5_net"].mean()),
        "fwd10_mean": float(frame["fwd10_net"].mean()),
        "fwd10_win": float((frame["fwd10_net"] > 0).mean()),
        "yearly_fwd10": {
            str(int(year)): float(group["fwd10_net"].mean())
            for year, group in frame.groupby(frame["datetime"].dt.year)
        },
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Walk-Forward Active-Value Brick ML",
        "",
        f"- Market gate: {report['definition']['market_gate']}",
        f"- Stock gate: {report['definition']['stock_gate']}",
        f"- Label: {report['definition']['label']}",
        f"- Purge: {report['definition']['purge']}",
        f"- Scored rows/dates: {report['rows_scored']} / {report['dates_scored']}",
        "",
        "## TopK",
        "",
    ]
    for row in report["topk"]:
        lines.append(f"- `{json.dumps(row, ensure_ascii=False)}`")
    lines.extend(["", "## Paired Bootstrap", ""])
    for row in report["paired_bootstrap"]:
        lines.append(f"- `{json.dumps(row, ensure_ascii=False)}`")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in report["notes"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
