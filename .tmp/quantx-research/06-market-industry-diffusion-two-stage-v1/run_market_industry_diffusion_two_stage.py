"""QMT-only market and industry diffusion two-stage ranker experiment.

All artifacts stay under .tmp/quantx-research. This script is intentionally
standalone so the experiment can be archived without promoting code.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRanker

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantx.core.data.qlib_reader import QlibBinReader
from quantx.core.decision.clock import MarketTime, SessionPhase
from quantx.core.decision.predictions import PredictionRecord, PredictionStore
from quantx.tools.run_backtest import load_a_share_symbols


ARTIFACT_ID = "market_industry_diffusion_two_stage_v1"
FEATURE_SCHEMA_HASH = "market_industry_diffusion_two_stage_v1"
ROOT = Path(".tmp/quantx-research/06-market-industry-diffusion-two-stage-v1")
PROVIDER = Path("data/qlib_data_fixed")
META = Path("data/meta/quantx_meta.sqlite")
START = "2021-01-04"
END = "2026-07-15"
BACKTEST_START = "2022-01-04"
SIGNAL_STEP_SESSIONS = 5
PREDICTION_POOL_TOPK = 200
PORTFOLIO_TOPK = 20
MIN_INDUSTRY_SIZE = 12
INDUSTRY_TOP_QUANTILE = 0.82
MARKET_BREADTH_FLOOR = 0.28


@dataclass
class ExperimentResult:
    ok: bool
    prediction_store: str
    prediction_checksum: str
    config_path: str
    run_dir: str
    summary: dict
    trade_audit: dict
    label_diagnostic: dict
    feature_columns: list[str]
    notes: list[str]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(ROOT))
    parser.add_argument("--provider-uri", default=str(PROVIDER))
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--backtest-start", default=BACKTEST_START)
    args = parser.parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    write_progress(root, "start", {"artifact_id": ARTIFACT_ID, "signal_step_sessions": SIGNAL_STEP_SESSIONS})

    provider = Path(args.provider_uri)
    symbols = load_a_share_symbols(provider, args.start, args.end, universe="all_mainboard")
    write_progress(root, "symbols_loaded", {"symbol_count": len(symbols)})
    reader = QlibBinReader(provider)
    quote = reader.features(symbols, ["$open", "$high", "$low", "$close", "$volume", "$amount", "$vwap"], args.start, args.end)
    quote = normalize_quote(quote)
    symbols = sorted(set(quote.index.get_level_values("instrument")))
    write_progress(root, "quote_loaded", {"rows": int(len(quote)), "symbol_count": len(symbols)})
    industry, concepts, risk_names = load_meta(symbols)
    write_progress(root, "meta_loaded", {"industry_count": len(industry), "concept_symbol_count": len(concepts), "risk_name_count": len(risk_names)})
    panel = build_panel(quote, industry, concepts, risk_names)
    write_progress(root, "panel_built", {"rows": int(len(panel)), "columns": int(len(panel.columns))})
    scored, feature_columns, label_diagnostic = walk_forward_rank(panel, root=root, backtest_start=args.backtest_start)
    write_progress(root, "scored", {"rows": int(len(scored)), "feature_count": len(feature_columns)})
    prediction_path = root / "predictions.json"
    checksum = write_prediction_store(scored, prediction_path)
    write_progress(root, "predictions_written", {"path": str(prediction_path), "checksum": checksum})
    config_path = write_backtest_config(root, prediction_path, checksum, args.backtest_start, args.end)
    summary = run_backtest(config_path, root)
    write_progress(root, "backtest_done", {"run_dir": summary.get("run_dir"), "summary": summary.get("summary")})
    run_dir = Path(summary["run_dir"])
    trade_audit = audit_trades(run_dir, quote, risk_names)
    write_progress(root, "trade_audit_done", trade_audit)
    result = ExperimentResult(
        ok=True,
        prediction_store=str(prediction_path),
        prediction_checksum=checksum,
        config_path=str(config_path),
        run_dir=str(run_dir),
        summary=summary,
        trade_audit=trade_audit,
        label_diagnostic=label_diagnostic,
        feature_columns=feature_columns,
        notes=[
            "Annual walk-forward: each test year uses only rows with year < test_year.",
            "Training labels penalize next-open untradable bars, but prediction features use only T and earlier data.",
            "Industry membership is current local MetaStore snapshot, not point-in-time history.",
            "The industry gate uses only T and earlier market/industry state, then the stock model ranks within gated industries.",
        ],
    )
    (root / "result.json").write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
    return 0


def write_progress(root: Path, step: str, payload: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "progress.jsonl"
    row = {
        "ts": datetime.now(tz=ZoneInfo("Asia/Shanghai")).isoformat(),
        "elapsed_epoch": time.time(),
        "step": step,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({"progress": step, **payload}, ensure_ascii=False, default=str), flush=True)


def normalize_quote(quote: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(quote.index, pd.MultiIndex):
        raise ValueError("quote must use MultiIndex")
    names = list(quote.index.names)
    if names != ["datetime", "instrument"]:
        quote = quote.reorder_levels(["datetime", "instrument"]) if "datetime" in names else quote.swaplevel(0, 1)
    quote = quote.sort_index()
    quote.index = quote.index.set_names(["datetime", "instrument"])
    return quote


def load_meta(symbols: list[str]) -> tuple[dict[str, str], dict[str, list[str]], set[str]]:
    industry: dict[str, str] = {}
    concepts: dict[str, list[str]] = {symbol: [] for symbol in symbols}
    risk_names: set[str] = set()
    if not META.exists():
        return industry, concepts, risk_names
    with sqlite3.connect(META) as conn:
        for symbol, industry_name in conn.execute("select symbol, industry_name from industry_membership where level=1"):
            industry[str(symbol)] = str(industry_name or "")
        for symbol, sector_name in conn.execute("select symbol, sector_name from sector_membership where sector_type='concept'"):
            symbol = normalize_symbol(str(symbol))
            name = str(sector_name or "")
            if name and not noisy_concept(name):
                concepts.setdefault(symbol, []).append(name)
        for symbol, name in conn.execute("select symbol, name from security_master"):
            upper = str(name or "").upper()
            if "ST" in upper or "退" in upper:
                risk_names.add(normalize_symbol(str(symbol)))
    return industry, concepts, risk_names


def normalize_symbol(symbol: str) -> str:
    value = symbol.strip().upper()
    if "." in value:
        code, exchange = value.split(".", 1)
        return f"{exchange}{code}"
    return value


def noisy_concept(name: str) -> bool:
    bad = ("昨日", "ST", "退市", "融资融券", "沪股通", "深股通", "创业板", "预亏", "预盈")
    return any(token in name for token in bad)


def build_panel(quote: pd.DataFrame, industry: dict[str, str], concepts: dict[str, list[str]], risk_names: set[str]) -> pd.DataFrame:
    df = quote.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ret1"] = df.groupby(level="instrument")["$close"].pct_change(fill_method=None)
    for window in (5, 10, 20, 60):
        df[f"ret{window}"] = df.groupby(level="instrument")["$close"].pct_change(window, fill_method=None)
        df[f"amount_mean{window}"] = df.groupby(level="instrument")["$amount"].rolling(window, min_periods=max(3, window // 3)).mean().reset_index(level=0, drop=True)
        df[f"vol{window}"] = df.groupby(level="instrument")["ret1"].rolling(window, min_periods=max(3, window // 3)).std().reset_index(level=0, drop=True)
    df["vwap_gap"] = df["$close"] / df["$vwap"] - 1.0
    df["intraday_pos"] = (df["$close"] - df["$low"]) / (df["$high"] - df["$low"]).replace(0, np.nan)
    df["range_pct"] = df["$high"] / df["$low"] - 1.0
    for col in ["ret1", "ret5", "ret10", "ret20", "ret60", "amount_mean20", "amount_mean60", "vol20", "vol60", "vwap_gap", "intraday_pos", "range_pct"]:
        df[f"{col}_rank"] = df.groupby(level="datetime")[col].rank(pct=True)
    market = df.groupby(level="datetime").agg(
        market_ret20_median=("ret20", "median"),
        market_ret60_median=("ret60", "median"),
        market_disp20=("ret20", "std"),
        market_breadth20=("ret20", lambda x: float((x > 0).mean())),
        market_amount_disp20=("amount_mean20", "std"),
    )
    df = df.join(market, on="datetime")
    reset = df.reset_index()
    reset["industry"] = reset["instrument"].map(industry).fillna("")
    reset["concepts"] = reset["instrument"].map(lambda s: concepts.get(s, []))
    reset["is_current_risk_name"] = reset["instrument"].isin(risk_names)
    reset = add_group_features(reset)
    reset = add_labels_and_audit(reset)
    feature_cols = [
        c for c in reset.columns
        if c.endswith("_rank") or c.startswith("market_") or c.startswith("ind_") or c.startswith("member_")
    ]
    for col in feature_cols:
        reset[col] = pd.to_numeric(reset[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    reset = reset.set_index(["datetime", "instrument"]).sort_index()
    return reset


def add_group_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["datetime", "instrument"]).copy()
    frame["amount5_20_rel"] = frame["amount_mean5"] / frame["amount_mean20"] - 1.0
    for window in (5, 20):
        ind_mean = frame.groupby(["datetime", "industry"])[f"ret{window}"].transform("mean")
        frame[f"ind_ret{window}_mean"] = ind_mean
        frame[f"ind_rel_ret{window}"] = frame[f"ret{window}"] - ind_mean
        frame[f"ind_ret{window}_rank"] = frame.groupby("datetime")[f"ind_ret{window}_mean"].rank(pct=True)
        frame[f"ind_rel_ret{window}_rank"] = frame.groupby("datetime")[f"ind_rel_ret{window}"].rank(pct=True)
    grouped = frame.groupby(["datetime", "industry"], sort=False)
    industry_state = grouped.agg(
        ind_member_count=("instrument", "size"),
        ind_ret5_mean=("ret5", "mean"),
        ind_ret20_mean=("ret20", "mean"),
        ind_breadth5=("ret5", lambda s: float((s > 0).mean())),
        ind_breadth20=("ret20", lambda s: float((s > 0).mean())),
        ind_amount_rel_mean=("amount5_20_rel", "mean"),
        ind_vol20_mean=("vol20", "mean"),
        ind_range_mean=("range_pct", "mean"),
    ).reset_index()
    industry_state = industry_state[industry_state["industry"] != ""].copy()
    for col in ["ind_ret5_mean", "ind_ret20_mean"]:
        industry_state[f"{col}_rank"] = industry_state.groupby("datetime")[col].rank(pct=True)
    for col in [
        "ind_breadth5",
        "ind_breadth20",
        "ind_amount_rel_mean",
        "ind_vol20_mean",
        "ind_range_mean",
        "ind_member_count",
    ]:
        industry_state[f"{col}_rank"] = industry_state.groupby("datetime")[col].rank(pct=True)
    industry_state["ind_diffusion_score"] = (
        0.25 * industry_state["ind_ret5_mean_rank"]
        + 0.15 * industry_state["ind_ret20_mean_rank"]
        + 0.25 * industry_state["ind_breadth5_rank"]
        + 0.15 * industry_state["ind_breadth20_rank"]
        + 0.15 * industry_state["ind_amount_rel_mean_rank"]
        + 0.05 * industry_state["ind_range_mean_rank"]
    )
    industry_state["ind_diffusion_rank"] = industry_state.groupby("datetime")["ind_diffusion_score"].rank(pct=True)
    keep_cols = [
        "datetime",
        "industry",
        "ind_ret5_mean_rank",
        "ind_ret20_mean_rank",
        "ind_member_count",
        "ind_breadth5",
        "ind_breadth20",
        "ind_amount_rel_mean",
        "ind_vol20_mean",
        "ind_range_mean",
        "ind_breadth5_rank",
        "ind_breadth20_rank",
        "ind_amount_rel_mean_rank",
        "ind_vol20_mean_rank",
        "ind_range_mean_rank",
        "ind_member_count_rank",
        "ind_diffusion_score",
        "ind_diffusion_rank",
    ]
    frame = frame.merge(industry_state[keep_cols], on=["datetime", "industry"], how="left")
    for col in ["ret5", "ret20", "ret60", "amount5_20_rel", "amount_mean20", "vol20", "vwap_gap", "intraday_pos", "range_pct"]:
        frame[f"member_{col}_rank"] = frame.groupby(["datetime", "industry"])[col].rank(pct=True)
    return frame


def add_labels_and_audit(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["instrument", "datetime"]).copy()
    grouped = frame.groupby("instrument", group_keys=False)
    frame["entry_open"] = grouped["$open"].shift(-1)
    frame["exit_open"] = grouped["$open"].shift(-6)
    frame["label_raw5"] = frame["exit_open"] / frame["entry_open"] - 1.0
    day_mean = frame.groupby("datetime")["label_raw5"].transform("mean")
    industry_mean = frame.groupby(["datetime", "industry"])["label_raw5"].transform("mean")
    frame["label_excess5"] = frame["label_raw5"] - day_mean
    frame["label_ind_excess5"] = frame["label_raw5"] - industry_mean
    frame["next_high"] = grouped["$high"].shift(-1)
    frame["next_low"] = grouped["$low"].shift(-1)
    frame["next_close"] = grouped["$close"].shift(-1)
    frame["next_volume"] = grouped["$volume"].shift(-1)
    frame["next_amount"] = grouped["$amount"].shift(-1)
    frame["open_gap"] = frame["entry_open"] / frame["$close"] - 1.0
    one_price = (frame["next_high"].sub(frame["next_low"]).abs() <= 1e-6) & (frame["entry_open"].sub(frame["next_close"]).abs() <= 1e-6)
    frame["entry_one_price_limit_up"] = one_price & (frame["next_close"] / frame["$close"] - 1.0 >= 0.045)
    frame["entry_zero_volume_or_amount"] = (frame["next_volume"] <= 0) | (frame["next_amount"] <= 0)
    frame["entry_open_gap_ge_3pct"] = frame["open_gap"] >= 0.03 - 1e-12
    abs_ret = frame.groupby("instrument")["$close"].pct_change(fill_method=None).abs()
    five = (abs_ret >= 0.045) & (abs_ret <= 0.055)
    ten = abs_ret >= 0.075
    frame["st_like_limit_history"] = (
        five.groupby(frame["instrument"]).rolling(60, min_periods=1).sum().reset_index(level=0, drop=True) >= 2
    ) & (
        ten.groupby(frame["instrument"]).rolling(60, min_periods=1).sum().reset_index(level=0, drop=True) == 0
    )
    contam = frame["entry_one_price_limit_up"] | frame["entry_zero_volume_or_amount"] | frame["entry_open_gap_ge_3pct"] | frame["st_like_limit_history"] | frame["is_current_risk_name"]
    frame["label_exec5"] = frame["label_excess5"].where(~contam, -0.20)
    frame["label_ind_exec5"] = frame["label_ind_excess5"].where(~contam, -0.20)
    return frame


def qcut_labels(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    out = pd.Series(np.nan, index=series.index)
    if len(valid) < 10:
        return out
    ranks = valid.rank(method="first")
    out.loc[valid.index] = pd.qcut(ranks, 5, labels=False, duplicates="drop")
    return out


def walk_forward_rank(panel: pd.DataFrame, *, root: Path, backtest_start: str) -> tuple[pd.DataFrame, list[str], dict]:
    feature_columns = [
        c for c in panel.columns
        if c.endswith("_rank") or c.startswith("market_") or c.startswith("ind_") or c.startswith("member_")
    ]
    data = panel.reset_index()
    signal_dates = weekly_signal_dates(data["datetime"], backtest_start=backtest_start, step=SIGNAL_STEP_SESSIONS)
    data = data[data["datetime"].isin(signal_dates)].copy()
    gate = (
        (data["industry"] != "")
        & (data["ind_member_count"] >= MIN_INDUSTRY_SIZE)
        & (data["ind_diffusion_rank"] >= INDUSTRY_TOP_QUANTILE)
        & (data["market_breadth20"] >= MARKET_BREADTH_FLOOR)
    )
    data = data[gate].copy()
    data["year"] = pd.to_datetime(data["datetime"]).dt.year
    data = data.dropna(subset=["label_ind_exec5", "label_exec5", *feature_columns]).copy()
    data["label_q5"] = data.groupby(["datetime", "industry"])["label_ind_exec5"].transform(qcut_labels)
    data = data.dropna(subset=["label_q5"]).copy()
    scored_parts = []
    diagnostics = {
        "years": {},
        "rows": int(len(data)),
        "signal_dates": len(signal_dates),
        "signal_step_sessions": SIGNAL_STEP_SESSIONS,
        "industry_top_quantile": INDUSTRY_TOP_QUANTILE,
        "market_breadth_floor": MARKET_BREADTH_FLOOR,
        "min_industry_size": MIN_INDUSTRY_SIZE,
    }
    write_progress(root, "walk_forward_input", diagnostics)
    for year in range(2022, 2027):
        train = data[data["year"] < year].copy()
        test = data[data["year"] == year].copy()
        if train.empty or test.empty:
            continue
        train = train.sort_values(["datetime", "industry", "instrument"])
        test = test.sort_values(["datetime", "industry", "instrument"])
        group = train.groupby(["datetime", "industry"], sort=False).size().to_numpy()
        model = LGBMRanker(
            objective="lambdarank",
            n_estimators=90,
            learning_rate=0.045,
            num_leaves=15,
            min_child_samples=120,
            subsample=0.80,
            colsample_bytree=0.80,
            random_state=20260715 + year,
            n_jobs=4,
            verbose=-1,
        )
        write_progress(root, "fit_start", {"year": year, "train_rows": int(len(train)), "test_rows": int(len(test)), "groups": int(len(group))})
        model.fit(train[feature_columns], train["label_q5"].astype(int), group=group)
        test["score"] = model.predict(test[feature_columns])
        test["score"] = test["score"] + 0.25 * test["ind_diffusion_score"] + 0.10 * test["member_amount5_20_rel_rank"]
        test["rank"] = test.groupby("datetime")["score"].rank(ascending=False, method="first")
        scored_parts.append(test)
        top = test[test["rank"] <= 20]
        diagnostics["years"][str(year)] = {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "top20_mean_exec_label": float(top["label_exec5"].mean()) if not top.empty else None,
            "top20_count_per_day": float(top.groupby("datetime").size().mean()) if not top.empty else 0.0,
            "candidate_count_per_day": float(test.groupby("datetime").size().mean()) if not test.empty else 0.0,
            "candidate_industries_per_day": float(test.groupby("datetime")["industry"].nunique().mean()) if not test.empty else 0.0,
            "top20_contamination_current_risk_name": int(top["is_current_risk_name"].sum()),
            "top20_contamination_st_like": int(top["st_like_limit_history"].sum()),
        }
        write_progress(root, "fit_done", {"year": year, **diagnostics["years"][str(year)]})
    scored = pd.concat(scored_parts, ignore_index=True).sort_values(["datetime", "rank"])
    return scored, feature_columns, diagnostics


def weekly_signal_dates(values: pd.Series, *, backtest_start: str, step: int) -> set[pd.Timestamp]:
    dates = pd.Series(pd.to_datetime(values).unique()).sort_values(ignore_index=True)
    backtest_start_ts = pd.Timestamp(backtest_start)
    anchor_idx = int(dates.searchsorted(backtest_start_ts, side="left"))
    selected = dates.iloc[anchor_idx::step].tolist()
    train_selected = dates.iloc[:anchor_idx:step].tolist()
    return set(pd.Timestamp(date) for date in [*train_selected, *selected])


def write_prediction_store(scored: pd.DataFrame, path: Path) -> str:
    records = []
    tz = ZoneInfo("Asia/Shanghai")
    for row in scored[scored["rank"] <= PREDICTION_POOL_TOPK].itertuples(index=False):
        session = pd.Timestamp(row.datetime).strftime("%Y-%m-%d")
        records.append(PredictionRecord(
            signal_time=MarketTime(
                session=session,
                phase=SessionPhase.AFTER_CLOSE,
                timestamp=pd.Timestamp(row.datetime).to_pydatetime().replace(hour=15, minute=0, tzinfo=tz),
            ),
            instrument=str(row.instrument),
            score=float(row.score),
            prediction_horizon=5,
            artifact_id=ARTIFACT_ID,
            fold_id=f"year-{int(row.year)}",
            feature_schema_hash=FEATURE_SCHEMA_HASH,
            evaluation_tier="development_oos" if int(row.year) < 2026 else "live_shadow",
            rank=int(row.rank),
        ))
    return PredictionStore(records).write(path)


def write_backtest_config(root: Path, prediction_path: Path, checksum: str, start: str, end: str) -> Path:
    config = {
        "name": ARTIFACT_ID,
        "title": "QMT市场行业扩散二阶段Ranker v1",
        "version": 1,
        "data": {
            "provider_uri": str(PROVIDER),
            "universe": "prediction_store",
            "prediction_pool_topk": PREDICTION_POOL_TOPK,
            "start": start,
            "end": end,
            "look_back_days": 80,
        },
        "strategy": {
            "type": "decision_pipeline",
            "rebalance_interval_sessions": 5,
            "alpha": {
                "type": "predictions",
                "path": str(prediction_path),
                "checksum": checksum,
                "artifact_id": ARTIFACT_ID,
                "feature_schema_hash": FEATURE_SCHEMA_HASH,
            },
            "portfolio": {"type": "topk_equal", "top_k": PORTFOLIO_TOPK, "gross_exposure": 0.98},
            "risk": {"type": "noop"},
            "exit": {"max_loss_pct": -0.10},
            "buy_filter": {
                "exclude_st": True,
                "meta_store_uri": "data/meta/quantx_meta.sqlite",
                "exclude_st_like_limit_rate": True,
                "st_like_lookback_sessions": 60,
                "st_like_min_limit_hits": 2,
                "max_buy_open_gap_pct": 0.03,
                "cooldown_sessions_after_sell": 5,
            },
            "order_planner": {"type": "standard", "lot_size": 100},
        },
        "execution": {"deal_price": "open"},
        "cost": {"commission_rate": 0.0005, "min_commission": 5.0, "stamp_tax_rate": 0.0001, "stamp_tax_on_buy": True, "transfer_fee_rate": 0.0, "slippage": 0.0},
        "engine": {"init_cash": 1000000, "validate_trading_rules": True, "deal_price": "open", "max_workers": 1, "error_policy": "fail_fast", "legacy_cost_price": False, "auto_adjust_buy_quantity": True},
    }
    path = root / "backtest_config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def run_backtest(config_path: Path, root: Path) -> dict:
    cmd = [
        "/Users/mingxiaoli/anaconda3/envs/test/bin/python",
        "-m", "quantx.tools.run_backtest",
        "--config", str(config_path),
        "--output-dir", str(root / "runs"),
        "--run-id", f"{ARTIFACT_ID}_top20_reb5",
        "--json",
    ]
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    start = proc.stdout.find("{")
    if start < 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    (root / "backtest_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (root / "backtest_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    return json.loads(proc.stdout[start:])


def audit_trades(run_dir: Path, quote: pd.DataFrame, risk_names: set[str]) -> dict:
    trades = json.loads((run_dir / "trades.json").read_text())
    buys = [t for t in trades if t.get("action") == "BUY" and not t.get("reject_reason")]
    violations = {"current_st_or_delist_name": 0, "missing_bar": 0, "zero_volume_or_amount": 0, "one_price_limit_up": 0, "open_gap_ge_3pct": 0}
    buy_counts: dict[str, int] = {}
    for trade in buys:
        symbol = str(trade["symbol"])
        date = pd.Timestamp(trade["date"])
        buy_counts[symbol] = buy_counts.get(symbol, 0) + 1
        if symbol in risk_names:
            violations["current_st_or_delist_name"] += 1
        key = (date, symbol)
        if key not in quote.index:
            violations["missing_bar"] += 1
            continue
        row = quote.loc[key]
        if float(row.get("$volume", np.nan) or 0) <= 0 or float(row.get("$amount", np.nan) or 0) <= 0:
            violations["zero_volume_or_amount"] += 1
        try:
            hist = quote.xs(symbol, level="instrument")
            pos = hist.index.get_loc(date)
            preclose = float(hist.iloc[pos - 1]["$close"]) if isinstance(pos, int) and pos > 0 else np.nan
        except Exception:
            preclose = np.nan
        if np.isfinite(preclose) and preclose > 0:
            open_, high, low, close = (float(row[f]) for f in ("$open", "$high", "$low", "$close"))
            if max(abs(high - low), abs(open_ - close)) <= 1e-6 and close / preclose - 1.0 >= 0.045:
                violations["one_price_limit_up"] += 1
            if open_ / preclose - 1.0 >= 0.03 - 1e-12:
                violations["open_gap_ge_3pct"] += 1
    top_repeat = sorted(buy_counts.items(), key=lambda item: item[1], reverse=True)[:20]
    return {
        "effective_buys": len(buys),
        "trade_rows": len(trades),
        "reject_rows": sum(1 for t in trades if t.get("reject_reason")),
        "violations": violations,
        "unique_buy_symbols": len(buy_counts),
        "top_repeat_buys": top_repeat,
    }


if __name__ == "__main__":
    raise SystemExit(main())
