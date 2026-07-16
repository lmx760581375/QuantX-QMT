"""Train Brick forward-label model through 2024 and replay 2025-2026 OOS.

This script reuses the candidate/label parquet produced by
run_brick_pre2020_oos.py. It does not use public Top5 teacher labels and does
not search 2025-2026 execution thresholds.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(".tmp/quantx-research/brick-pre2020-oos-v1")
BASE_SCRIPT = ROOT / "run_brick_pre2020_oos.py"
TRAIN_END = "2024-12-31"
OOS_START = "2025-01-02"
END = "2026-07-15"
RUN_ID = "brick_train2024_forward_label_oos_top5_maxpos10"


@dataclass
class Result:
    ok: bool
    score_path: str
    config_path: str
    run_dir: str
    summary: dict
    yearly: dict
    position_diagnostic: dict
    trade_audit: dict
    label_diagnostic: dict
    notes: list[str]


def load_base():
    spec = importlib.util.spec_from_file_location("brick_pre2020_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    base = load_base()
    candidate_path = ROOT / "candidates_with_labels.parquet"
    if not candidate_path.exists():
        raise FileNotFoundError(f"missing candidate labels: {candidate_path}")
    candidates = pd.read_parquet(candidate_path)
    scored, diag = base.train_and_score(candidates, train_end=TRAIN_END, oos_start=OOS_START)

    score_path = ROOT / "scores_train2024_oos_2025_2026.parquet"
    scored[["date", "instrument", "score"]].to_parquet(score_path, index=False)

    config_path = base.write_backtest_config(ROOT, score_path, OOS_START, END)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["name"] = "brick_train2024_forward_label_oos_v1"
    config["description"] = (
        "Brick forward-label model trained through 2024 and replayed on 2025-2026. "
        "No public Top5 teacher labels, no OOS score_floor search, no open-gap hard filter."
    )
    config["execution"].get("buy", {}).pop("max_open_gap_pct", None)
    config_path = ROOT / "backtest_config_train2024_oos.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    summary = run_backtest(config_path, ROOT, RUN_ID)
    run_dir = Path(summary["run_dir"])
    yearly, position_diag = base.analyze_run(run_dir)
    trade_audit = audit_existing_run(base, run_dir, config_path)
    result = Result(
        ok=True,
        score_path=str(score_path),
        config_path=str(config_path),
        run_dir=str(run_dir),
        summary=summary,
        yearly=yearly,
        position_diagnostic=position_diag,
        trade_audit=trade_audit,
        label_diagnostic=diag,
        notes=[
            "Training labels use only candidate rows with signal date and dynamic-exit exit_date <= 2024-12-31.",
            "2025-2026 is used only for scoring, QuantX account replay, and post-run diagnostics.",
            "No open-gap >=3% hard filter is applied in this main Brick replay; open-gap buys are diagnostic only.",
            "No public Top5 teacher labels and no OOS score_floor/max_positions search are used.",
        ],
    )
    out = ROOT / "result_train2024_oos.json"
    out.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
    return 0


def run_backtest(config_path: Path, root: Path, run_id: str) -> dict:
    import subprocess

    cmd = [
        "/Users/mingxiaoli/anaconda3/envs/test/bin/python",
        "-m", "quantx.tools.run_backtest",
        "--config", str(config_path),
        "--output-dir", str(root / "runs"),
        "--run-id", run_id,
        "--json",
    ]
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    (root / "backtest_train2024_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (root / "backtest_train2024_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    start = proc.stdout.find("{")
    if start < 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return json.loads(proc.stdout[start:])


def audit_existing_run(base, run_dir: Path, config_path: Path) -> dict:
    from quantx.core.data.qlib_reader import QlibBinReader
    from quantx.tools.run_backtest import load_config, load_symbols

    config = load_config(config_path)
    symbols = load_symbols(config)
    reader = QlibBinReader("data/qlib_data_fixed")
    quote = reader.features(symbols, ["$open", "$high", "$low", "$close", "$volume", "$amount"], OOS_START, END)
    quote = quote.reorder_levels(["datetime", "instrument"]).sort_index()
    risk_names = base.load_current_risk_names()
    audit = base.audit_trades(run_dir, quote, risk_names)
    audit["open_gap_ge_3pct_is_diagnostic_only"] = True
    return audit


if __name__ == "__main__":
    raise SystemExit(main())
