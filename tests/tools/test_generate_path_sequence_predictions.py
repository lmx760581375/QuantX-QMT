"""Path-sequence PredictionStore generation tests."""

import json

import pandas as pd
import pytest

from quantx.core.decision.predictions import PredictionStore
from quantx.tools.generate_path_sequence_predictions import main


def test_generate_path_sequence_predictions_writes_loadable_store(tmp_path):
    panel = tmp_path / "panel.parquet"
    write_panel(panel)
    output = tmp_path / "predictions.json"

    rc = main([
        "--panel", str(panel),
        "--signal-date", "2026-07-10",
        "--output", str(output),
        "--topk-pool", "2",
        "--min-records", "2",
        "--training-information-end", "2026-07-09",
    ])

    store = PredictionStore.load(output)
    records = store.records_for("2026-07-10", artifact_id="path_sequence_top7_bridge_v1")
    manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert len(records) == 2
    assert records[0].rank == 1
    assert manifest["record_count"] == 2


def test_generate_path_sequence_predictions_requires_explicit_dl_score(tmp_path):
    panel = tmp_path / "panel.parquet"
    write_panel(panel)
    output = tmp_path / "predictions.json"

    with pytest.raises(ValueError, match="DL score column"):
        main([
            "--panel", str(panel),
            "--signal-date", "2026-07-10",
            "--output", str(output),
            "--score-mode", "dl_blend",
            "--training-information-end", "2026-07-09",
        ])


def write_panel(path):
    rows = []
    for day in ("2026-07-09", "2026-07-10"):
        for i, symbol in enumerate(("SZ000001", "SH600000", "SZ000002"), start=1):
            rows.append({
                "session": day,
                "instrument": symbol,
                "part_base": 1.0 / i,
                "part_path_ev": 0.2 * i,
                "part_path_spread": 0.3 * i,
                "part_path_topq": 0.4 * i,
                "part_exec_cls": 0.5 * i,
            })
    pd.DataFrame(rows).to_parquet(path, index=False)
