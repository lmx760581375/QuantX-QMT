"""Active-value brick ML score export tests."""

import json

import pandas as pd
import pytest

from quantx.tools.export_active_brick_ml_scores import export_score_asset


def test_export_score_asset_filters_and_writes_metadata(tmp_path):
    source = tmp_path / "scores.parquet"
    output = tmp_path / "derived" / "scores.parquet"
    metadata = tmp_path / "derived" / "metadata.json"
    pd.DataFrame([
        {"datetime": "2025-01-01", "instrument": "sz000001", "ml_consensus_score": 0.9},
        {"datetime": "2025-01-02", "instrument": "sz000002", "ml_consensus_score": None},
        {"datetime": "2025-01-03", "instrument": "sh600000", "ml_consensus_score": 0.4},
        {"datetime": "2025-01-03", "instrument": "sz000003", "ml_consensus_score": 0.8},
    ]).to_parquet(source, index=False)

    report = export_score_asset(
        source=source,
        output=output,
        metadata_output=metadata,
        start="2025-01-02",
        end="2025-01-31",
        min_rows=2,
        min_signal_days=1,
    )

    exported = pd.read_parquet(output)
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    assert report["rows"] == 2
    assert report["signal_days"] == 1
    assert exported["instrument"].tolist() == ["SZ000003", "SH600000"]
    assert exported["ml_consensus_score"].tolist() == [0.8, 0.4]
    assert meta["output_rows"] == 2
    assert meta["source_sha256"].startswith("sha256:")
    assert meta["output_sha256"].startswith("sha256:")


def test_export_score_asset_rejects_duplicate_date_instrument(tmp_path):
    source = tmp_path / "scores.parquet"
    output = tmp_path / "scores_out.parquet"
    metadata = tmp_path / "metadata.json"
    pd.DataFrame([
        {"datetime": "2025-01-03", "instrument": "SZ000001", "ml_consensus_score": 0.4},
        {"datetime": "2025-01-03", "instrument": "SZ000001", "ml_consensus_score": 0.5},
    ]).to_parquet(source, index=False)

    with pytest.raises(ValueError, match="Duplicate score rows"):
        export_score_asset(source=source, output=output, metadata_output=metadata)

