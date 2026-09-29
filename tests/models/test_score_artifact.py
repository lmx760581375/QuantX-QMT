from __future__ import annotations

import json

import pandas as pd
import pytest

from models.reward.pipeline.artifact import (
    MarketScoreArtifactError,
    merge_market_score_artifacts,
    write_market_score_artifact,
)


def _write_score(path, rows):
    return write_market_score_artifact(
        pd.DataFrame(rows),
        path,
        score_columns=["reward_score_7d"],
    )


def test_merge_market_score_artifacts_uses_strict_cutoff(tmp_path):
    historical = tmp_path / "historical.parquet"
    incremental = tmp_path / "incremental.parquet"
    output = tmp_path / "merged.parquet"
    _write_score(
        historical,
        [
            {"signal_date": "2026-06-01", "instrument": "SH600000", "reward_score_7d": 0.1},
            {"signal_date": "2026-06-02", "instrument": "SH600000", "reward_score_7d": 0.2},
        ],
    )
    _write_score(
        incremental,
        [
            {"signal_date": "2026-06-02", "instrument": "SH600000", "reward_score_7d": 0.9},
            {"signal_date": "2026-06-03", "instrument": "SH600000", "reward_score_7d": 0.3},
        ],
    )

    manifest = merge_market_score_artifacts(
        historical,
        incremental,
        output,
        cutoff_date="2026-06-02",
    )

    merged = pd.read_parquet(output)
    assert merged["signal_date"].tolist() == ["2026-06-01", "2026-06-02", "2026-06-03"]
    assert merged["reward_score_7d"].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert manifest["rows"] == 3
    metadata = json.loads(output.with_suffix(".json").read_text())["metadata"]
    assert metadata["formal_status"] == "score_merged_ready_for_quantx"
    assert metadata["overlap_validation"]["dates"] == 1
    assert metadata["overlap_validation"]["minimum_top5_overlap"] == 1


def test_merge_market_score_artifacts_rejects_empty_increment_after_cutoff(tmp_path):
    historical = tmp_path / "historical.parquet"
    incremental = tmp_path / "incremental.parquet"
    _write_score(
        historical,
        [{"signal_date": "2026-06-02", "instrument": "SH600000", "reward_score_7d": 0.2}],
    )
    _write_score(
        incremental,
        [{"signal_date": "2026-06-02", "instrument": "SH600000", "reward_score_7d": 0.3}],
    )

    with pytest.raises(MarketScoreArtifactError, match="after cutoff"):
        merge_market_score_artifacts(
            historical,
            incremental,
            tmp_path / "merged.parquet",
            cutoff_date="2026-06-02",
        )
