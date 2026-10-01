from argparse import Namespace

import torch

from quantx_reward.reproduce import build_reproduction_plan


def test_reproduction_plan_contains_complete_pipeline(tmp_path):
    args = Namespace(
        data_root=str(tmp_path / "data"),
        feature_root=str(tmp_path / "features"),
        reward_root=str(tmp_path / "reward"),
        inference_root=str(tmp_path / "inference"),
        historical_score_path=str(tmp_path / "historical.parquet"),
        incremental_score_path=str(tmp_path / "incremental.parquet"),
        score_path=str(tmp_path / "score.parquet"),
        output_dir=str(tmp_path / "runs"),
        data_start="2010-01-01",
        score_start="2020-01-02",
        as_of="2026-09-28",
        historical_data_end="2026-07-15",
        historical_cutoff="2026-06-02",
        overlap_start=None,
        overlap_trading_days=7,
        run_id="reproduce",
        nproc_per_node=1,
        skip_data_download=False,
        resume_data=False,
        update_data=False,
        skip_dataset=False,
        skip_historical_inference=False,
        skip_inference_index=False,
        skip_incremental_inference=False,
        skip_merge=False,
        force=False,
    )

    plan = build_reproduction_plan(args)

    assert set(plan["commands"]) == {
        "bootstrap_data",
        "prepare_features",
        "prepare_reward_dataset",
        "historical_infer",
        "prepare_inference_index",
        "incremental_infer",
        "merge_scores",
        "dual_sleeve",
    }
    assert plan["as_of"] == "2026-09-28"
    assert plan["signal_end"] == "2026-09-24"
    assert "models.reward.infer" in plan["commands"]["historical_infer"]
    assert "--standalone" not in plan["commands"]["historical_infer"]
    assert "--output-dir" in plan["commands"]["dual_sleeve"]

    args.nproc_per_node = 8
    distributed = build_reproduction_plan(args)["commands"]["historical_infer"]
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and not torch.cuda.is_available():
        assert "--standalone" not in distributed
        assert "--device" in distributed and "mps" in distributed
        assert "--ddp" in distributed and "off" in distributed
    else:
        assert "--standalone" in distributed
        assert "--" in distributed
