from models.reward.launcher import build_command
from quantx_reward.cli import STRATEGY_CONFIGS, STRATEGY_SCORE_PATHS, build_parser
from quantx_reward.config import load_yaml, resolve_data_paths
from quantx_reward.verify import restore_weights, verify_environment, verify_weights


def test_resolve_data_paths_accepts_provider_directory(tmp_path):
    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    (provider / "calendars" / "day.txt").write_text("2021-01-04\n", encoding="utf-8")

    paths = resolve_data_paths(provider)

    assert paths["provider_uri"] == provider
    assert paths["data_root"] == tmp_path


def test_train_config_builds_versioned_torchrun_command():
    command = build_command("configs/model/reward_v1_train.yaml")

    assert "torch.distributed.run" in command
    assert "--standalone" in command
    assert "--nproc_per_node=8" in command
    assert "models.reward.train_h7_absolute" in command
    assert "--" in command
    assert "--reward-d-model" in command
    assert "768" in command


def test_train_config_supports_multi_node_torchrun():
    command = build_command(
        "configs/model/reward_v1_train.yaml",
        nproc_per_node=8,
        nnodes=4,
        node_rank=2,
        rdzv_endpoint="host:29400",
        rdzv_id="reward-h7",
    )

    assert "--standalone" not in command
    assert "--nnodes=4" in command
    assert "--node_rank=2" in command
    assert "--rdzv_endpoint=host:29400" in command
    assert "--rdzv_id=reward-h7" in command


def test_versioned_weight_manifest_matches_files():
    payload = verify_weights()
    assert payload["ok"] is True
    assert payload["checks"]["preprocessing_scaler"]["ok"] is True


def test_reference_environment_matches_current_test_environment():
    payload = verify_environment()

    assert payload["ok"] is True


def test_restore_weights_accepts_already_valid_files():
    payload = restore_weights()
    assert payload["ok"] is True
    assert payload["restored"]["reward_checkpoint"]["status"] in {
        "already_valid",
        "restored",
    }
    assert payload["restored"]["frozen_autoencoder"]["status"] == "already_valid"
    assert payload["restored"]["preprocessing_scaler"]["status"] == "verified"


def test_restore_h2_weights_reuses_the_versioned_autoencoder():
    payload = restore_weights(bundle="reward_h2_event_v1")

    assert payload["ok"] is True
    assert payload["bundle"] == "reward_h2_event_v1"
    assert payload["restored"]["reward_checkpoint"]["status"] in {
        "already_valid",
        "restored",
    }
    assert payload["restored"]["frozen_autoencoder"]["status"] in {
        "already_valid",
        "verified",
    }
    assert payload["restored"]["preprocessing_scaler"]["status"] == "verified"


def test_inference_config_uses_versioned_immutable_scaler():
    config = load_yaml("configs/model/reward_v1_infer.yaml")

    assert config["args"]["checkpoint"].endswith("reward_epoch009_inference.pt")
    assert (
        config["args"]["scaler_path"]
        == "weights/reward_v1/preprocessing/pre2020_eval2020_2026_raw_relative_scaler.json"
    )


def test_h2_inference_config_uses_its_own_scaler_and_epoch18_checkpoint():
    config = load_yaml("configs/model/reward_h2_event_v1_infer.yaml")

    assert config["weight_bundle"] == "reward_h2_event_v1"
    assert config["args"]["rank_horizon"] == 2
    assert config["args"]["checkpoint"].endswith("reward_epoch018_inference.pt")
    assert "h2_event" in config["args"]["scaler_path"]


def test_h2_strategy_is_registered_with_its_default_score_path():
    assert STRATEGY_CONFIGS["reward-h2-event"].name == "reward_h2_event.yaml"
    assert STRATEGY_SCORE_PATHS["reward-h2-event"].endswith("reward_h2_event_v1.parquet")


def test_concentrated_weak_to_strong_strategy_is_registered():
    assert STRATEGY_CONFIGS["weak-to-strong-pos3"].name == "weak_to_strong_pos3.yaml"


def test_h7_runner_strategy_is_registered():
    assert STRATEGY_CONFIGS["reward-h7-runner"].name == "reward_h7_runner.yaml"
    assert STRATEGY_SCORE_PATHS["reward-h7-runner"].endswith("reward_v1.parquet")


def test_quarterly_dual_sleeve_config_is_current_default():
    config = load_yaml("configs/sleeve/quarterly_wts3_reward5_65_35.yaml")

    assert config["wts_weight"] == 0.65
    assert config["reward_weight"] == 0.35
    assert config["initial_cash"] == 100_000_000
    assert config["schedule"] == "quarterly"


def test_cli_registers_current_score_and_dual_sleeve_commands():
    parser = build_parser()

    assert parser.parse_args([
        "prepare-inference-index",
        "--start",
        "2026-09-24",
        "--end",
        "2026-09-24",
        "--dry-run",
    ]).command == "prepare-inference-index"
    assert parser.parse_args([
        "dual-sleeve",
        "--data-root",
        "data",
        "--run-id",
        "smoke",
        "--dry-run",
    ]).command == "dual-sleeve"
    assert parser.parse_args([
        "bootstrap-data",
        "--data-root",
        "data",
        "--end",
        "2026-09-28",
        "--dry-run",
    ]).command == "bootstrap-data"
    assert parser.parse_args([
        "data-manifest",
        "verify",
        "--data-root",
        "data",
    ]).command == "data-manifest"
    assert parser.parse_args(["environment"]).command == "environment"
    assert parser.parse_args([
        "reproduce-full",
        "--data-root",
        "data",
        "--dry-run",
    ]).command == "reproduce-full"
