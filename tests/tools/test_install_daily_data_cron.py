"""Cron installer tests."""

from quantx.tools.install_daily_data_cron import BEGIN, END, cron_block, install_block


def test_cron_block_uses_1730_daily_data_stage():
    block = cron_block(
        root="/repo/quantx",
        profile="configs/production/daily_default.yaml",
        python_bin="/env/bin/python",
        log_file="/tmp/quantx_daily_data.log",
    )

    assert block.startswith(BEGIN)
    assert block.endswith(END)
    assert "30 17 * * *" in block
    assert "--stage data" in block
    assert "quantx.tools.run_daily_pipeline" in block


def test_cron_block_can_use_weekdays_only():
    block = cron_block(
        root="/repo/quantx",
        profile="configs/production/daily_default.yaml",
        python_bin="/env/bin/python",
        log_file="/tmp/quantx_daily_data.log",
        days="1-5",
    )

    assert "30 17 * * 1-5" in block


def test_cron_block_can_schedule_2200_full_pipeline():
    block = cron_block(
        root="/repo/quantx",
        profile="configs/production/daily_default.yaml",
        python_bin="/env/bin/python",
        log_file="/tmp/quantx_daily_full.log",
        hour=22,
        minute=0,
        stage="all",
        extra_args=["--skip-data-update"],
        label="DAILY FULL RUN",
    )

    assert block.startswith("# BEGIN QUANTX DAILY FULL RUN")
    assert block.endswith("# END QUANTX DAILY FULL RUN")
    assert "0 22 * * *" in block
    assert "--stage all" in block
    assert "--skip-data-update" in block
    assert "/tmp/quantx_daily_full.log" in block


def test_install_block_replaces_existing_marked_block(monkeypatch):
    calls = {}

    def fake_current():
        return "0 1 * * * old\n# BEGIN QUANTX DAILY DATA UPDATE\nbad\n# END QUANTX DAILY DATA UPDATE\n"

    def fake_run(cmd, input=None, text=None, check=None, **kwargs):
        calls["cmd"] = cmd
        calls["input"] = input

    monkeypatch.setattr("quantx.tools.install_daily_data_cron._current_crontab", fake_current)
    monkeypatch.setattr("subprocess.run", fake_run)

    new_text = install_block("new block")

    assert calls["cmd"] == ["crontab", "-"]
    assert "old" in new_text
    assert "bad" not in new_text
    assert "new block" in new_text


def test_install_block_replaces_only_matching_marked_block(monkeypatch):
    calls = {}

    def fake_current():
        return (
            "# BEGIN QUANTX DAILY DATA UPDATE\n"
            "data\n"
            "# END QUANTX DAILY DATA UPDATE\n"
            "# BEGIN QUANTX DAILY FULL RUN\n"
            "old-full\n"
            "# END QUANTX DAILY FULL RUN\n"
        )

    def fake_run(cmd, input=None, text=None, check=None, **kwargs):
        calls["input"] = input

    monkeypatch.setattr("quantx.tools.install_daily_data_cron._current_crontab", fake_current)
    monkeypatch.setattr("subprocess.run", fake_run)

    install_block("# BEGIN QUANTX DAILY FULL RUN\nnew-full\n# END QUANTX DAILY FULL RUN")

    assert "data" in calls["input"]
    assert "old-full" not in calls["input"]
    assert "new-full" in calls["input"]
