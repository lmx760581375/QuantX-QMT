"""Legacy Wufu command delegates to the standard config runner."""

from quantx.tools import run_etf_wufu_backtest


def test_wufu_compat_entrypoint_forwards_arguments(monkeypatch):
    captured = {}

    def fake_standard(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(run_etf_wufu_backtest, "run_standard_backtest", fake_standard)

    result = run_etf_wufu_backtest.main(["--dry-run", "--json"])

    assert result == 0
    assert captured["args"] == [
        "--config",
        run_etf_wufu_backtest.DEFAULT_CONFIG,
        "--dry-run",
        "--json",
    ]
