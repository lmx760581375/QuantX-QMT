"""Named instrument universes shared by data sync and standard backtests."""

from pathlib import Path

from quantx.strategies.market_regime_rotation_model import DEFAULT_SOURCE_STRATEGY, _all_pool_symbols


def wufu_etf_symbols() -> list[str]:
    symbols = _all_pool_symbols(Path(DEFAULT_SOURCE_STRATEGY))
    return list(dict.fromkeys([*symbols, "SH511880"]))
