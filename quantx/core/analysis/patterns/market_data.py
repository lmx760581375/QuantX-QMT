"""Daily OHLCV loading for trade pattern windows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from quantx.core.data.repository import LocalDataRepository

logger = logging.getLogger(__name__)


class DailyBarLoader:
    """Load daily bars, preferring Qlib and falling back to local CSV."""

    def __init__(
        self,
        raw_data_dir: str | Path = "data/raw/baostock",
        provider_uri: str | Path = "data/qlib_data_fixed",
        prefer_qlib: bool = True,
    ):
        self.repository = LocalDataRepository(str(raw_data_dir))
        self.provider_uri = str(provider_uri)
        self.prefer_qlib = prefer_qlib
        self._qlib_initialized = False
        self._cache: Dict[str, pd.DataFrame] = {}

    def load_symbol(self, symbol: str) -> pd.DataFrame:
        if symbol in self._cache:
            return self._cache[symbol].copy()
        frame = self._load_symbol_qlib(symbol) if self.prefer_qlib else pd.DataFrame()
        if frame.empty:
            frame = self._load_symbol_csv(symbol)
        self._cache[symbol] = frame
        return frame.copy()

    def preload_symbols(self, symbols: Iterable[str], start: str | None = None, end: str | None = None) -> None:
        """Warm the cache with a single batch Qlib request when possible."""
        missing = sorted(set(symbols) - set(self._cache))
        if not missing:
            return
        if self.prefer_qlib:
            batch = self._load_symbols_qlib(missing, start=start, end=end)
            for symbol, frame in batch.items():
                self._cache[symbol] = frame
        for symbol in missing:
            if symbol not in self._cache:
                self._cache[symbol] = self._load_symbol_csv(symbol)

    def load_symbols(self, symbols: Iterable[str]) -> pd.DataFrame:
        self.preload_symbols(symbols)
        frames: List[pd.DataFrame] = []
        for symbol in sorted(set(symbols)):
            frame = self.load_symbol(symbol)
            if not frame.empty:
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _load_symbol_qlib(self, symbol: str) -> pd.DataFrame:
        return self._load_symbols_qlib([symbol]).get(symbol, pd.DataFrame())

    def _load_symbols_qlib(
        self,
        symbols: List[str],
        start: str | None = None,
        end: str | None = None,
    ) -> Dict[str, pd.DataFrame]:
        try:
            self._init_qlib()
            from qlib.data import D

            fields = ["$open", "$high", "$low", "$close", "$volume", "$vwap", "$change"]
            frame = D.features(symbols, fields, start or "1900-01-01", end or "2100-12-31", freq="day")
        except Exception as exc:
            logger.debug("Qlib daily bar load failed for %s symbols: %s", len(symbols), exc)
            return {}
        if frame.empty:
            return {}
        frame = frame.reset_index()
        return {
            symbol: self._normalize_qlib_frame(symbol, symbol_frame)
            for symbol, symbol_frame in frame.groupby("instrument", sort=False)
        }

    def _normalize_qlib_frame(self, symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "instrument": "symbol",
            "datetime": "date",
            "$open": "open",
            "$high": "high",
            "$low": "low",
            "$close": "close",
            "$volume": "volume",
            "$vwap": "vwap",
            "$change": "return_pct",
        }
        frame = frame.rename(columns=rename_map)
        frame["date"] = pd.to_datetime(frame["date"])
        frame["symbol"] = symbol
        frame["amount"] = frame.get("vwap", frame["close"]) * frame["volume"] * 100
        frame["turnover"] = 0.0
        frame["preclose"] = frame["close"].shift(1)
        frame["pct_chg"] = frame["return_pct"] * 100
        columns = ["date", "symbol", "open", "high", "low", "close", "preclose", "volume", "amount", "turnover", "pct_chg"]
        return frame[columns].sort_values("date").reset_index(drop=True)

    def _load_symbol_csv(self, symbol: str) -> pd.DataFrame:
        frame = self.repository.load_symbol(symbol)
        if frame.empty:
            return pd.DataFrame()
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["symbol"] = symbol
        for column in ["open", "high", "low", "close", "preclose", "volume", "amount", "turnover", "pct_chg"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if "amount" not in frame.columns:
            frame["amount"] = 0.0
        if "turnover" not in frame.columns:
            frame["turnover"] = 0.0
        return frame.sort_values("date").reset_index(drop=True)

    def _init_qlib(self) -> None:
        if self._qlib_initialized:
            return
        import qlib

        qlib.init(provider_uri=self.provider_uri, region="cn")
        self._qlib_initialized = True
