"""QMT xtdata adapter backed by a remote xqshare service."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Iterable, List, Literal, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_XQSHARE_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"

QMT_FIELD_RENAME_MAP = {
    "preClose": "preclose",
    "turnoverrate": "turnover",
    "stock_code": "code",
}


class QMTClient:
    """Thin wrapper around ``xtquant.xtdata`` exposed through xqshare."""

    DEFAULT_SECTORS = ["沪深A股"]

    def __init__(
        self,
        dividend_type: str = "front_ratio",
        fill_data: bool = True,
        pause_seconds: float = 0.0,
        max_retries: int = 3,
        env_file: str | Path | None = None,
    ):
        self.dividend_type = dividend_type
        self.fill_data = bool(fill_data)
        self.pause_seconds = float(pause_seconds)
        self.max_retries = int(max_retries)
        configured_env_file = env_file or os.environ.get("QUANTX_ENV_FILE") or DEFAULT_XQSHARE_ENV_FILE
        self.env_file = Path(configured_env_file).expanduser()
        self._remote = None
        self._xtdata = None

    def __enter__(self) -> "QMTClient":
        self._load_xtdata()
        return self

    def __exit__(self, *args) -> Literal[False]:
        if self._remote is not None:
            try:
                self._remote.close()
            finally:
                self._remote = None
                self._xtdata = None
        return False

    @property
    def xtdata(self):
        return self._load_xtdata()

    def _load_xtdata(self):
        if self._xtdata is None:
            try:
                from xqshare import XtQuantRemote
            except ImportError as exc:
                raise ImportError(
                    "QMT data source requires xqshare. Install the project dependencies "
                    "and configure the remote service in .env."
                ) from exc
            self._remote = XtQuantRemote(env_file=str(self.env_file))
            self._xtdata = self._remote.xtdata
        return self._xtdata

    def query_history_k_data_with_retry(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None,
        dividend_type: Optional[str] = None,
        download_first: bool = True,
    ) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(max(1, self.max_retries)):
            try:
                frame = self.query_history_k_data(
                    symbol,
                    start_date,
                    end_date,
                    fields=fields,
                    dividend_type=dividend_type,
                    download_first=download_first,
                )
                if not frame.empty:
                    return frame
            except Exception as exc:  # pragma: no cover - provider errors are environment-specific.
                last_error = exc
                logger.warning("QMT query attempt %s failed for %s: %s", attempt + 1, symbol, exc)
            if attempt < self.max_retries - 1 and self.pause_seconds > 0:
                time.sleep(self.pause_seconds * (attempt + 1))
        if last_error is not None:
            logger.warning("QMT query failed for %s after retries: %s", symbol, last_error)
        return pd.DataFrame()

    def query_history_k_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None,
        dividend_type: Optional[str] = None,
        download_first: bool = True,
    ) -> pd.DataFrame:
        xtdata = self.xtdata
        qmt_symbol = to_qmt_symbol(symbol)
        start = to_qmt_date(start_date)
        end = to_qmt_date(end_date)
        qmt_dividend_type = dividend_type or self.dividend_type

        if download_first:
            self.download_history_data(qmt_symbol, start, end)

        query_started = time.perf_counter()
        field_list = _to_qmt_fields(fields)
        data = xtdata.get_market_data_ex(
            field_list=field_list,
            stock_list=[qmt_symbol],
            period="1d",
            start_time=start,
            end_time=end,
            count=-1,
            dividend_type=qmt_dividend_type,
            fill_data=self.fill_data,
        )
        raw = data.get(qmt_symbol) if isinstance(data, dict) else None
        if raw is None or raw.empty:
            logger.warning(
                "QMT query returned empty symbol=%s start=%s end=%s elapsed=%.2fs",
                qmt_symbol,
                start,
                end,
                time.perf_counter() - query_started,
            )
            return pd.DataFrame()
        frame = normalize_qmt_bars(raw, qmt_symbol)
        logger.info(
            "QMT query done symbol=%s rows=%s elapsed=%.2fs",
            qmt_symbol,
            len(frame),
            time.perf_counter() - query_started,
        )
        return frame

    def download_history_data(self, symbol: str, start_date: str, end_date: str, period: str = "1d") -> bool:
        xtdata = self.xtdata
        qmt_symbol = to_qmt_symbol(symbol)
        start = to_qmt_date(start_date)
        end = to_qmt_date(end_date)
        download_started = time.perf_counter()
        logger.info("QMT download start symbol=%s period=%s start=%s end=%s", qmt_symbol, period, start, end)
        try:
            xtdata.download_history_data(qmt_symbol, period=period, start_time=start, end_time=end)
        except Exception:
            logger.exception("QMT download failed symbol=%s period=%s start=%s end=%s", qmt_symbol, period, start, end)
            raise
        logger.info(
            "QMT download done symbol=%s period=%s elapsed=%.2fs",
            qmt_symbol,
            period,
            time.perf_counter() - download_started,
        )
        return True

    def get_stock_codes(self, sectors: Optional[Iterable[str]] = None) -> List[str]:
        xtdata = self.xtdata
        symbols: list[str] = []
        for sector in sectors or self.DEFAULT_SECTORS:
            try:
                symbols.extend(xtdata.get_stock_list_in_sector(sector))
            except Exception as exc:  # pragma: no cover - provider catalog varies by install.
                logger.warning("Failed to load QMT sector %s: %s", sector, exc)
        return sorted({normalize_symbol(symbol) for symbol in symbols if _is_supported_market(symbol)})

    def query_all_stocks(self, sectors: Optional[Iterable[str]] = None) -> pd.DataFrame:
        rows = [{"code": symbol} for symbol in self.get_stock_codes(sectors)]
        return pd.DataFrame(rows, columns=["code"])

    def query_dividend_data(
        self,
        symbol: str,
        start_date: str = "2010-01-01",
        end_date: str = "2030-12-31",
    ) -> pd.DataFrame:
        xtdata = self.xtdata
        qmt_symbol = to_qmt_symbol(symbol)
        frame = xtdata.get_divid_factors(qmt_symbol, start_time=to_qmt_date(start_date), end_time=to_qmt_date(end_date))
        if frame is None:
            return pd.DataFrame()
        out = pd.DataFrame(frame).copy()
        if out.empty:
            return out
        out["symbol"] = normalize_symbol(qmt_symbol)
        out["qmt_symbol"] = qmt_symbol
        return out

    def get_trading_dates(self, start_date: str, end_date: str, market: str = "SH") -> List[str]:
        dates = self.xtdata.get_trading_dates(market, start_time=to_qmt_date(start_date), end_time=to_qmt_date(end_date))
        return [to_standard_date(date) for date in dates]


def normalize_qmt_bars(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = pd.DataFrame(frame).copy()
    if out.empty:
        return pd.DataFrame()

    out = out.rename(columns=QMT_FIELD_RENAME_MAP)
    out["date"] = _extract_dates(out)
    out["code"] = normalize_symbol(symbol)

    for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "turnover"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "preclose" not in out.columns:
        out["preclose"] = out["close"].shift(1).fillna(out["close"])
    if "amount" not in out.columns:
        out["amount"] = out["close"] * out["volume"] * 100
    if "turnover" not in out.columns:
        out["turnover"] = 0.0
    if "pct_chg" not in out.columns:
        preclose = pd.to_numeric(out["preclose"], errors="coerce")
        close = pd.to_numeric(out["close"], errors="coerce")
        out["pct_chg"] = ((close - preclose) / preclose * 100.0).where(preclose > 0, 0.0)
    if "is_st" not in out.columns:
        out["is_st"] = 0

    columns = [
        "date",
        "code",
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "turnover",
        "pct_chg",
        "is_st",
    ]
    return out[columns].dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)


def normalize_symbol(symbol: str) -> str:
    text = str(symbol).strip()
    if not text:
        return text
    upper = text.upper()
    if "." in upper:
        left, right = upper.split(".", 1)
        if left in {"SH", "SZ", "BJ"}:
            return f"{left}{right}"
        return f"{right}{left}"
    if len(upper) >= 8 and upper[:2] in {"SH", "SZ", "BJ"}:
        return upper
    if len(upper) >= 8 and upper[-2:] in {"SH", "SZ", "BJ"}:
        return f"{upper[-2:]}{upper[:-2]}"
    if upper.startswith("6"):
        return f"SH{upper}"
    if upper.startswith(("0", "3")):
        return f"SZ{upper}"
    if upper.startswith(("4", "8", "9")):
        return f"BJ{upper}"
    return upper


def to_qmt_symbol(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    market = normalized[:2]
    code = normalized[2:]
    if market not in {"SH", "SZ", "BJ"} or not code:
        return str(symbol).upper()
    return f"{code}.{market}"


def to_qmt_date(date: str | pd.Timestamp | None) -> str:
    if date in {None, ""}:
        return ""
    return pd.Timestamp(date).strftime("%Y%m%d")


def to_standard_date(value) -> str:
    text = str(value)
    if text.isdigit() and len(text) == 8:
        return pd.to_datetime(text, format="%Y%m%d").strftime("%Y-%m-%d")
    if text.isdigit() and len(text) >= 12:
        return pd.to_datetime(int(text), unit="ms").strftime("%Y-%m-%d")
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _extract_dates(frame: pd.DataFrame) -> pd.Series:
    if "date" in frame.columns:
        return pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    index_text = pd.Series(frame.index, index=frame.index).astype(str)
    if index_text.str.match(r"^\d{8}$").all():
        return pd.to_datetime(index_text, format="%Y%m%d").dt.strftime("%Y-%m-%d")
    if "time" in frame.columns:
        time_values = pd.to_numeric(frame["time"], errors="coerce")
        if bool((time_values.dropna() > 10**11).all()):
            return pd.to_datetime(time_values, unit="ms").dt.strftime("%Y-%m-%d")
        return pd.to_datetime(time_values.astype("Int64").astype(str), format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    return pd.to_datetime(frame.index).strftime("%Y-%m-%d")


def _to_qmt_fields(fields: Optional[List[str]]) -> List[str]:
    if not fields:
        return []
    mapped = []
    for field in fields:
        clean = str(field).lstrip("$")
        mapped.append("preClose" if clean == "preclose" else clean)
    return mapped


def _is_supported_market(symbol: str) -> bool:
    normalized = normalize_symbol(symbol)
    return normalized.startswith(("SH", "SZ", "BJ")) and len(normalized) >= 8
