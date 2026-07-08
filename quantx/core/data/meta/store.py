"""SQLite-backed security metadata store."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


def normalize_symbol(value: str) -> str:
    text = str(value).strip()
    if not text:
        return text
    if "." in text:
        left, right = text.split(".", 1)
        if left.lower() in {"sh", "sz", "bj"}:
            return f"{left.upper()}{right}"
        if right.upper() in {"SH", "SZ", "BJ"}:
            return f"{right.upper()}{left}"
    upper = text.upper()
    if upper.startswith(("SH", "SZ", "BJ")):
        return upper
    if len(upper) == 6 and upper.isdigit():
        if upper.startswith("6"):
            return f"SH{upper}"
        if upper.startswith(("0", "3")):
            return f"SZ{upper}"
        if upper.startswith(("4", "8", "9")):
            return f"BJ{upper}"
    return upper


class MetaStore:
    """Store latest security metadata snapshots in SQLite."""

    def __init__(self, uri: str | Path = "data/meta/quantx_meta.sqlite"):
        self.uri = Path(uri)
        self.uri.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def connect(self):
        return sqlite3.connect(self.uri)

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS security_master (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    exchange TEXT,
                    board TEXT,
                    list_date TEXT,
                    delist_date TEXT,
                    source TEXT,
                    snapshot_date TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS industry_membership (
                    symbol TEXT,
                    industry_source TEXT,
                    level INTEGER,
                    industry_code TEXT,
                    industry_name TEXT,
                    source TEXT,
                    snapshot_date TEXT,
                    updated_at TEXT,
                    PRIMARY KEY (symbol, industry_source, level)
                );

                CREATE TABLE IF NOT EXISTS sector_membership (
                    symbol TEXT,
                    sector_type TEXT,
                    sector_code TEXT,
                    sector_name TEXT,
                    weight REAL,
                    source TEXT,
                    snapshot_date TEXT,
                    updated_at TEXT,
                    PRIMARY KEY (symbol, sector_type, sector_code)
                );

                CREATE TABLE IF NOT EXISTS symbol_map (
                    symbol TEXT PRIMARY KEY,
                    baostock_code TEXT,
                    akshare_code TEXT,
                    tushare_code TEXT,
                    qlib_code TEXT
                );
                """
            )

    def upsert_security_master(self, frame: pd.DataFrame, source: str = "manual", snapshot_date: str | None = None) -> int:
        if frame is None or frame.empty:
            return 0
        rows = _security_rows(frame, source=source, snapshot_date=snapshot_date)
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO security_master(symbol, name, exchange, board, list_date, delist_date, source, snapshot_date, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    name=excluded.name,
                    exchange=excluded.exchange,
                    board=excluded.board,
                    list_date=excluded.list_date,
                    delist_date=excluded.delist_date,
                    source=excluded.source,
                    snapshot_date=excluded.snapshot_date,
                    updated_at=excluded.updated_at
                """,
                rows,
            )
        return len(rows)

    def upsert_industry_membership(self, frame: pd.DataFrame, source: str = "manual", snapshot_date: str | None = None) -> int:
        if frame is None or frame.empty:
            return 0
        rows = _industry_rows(frame, source=source, snapshot_date=snapshot_date)
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO industry_membership(symbol, industry_source, level, industry_code, industry_name, source, snapshot_date, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, industry_source, level) DO UPDATE SET
                    industry_code=excluded.industry_code,
                    industry_name=excluded.industry_name,
                    source=excluded.source,
                    snapshot_date=excluded.snapshot_date,
                    updated_at=excluded.updated_at
                """,
                rows,
            )
        return len(rows)

    def upsert_sector_membership(self, frame: pd.DataFrame, source: str = "manual", snapshot_date: str | None = None) -> int:
        if frame is None or frame.empty:
            return 0
        rows = _sector_rows(frame, source=source, snapshot_date=snapshot_date)
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO sector_membership(symbol, sector_type, sector_code, sector_name, weight, source, snapshot_date, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, sector_type, sector_code) DO UPDATE SET
                    sector_name=excluded.sector_name,
                    weight=excluded.weight,
                    source=excluded.source,
                    snapshot_date=excluded.snapshot_date,
                    updated_at=excluded.updated_at
                """,
                rows,
            )
        return len(rows)

    def get_symbol_meta(self, symbols: Iterable[str] | None = None, industry_source: str = "eastmoney", level: int = 1) -> pd.DataFrame:
        symbols_list = [normalize_symbol(sym) for sym in symbols] if symbols is not None else []
        if symbols_list:
            values = ",".join("(?)" for _ in symbols_list)
            symbol_source = f"VALUES {values}"
            params: list[object] = [*symbols_list, industry_source, int(level)]
        else:
            symbol_source = """
                SELECT symbol FROM security_master
                UNION
                SELECT symbol FROM industry_membership
                WHERE industry_source = ? AND level = ?
            """
            params = [industry_source, int(level), industry_source, int(level)]
        query = f"""
            WITH symbols(symbol) AS (
                {symbol_source}
            )
            SELECT
                symbols.symbol,
                sm.name,
                sm.exchange,
                sm.board,
                sm.list_date,
                sm.delist_date,
                im.industry_source,
                im.level AS industry_level,
                im.industry_code,
                im.industry_name,
                (
                    SELECT GROUP_CONCAT(sector_code, '|')
                    FROM sector_membership smem
                    WHERE smem.symbol = symbols.symbol
                      AND smem.sector_type = 'concept'
                ) AS concept_codes,
                (
                    SELECT GROUP_CONCAT(sector_name, '|')
                    FROM sector_membership smem
                    WHERE smem.symbol = symbols.symbol
                      AND smem.sector_type = 'concept'
                ) AS concept_names
            FROM symbols
            LEFT JOIN security_master sm
              ON symbols.symbol = sm.symbol
            LEFT JOIN industry_membership im
              ON symbols.symbol = im.symbol
             AND im.industry_source = ?
             AND im.level = ?
            ORDER BY symbols.symbol
        """
        with self.connect() as conn:
            frame = pd.read_sql_query(query, conn, params=params)
        if not frame.empty:
            frame = frame.set_index("symbol", drop=False)
        return frame

    def get_industries(self, industry_source: str = "eastmoney", level: int = 1) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(
                """
                SELECT industry_source, level, industry_code, industry_name, COUNT(*) AS symbol_count
                FROM industry_membership
                WHERE industry_source = ? AND level = ?
                GROUP BY industry_source, level, industry_code, industry_name
                ORDER BY symbol_count DESC, industry_name
                """,
                conn,
                params=[industry_source, int(level)],
            )

    def get_sectors(self, sector_type: str = "concept") -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(
                """
                SELECT sector_type, sector_code, sector_name, COUNT(*) AS symbol_count
                FROM sector_membership
                WHERE sector_type = ?
                GROUP BY sector_type, sector_code, sector_name
                ORDER BY symbol_count DESC, sector_name
                """,
                conn,
                params=[sector_type],
            )


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _snapshot(snapshot_date: str | None) -> str:
    return snapshot_date or datetime.now().strftime("%Y-%m-%d")


def _row_snapshot(row, snapshot_date: str | None) -> str:
    return str(_col(row, "snapshot_date", "快照日期", default=None) or _snapshot(snapshot_date))


def _col(row, *names, default=None):
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


def _security_rows(frame: pd.DataFrame, source: str, snapshot_date: str | None):
    updated = _now()
    rows = []
    for _, row in frame.iterrows():
        snap = _row_snapshot(row, snapshot_date)
        symbol = normalize_symbol(_col(row, "symbol", "code", "股票代码", "代码"))
        if not symbol:
            continue
        exchange = _col(row, "exchange", default=symbol[:2])
        rows.append((
            symbol,
            _col(row, "name", "code_name", "股票简称", "名称", default=symbol),
            exchange,
            _col(row, "board", "市场", default=_infer_board(symbol)),
            _col(row, "list_date", "ipoDate", "上市日期"),
            _col(row, "delist_date", "outDate", "退市日期"),
            source,
            snap,
            updated,
        ))
    return rows


def _industry_rows(frame: pd.DataFrame, source: str, snapshot_date: str | None):
    updated = _now()
    rows = []
    for _, row in frame.iterrows():
        snap = _row_snapshot(row, snapshot_date)
        symbol = normalize_symbol(_col(row, "symbol", "code", "股票代码", "代码"))
        industry_name = _col(row, "industry_name", "industry", "行业", "板块名称")
        if not symbol or not industry_name:
            continue
        rows.append((
            symbol,
            str(_col(row, "industry_source", default="eastmoney")),
            int(_col(row, "level", default=1)),
            _col(row, "industry_code", "板块代码", default=str(industry_name)),
            industry_name,
            source,
            snap,
            updated,
        ))
    return rows


def _sector_rows(frame: pd.DataFrame, source: str, snapshot_date: str | None):
    updated = _now()
    rows = []
    for _, row in frame.iterrows():
        snap = _row_snapshot(row, snapshot_date)
        symbol = normalize_symbol(_col(row, "symbol", "code", "股票代码", "代码"))
        sector_name = _col(row, "sector_name", "concept", "板块名称", "概念名称")
        if not symbol or not sector_name:
            continue
        rows.append((
            symbol,
            str(_col(row, "sector_type", default="concept")),
            str(_col(row, "sector_code", "板块代码", default=sector_name)),
            sector_name,
            _col(row, "weight", default=None),
            source,
            snap,
            updated,
        ))
    return rows


def _infer_board(symbol: str) -> str:
    code = symbol[2:]
    if symbol.startswith("BJ") or code.startswith(("4", "8", "9")):
        return "beijing"
    if code.startswith("688"):
        return "star"
    if code.startswith("300"):
        return "chi_next"
    return "mainboard"
