"""BaoStock CSV → Qlib 二进制格式转换器

默认使用前复权价格（adjustflag="2"），$factor = 1.0。
保留 query_adjust_factor 相关代码，切换不复权模式时可用。
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from .baostock_client import BaoStockClient

logger = logging.getLogger(__name__)


class BaostockToQlibConverter:
    """BaoStock CSV → Qlib 二进制格式转换器

    默认使用前复权价格，$factor = 1.0。
    保留 _get_adjust_factors / _map_factors_to_dates 方法，
    切换到不复权模式（adjustflag="3"）时可直接使用。
    """

    CALENDARS_DIR = "calendars"
    FEATURES_DIR = "features"
    INSTRUMENTS_DIR = "instruments"
    INSTRUMENTS_FILE = "all.txt"
    DUMP_SUFFIX = ".bin"

    def __init__(
        self,
        qlib_dir: str = "data/qlib_data",
        csv_dir: str = "data/raw/baostock/stocks",
        freq: str = "day",
        max_workers: int = 16,
    ):
        self.qlib_dir = Path(qlib_dir)
        self.csv_dir = Path(csv_dir)
        self.freq = freq
        self.max_workers = max_workers
        # 缓存复权因子，避免重复查询 BaoStock
        self._factor_cache: Dict[str, pd.DataFrame] = {}

    def _get_adjust_factors(self, symbol: str) -> pd.DataFrame:
        """获取某只股票的复权因子（带缓存）"""
        if symbol not in self._factor_cache:
            try:
                with BaoStockClient() as client:
                    baostock_sym = BaoStockClient.to_baostock_format(symbol)
                    df = client.query_adjust_factor(baostock_sym)
                    if not df.empty:
                        self._factor_cache[symbol] = df
                    else:
                        self._factor_cache[symbol] = pd.DataFrame()
            except Exception as e:
                logger.warning(f"Failed to get adjust factors for {symbol}: {e}")
                self._factor_cache[symbol] = pd.DataFrame()
        return self._factor_cache[symbol]

    # ============================================================
    # 全量转换
    # ============================================================

    def convert_all(
        self, start: str, end: str, symbols: Optional[List[str]] = None
    ) -> None:
        """全量转换所有股票数据

        Args:
            start: 起始日期
            end: 结束日期
            symbols: 股票列表，None 表示转换所有已存储的股票
        """
        logger.info(f"Starting full conversion: {start} ~ {end}")

        # 1. 收集 CSV 文件
        if symbols is None:
            csv_files = list(self.csv_dir.glob("*.csv"))
        else:
            csv_files = [self.csv_dir / f"{sym}.csv" for sym in symbols]

        csv_files = [f for f in csv_files if f.exists()]
        if not csv_files:
            logger.warning("No CSV files found for conversion")
            return

        # 2. 加载所有数据，收集交易日历
        all_data = {}
        all_dates: Set[pd.Timestamp] = set()
        instruments = []

        for csv_file in csv_files:
            symbol = csv_file.stem  # 如 SH600519
            df = self._load_and_process(csv_file, symbol, start, end)
            if df.empty:
                continue
            all_data[symbol] = df
            dates = pd.to_datetime(df["date"])
            all_dates.update(dates)
            instruments.append({
                "symbol": symbol,
                "start": dates.min().strftime("%Y-%m-%d"),
                "end": dates.max().strftime("%Y-%m-%d"),
            })

        if not all_data:
            logger.warning("No data to convert")
            return

        # 3. 排序日历
        calendar_list = sorted(all_dates)
        calendar_strs = [d.strftime("%Y-%m-%d") for d in calendar_list]

        # 4. 写入日历
        self._write_calendars(calendar_strs)

        # 5. 写入 instruments
        self._write_instruments(instruments)

        # 6. 写入 features（每只股票每个字段一个 .bin 文件）
        self._write_features(all_data, calendar_list)

        logger.info(f"Conversion complete: {len(all_data)} stocks, {len(calendar_list)} days")

    # ============================================================
    # 增量转换
    # ============================================================

    def convert_incremental(self, symbols: List[str]) -> None:
        """增量转换：为新数据追加 bin 文件，已有股票重写全量以更新前复权价格"""
        logger.info(f"Starting incremental conversion: {len(symbols)} symbols")

        calendar_list = self._read_calendars()
        if not calendar_list:
            logger.warning("No existing calendar, run convert_all first")
            return

        all_data = {}
        new_dates: Set[pd.Timestamp] = set()

        for sym in symbols:
            csv_file = self.csv_dir / f"{sym}.csv"
            if not csv_file.exists():
                continue
            df = self._load_and_process(csv_file, sym, None, None)
            if df.empty:
                continue
            all_data[sym] = df
            dates = pd.to_datetime(df["date"])
            new_dates.update(dates)

        if not all_data:
            return

        # 合并新日期到日历
        existing_set = set(calendar_list)
        added_dates = sorted(new_dates - existing_set)
        if added_dates:
            calendar_list = sorted(set(calendar_list) | new_dates)
            self._write_calendars([d.strftime("%Y-%m-%d") for d in calendar_list])

        # 前复权价格可能整体变化，重写完整 bin 文件
        self._write_features(all_data, calendar_list, mode="overwrite")
        self._merge_instruments(all_data)

        logger.info(f"Incremental conversion complete: {len(all_data)} stocks")

    # ============================================================
    # 覆盖范围检查
    # ============================================================

    def check_coverage(self, start: str, end: str) -> bool:
        """检查 Qlib 数据是否覆盖所需日期范围"""
        calendar_path = self.qlib_dir / self.CALENDARS_DIR / f"{self.freq}.txt"
        if not calendar_path.exists():
            return False

        dates = pd.read_csv(calendar_path, header=None, names=["date"])
        dates["date"] = pd.to_datetime(dates["date"])
        min_date = dates["date"].min()
        max_date = dates["date"].max()

        return min_date <= pd.Timestamp(start) and max_date >= pd.Timestamp(end)

    # ============================================================
    # 内部方法：数据加载
    # ============================================================

    def _load_and_process(
        self, csv_file: Path, symbol: str, start: Optional[str], end: Optional[str]
    ) -> pd.DataFrame:
        """加载 CSV 并处理为 Qlib 格式"""
        try:
            df = pd.read_csv(csv_file, dtype={"code": str})
        except Exception as e:
            logger.warning(f"Failed to read {csv_file}: {e}")
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        # 确保必需列存在
        required_cols = ["date", "open", "high", "low", "close", "preclose", "volume", "amount"]
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Missing column {col} in {csv_file}")
                return pd.DataFrame()

        # 日期过滤
        df["date"] = pd.to_datetime(df["date"])
        if start:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end:
            df = df[df["date"] <= pd.Timestamp(end)]

        if df.empty:
            return pd.DataFrame()

        # 转换为 Qlib 格式
        result = pd.DataFrame()
        result["date"] = df["date"].dt.strftime("%Y-%m-%d")
        result["symbol"] = symbol

        # 字段映射和数值转换
        for field in ["open", "high", "low", "close", "volume"]:
            result[f"${field}"] = pd.to_numeric(df[field], errors="coerce")

        amount = pd.to_numeric(df["amount"], errors="coerce")
        volume = pd.to_numeric(df["volume"], errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")
        result["$vwap"] = np.where(volume > 0, amount / (volume * 100), close)

        # $factor: 前复权模式下为 1.0
        # 不复权模式时调用 _get_adjust_factors 获取复权因子
        result["$factor"] = 1.0

        preclose = pd.to_numeric(df["preclose"], errors="coerce")
        result["$change"] = np.where(preclose > 0, (close - preclose) / preclose, 0.0)

        return result

    def _map_factors_to_dates(
        self, dates: pd.Series, factors_df: pd.DataFrame, symbol: str
    ) -> np.ndarray:
        """将复权因子映射到每个日期

        foreAdjustFactor 是阶梯函数，只在除权日变化。
        前向填充：每个日期使用最近一次除权日的因子值。
        """
        n = len(dates)
        if factors_df.empty:
            logger.warning(f"No adjust factors for {symbol}, using 1.0")
            return np.ones(n, dtype=np.float32)

        factor_dates = pd.to_datetime(factors_df["dividOperateDate"])
        factor_values = factors_df["foreAdjustFactor"].values.astype(np.float32)
        sort_idx = np.argsort(factor_dates)
        factor_dates = factor_dates.iloc[sort_idx]
        factor_values = factor_values[sort_idx]

        result = np.ones(n, dtype=np.float32)
        dates_pd = pd.to_datetime(dates)
        for i, d in enumerate(dates_pd):
            mask = factor_dates <= d
            if mask.any():
                result[i] = factor_values[mask][-1]
        return result

    # ============================================================
    # 内部方法：日历和 instruments
    # ============================================================

    def _write_calendars(self, dates: List[str]) -> None:
        """写入交易日历文件"""
        cal_dir = self.qlib_dir / self.CALENDARS_DIR
        cal_dir.mkdir(parents=True, exist_ok=True)
        cal_path = cal_dir / f"{self.freq}.txt"
        with open(cal_path, "w") as f:
            f.write("\n".join(dates))
        logger.info(f"Calendars written: {cal_path} ({len(dates)} days)")

    def _read_calendars(self) -> List[pd.Timestamp]:
        """读取交易日历"""
        cal_path = self.qlib_dir / self.CALENDARS_DIR / f"{self.freq}.txt"
        if not cal_path.exists():
            return []
        dates = pd.read_csv(cal_path, header=None, names=["date"])
        return sorted(pd.to_datetime(dates["date"]))

    def _write_instruments(self, instruments: List[dict]) -> None:
        """写入 instruments 文件"""
        inst_dir = self.qlib_dir / self.INSTRUMENTS_DIR
        inst_dir.mkdir(parents=True, exist_ok=True)
        inst_path = inst_dir / self.INSTRUMENTS_FILE

        with open(inst_path, "w") as f:
            for inst in instruments:
                f.write(f"{inst['symbol']}\t{inst['start']}\t{inst['end']}\n")
        logger.info(f"Instruments written: {inst_path} ({len(instruments)} symbols)")

    def _read_instruments(self) -> Dict[str, dict]:
        inst_path = self.qlib_dir / self.INSTRUMENTS_DIR / self.INSTRUMENTS_FILE
        if not inst_path.exists():
            return {}
        rows: Dict[str, dict] = {}
        for line in inst_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            rows[parts[0]] = {"symbol": parts[0], "start": parts[1], "end": parts[2]}
        return rows

    def _merge_instruments(self, all_data: Dict[str, pd.DataFrame]) -> None:
        instruments = self._read_instruments()
        for symbol, df in all_data.items():
            if df.empty:
                continue
            dates = pd.to_datetime(df["date"])
            instruments[symbol] = {
                "symbol": symbol,
                "start": dates.min().strftime("%Y-%m-%d"),
                "end": dates.max().strftime("%Y-%m-%d"),
            }
        self._write_instruments([instruments[symbol] for symbol in sorted(instruments)])

    # ============================================================
    # 内部方法：features 二进制写入
    # ============================================================

    def _write_features(
        self,
        all_data: dict,
        calendar_list: List[pd.Timestamp],
        mode: str = "overwrite",
    ) -> None:
        """写入 features 二进制文件

        mode='overwrite': 重写整个 bin 文件（全量转换）
        mode='append': 追加新数据（增量转换，不复权价格不变）
        """
        features_dir = self.qlib_dir / self.FEATURES_DIR
        features_dir.mkdir(parents=True, exist_ok=True)

        date_to_idx = {d: i for i, d in enumerate(calendar_list)}
        fields = ["$open", "$high", "$low", "$close", "$volume", "$vwap", "$factor", "$change"]

        for symbol, df in all_data.items():
            if df.empty:
                continue

            sym_dir = features_dir / symbol.lower()
            sym_dir.mkdir(parents=True, exist_ok=True)
            df_dates = pd.to_datetime(df["date"])

            for field in fields:
                if field not in df.columns:
                    continue

                bin_path = sym_dir / f"{field.lower().lstrip('$')}.{self.freq}{self.DUMP_SUFFIX}"
                values = df[field].values.astype(np.float32)
                if np.isnan(values).all():
                    continue

                date_indices = np.array([date_to_idx.get(d, -1) for d in df_dates], dtype=np.int64)
                valid = date_indices >= 0
                if not valid.any():
                    continue

                sorted_order = np.argsort(date_indices[valid])
                sorted_indices = date_indices[valid][sorted_order]
                sorted_values = values[valid][sorted_order]

                start_idx = int(sorted_indices[0])
                end_idx = int(sorted_indices[-1])
                aligned = np.full(end_idx - start_idx + 1, np.nan, dtype=np.float32)
                aligned[sorted_indices - start_idx] = sorted_values.astype(np.float32)

                if mode == "append" and bin_path.exists():
                    # Qlib bin 格式为 [start_index, value0, value1, ...]。
                    # 为避免前复权/日历变更造成错位，增量转换也重写该股票字段文件。
                    logger.debug("Rewriting %s in append mode to preserve qlib bin alignment", bin_path)

                np.hstack([np.array([start_idx], dtype=np.float32), aligned]).astype("<f").tofile(str(bin_path))

        logger.info(f"Features written ({mode}): {len(all_data)} stocks, {len(fields)} fields")
