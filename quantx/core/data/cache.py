"""数据缓存（可选模块）

提供 MultiIndex DataFrame 的 pickle 缓存，加速数据加载。
使用 Qlib 二进制格式后加载已很快，本缓存层作为可选加速。
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataCache:
    """MultiIndex DataFrame pickle 缓存

    缓存键: MD5 哈希 {start}_{end}_{adjust}_{sorted_stocks}
    旁路文件: .meta.json 记录缓存参数
    """

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(
        self, start: str, end: str, adjust: str, symbols: list
    ) -> str:
        """生成缓存键"""
        raw = f"{start}_{end}_{adjust}_{','.join(sorted(symbols))}"
        return hashlib.md5(raw.encode()).hexdigest()

    def save(
        self, df: pd.DataFrame, start: str, end: str, adjust: str, symbols: list
    ) -> None:
        """缓存 DataFrame"""
        key = self._cache_key(start, end, adjust, symbols)
        pkl_path = self.cache_dir / f"{key}.pkl"
        meta_path = self.cache_dir / f"{key}.meta.json"

        df.to_pickle(pkl_path)

        meta = {
            "start": start,
            "end": end,
            "adjust": adjust,
            "n_symbols": len(symbols),
            "shape": list(df.shape),
            "created_at": datetime.now().isoformat(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Cache saved: {pkl_path}")

    def load(
        self, start: str, end: str, adjust: str, symbols: list
    ) -> Optional[pd.DataFrame]:
        """读取缓存"""
        key = self._cache_key(start, end, adjust, symbols)
        pkl_path = self.cache_dir / f"{key}.pkl"

        if not pkl_path.exists():
            return None

        logger.info(f"Cache hit: {pkl_path}")
        return pd.read_pickle(pkl_path)

    def exists(
        self, start: str, end: str, adjust: str, symbols: list
    ) -> bool:
        """检查缓存是否存在"""
        key = self._cache_key(start, end, adjust, symbols)
        return (self.cache_dir / f"{key}.pkl").exists()

    def invalidate(
        self, start: str, end: str, adjust: str, symbols: list
    ) -> None:
        """失效缓存"""
        key = self._cache_key(start, end, adjust, symbols)
        pkl_path = self.cache_dir / f"{key}.pkl"
        meta_path = self.cache_dir / f"{key}.meta.json"

        if pkl_path.exists():
            pkl_path.unlink()
        if meta_path.exists():
            meta_path.unlink()

        logger.info(f"Cache invalidated: {key}")

    def clear(self) -> int:
        """清理所有缓存，返回清理的文件数"""
        count = 0
        for pkl_file in self.cache_dir.glob("*.pkl"):
            pkl_file.unlink()
            count += 1
        for meta_file in self.cache_dir.glob("*.meta.json"):
            meta_file.unlink()
            count += 1
        logger.info(f"Cache cleared: {count} files")
        return count