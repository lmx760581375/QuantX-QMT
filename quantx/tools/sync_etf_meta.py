"""Build a local ETF metadata snapshot for ETF rotation research."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_OUTPUT = Path("data/meta/snapshots/etf_master.csv")
DEFAULT_RAW_DIR = Path("data/raw/eastmoney/etfs")

FUND_COMPANIES = [
    "易方达", "广发", "华夏", "华安", "嘉实", "富国", "招商", "鹏华", "南方", "汇添富",
    "国泰", "平安", "银华", "天弘", "建信", "工银", "华泰柏瑞", "博时", "景顺长城", "景顺",
    "华宝", "申万菱信", "万家", "中欧", "永赢", "大成", "摩根", "中信", "中银", "国投",
]

NOISE_WORDS = [
    "ETF基金", "ETF联接", "ETF", "LOF基金", "LOF", "基金", "联接", "指数", "指数基金", "指数ETF",
    "主题", "增强", "龙头", "策略", "场内", "发起式", "A类", "C类", "A", "C",
]

SPECIAL_GROUPS = [
    ("hong_kong", ["恒生", "恒指", "港股", "港股通", "H股", "香港", "中概", "HK", "H股"]),
    ("star_market", ["科创", "科创板", "科创创业", "双创"]),
    ("chinext", ["创业板", "创业", "创成长"]),
    ("us_index", ["标普", "纳指", "纳斯达克", "道琼斯"]),
]

MONEY_KEYWORDS = ["货币", "现金", "快线", "添富快钱", "银华日利", "保证金"]
BOND_KEYWORDS = ["债", "转债", "国债", "城投", "政金", "可转债", "短融"]
CROSS_BORDER_KEYWORDS = [
    "恒生", "恒指", "港股", "港股通", "H股", "香港", "中概", "纳指", "纳斯达克", "标普",
    "道琼斯", "日经", "德国", "法国", "印度", "沙特", "东南亚", "中韩", "海外", "QDII",
]
BROAD_INDEX_KEYWORDS = [
    "沪深300", "中证500", "中证1000", "中证2000", "中证A500", "A500", "上证50", "上证180",
    "深证100", "创业板50", "创业板", "科创50", "科创100", "上证", "深证", "沪深", "中证", "MSCI",
]
STYLE_KEYWORDS = ["红利", "低波", "央企", "国企", "民企", "价值", "成长", "质量", "现金流", "ESG"]
COMMODITY_KEYWORDS = ["黄金", "白银", "豆粕", "原油", "油气", "商品"]


def normalize_fund_symbol(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    lower = text.lower()
    match = re.match(r"^(sh|sz)(\d{6})$", lower)
    if match:
        return f"{match.group(1).upper()}{match.group(2)}"
    if "." in text:
        left, right = text.split(".", 1)
        if left.lower() in {"sh", "sz"}:
            return f"{left.upper()}{right}"
        if right.upper() in {"SH", "SZ"}:
            return f"{right.upper()}{left}"
    if len(text) == 6 and text.isdigit():
        return ("SH" if text.startswith("5") else "SZ") + text
    upper = text.upper()
    return upper if upper.startswith(("SH", "SZ")) else upper


def classify_etf_name(name: str) -> dict[str, Any]:
    name = str(name or "")
    if _has_any(name, MONEY_KEYWORDS):
        category = "money"
    elif _has_any(name, BOND_KEYWORDS):
        category = "bond"
    elif _has_any(name, CROSS_BORDER_KEYWORDS):
        category = "cross_border"
    elif _has_any(name, COMMODITY_KEYWORDS):
        category = "commodity"
    elif _has_any(name, BROAD_INDEX_KEYWORDS) and not _looks_sector_theme(name):
        category = "broad_index"
    elif _has_any(name, STYLE_KEYWORDS) and not _looks_sector_theme(name):
        category = "style"
    else:
        category = "sector_theme"

    special_group = ""
    for group, keywords in SPECIAL_GROUPS:
        if _has_any(name, keywords):
            special_group = group
            break

    group_name = _clean_group_name(name)
    is_dynamic_theme_candidate = category == "sector_theme"
    return {
        "category": category,
        "special_group": special_group,
        "theme_group": group_name,
        "is_dynamic_theme_candidate": is_dynamic_theme_candidate,
    }


def _has_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword and keyword in text for keyword in keywords)


def _looks_sector_theme(name: str) -> bool:
    sector_markers = [
        "半导", "芯片", "通信", "软件", "人工智能", "计算机", "机器人", "新能源", "光伏", "电池",
        "医药", "医疗", "创新药", "证券", "银行", "煤炭", "钢铁", "有色", "化工", "军工", "消费",
        "酒", "食品", "农业", "养殖", "传媒", "游戏", "旅游", "汽车", "电力", "地产", "稀土",
        "金融科技", "大数据", "云计算", "智能驾驶", "教育", "文娱", "航空", "高端装备", "工业母机",
    ]
    return _has_any(name, sector_markers)


def _clean_group_name(name: str) -> str:
    cleaned = str(name or "")
    for word in FUND_COMPANIES:
        cleaned = cleaned.replace(word, "")
    for word in NOISE_WORDS:
        cleaned = cleaned.replace(word, "")
    cleaned = re.sub(r"[\s\-_（）()]+", "", cleaned)
    return cleaned[:8] or str(name or "")[:8]


def fetch_akshare_etf_meta(categories: list[str] | None = None) -> pd.DataFrame:
    import akshare as ak

    categories = categories or ["ETF基金", "LOF基金"]
    frames: list[pd.DataFrame] = []
    for category in categories:
        frame = ak.fund_etf_category_sina(symbol=category)
        if frame is None or frame.empty:
            continue
        frame = frame.copy()
        frame["fund_market_type"] = category
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_etf_master(frame: pd.DataFrame, snapshot_date: str | None = None) -> pd.DataFrame:
    snapshot = snapshot_date or datetime.now().strftime("%Y-%m-%d")
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        symbol = normalize_fund_symbol(row.get("代码") or row.get("symbol"))
        name = str(row.get("名称") or row.get("name") or symbol)
        if not symbol or not symbol.startswith(("SH", "SZ")):
            continue
        cls = classify_etf_name(name)
        rows.append({
            "symbol": symbol,
            "name": name,
            "exchange": symbol[:2],
            "fund_market_type": row.get("fund_market_type") or row.get("fund_type") or "",
            "category": cls["category"],
            "special_group": cls["special_group"],
            "theme_group": cls["theme_group"],
            "is_dynamic_theme_candidate": bool(cls["is_dynamic_theme_candidate"]),
            "latest_price": _num(row.get("最新价")),
            "latest_change_pct": _num(row.get("涨跌幅")),
            "latest_volume": _num(row.get("成交量")),
            "latest_amount": _num(row.get("成交额")),
            "source": "akshare.fund_etf_category_sina",
            "snapshot_date": snapshot,
        })
    result = pd.DataFrame(rows).drop_duplicates("symbol", keep="first")
    return result.sort_values("symbol").reset_index(drop=True) if not result.empty else result


def build_fallback_from_raw(raw_dir: Path, snapshot_date: str | None = None) -> pd.DataFrame:
    snapshot = snapshot_date or datetime.now().strftime("%Y-%m-%d")
    rows = []
    for path in sorted(raw_dir.glob("*.csv")):
        symbol = normalize_fund_symbol(path.stem)
        cls = classify_etf_name(symbol)
        rows.append({
            "symbol": symbol,
            "name": symbol,
            "exchange": symbol[:2],
            "fund_market_type": "local_raw",
            "category": cls["category"],
            "special_group": cls["special_group"],
            "theme_group": cls["theme_group"],
            "is_dynamic_theme_candidate": False,
            "latest_price": None,
            "latest_change_pct": None,
            "latest_volume": None,
            "latest_amount": None,
            "source": "local_raw_fallback",
            "snapshot_date": snapshot,
        })
    return pd.DataFrame(rows)


def _num(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        return out if pd.notna(out) else None
    except Exception:
        return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    raw_dir = Path(args.raw_dir)
    try:
        source_frame = fetch_akshare_etf_meta(args.categories)
        master = build_etf_master(source_frame, snapshot_date=args.date)
        source = "akshare"
    except Exception as exc:
        if not args.fallback_raw:
            raise
        master = build_fallback_from_raw(raw_dir, snapshot_date=args.date)
        source = f"local_raw_fallback: {exc}"
    output.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output, index=False)
    return {
        "ok": True,
        "output": str(output),
        "source": source,
        "rows": int(len(master)),
        "dynamic_theme_candidates": int(master["is_dynamic_theme_candidate"].sum()) if not master.empty else 0,
        "categories": master["category"].value_counts().to_dict() if not master.empty else {},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--date", help="Snapshot date, default today.")
    parser.add_argument("--categories", nargs="+", default=["ETF基金", "LOF基金"])
    parser.add_argument("--fallback-raw", action="store_true", default=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
