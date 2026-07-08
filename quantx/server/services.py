"""Services for the local QuantX web workspace."""

from __future__ import annotations

import json
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from quantx.core.analysis import annual_returns, build_next_session_guide, load_run_artifacts
from quantx.core.analysis.patterns import find_similar_samples
from quantx.core.analysis.patterns.config import LabelConfig, PatternAnalysisConfig
from quantx.core.analysis.patterns.dataset import build_pattern_analysis
from quantx.core.analysis.patterns.io import read_json, read_table
from quantx.core.data.meta import MetaStore
from quantx.core.engine.exchange import AStockExchange
from quantx.core.strategy.config_strategy import build_formula_strategy, explain_strategy_config
from quantx.tools.run_backtest import build_cost, load_config, run_config_with_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "strategies"
PRODUCTION_PROFILE = PROJECT_ROOT / "configs" / "production" / "daily_default.yaml"
RUNS_ROOT = PROJECT_ROOT / "runs"
PATTERN_ANALYSIS_ROOT = PROJECT_ROOT / "artifacts" / "pattern_analysis"
META_URI = PROJECT_ROOT / "data" / "meta" / "quantx_meta.sqlite"


def _relative_to_root(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _safe_child(root: Path, rel_path: str) -> Path:
    path = (root / rel_path).resolve()
    if root.resolve() not in path.parents and path != root.resolve():
        raise ValueError(f"Path escapes root: {rel_path}")
    return path


def _candidate_symbols(candidates: Dict[str, Any]) -> List[str]:
    return [
        str(row.get("symbol"))
        for key in ("raw_candidates", "selected_candidates")
        for row in candidates.get(key) or []
        if row.get("symbol")
    ]


def _meta_fields(symbol: Any, meta: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    row = meta.get(str(symbol), {})
    return {
        "name": row.get("name") or symbol,
        "industry_name": row.get("industry_name") or "",
        "industry_code": row.get("industry_code") or "",
    }


def _clip_title(value: Any, limit: int = 10) -> str:
    text = str(value or "").strip()
    return text[:limit]


def _infer_display_title(
    summary: Dict[str, Any] | None,
    run_id: str = "",
    config: Dict[str, Any] | None = None,
) -> str:
    summary = summary if isinstance(summary, dict) else {}
    config = config if isinstance(config, dict) else {}
    for source in (summary, config):
        for key in ("display_title", "title"):
            title = _clip_title(source.get(key))
            if title:
                return title

    text = " ".join(
        str(part or "")
        for part in (
            summary.get("name"),
            run_id,
            summary.get("description"),
            config.get("name"),
            config.get("description"),
        )
    ).lower()

    if "shuijiao" in text or "sleep" in text:
        return "睡觉策略"
    if "supertrend" in text:
        return "超趋势"
    if "dynamic_add" in text or "independent_add" in text or "add_existing" in text:
        return "强股加仓"
    if "watchlist" in text or "second_buy" in text or "late_confirm" in text or "peak_confirm" in text:
        return "二买确认"
    if "cluster" in text:
        return "形态聚类"
    if "ml_balance" in text or "quality" in text:
        return "质量过滤"
    if "industry" in text:
        return "行业过滤"
    if "concept" in text:
        return "概念增强"
    if "weighted" in text or "rank_weight" in text or "tail" in text:
        return "尾仓分层"
    if "d15_shape" in text or "15d" in text or "d15" in text:
        return "十五日形"
    if "bbi" in text or "short_long" in text or "weak" in text:
        match = re.search(r"(?:^|[_\-])pos(\d+)(?:[_\-]|$)", text)
        if not match:
            match = re.search(r"max[_\-]?positions[_\-]?(\d+)", text)
        if match:
            return _clip_title(f"弱转强{match.group(1)}仓")
        return "弱转强"
    if summary.get("name"):
        token = re.split(r"[_\-\s]+", str(summary["name"]).strip())[0]
        return _clip_title(token) or "策略实验"
    return "策略实验"


def _infer_report_display_title(
    summary: Dict[str, Any] | None,
    run_id: str = "",
    config: Dict[str, Any] | None = None,
    prefer_config: bool = False,
) -> str:
    if prefer_config:
        config_title = _infer_display_title({}, run_id, config)
        if config_title != "策略实验":
            return config_title
    return _infer_display_title(summary, run_id, config)


def _profile_config_paths(config_root: Path, production_profile: Path | None) -> List[Path]:
    if production_profile is None or not production_profile.exists():
        return []
    try:
        profile = yaml.safe_load(production_profile.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(profile, dict):
        return []
    strategies = profile.get("strategies")
    if not isinstance(strategies, dict):
        return []
    paths: List[Path] = []
    for item in strategies.get("include") or []:
        raw_path = Path(str(item))
        path = raw_path if raw_path.is_absolute() else (PROJECT_ROOT / raw_path)
        try:
            path = path.resolve()
            if config_root.resolve() in path.parents and path.exists() and path.suffix in {".yaml", ".yml"}:
                paths.append(path)
        except Exception:
            continue
    return paths


def _profile_report_entries(production_profile: Path | None) -> List[Dict[str, Any]]:
    if production_profile is None or not production_profile.exists():
        return []
    try:
        profile = yaml.safe_load(production_profile.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(profile, dict):
        return []
    reports = profile.get("reports")
    if not isinstance(reports, dict):
        return []
    entries: List[Dict[str, Any]] = []
    for item in reports.get("include") or []:
        if isinstance(item, str):
            entries.append({"run_id": item})
        elif isinstance(item, dict):
            entries.append(dict(item))
    return entries


def _report_sort_key(row: Dict[str, Any]) -> tuple[str, float, str]:
    end_date = str(row.get("end_date") or "")
    mtime = row.get("_mtime")
    if mtime is None and row.get("updated_at"):
        try:
            mtime = datetime.fromisoformat(str(row["updated_at"])).timestamp()
        except Exception:
            mtime = 0.0
    return (end_date, float(mtime or 0.0), str(row.get("run_id") or ""))


class ConfigService:
    def __init__(self, root: Path = CONFIG_ROOT, production_profile: Path | None = None):
        self.root = root
        self.production_profile = production_profile if production_profile is not None else (
            PRODUCTION_PROFILE if root.resolve() == CONFIG_ROOT.resolve() else None
        )
        self.root.mkdir(parents=True, exist_ok=True)

    def list_configs(self) -> List[Dict[str, Any]]:
        rows = []
        paths = _profile_config_paths(self.root, self.production_profile) or sorted(self.root.rglob("*.yaml"))
        for path in paths:
            stat = path.stat()
            description = ""
            config_name = path.stem
            config: Dict[str, Any] = {}
            try:
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
                if isinstance(config, dict):
                    config_name = str(config.get("name") or config_name)
                    description = str(config.get("description") or "")
            except Exception:
                pass
            rows.append({
                "path": _relative_to_root(path, self.root),
                "name": config_name,
                "display_title": _infer_display_title(config, path.stem, config),
                "title": _infer_display_title(config, path.stem, config),
                "description": description,
                "size": stat.st_size,
                "updated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            })
        return rows

    def read_config(self, rel_path: str) -> Dict[str, Any]:
        path = _safe_child(self.root, rel_path)
        return {
            "path": _relative_to_root(path, self.root),
            "content": path.read_text(encoding="utf-8"),
        }

    def write_config(self, rel_path: str, content: str) -> Dict[str, Any]:
        path = _safe_child(self.root, rel_path)
        if path.suffix not in {".yaml", ".yml"}:
            raise ValueError("Config path must end with .yaml or .yml")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return self.read_config(_relative_to_root(path, self.root))

    def delete_config(self, rel_path: str) -> Dict[str, Any]:
        path = _safe_child(self.root, rel_path)
        if path.suffix not in {".yaml", ".yml"}:
            raise ValueError("Config path must end with .yaml or .yml")
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {rel_path}")
        path.unlink()
        return {"ok": True, "path": rel_path}

    def validate_config(self, content: str) -> Dict[str, Any]:
        try:
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Config must be a YAML mapping")
            cost = build_cost(config)
            build_formula_strategy(config, cost)
            explain = explain_strategy_config(config)
            return {
                "ok": True,
                "name": config.get("name"),
                "display_title": _infer_display_title(config, str(config.get("name") or ""), config),
                "title": _infer_display_title(config, str(config.get("name") or ""), config),
                "description": config.get("description"),
                "version": config.get("version"),
                "formula_count": explain["formula_count"],
                "formula_order": explain["formula_order"],
                "dependencies": explain["dependencies"],
                "selector": explain["selector"],
                "rebalance": explain["rebalance"],
                "execution": explain["execution"],
            }
        except Exception as exc:
            return {"ok": False, "error_type": type(exc).__name__, "message": str(exc)}


class ReportService:
    def __init__(
        self,
        root: Path = RUNS_ROOT,
        meta_store: MetaStore | None = None,
        config_root: Path = CONFIG_ROOT,
        production_profile: Path | None = None,
    ):
        self.root = root
        self.meta_store = meta_store or MetaStore(META_URI)
        self.config_root = config_root
        self.production_profile = production_profile if production_profile is not None else (
            PRODUCTION_PROFILE if root.resolve() == RUNS_ROOT.resolve() and config_root.resolve() == CONFIG_ROOT.resolve() else None
        )
        self._profile_config_cache: Dict[str, Dict[str, Any]] | None = None
        self._profile_report_cache: Dict[str, Dict[str, Any]] | None = None
        self._description_cache: Dict[str, str] = {}
        self._description_cache_fingerprint: tuple[int, int, int] | None = None
        self.root.mkdir(parents=True, exist_ok=True)

    def list_reports(self) -> List[Dict[str, Any]]:
        rows = []
        for run_dir in sorted(self.root.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                continue
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            metrics = self._metrics_from_run_dir(run_dir)
            self._merge_list_metrics(summary, metrics)
            if not summary.get("description"):
                summary["description"] = self._description_from_run_dir(run_dir, summary)
            config = self._config_from_run_dir(run_dir)
            if not self._is_profile_report(summary, config):
                continue
            profile_config = self._profile_config_for_report(summary, config)
            profile_report = self._profile_report_for_report(run_dir.name, summary, config)
            if profile_report:
                if profile_report.get("description"):
                    summary["description"] = str(profile_report["description"])
                for key in ("display_title", "title"):
                    if profile_report.get(key):
                        summary[key] = profile_report[key]
            config = profile_config or profile_report or config
            title = _infer_report_display_title(summary, run_dir.name, config, prefer_config=profile_config is not None)
            summary["display_title"] = title
            summary["title"] = title
            rows.append({
                **summary,
                "run_dir": str(run_dir),
                "run_id": run_dir.name,
                "_profile_report_included": profile_report is not None,
                "updated_at": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(timespec="seconds"),
                "_mtime": run_dir.stat().st_mtime,
            })
        return self._latest_reports_by_strategy(rows)

    @staticmethod
    def _latest_reports_by_strategy(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        latest: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            if row.get("_profile_report_included"):
                key = str(row.get("run_id") or row.get("run_dir"))
            else:
                key = str(row.get("name") or row.get("run_id") or row.get("run_dir"))
            current = latest.get(key)
            if current is None or _report_sort_key(row) > _report_sort_key(current):
                latest[key] = row
        result = []
        for row in latest.values():
            clean = dict(row)
            clean.pop("_mtime", None)
            clean.pop("_profile_report_included", None)
            clean["is_latest_for_strategy"] = True
            result.append(clean)
        return sorted(result, key=_report_sort_key, reverse=True)

    def read_report(self, run_id: str) -> Dict[str, Any]:
        run_dir = _safe_child(self.root, run_id)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run not found: {run_id}")
        data = load_run_artifacts(run_dir)
        data["run_id"] = run_id
        self._enrich_report(data)
        return data

    def _description_from_run_dir(self, run_dir: Path, summary: Dict[str, Any] | None = None) -> str:
        for artifact in ("summary.json", "explain.json", "config.yaml"):
            try:
                if artifact == "summary.json":
                    summary = json.loads((run_dir / artifact).read_text(encoding="utf-8"))
                    if summary.get("description"):
                        return str(summary["description"])
                elif artifact == "explain.json":
                    explain = json.loads((run_dir / artifact).read_text(encoding="utf-8"))
                    desc = (explain.get("config") or {}).get("description")
                    if desc:
                        return str(desc)
                elif artifact == "config.yaml" and (run_dir / artifact).exists():
                    config = yaml.safe_load((run_dir / artifact).read_text(encoding="utf-8"))
                    if isinstance(config, dict) and config.get("description"):
                        return str(config["description"])
                    if isinstance(config, dict) and config.get("name"):
                        desc = self._description_from_current_config(str(config["name"]))
                        if desc:
                            return desc
            except Exception:
                continue
        if summary and summary.get("name"):
            return self._description_from_current_config(str(summary["name"]))
        return ""

    @staticmethod
    def _config_from_run_dir(run_dir: Path) -> Dict[str, Any]:
        for artifact in ("explain.json", "config.yaml"):
            try:
                path = run_dir / artifact
                if not path.exists():
                    continue
                if artifact == "explain.json":
                    explain = json.loads(path.read_text(encoding="utf-8"))
                    config = explain.get("config") or {}
                    if isinstance(config, dict):
                        return config
                else:
                    config = yaml.safe_load(path.read_text(encoding="utf-8"))
                    if isinstance(config, dict):
                        return config
            except Exception:
                continue
        return {}

    @staticmethod
    def _metrics_from_run_dir(run_dir: Path) -> Dict[str, Any]:
        try:
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        except Exception:
            return {}
        return metrics if isinstance(metrics, dict) else {}

    @staticmethod
    def _merge_list_metrics(summary: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        for key in (
            "total_return",
            "annual_return",
            "max_drawdown",
            "sharpe",
            "win_rate",
            "profit_factor",
            "avg_holding_days",
            "avg_position_count",
            "max_position_count",
            "final_value",
        ):
            if key in metrics:
                summary[key] = metrics[key]
        if "trade_count" in metrics:
            summary["trade_count"] = metrics["trade_count"]
            summary.setdefault("trades", metrics["trade_count"])

    def _description_from_current_config(self, name: str) -> str:
        if not name:
            return ""
        return self._config_description_map().get(name, "")

    def _profile_config_map(self) -> Dict[str, Dict[str, Any]]:
        if self._profile_config_cache is not None:
            return self._profile_config_cache
        configs: Dict[str, Dict[str, Any]] = {}
        for path in _profile_config_paths(self.config_root, self.production_profile):
            try:
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(config, dict) and config.get("name"):
                configs[str(config["name"])] = config
        self._profile_config_cache = configs
        return configs

    def _profile_report_map(self) -> Dict[str, Dict[str, Any]]:
        if self._profile_report_cache is not None:
            return self._profile_report_cache
        reports: Dict[str, Dict[str, Any]] = {}
        for entry in _profile_report_entries(self.production_profile):
            for key in ("run_id", "name"):
                value = str(entry.get(key) or "").strip()
                if value:
                    reports[value] = entry
        self._profile_report_cache = reports
        return reports

    def _is_profile_report(self, summary: Dict[str, Any], config: Dict[str, Any]) -> bool:
        profile_configs = self._profile_config_map()
        profile_reports = self._profile_report_map()
        if not profile_configs and not profile_reports:
            return True
        name = str(summary.get("name") or config.get("name") or "")
        run_id = str(summary.get("run_id") or "")
        return name in profile_configs or name in profile_reports or run_id in profile_reports

    def _profile_config_for_report(self, summary: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any] | None:
        name = str(summary.get("name") or config.get("name") or "")
        return self._profile_config_map().get(name)

    def _profile_report_for_report(
        self,
        run_id: str,
        summary: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        name = str(summary.get("name") or config.get("name") or "")
        report_map = self._profile_report_map()
        return report_map.get(str(summary.get("run_id") or run_id)) or report_map.get(name)

    def _config_description_map(self) -> Dict[str, str]:
        if not self.config_root.exists():
            return {}
        paths = list(self.config_root.rglob("*.yaml"))
        fingerprint = (
            len(paths),
            max((path.stat().st_mtime_ns for path in paths), default=0),
            sum((path.stat().st_size for path in paths), 0),
        )
        if self._description_cache_fingerprint == fingerprint:
            return self._description_cache
        descriptions: Dict[str, str] = {}
        for path in paths:
            try:
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(config, dict) or not config.get("name"):
                continue
            descriptions[str(config["name"])] = str(config.get("description") or "")
        self._description_cache = descriptions
        self._description_cache_fingerprint = fingerprint
        return descriptions

    def _enrich_report(self, data: Dict[str, Any]) -> None:
        summary = data.get("summary") or {}
        explain = data.get("explain") or {}
        config = explain.get("config") or {}
        if config.get("description") and not summary.get("description"):
            summary["description"] = config.get("description")
        if not summary.get("description") and config.get("name"):
            summary["description"] = self._description_from_current_config(str(config["name"]))
        if not summary.get("description") and summary.get("name"):
            summary["description"] = self._description_from_current_config(str(summary["name"]))
        profile_config = self._profile_config_for_report(summary, config)
        profile_report = self._profile_report_for_report(str(data.get("run_id") or ""), summary, config)
        if profile_report:
            if profile_report.get("description"):
                summary["description"] = str(profile_report["description"])
            for key in ("display_title", "title"):
                if profile_report.get(key):
                    summary[key] = profile_report[key]
        config = profile_config or profile_report or config
        title = _infer_report_display_title(
            summary,
            str(data.get("run_id") or ""),
            config,
            prefer_config=profile_config is not None,
        )
        summary["display_title"] = title
        summary["title"] = title
        data["display_title"] = title
        data["title"] = title
        data["best_closed_positions"] = self._rank_closed_positions(data.get("closed_positions") or [])
        data["current_positions"] = self._current_positions(
            data.get("positions") or [],
            data.get("trades") or [],
            current_date=summary.get("end_date") or self._latest_nav_date(data.get("daily_nav") or []),
        )
        data["closed_position_summary"] = self._closed_position_summary(data.get("closed_positions") or [])
        data["annual_returns"] = annual_returns(data.get("daily_nav") or [])
        data["next_session_candidates"] = self._latest_daily_selection_candidates(
            data.get("daily_selection_candidates") or [],
            summary.get("end_date") or self._latest_nav_date(data.get("daily_nav") or []),
        )
        meta = self._meta_map(_candidate_symbols(data.get("next_session_candidates") or {}))
        self._enrich_candidate_rows(data.get("next_session_candidates") or {}, meta)
        data["next_session_guide"] = build_next_session_guide(
            data.get("next_session_candidates") or {},
            data.get("current_positions") or [],
            config=config,
            latest_date=summary.get("end_date") or self._latest_nav_date(data.get("daily_nav") or []),
        )

    @staticmethod
    def _latest_nav_date(daily_nav: List[Dict[str, Any]]) -> str | None:
        dates = [str(row.get("date")) for row in daily_nav if row.get("date")]
        return max(dates) if dates else None

    @staticmethod
    def _rank_closed_positions(closed_positions: List[Dict[str, Any]], limit: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        valid = [row for row in closed_positions if row.get("net_pnl") is not None]
        gains = sorted(valid, key=lambda row: float(row.get("net_pnl") or 0.0), reverse=True)[:limit]
        losses = sorted(valid, key=lambda row: float(row.get("net_pnl") or 0.0))[:limit]
        return {"gains": gains, "losses": losses}

    @staticmethod
    def _current_positions(
        positions: List[Dict[str, Any]],
        trades: List[Dict[str, Any]],
        current_date: str | None = None,
    ) -> List[Dict[str, Any]]:
        if not positions:
            return []
        latest_date = str(current_date) if current_date else max(str(row.get("date")) for row in positions if row.get("date"))
        latest = [dict(row) for row in positions if str(row.get("date")) == latest_date and int(row.get("quantity") or 0) > 0]
        last_trade_by_symbol: Dict[str, Dict[str, Any]] = {}
        for trade in trades:
            if trade.get("reject_reason"):
                continue
            symbol = trade.get("symbol")
            if symbol:
                last_trade_by_symbol[symbol] = trade
        for row in latest:
            avg_cost = float(row.get("avg_cost") or 0.0)
            quantity = int(row.get("quantity") or 0)
            market_value = float(row.get("market_value") or 0.0)
            current_price = market_value / quantity if quantity > 0 else 0.0
            pnl = (current_price - avg_cost) * quantity
            row.update({
                "current_price": current_price,
                "unrealized_pnl": pnl,
                "unrealized_return": current_price / avg_cost - 1 if avg_cost > 0 else 0.0,
                "status": "holding",
                "last_trade_date": last_trade_by_symbol.get(row.get("symbol"), {}).get("date"),
            })
        return sorted(latest, key=lambda row: float(row.get("unrealized_pnl") or 0.0), reverse=True)

    @staticmethod
    def _closed_position_summary(closed_positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = []
        for row in closed_positions:
            item = dict(row)
            pnl = float(item.get("net_pnl") or 0.0)
            item["status"] = "closed"
            item["is_profit"] = pnl > 0
            item["profit_label"] = "盈利" if pnl > 0 else ("亏损" if pnl < 0 else "持平")
            rows.append(item)
        return rows

    @staticmethod
    def _latest_daily_selection_candidates(rows: List[Dict[str, Any]], latest_date: str | None) -> Dict[str, Any]:
        if not rows:
            return {}
        exact = [row for row in rows if latest_date and str(row.get("date")) == str(latest_date)]
        item = dict(exact[-1] if exact else rows[-1])
        item["is_latest_signal_date"] = bool(latest_date and str(item.get("date")) == str(latest_date))
        return item

    def _meta_map(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        symbols = sorted({symbol for symbol in symbols if symbol})
        if not symbols:
            return {}
        frame = self.meta_store.get_symbol_meta(symbols)
        if frame.empty:
            return {}
        return {str(idx): row.dropna().to_dict() for idx, row in frame.iterrows()}

    @staticmethod
    def _enrich_candidate_rows(candidates: Dict[str, Any], meta: Dict[str, Dict[str, Any]]) -> None:
        for key in ("raw_candidates", "selected_candidates"):
            rows = []
            for row in candidates.get(key) or []:
                item = dict(row)
                item.update(_meta_fields(item.get("symbol"), meta))
                rows.append(item)
            candidates[key] = rows

    def read_artifact(self, run_id: str, artifact: str) -> Any:
        artifact = artifact.replace("-", "_")
        allowed = {
            "summary": "summary.json",
            "metrics": "metrics.json",
            "daily_nav": "daily_nav.json",
            "trades": "trades.json",
            "positions": "positions.json",
            "closed_positions": "closed_positions.json",
            "selection_candidates": "selection_candidates.json",
            "daily_selection_candidates": "daily_selection_candidates.json",
            "explain": "explain.json",
        }
        if artifact not in allowed:
            raise ValueError(f"Unsupported artifact: {artifact}")
        run_dir = _safe_child(self.root, run_id)
        return json.loads((run_dir / allowed[artifact]).read_text(encoding="utf-8"))

    def read_symbol_detail(self, run_id: str, symbol: str) -> Dict[str, Any]:
        report = self.read_report(run_id)
        summary = report["summary"]
        provider_uri = (
            report.get("explain", {})
            .get("config", {})
            .get("data", {})
            .get("provider_uri", "data/qlib_data_fixed")
        )
        start = summary["start_date"]
        end = summary["end_date"]
        trades = [trade for trade in report["trades"] if trade.get("symbol") == symbol and not trade.get("reject_reason")]
        closed = [pos for pos in report.get("closed_positions", []) if pos.get("symbol") == symbol]
        position_rows = [pos for pos in report.get("positions", []) if pos.get("symbol") == symbol]

        exchange = AStockExchange(provider_uri=provider_uri)
        exchange.load_quote_data([symbol], start, end)
        bars = self._quote_to_bars(exchange.quote, symbol)
        meta = self.meta_store.get_symbol_meta([symbol])
        meta_row = meta.iloc[0].to_dict() if not meta.empty else {}
        return {
            "run_id": run_id,
            "symbol": symbol,
            "name": meta_row.get("name") or symbol,
            "meta": meta_row,
            "summary": summary,
            "provider_uri": provider_uri,
            "bars": bars,
            "trades": trades,
            "round_trips": closed or self._round_trips(trades),
            "positions": position_rows,
        }

    @staticmethod
    def _quote_to_bars(quote, symbol: str) -> List[Dict[str, Any]]:
        if quote is None or quote.empty:
            return []
        try:
            frame = quote.xs(symbol, level="instrument").sort_index()
        except Exception:
            return []
        rows: List[Dict[str, Any]] = []
        for dt, row in frame.iterrows():
            rows.append({
                "date": dt.strftime("%Y-%m-%d"),
                "open": _none_if_nan(row.get("$open")),
                "high": _none_if_nan(row.get("$high")),
                "low": _none_if_nan(row.get("$low")),
                "close": _none_if_nan(row.get("$close")),
                "volume": _none_if_nan(row.get("$volume")),
                "change": _none_if_nan(row.get("$change")),
                "vwap": _none_if_nan(row.get("$vwap")),
            })
        return rows

    @staticmethod
    def _round_trips(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        open_lots: List[Dict[str, Any]] = []
        trips: List[Dict[str, Any]] = []
        for trade in sorted(trades, key=lambda item: item.get("date", "")):
            qty = int(trade.get("quantity") or 0)
            if qty <= 0:
                continue
            if trade.get("action") == "BUY":
                open_lots.append({
                    "date": trade["date"],
                    "quantity": qty,
                    "price": float(trade["price"]),
                })
                continue
            if trade.get("action") != "SELL":
                continue
            remaining = qty
            while remaining > 0 and open_lots:
                lot = open_lots[0]
                matched = min(remaining, int(lot["quantity"]))
                entry_price = float(lot["price"])
                exit_price = float(trade["price"])
                trips.append({
                    "entry_date": lot["date"],
                    "exit_date": trade["date"],
                    "quantity": matched,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return": exit_price / entry_price - 1 if entry_price else 0.0,
                })
                lot["quantity"] -= matched
                remaining -= matched
                if lot["quantity"] <= 0:
                    open_lots.pop(0)
        return trips


@dataclass
class BacktestTask:
    task_id: str
    config_path: str
    status: str = "queued"
    run_id: Optional[str] = None
    run_dir: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def log(self, message: str) -> None:
        self.logs.append(f"{datetime.now().isoformat(timespec='seconds')} {message}")
        self.updated_at = datetime.now().isoformat(timespec="seconds")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "config_path": self.config_path,
            "status": self.status,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "result": self.result,
            "error": self.error,
            "logs": self.logs,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class BacktestTaskService:
    def __init__(self, config_root: Path = CONFIG_ROOT, runs_root: Path = RUNS_ROOT):
        self.config_root = config_root
        self.runs_root = runs_root
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.tasks: Dict[str, BacktestTask] = {}

    def submit(self, config_path: str, symbol_limit: int | None = None) -> Dict[str, Any]:
        task_id = datetime.now().strftime("task_%Y%m%d_%H%M%S_%f")
        task = BacktestTask(task_id=task_id, config_path=config_path)
        self.tasks[task_id] = task
        self.executor.submit(self._run_task, task, symbol_limit)
        return task.to_dict()

    def get(self, task_id: str) -> Dict[str, Any]:
        if task_id not in self.tasks:
            raise KeyError(task_id)
        return self.tasks[task_id].to_dict()

    def _run_task(self, task: BacktestTask, symbol_limit: int | None) -> None:
        task.status = "running"
        task.log("backtest started")
        try:
            path = _safe_child(self.config_root, task.config_path)
            config = load_config(path)
            run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.get('name', path.stem)}"
            task.run_id = run_id
            summary = run_config_with_artifacts(
                path,
                self.runs_root,
                symbol_limit=symbol_limit,
                run_id=run_id,
            )
            task.result = summary
            task.run_dir = summary.get("run_dir")
            task.status = "success"
            task.log("backtest completed")
        except Exception as exc:
            task.status = "failed"
            task.error = f"{type(exc).__name__}: {exc}"
            task.log(task.error)
            task.logs.append(traceback.format_exc())
        finally:
            task.updated_at = datetime.now().isoformat(timespec="seconds")


class DataService:
    def status(self, provider_uri: str = "data/qlib_data_fixed") -> Dict[str, Any]:
        base = (PROJECT_ROOT / provider_uri).resolve()
        calendar_path = base / "calendars" / "day.txt"
        instruments_path = base / "instruments" / "all.txt"
        dates = []
        if calendar_path.exists():
            dates = [line.strip() for line in calendar_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        instruments = []
        if instruments_path.exists():
            instruments = [line.strip() for line in instruments_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        feature_count = len(list((base / "features").glob("*/*.bin"))) if (base / "features").exists() else 0
        return {
            "provider_uri": provider_uri,
            "exists": base.exists(),
            "calendar_start": dates[0] if dates else None,
            "calendar_end": dates[-1] if dates else None,
            "calendar_days": len(dates),
            "instrument_count": len(instruments),
            "feature_file_count": feature_count,
        }

    def health_check(self, provider_uri: str = "data/qlib_data_fixed") -> Dict[str, Any]:
        status = self.status(provider_uri)
        ok = bool(status["exists"] and status["calendar_days"] and status["instrument_count"] and status["feature_file_count"])
        return {"ok": ok, **status}

    def update_plan(self, provider_uri: str = "data/qlib_data_fixed") -> Dict[str, Any]:
        status = self.status(provider_uri)
        return {
            "ok": True,
            "provider_uri": provider_uri,
            "current": status,
            "commands": [
                "conda run -n test python -m quantx.tools.sync_daily_data --provider-uri data/qlib_data_fixed --raw-dir data/raw/baostock --limit 5 --dry-run --json",
                "conda run -n test python -m quantx.tools.sync_daily_data --provider-uri data/qlib_data_fixed --raw-dir data/raw/baostock --mode incremental --json",
                "conda run -n test python -m quantx.tools.install_daily_data_cron --install",
                "conda run -n test python -m quantx.tools.run_backtest --config configs/strategies/shuijiao_legacy.yaml --dry-run --json",
            ],
        }


class PatternAnalysisService:
    def __init__(self, root: Path = PATTERN_ANALYSIS_ROOT, runs_root: Path = RUNS_ROOT):
        self.root = root
        self.runs_root = runs_root
        self.root.mkdir(parents=True, exist_ok=True)

    def list_analyses(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for analysis_dir in sorted(self.root.iterdir(), reverse=True):
            if not analysis_dir.is_dir():
                continue
            summary = read_json(analysis_dir / "pattern_summary.json", default={}) or {}
            config = read_json(analysis_dir / "config.json", default={}) or {}
            if not summary and not config:
                continue
            rows.append({
                "analysis_id": analysis_dir.name,
                "analysis_dir": str(analysis_dir),
                "sample_count": summary.get("sample_count", 0),
                "date_start": summary.get("date_start"),
                "date_end": summary.get("date_end"),
                "avg_return": summary.get("avg_return"),
                "avg_mfe": summary.get("avg_mfe"),
                "avg_mae": summary.get("avg_mae"),
                "outcome_counts": summary.get("outcome_counts", {}),
                "runs": config.get("runs", []),
                "updated_at": datetime.fromtimestamp(analysis_dir.stat().st_mtime).isoformat(timespec="seconds"),
            })
        return rows

    def read_analysis(self, analysis_id: str) -> Dict[str, Any]:
        analysis_dir = _safe_child(self.root, analysis_id)
        if not analysis_dir.is_dir():
            raise FileNotFoundError(f"Pattern analysis not found: {analysis_id}")
        summary = read_json(analysis_dir / "pattern_summary.json", default={}) or {}
        config = read_json(analysis_dir / "config.json", default={}) or {}
        clusters = self._cluster_table(analysis_dir)
        return {
            "analysis_id": analysis_id,
            "analysis_dir": str(analysis_dir),
            "summary": summary,
            "config": config,
            "clusters": clusters,
        }

    def cluster_samples(self, analysis_id: str, cluster_id: str, limit: int = 100) -> Dict[str, Any]:
        analysis_dir = _safe_child(self.root, analysis_id)
        clusters = read_table(analysis_dir / "cluster_results.parquet")
        rows = clusters[clusters["cluster_id"] == cluster_id].sort_values("return", ascending=False).head(limit)
        return {"analysis_id": analysis_id, "cluster_id": cluster_id, "samples": rows.to_dict(orient="records")}

    def similar(self, analysis_id: str, sample_id: str, top_k: int = 20, same_cluster_only: bool = False) -> Dict[str, Any]:
        analysis_dir = _safe_child(self.root, analysis_id)
        return find_similar_samples(analysis_dir, sample_id, top_k=top_k, same_cluster_only=same_cluster_only)

    def figure_path(self, analysis_id: str, figure_path: str) -> Path:
        analysis_dir = _safe_child(self.root, analysis_id)
        path = _safe_child(analysis_dir, figure_path)
        if not path.is_file():
            raise FileNotFoundError(f"Figure not found: {figure_path}")
        return path

    def build(
        self,
        runs: List[str],
        analysis_id: str | None = None,
        pre_n: int = 40,
        post_n: int = 15,
        n_clusters: int = 8,
        models: List[str] | None = None,
    ) -> Dict[str, Any]:
        config = PatternAnalysisConfig(
            runs=[self._resolve_run_path(run) for run in runs],
            output_dir=self.root,
            analysis_id=analysis_id,
            pre_n=pre_n,
            post_n=post_n,
            n_clusters=n_clusters,
            models=models or ["logistic_regression", "random_forest", "lightgbm"],
            label_config=LabelConfig(),
        )
        result = build_pattern_analysis(config)
        return {
            "ok": result.sample_count > 0,
            "analysis_id": result.analysis_id,
            "sample_count": result.sample_count,
            "output_dir": str(result.output_dir),
            "report_path": str(result.report_path) if result.report_path else None,
            "summary": result.summary,
        }

    def _resolve_run_path(self, run: str) -> Path:
        path = Path(run)
        if path.is_absolute():
            return path
        if path.parts and path.parts[0] == self.runs_root.name:
            return (self.runs_root.parent / path).resolve()
        return (self.runs_root / path).resolve()

    @staticmethod
    def _cluster_table(analysis_dir: Path) -> List[Dict[str, Any]]:
        try:
            frame = read_table(analysis_dir / "cluster_results.parquet")
        except FileNotFoundError:
            return []
        if frame.empty:
            return []
        rows = []
        for cluster_id, group in frame.groupby("cluster_id"):
            rows.append({
                "cluster_id": cluster_id,
                "sample_count": int(len(group)),
                "success_rate": float(group["binary_success"].mean()) if "binary_success" in group else None,
                "avg_return": float(group["return"].mean()),
                "avg_mfe": float(group["max_favorable_excursion"].mean()),
                "avg_mae": float(group["max_adverse_excursion"].mean()),
                "early_confirm_rate": float(group["early_confirmed"].mean()) if "early_confirmed" in group else None,
                "opportunity_rate": float(group["had_opportunity"].mean()) if "had_opportunity" in group else None,
                "fade_after_peak_rate": float(group["faded_after_peak"].mean()) if "faded_after_peak" in group else None,
                "efficient_capture_rate": float(group["efficient_capture"].mean()) if "efficient_capture" in group else None,
            })
        return sorted(rows, key=lambda row: str(row["cluster_id"]))


class MetaService:
    def __init__(self, store: MetaStore | None = None):
        self.store = store or MetaStore(META_URI)

    def symbol(self, symbol: str) -> Dict[str, Any]:
        frame = self.store.get_symbol_meta([symbol])
        if frame.empty:
            return {"symbol": symbol, "found": False}
        row = frame.iloc[0].to_dict()
        row["found"] = True
        return row

    def industries(self) -> List[Dict[str, Any]]:
        return self.store.get_industries().to_dict(orient="records")

    def concepts(self) -> List[Dict[str, Any]]:
        return self.store.get_sectors("concept").to_dict(orient="records")


def _none_if_nan(value: Any) -> Any:
    try:
        if value != value:
            return None
    except Exception:
        pass
    try:
        item = value.item()
        return None if item != item else item
    except Exception:
        return value
