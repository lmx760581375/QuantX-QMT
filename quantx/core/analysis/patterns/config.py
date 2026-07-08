"""Configuration objects for trade pattern analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass
class LabelConfig:
    """Thresholds used to label completed trades."""

    success_return_threshold: float = 0.05
    failure_return_threshold: float = 0.0
    hard_loss_threshold: float = 0.05
    max_adverse_threshold: float = 0.08
    quick_confirm_days: int = 5
    quick_confirm_return: float = 0.04
    opportunity_return_threshold: float = 0.08
    fade_drawdown_threshold: float = 0.06
    recover_improvement_threshold: float = 0.02
    recover_profit_threshold: float = 0.0


@dataclass
class PatternAnalysisConfig:
    """Configuration for building a trade pattern analysis artifact."""

    runs: Sequence[Path]
    output_dir: Path = Path("artifacts/pattern_analysis")
    analysis_id: str | None = None
    raw_data_dir: Path = Path("data/raw/baostock")
    provider_uri: Path = Path("data/qlib_data_fixed")
    prefer_qlib: bool = True
    pre_n: int = 40
    post_n: int = 15
    min_pre_bars: int = 20
    min_post_bars: int = 0
    label_config: LabelConfig = field(default_factory=LabelConfig)
    models: Sequence[str] = field(default_factory=lambda: ["logistic_regression", "random_forest", "lightgbm"])
    n_clusters: int = 8
    random_state: int = 42
    build_models: bool = True
    build_clusters: bool = True
    render_report: bool = True
    feature_scope: str = "entry"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["runs"] = [str(path) for path in self.runs]
        data["output_dir"] = str(self.output_dir)
        data["raw_data_dir"] = str(self.raw_data_dir)
        data["provider_uri"] = str(self.provider_uri)
        return data

    @property
    def train_prefixes(self) -> List[str]:
        return ["pre_", "entry_"]
