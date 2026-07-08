# Metrics Extension

Use this reference when the user asks to add a new performance metric or expose new report analysis to Agent/Web.

## Target Design

Metrics should move toward a registry instead of being hard-coded in one report function.

Recommended module layout:

```text
quantx/core/analysis/metrics/
├── __init__.py
├── base.py
├── registry.py
├── builtin.py
└── custom.py
```

## Metric Interface

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

@dataclass(frozen=True)
class MetricSpec:
    name: str
    display_name: str
    category: str
    required_artifacts: tuple[str, ...]
    description: str = ""

@dataclass
class MetricContext:
    summary: Dict[str, Any]
    daily_nav: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    positions: List[Dict[str, Any]]
    closed_positions: List[Dict[str, Any]]

class MetricFn(Protocol):
    spec: MetricSpec

    def __call__(self, context: MetricContext) -> Any:
        ...
```

## Registry Rules

- Metric names must be unique.
- Metric output must be JSON serializable.
- Metric computation must not mutate report artifacts or backtest state.
- Each new metric needs at least one unit test.
- Declare required artifacts so Agent can tell whether a metric can be computed from an existing run.

## Common Metrics To Add

- `sortino`: annual return divided by downside volatility.
- `calmar`: annual return divided by absolute max drawdown.
- `volatility`: annualized daily return standard deviation.
- `turnover`: traded value over average equity.
- `exposure`: average market value over total value.
- `largest_win`, `largest_loss`: extreme round-trip returns.
- `max_consecutive_loss`: consecutive losing trades.
- `industry_exposure`: position/trade distribution by industry.

## Agent Workflow

1. Identify formula and required artifacts.
2. If artifacts are missing, extend reporting first.
3. Implement metric in the metrics module.
4. Register it.
5. Add tests.
6. Recompute metrics for an existing run or run a small backtest.
7. Update Web/API display only after the metric is stable.
