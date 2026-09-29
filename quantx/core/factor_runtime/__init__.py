"""Dynamic in-memory factor runtime.

The runtime evaluates strategy formulas on full-market matrices shaped as
``[date, instrument]``. It is run-level only: no long-lived factor files are
required for strategy iteration.
"""

from .panel import MarketPanel
from .runtime import FactorRuntime, FormulaError
from .operators import OperatorRegistry, operator

__all__ = [
    "MarketPanel",
    "FactorRuntime",
    "FormulaError",
    "OperatorRegistry",
    "operator",
]
