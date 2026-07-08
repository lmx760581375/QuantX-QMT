"""回测引擎"""

from .types import Order, OrderAction, Trade, Position, Signal, DailySnapshot
from .cost import TransactionCost
from .board import BoardManager
from .exchange import AStockExchange
from .account import Account
from .executor import Executor
from .context import BacktestContext
from .signals import SignalMatrix
from .engine import BacktestEngine, BacktestConfig, BacktestResult

__all__ = [
    "Order", "OrderAction", "Trade", "Position", "Signal", "DailySnapshot",
    "TransactionCost", "BoardManager", "AStockExchange",
    "Account", "Executor", "BacktestContext", "SignalMatrix",
    "BacktestEngine", "BacktestConfig", "BacktestResult",
]
