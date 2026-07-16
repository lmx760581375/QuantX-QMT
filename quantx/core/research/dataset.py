"""Build causal features and theoretical next-open labels from Qlib data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantx.core.data.meta.groups import group_config_identity, load_meta_groups
from quantx.core.factor_runtime import FactorRuntime, MarketPanel

from .causality import FeatureCausalityValidator
from .schema import FeatureSchema
from .specs import FeatureSpec, LabelSpec
from .universe import PointInTimeUniverseProvider


@dataclass(frozen=True)
class ResearchDataset:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    feature_spec_hash: str
    label_spec_hash: str
    universe_version_id: str
    label_horizon_sessions: int = 1
    feature_schema: FeatureSchema | None = None
    label_winsorize: tuple[float, float] | None = None
    execution_audit: dict | None = None


class DatasetBuilder:
    def __init__(self, reader, universe: PointInTimeUniverseProvider):
        self.reader = reader
        self.universe = universe

    def build(
        self,
        feature_spec: FeatureSpec,
        label_spec: LabelSpec,
        *,
        start: str,
        end: str,
    ) -> ResearchDataset:
        causality = FeatureCausalityValidator()
        if feature_spec.require_causal:
            causality.require_causal(feature_spec)
        calendar = pd.DatetimeIndex(self.reader.calendar(None, end))
        start_position = int(calendar.searchsorted(pd.Timestamp(start), side="left"))
        load_position = max(0, start_position - feature_spec.lookback_sessions)
        load_start = calendar[load_position].strftime("%Y-%m-%d")
        raw_fields = tuple(dict.fromkeys((*feature_spec.raw_fields, "open", "high", "low", "close", "volume", "amount")))
        load_symbols = list(self.universe.all_symbols)
        if (
            label_spec.excess_return
            and label_spec.benchmark != "CROSS_SECTIONAL_MEAN"
            and label_spec.benchmark not in load_symbols
        ):
            load_symbols.append(label_spec.benchmark)
        data = self.reader.features(
            load_symbols,
            [f"${field}" for field in raw_fields],
            load_start,
            end,
        )
        panel = MarketPanel.from_frame(data)
        runtime = FactorRuntime(panel)
        runtime.values.update(load_meta_groups(panel.instruments, feature_spec.groups))
        computed = runtime.compute_formulas(dict(feature_spec.expressions)) if feature_spec.expressions else {}
        feature_names = tuple((*feature_spec.raw_fields, *feature_spec.expressions.keys()))
        matrices = {
            **{field: panel.get(field) for field in feature_spec.raw_fields},
            **computed,
        }
        index = pd.MultiIndex.from_product([panel.dates, panel.instruments], names=["signal_time", "instrument"])
        frame = pd.DataFrame(
            {name: _feature_values(values, panel).reshape(-1) for name, values in matrices.items()},
            index=index,
        )
        frame["valid_features"] = np.isfinite(frame[list(feature_names)]).all(axis=1)
        expression_windows = causality.analyze(feature_spec)
        read_windows = {
            **{field: (0, 0) for field in feature_spec.raw_fields},
            **{
                name: (window.min_offset_sessions, window.max_offset_sessions)
                for name, window in expression_windows.items()
            },
        }
        if feature_spec.groups:
            read_windows[f"__groups__:{group_config_identity(feature_spec.groups)}"] = (0, 0)
        schema = FeatureSchema.create(
            names=feature_names,
            dtypes=tuple(str(frame[name].dtype) for name in feature_names),
            lookback_sessions=feature_spec.lookback_sessions,
            frequency="day",
            read_windows=read_windows,
        )
        self._add_labels(frame, panel, label_spec)
        if label_spec.execution_diagnostic:
            self._add_execution_diagnostics(frame, panel)
        date_values = frame.index.get_level_values("signal_time")
        frame = frame[(date_values >= pd.Timestamp(start)) & (date_values <= pd.Timestamp(end))]
        membership = {
            session: set(self.universe.members(session))
            for session in frame.index.get_level_values("signal_time").unique()
        }
        member_mask = [instrument in membership[session] for session, instrument in frame.index]
        frame = frame[np.asarray(member_mask, dtype=bool)].sort_index()
        execution_audit = _execution_audit(frame) if label_spec.execution_diagnostic else None
        return ResearchDataset(
            frame=frame,
            feature_columns=feature_names,
            feature_spec_hash=schema.schema_hash,
            label_spec_hash=label_spec.spec_hash,
            universe_version_id=self.universe.version_id,
            label_horizon_sessions=label_spec.horizon_sessions,
            feature_schema=schema,
            label_winsorize=label_spec.winsorize,
            execution_audit=execution_audit,
        )

    @staticmethod
    def _add_labels(frame: pd.DataFrame, panel: MarketPanel, spec: LabelSpec) -> None:
        opens = panel.get("open")
        entry_values = np.full_like(opens, np.nan, dtype=float)
        exit_values = np.full_like(opens, np.nan, dtype=float)
        entry_sessions = np.full(opens.shape, None, dtype=object)
        end_sessions = np.full(opens.shape, None, dtype=object)
        for signal_index in range(len(panel.dates)):
            entry_index = signal_index + 1
            exit_index = entry_index + spec.horizon_sessions
            if exit_index >= len(panel.dates):
                continue
            entry_values[signal_index] = opens[entry_index]
            exit_values[signal_index] = opens[exit_index]
            entry_sessions[signal_index, :] = panel.dates[entry_index].strftime("%Y-%m-%d")
            end_sessions[signal_index, :] = panel.dates[exit_index].strftime("%Y-%m-%d")
        labels = exit_values / entry_values - 1.0
        if spec.excess_return:
            if spec.benchmark == "CROSS_SECTIONAL_MEAN":
                finite_counts = np.isfinite(labels).sum(axis=1)
                benchmark_returns = np.divide(
                    np.nansum(labels, axis=1),
                    finite_counts,
                    out=np.full(len(labels), np.nan, dtype=float),
                    where=finite_counts > 0,
                )
            else:
                try:
                    benchmark_index = panel.instruments.get_loc(spec.benchmark)
                except KeyError as exc:
                    raise ValueError(f"Benchmark quote data is missing: {spec.benchmark}") from exc
                benchmark_returns = labels[:, benchmark_index]
            labels = labels - benchmark_returns[:, np.newaxis]
        frame["theoretical_label"] = labels.reshape(-1)
        frame["label_entry_session"] = entry_sessions.reshape(-1)
        frame["label_end_session"] = end_sessions.reshape(-1)
        frame["label_available"] = np.isfinite(frame["theoretical_label"])

    @staticmethod
    def _add_execution_diagnostics(frame: pd.DataFrame, panel: MarketPanel) -> None:
        opens = _panel_field(panel, "open")
        if opens is None:
            return
        closes = _panel_field(panel, "close")
        highs = _panel_field(panel, "high")
        lows = _panel_field(panel, "low")
        volumes = _panel_field(panel, "volume")
        amounts = _panel_field(panel, "amount")
        shape = opens.shape
        entry_open = np.full(shape, np.nan, dtype=float)
        entry_high = np.full(shape, np.nan, dtype=float)
        entry_low = np.full(shape, np.nan, dtype=float)
        entry_close = np.full(shape, np.nan, dtype=float)
        entry_volume = np.full(shape, np.nan, dtype=float)
        entry_amount = np.full(shape, np.nan, dtype=float)
        preclose = np.full(shape, np.nan, dtype=float)
        for signal_index in range(len(panel.dates)):
            entry_index = signal_index + 1
            if entry_index >= len(panel.dates):
                continue
            entry_open[signal_index] = opens[entry_index]
            if highs is not None:
                entry_high[signal_index] = highs[entry_index]
            if lows is not None:
                entry_low[signal_index] = lows[entry_index]
            if closes is not None:
                entry_close[signal_index] = closes[entry_index]
                preclose[signal_index] = closes[signal_index]
            if volumes is not None:
                entry_volume[signal_index] = volumes[entry_index]
            if amounts is not None:
                entry_amount[signal_index] = amounts[entry_index]
        entry_has_bar = np.isfinite(entry_open) & (entry_open > 0)
        zero_volume = np.isfinite(entry_volume) & (entry_volume <= 0)
        zero_amount = np.isfinite(entry_amount) & (entry_amount <= 0)
        open_gap = np.divide(
            entry_open,
            preclose,
            out=np.full(shape, np.nan, dtype=float),
            where=np.isfinite(preclose) & (preclose > 0),
        ) - 1.0
        has_ohlc = (
            np.isfinite(entry_open)
            & np.isfinite(entry_high)
            & np.isfinite(entry_low)
            & np.isfinite(entry_close)
            & (entry_open > 0)
            & (entry_high > 0)
            & (entry_low > 0)
            & (entry_close > 0)
        )
        one_price = has_ohlc & (np.maximum(np.abs(entry_high - entry_low), np.abs(entry_open - entry_close)) <= 1e-6)
        one_price_limit_up = one_price & (open_gap >= 0.045)
        st_like_as_of_signal = _st_like_limit_history(closes) if closes is not None else np.full(shape, False, dtype=bool)
        frame["entry_has_bar"] = entry_has_bar.reshape(-1)
        frame["entry_zero_volume_or_amount"] = (zero_volume | zero_amount).reshape(-1)
        frame["entry_one_price_limit_up"] = one_price_limit_up.reshape(-1)
        frame["entry_open_gap_pct"] = open_gap.reshape(-1)
        frame["entry_open_gap_ge_3pct"] = (open_gap >= 0.03 - 1e-12).reshape(-1)
        frame["st_like_limit_history"] = st_like_as_of_signal.reshape(-1)


def _feature_values(values, panel: MarketPanel) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 0:
        return np.full((len(panel.dates), len(panel.instruments)), arr.item(), dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] == len(panel.dates):
            return np.repeat(arr[:, np.newaxis], len(panel.instruments), axis=1)
        if arr.shape[0] == len(panel.instruments):
            return np.repeat(arr[np.newaxis, :], len(panel.dates), axis=0)
    return arr


def _panel_field(panel: MarketPanel, field: str) -> np.ndarray | None:
    return panel.get(field) if panel.has(field) else None


def _st_like_limit_history(closes: np.ndarray, *, lookback: int = 60, min_hits: int = 2) -> np.ndarray:
    preclose = np.vstack([np.full((1, closes.shape[1]), np.nan), closes[:-1]])
    daily_ret = np.divide(
        closes,
        preclose,
        out=np.full_like(closes, np.nan, dtype=float),
        where=np.isfinite(preclose) & (preclose > 0),
    ) - 1.0
    abs_ret = np.abs(daily_ret)
    five_pct = (abs_ret >= 0.045) & (abs_ret <= 0.055)
    ten_pct = abs_ret >= 0.075
    five_counts = _rolling_count(five_pct, lookback)
    ten_counts = _rolling_count(ten_pct, lookback)
    # Signal T can only use history through T, so no shift is applied here.
    return (five_counts >= min_hits) & (ten_counts == 0)


def _rolling_count(mask: np.ndarray, window: int) -> np.ndarray:
    values = mask.astype(int)
    cumsum = np.cumsum(values, axis=0)
    out = cumsum.copy()
    if window < len(values):
        out[window:] = cumsum[window:] - cumsum[:-window]
    return out


def _execution_audit(frame: pd.DataFrame) -> dict:
    rows = int(len(frame))
    label_available = frame["label_available"].astype(bool) if "label_available" in frame else pd.Series(False, index=frame.index)
    available_count = int(label_available.sum())
    def count(column: str, mask=None) -> int:
        if column not in frame:
            return 0
        selected = frame[column].astype(bool)
        if mask is not None:
            selected = selected & mask
        return int(selected.sum())
    missing_bar = int((~frame["entry_has_bar"].astype(bool) & label_available).sum()) if "entry_has_bar" in frame else 0
    counts = {
        "missing_entry_bar": missing_bar,
        "zero_volume_or_amount": count("entry_zero_volume_or_amount", label_available),
        "one_price_limit_up": count("entry_one_price_limit_up", label_available),
        "open_gap_ge_3pct": count("entry_open_gap_ge_3pct", label_available),
        "st_like_limit_history": count("st_like_limit_history", label_available),
    }
    denominators = {key: available_count for key in counts}
    ratios = {key: (value / denominators[key] if denominators[key] else 0.0) for key, value in counts.items()}
    contaminated = pd.Series(False, index=frame.index)
    for column in ("entry_zero_volume_or_amount", "entry_one_price_limit_up", "entry_open_gap_ge_3pct", "st_like_limit_history"):
        if column in frame:
            contaminated = contaminated | frame[column].astype(bool)
    if "entry_has_bar" in frame:
        contaminated = contaminated | ~frame["entry_has_bar"].astype(bool)
    contaminated = contaminated & label_available
    return {
        "sample_count": rows,
        "label_available_count": available_count,
        "contaminated_count": int(contaminated.sum()),
        "contaminated_ratio": float(contaminated.sum() / available_count) if available_count else 0.0,
        "counts": counts,
        "ratios": ratios,
        "gate_columns": [
            "entry_has_bar",
            "entry_zero_volume_or_amount",
            "entry_one_price_limit_up",
            "entry_open_gap_ge_3pct",
            "st_like_limit_history",
        ],
        "notes": [
            "ST-like is a causal 5% limit-rate proxy, not point-in-time historical ST names.",
            "This audit reports contamination; experiments must decide whether to filter or model it.",
        ],
    }
