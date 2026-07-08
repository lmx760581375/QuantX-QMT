"""K-line plotting helpers for trade pattern reports."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle


def write_cluster_kline_figures(
    output_dir: Path,
    bars: pd.DataFrame,
    cluster_rows: pd.DataFrame,
    max_samples_per_group: int = 6,
) -> Dict[str, Dict[str, str]]:
    """Write representative and counter-example K-line grids for each cluster."""
    if bars.empty or cluster_rows.empty:
        return {}
    figures_dir = output_dir / "figures" / "clusters"
    figures_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Dict[str, str]] = {}
    for cluster_id, group in cluster_rows.groupby("cluster_id"):
        success_ids = _sample_ids(group.sort_values("return", ascending=False).head(max_samples_per_group))
        failure_ids = _sample_ids(group.sort_values("return", ascending=True).head(max_samples_per_group))
        cluster_figures: Dict[str, str] = {}
        if success_ids:
            path = figures_dir / f"{cluster_id}_representatives.png"
            plot_kline_grid(bars, success_ids, path, title=f"{cluster_id} representatives")
            cluster_figures["representatives"] = str(path.relative_to(output_dir))
        if failure_ids:
            path = figures_dir / f"{cluster_id}_counter_examples.png"
            plot_kline_grid(bars, failure_ids, path, title=f"{cluster_id} counter examples")
            cluster_figures["counter_examples"] = str(path.relative_to(output_dir))
        result[str(cluster_id)] = cluster_figures
    return result


def plot_kline_grid(bars: pd.DataFrame, sample_ids: Iterable[str], output_path: Path, title: str = "") -> Path:
    sample_ids = list(sample_ids)
    if not sample_ids:
        raise ValueError("sample_ids must not be empty")
    cols = min(3, len(sample_ids))
    rows = (len(sample_ids) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.8, rows * 3.4), squeeze=False)
    if title:
        fig.suptitle(title, fontsize=13)
    for ax in axes.flat:
        ax.axis("off")
    for ax, sample_id in zip(axes.flat, sample_ids):
        sample = bars[bars["sample_id"] == sample_id].sort_values("offset_from_entry")
        _plot_single_kline(ax, sample)
    fig.tight_layout(rect=(0, 0, 1, 0.96) if title else None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    return output_path


def _plot_single_kline(ax, sample: pd.DataFrame) -> None:
    ax.axis("on")
    if sample.empty:
        ax.text(0.5, 0.5, "empty", ha="center", va="center")
        return
    data = sample.copy()
    data["x"] = range(len(data))
    volume_ax = ax.twinx()
    volume_ax.set_yticks([])
    volume_ax.set_ylim(0, max(float(data["volume"].max()) * 4, 1.0))
    for _, row in data.iterrows():
        x = float(row["x"])
        open_price = float(row["open"])
        close_price = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        color = "#d94738" if close_price >= open_price else "#2f8f5b"
        ax.vlines(x, low, high, color=color, linewidth=0.8)
        bottom = min(open_price, close_price)
        height = max(abs(close_price - open_price), max(high - low, 1e-6) * 0.015)
        ax.add_patch(Rectangle((x - 0.28, bottom), 0.56, height, facecolor=color, edgecolor=color, linewidth=0.8))
        volume_ax.bar(x, float(row.get("volume", 0.0)), width=0.5, color=color, alpha=0.18)
    entry_rows = data[data["offset_from_entry"] == 0]
    exit_rows = data[data["offset_from_exit"] == 0]
    if not entry_rows.empty:
        ax.axvline(float(entry_rows.iloc[0]["x"]), color="#1f5fbf", linewidth=1.0, linestyle="--")
    if not exit_rows.empty:
        ax.axvline(float(exit_rows.iloc[0]["x"]), color="#7a3db8", linewidth=1.0, linestyle=":")
    symbol = str(data.iloc[0].get("symbol", ""))
    entry_date = pd.Timestamp(data.iloc[0].get("entry_date")).strftime("%Y-%m-%d")
    trade_return = _sample_return(data)
    ax.set_title(f"{symbol} {entry_date} {_pct(trade_return)}", fontsize=9)
    ax.grid(True, axis="y", alpha=0.18)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_xlim(-1, len(data))
    ticks = [0, len(data) // 2, len(data) - 1] if len(data) > 2 else list(range(len(data)))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(data.iloc[i]["offset_from_entry"])) for i in ticks])


def _sample_ids(frame: pd.DataFrame) -> List[str]:
    return [str(value) for value in frame.get("sample_id", pd.Series(dtype=str)).dropna().tolist()]


def _sample_return(sample: pd.DataFrame) -> float | None:
    entry = sample[sample["offset_from_entry"] == 0]
    exit_row = sample[sample["offset_from_exit"] == 0]
    if entry.empty or exit_row.empty:
        return None
    entry_close = float(entry.iloc[0]["close"])
    exit_close = float(exit_row.iloc[0]["close"])
    return exit_close / entry_close - 1 if entry_close else None


def _pct(value: float | None) -> str:
    return "" if value is None else f"{value * 100:.1f}%"
