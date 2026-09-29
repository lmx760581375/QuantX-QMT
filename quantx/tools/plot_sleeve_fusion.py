"""绘制两个独立 QuantX sleeve 的静态组合图表。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import PercentFormatter


DEFAULT_REWARD_RUN = "20260921_rewardh7_h15_only_v1"
DEFAULT_WTS_RUN = "20260921_wts_pos5_common_cost_v1"
DEFAULT_WTS_WEIGHT = 0.50


def _read_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return [dict(row) for row in value]


def load_daily_nav(run_dir: str | Path) -> pd.DataFrame:
    """读取并规范化一个 run 的日净值。"""
    path = Path(run_dir) / "daily_nav.json"
    rows = _read_json(path)
    frame = pd.DataFrame(rows)
    required = {"date", "total_value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["total_value"] = pd.to_numeric(frame["total_value"], errors="raise")
    frame = frame.dropna(subset=["date", "total_value"]).sort_values("date")
    if frame.empty:
        raise ValueError(f"No usable daily NAV rows: {path}")
    if frame["date"].duplicated().any():
        raise ValueError(f"Duplicate NAV dates: {path}")
    return frame[["date", "total_value"]].reset_index(drop=True)


def build_static_sleeve_nav(
    reward_nav: pd.DataFrame,
    wts_nav: pd.DataFrame,
    *,
    wts_weight: float,
) -> pd.DataFrame:
    """按初始资金固定比例合成，不进行再平衡或资金互转。"""
    if not 0.0 <= float(wts_weight) <= 1.0:
        raise ValueError("wts_weight must be within [0, 1]")
    reward = reward_nav.rename(columns={"total_value": "reward_value"})
    wts = wts_nav.rename(columns={"total_value": "wts_value"})
    frame = reward.merge(wts, on="date", how="inner", validate="one_to_one")
    if frame.empty:
        raise ValueError("Reward and weak-to-strong NAVs have no common dates")
    frame = frame.sort_values("date").reset_index(drop=True)
    reward_start = float(frame["reward_value"].iloc[0])
    wts_start = float(frame["wts_value"].iloc[0])
    if reward_start <= 0.0 or wts_start <= 0.0:
        raise ValueError("Initial sleeve NAV must be positive")
    reward_weight = 1.0 - float(wts_weight)
    frame["reward_normalized"] = frame["reward_value"] / reward_start
    frame["wts_normalized"] = frame["wts_value"] / wts_start
    frame["portfolio_normalized"] = (
        reward_weight * frame["reward_normalized"]
        + float(wts_weight) * frame["wts_normalized"]
    )
    frame["portfolio_return"] = frame["portfolio_normalized"] - 1.0
    frame["portfolio_drawdown"] = (
        frame["portfolio_normalized"] / frame["portfolio_normalized"].cummax() - 1.0
    )
    return frame


def monthly_returns(portfolio_nav: pd.DataFrame) -> pd.DataFrame:
    """按月末净值计算月收益，首月以首个有效日净值作为起点。"""
    frame = portfolio_nav[["date", "portfolio_normalized"]].copy()
    frame = frame.sort_values("date")
    frame["month"] = frame["date"].dt.to_period("M")
    rows: list[dict[str, Any]] = []
    previous_value: float | None = None
    for month, group in frame.groupby("month", sort=True):
        start_value = previous_value if previous_value is not None else float(group["portfolio_normalized"].iloc[0])
        end_value = float(group["portfolio_normalized"].iloc[-1])
        rows.append({
            "month": month.to_timestamp(),
            "monthly_return": end_value / start_value - 1.0 if start_value > 0.0 else 0.0,
        })
        previous_value = end_value
    return pd.DataFrame(rows)


def calendar_month_win_rates(monthly_return_frame: pd.DataFrame) -> pd.DataFrame:
    """按自然月统计收红概率：例如所有一月中收益为正的比例。"""
    required = {"month", "monthly_return"}
    missing = required.difference(monthly_return_frame.columns)
    if missing:
        raise ValueError(f"Monthly return frame is missing columns: {sorted(missing)}")
    frame = monthly_return_frame.copy()
    frame["month"] = pd.to_datetime(frame["month"], errors="raise")
    frame["month_number"] = frame["month"].dt.month
    frame["is_red_month"] = frame["monthly_return"] > 0.0
    grouped = frame.groupby("month_number", as_index=False).agg(
        observed_months=("is_red_month", "size"),
        red_months=("is_red_month", "sum"),
    )
    grouped["red_months"] = grouped["red_months"].astype(int)
    grouped["win_rate"] = grouped["red_months"] / grouped["observed_months"]
    return grouped.set_index("month_number").reindex(range(1, 13)).reset_index()


def monthly_return_heatmap(monthly_return_frame: pd.DataFrame) -> pd.DataFrame:
    """将月收益整理为年份×月份矩阵，缺失月保留 NaN。"""
    required = {"month", "monthly_return"}
    missing = required.difference(monthly_return_frame.columns)
    if missing:
        raise ValueError(f"Monthly return frame is missing columns: {sorted(missing)}")
    frame = monthly_return_frame.copy()
    frame["month"] = pd.to_datetime(frame["month"], errors="raise")
    frame["year"] = frame["month"].dt.year
    frame["month_number"] = frame["month"].dt.month
    matrix = frame.pivot(
        index="year",
        columns="month_number",
        values="monthly_return",
    )
    return matrix.reindex(columns=range(1, 13))


def load_closed_positions(run_dir: str | Path) -> list[dict[str, Any]]:
    return _read_json(Path(run_dir) / "closed_positions.json")


def configure_chinese_font() -> None:
    """尽量采用本机已有中文字体，缺失时保持 Matplotlib 默认字体。"""
    candidates = ["Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei"]
    available = {font.name for font in matplotlib.font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _format_month_axis(axis) -> None:
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in axis.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")


def write_figures(
    portfolio_nav: pd.DataFrame,
    monthly_return_frame: pd.DataFrame,
    monthly_win_rate_frame: pd.DataFrame,
    *,
    output_dir: str | Path,
    wts_weight: float,
    reward_label: str = "Reward H15",
    wts_label: str = "Weak-to-Strong Top-5",
) -> dict[str, Path]:
    """生成四张 PNG 图，返回逻辑名称到图片路径的映射。"""
    configure_chinese_font()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reward_weight = 1.0 - float(wts_weight)
    paths = {
        "equity_curve": output / "01_收益曲线.png",
        "drawdown": output / "02_回撤曲线.png",
        "monthly_returns": output / "03_月度收益.png",
        "monthly_win_rate": output / "04_月度胜率.png",
    }

    fig, axis = plt.subplots(figsize=(14, 6))
    axis.plot(
        portfolio_nav["date"],
        portfolio_nav["reward_normalized"] - 1.0,
        color="#d62728",
        alpha=0.65,
        linewidth=1.3,
        label=reward_label,
    )
    axis.plot(
        portfolio_nav["date"],
        portfolio_nav["wts_normalized"] - 1.0,
        color="#1f77b4",
        alpha=0.65,
        linewidth=1.3,
        label=wts_label,
    )
    axis.plot(
        portfolio_nav["date"],
        portfolio_nav["portfolio_return"],
        color="#111111",
        linewidth=2.2,
        label=f"Static Portfolio (Reward {reward_weight:.0%} / WTS {wts_weight:.0%})",
    )
    axis.axhline(0.0, color="#888888", linewidth=0.8)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_title("Cumulative Return")
    axis.set_ylabel("Cumulative Return")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.22)
    _format_month_axis(axis)
    fig.tight_layout()
    fig.savefig(paths["equity_curve"], dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(14, 5))
    axis.fill_between(
        portfolio_nav["date"],
        portfolio_nav["portfolio_drawdown"],
        0.0,
        color="#2ca02c",
        alpha=0.35,
    )
    axis.plot(portfolio_nav["date"], portfolio_nav["portfolio_drawdown"], color="#1b7f32", linewidth=1.2)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_title("Portfolio Drawdown")
    axis.set_ylabel("Drawdown")
    axis.grid(alpha=0.22)
    _format_month_axis(axis)
    fig.tight_layout()
    fig.savefig(paths["drawdown"], dpi=150)
    plt.close(fig)

    heatmap = monthly_return_heatmap(monthly_return_frame)
    values = heatmap.to_numpy(dtype=float)
    max_abs = max(float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 0.01, 0.01)
    cmap = LinearSegmentedColormap.from_list(
        "red_green_return",
        ["#1b7f32", "#ffffff", "#c51f1a"],
    )
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    fig, axis = plt.subplots(figsize=(14, max(4.5, 0.72 * len(heatmap) + 1.8)))
    image = axis.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    axis.set_xticks(
        range(12),
        labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    )
    axis.set_yticks(range(len(heatmap.index)), labels=[str(year) for year in heatmap.index])
    axis.set_title("Monthly Return Heatmap (Red Positive / Green Negative)")
    axis.set_xlabel("Month")
    axis.set_ylabel("Year")
    for row_index, year in enumerate(heatmap.index):
        for column_index, month in enumerate(heatmap.columns):
            value = heatmap.loc[year, month]
            if pd.notna(value):
                text_color = "white" if abs(float(value)) >= max_abs * 0.52 else "#222222"
                axis.text(
                    column_index,
                    row_index,
                    f"{float(value):.1%}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                )
    colorbar = fig.colorbar(image, ax=axis, pad=0.015)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    colorbar.set_label("Monthly Return")
    fig.tight_layout()
    fig.savefig(paths["monthly_returns"], dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5))
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    colors = [
        "#d62728" if value >= 0.5 else "#2ca02c"
        for value in monthly_win_rate_frame["win_rate"].fillna(0.0)
    ]
    bars = axis.bar(
        month_names,
        monthly_win_rate_frame["win_rate"],
        color=colors,
    )
    for bar, observed, red in zip(
        bars,
        monthly_win_rate_frame["observed_months"],
        monthly_win_rate_frame["red_months"],
    ):
        if pd.notna(observed):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                float(bar.get_height()) + 0.02,
                f"{int(red)}/{int(observed)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_ylim(0.0, 1.12)
    axis.set_title("Calendar-Month Win Rate (Positive Monthly Return)")
    axis.set_ylabel("Positive-Month Rate")
    axis.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(paths["monthly_win_rate"], dpi=150)
    plt.close(fig)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="绘制 Reward H15 与弱转强静态 sleeve 组合图")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--reward-run", default=DEFAULT_REWARD_RUN)
    parser.add_argument("--wts-run", default=DEFAULT_WTS_RUN)
    parser.add_argument("--wts-weight", type=float, default=DEFAULT_WTS_WEIGHT)
    parser.add_argument("--reward-label", default="Reward H15")
    parser.add_argument("--wts-label", default="Weak-to-Strong Top-5")
    parser.add_argument(
        "--output-dir",
        default="artifacts/sleeve_fusion/reward_h15_wts_pos5_static_50_50",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs_root = Path(args.runs_root)
    reward_dir = runs_root / args.reward_run
    wts_dir = runs_root / args.wts_run
    portfolio_nav = build_static_sleeve_nav(
        load_daily_nav(reward_dir),
        load_daily_nav(wts_dir),
        wts_weight=args.wts_weight,
    )
    month_return = monthly_returns(portfolio_nav)
    month_win_rate = calendar_month_win_rates(month_return)
    output_dir = Path(args.output_dir)
    paths = write_figures(
        portfolio_nav,
        month_return,
        month_win_rate,
        output_dir=output_dir,
        wts_weight=args.wts_weight,
        reward_label=args.reward_label,
        wts_label=args.wts_label,
    )
    month_return["is_red_month"] = month_return["monthly_return"] > 0.0
    merged_months = month_return.copy()
    merged_months.to_csv(output_dir / "monthly_metrics.csv", index=False, encoding="utf-8-sig")
    summary = {
        "reward_run": args.reward_run,
        "wts_run": args.wts_run,
        "reward_weight": 1.0 - args.wts_weight,
        "wts_weight": args.wts_weight,
        "start_date": portfolio_nav["date"].iloc[0].strftime("%Y-%m-%d"),
        "end_date": portfolio_nav["date"].iloc[-1].strftime("%Y-%m-%d"),
        "method": "initial-capital static sleeves; no rebalancing or cross-sleeve cash transfer",
        "monthly_win_rate_definition": "positive monthly return; calendar-month chart aggregates positive-month frequency across available years",
        "figures": {name: str(path) for name, path in paths.items()},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "output_dir": str(output_dir), **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
