"""Trade outcome labels for pattern analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LabelConfig


def build_labels(features: pd.DataFrame, config: LabelConfig) -> pd.DataFrame:
    """Build success/failure labels from trade outcomes."""
    rows = []
    for _, row in features.iterrows():
        trade_return = float(row.get("return", 0.0) or 0.0)
        mae = float(row.get("max_adverse_excursion", row.get("hold_min_return", 0.0)) or 0.0)
        mfe = float(row.get("max_favorable_excursion", row.get("hold_max_return", 0.0)) or 0.0)
        first_3d = row.get("hold_first_3d_return", np.nan)
        first_5d = row.get("hold_first_5d_return", np.nan)
        first_10d = row.get("hold_first_10d_return", np.nan)
        drawdown_after_peak = row.get("hold_drawdown_after_peak", np.nan)
        post_20d_best_close = row.get("post_20d_best_close_from_entry", np.nan)
        post_20d_best_high = row.get("post_20d_best_high_from_entry", np.nan)
        post_20d_close = row.get("post_20d_close_from_entry", np.nan)
        post_20d_best_high_improvement = row.get("post_20d_best_high_improvement", np.nan)
        post_20d_best_close_improvement = row.get("post_20d_best_close_improvement", np.nan)
        post_20d_close_improvement = row.get("post_20d_close_improvement", np.nan)
        post_20d_first_close_profit_day = row.get("post_20d_first_close_profit_day", np.nan)
        post_20d_first_high_profit_day = row.get("post_20d_first_high_profit_day", np.nan)
        post_20d_first_close_improve_2pct_day = row.get("post_20d_first_close_improve_2pct_day", np.nan)
        label = "neutral"
        if trade_return >= config.success_return_threshold and mae >= -config.max_adverse_threshold:
            label = "success"
        if trade_return <= config.failure_return_threshold or mae <= -config.hard_loss_threshold:
            label = "failure"

        first_window_return = first_5d if pd.notna(first_5d) else first_3d
        early_confirmed = bool(pd.notna(first_window_return) and float(first_window_return) >= config.quick_confirm_return)
        early_failed = bool(pd.notna(first_window_return) and float(first_window_return) <= -config.hard_loss_threshold)
        had_opportunity = bool(mfe >= config.opportunity_return_threshold)
        faded_after_peak = bool(
            pd.notna(drawdown_after_peak)
            and float(drawdown_after_peak) <= -config.fade_drawdown_threshold
            and trade_return < config.success_return_threshold
        )
        efficient_capture = bool(had_opportunity and trade_return >= config.success_return_threshold and not faded_after_peak)
        is_failed_trade = trade_return <= config.failure_return_threshold
        loss_reducible = bool(
            is_failed_trade
            and pd.notna(post_20d_close_improvement)
            and float(post_20d_close_improvement) >= config.recover_improvement_threshold
        )
        close_turn_profitable = bool(
            is_failed_trade
            and pd.notna(post_20d_best_close)
            and float(post_20d_best_close) > config.recover_profit_threshold
        )
        high_turn_profitable = bool(
            is_failed_trade
            and pd.notna(post_20d_best_high)
            and float(post_20d_best_high) > config.recover_profit_threshold
        )
        recoverable_failure = bool(loss_reducible or close_turn_profitable or high_turn_profitable)

        behavior = "neutral"
        if early_confirmed:
            behavior = "quick_confirm_success"
        elif mae <= -config.hard_loss_threshold and mfe < config.quick_confirm_return:
            behavior = "buy_and_drop_failure"
        elif mfe >= config.quick_confirm_return and trade_return <= config.failure_return_threshold:
            behavior = "spike_and_fade_failure"
        elif trade_return > 0:
            behavior = "slow_success"
        elif trade_return <= config.failure_return_threshold:
            behavior = "sideways_failure"

        rows.append({
            "sample_id": row["sample_id"],
            "symbol": row.get("symbol"),
            "entry_date": row.get("entry_date"),
            "exit_date": row.get("exit_date"),
            "exit_reason": row.get("exit_reason"),
            "return": trade_return,
            "max_favorable_excursion": mfe,
            "max_adverse_excursion": mae,
            "outcome_label": label,
            "binary_success": 1 if label == "success" else 0 if label == "failure" else np.nan,
            "behavior_label": behavior,
            "early_confirmed": early_confirmed,
            "early_failed": early_failed,
            "opportunity_label": "opportunity" if had_opportunity else "no_opportunity",
            "had_opportunity": had_opportunity,
            "fade_label": "fade_after_peak" if faded_after_peak else "held_or_no_peak_fade",
            "faded_after_peak": faded_after_peak,
            "efficient_capture": efficient_capture,
            "is_failed_trade": is_failed_trade,
            "recoverable_failure": recoverable_failure,
            "loss_reducible_after_exit": loss_reducible,
            "close_turn_profitable_after_exit": close_turn_profitable,
            "high_turn_profitable_after_exit": high_turn_profitable,
            "recovery_label": "recoverable_failure" if recoverable_failure else "not_recoverable_failure" if is_failed_trade else "not_failed",
            "post_20d_best_close_from_entry": float(post_20d_best_close) if pd.notna(post_20d_best_close) else np.nan,
            "post_20d_best_high_from_entry": float(post_20d_best_high) if pd.notna(post_20d_best_high) else np.nan,
            "post_20d_close_from_entry": float(post_20d_close) if pd.notna(post_20d_close) else np.nan,
            "post_20d_best_high_improvement": float(post_20d_best_high_improvement) if pd.notna(post_20d_best_high_improvement) else np.nan,
            "post_20d_best_close_improvement": float(post_20d_best_close_improvement) if pd.notna(post_20d_best_close_improvement) else np.nan,
            "post_20d_close_improvement": float(post_20d_close_improvement) if pd.notna(post_20d_close_improvement) else np.nan,
            "post_20d_first_close_profit_day": float(post_20d_first_close_profit_day) if pd.notna(post_20d_first_close_profit_day) else np.nan,
            "post_20d_first_high_profit_day": float(post_20d_first_high_profit_day) if pd.notna(post_20d_first_high_profit_day) else np.nan,
            "post_20d_first_close_improve_2pct_day": float(post_20d_first_close_improve_2pct_day) if pd.notna(post_20d_first_close_improve_2pct_day) else np.nan,
            "first_3d_return": float(first_3d) if pd.notna(first_3d) else np.nan,
            "first_5d_return": float(first_5d) if pd.notna(first_5d) else np.nan,
            "first_10d_return": float(first_10d) if pd.notna(first_10d) else np.nan,
            "drawdown_after_peak": float(drawdown_after_peak) if pd.notna(drawdown_after_peak) else np.nan,
        })
    return pd.DataFrame(rows)
