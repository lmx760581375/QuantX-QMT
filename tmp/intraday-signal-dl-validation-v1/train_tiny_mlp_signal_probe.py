"""Tiny MLP smoke test for cached 5m intraday proxies.

The goal is not to tune a production deep model. It answers a narrower question:
with a chronological split, do cached 5m path features add predictive signal over
daily features for short-horizon labels?
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "cached_5m_signal_panel.csv"
OUT_JSON = ROOT / "tiny_mlp_signal_probe_summary.json"

FEATURES_DAILY = [
    "daily_ret",
    "daily_high_ret",
    "daily_close_from_high",
    "daily_close_vs_vwap",
    "daily_volume_z20",
    "daily_amount_z20",
]
FEATURES_MINUTE = [
    "m_close_ret",
    "m_high_ret",
    "m_low_ret",
    "m_close_from_high",
    "m_vwap_support",
    "m_tail_ret_30m",
    "m_tail_volume_share_30m",
    "m_volume_concentration_top20pct",
    "m_first_high_frac",
    "m_last_bar_ret",
    "m_near_limit_ratio",
    "m_limit_touch_flag",
    "m_break_proxy_count",
]
LABELS = ["next_open_to_high_ret", "next_close_ret"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


class MLP(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        width = min(32, max(8, n_features * 2))
        self.net = nn.Sequential(
            nn.Linear(n_features, width),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(width, width // 2),
            nn.ReLU(),
            nn.Linear(width // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    sa = pd.Series(a).rank().to_numpy(dtype=float)
    sb = pd.Series(b).rank().to_numpy(dtype=float)
    if np.std(sa) <= 0 or np.std(sb) <= 0:
        return 0.0
    return float(np.corrcoef(sa, sb)[0, 1])


def top_bottom_spread(frame: pd.DataFrame, label: str) -> dict[str, float]:
    spreads = []
    for _, group in frame.groupby("date"):
        if len(group) < 5:
            continue
        ranked = group.sort_values("pred")
        k = max(1, len(ranked) // 5)
        spreads.append(float(ranked.tail(k)[label].mean() - ranked.head(k)[label].mean()))
    if not spreads:
        return {"mean": 0.0, "hit_rate": 0.0, "n": 0}
    arr = np.asarray(spreads, dtype=float)
    return {"mean": float(arr.mean()), "hit_rate": float((arr > 0).mean()), "n": int(len(arr))}


def prepare(frame: pd.DataFrame, features: list[str], label: str) -> dict[str, Any]:
    cols = ["date", "symbol", label] + features
    data = frame[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    dates = sorted(data["date"].unique())
    split = dates[int(len(dates) * 0.6)]
    train = data[data["date"] < split].copy()
    test = data[data["date"] >= split].copy()
    x_train = train[features].to_numpy(dtype=np.float32)
    x_test = test[features].to_numpy(dtype=np.float32)
    y_train = train[label].to_numpy(dtype=np.float32)
    y_test = test[label].to_numpy(dtype=np.float32)
    mu = np.nanmean(x_train, axis=0)
    sigma = np.nanstd(x_train, axis=0)
    sigma[sigma == 0] = 1.0
    x_train = np.nan_to_num((x_train - mu) / sigma).astype(np.float32)
    x_test = np.nan_to_num((x_test - mu) / sigma).astype(np.float32)
    y_mu = float(y_train.mean())
    y_sigma = float(y_train.std()) or 1.0
    y_train_z = ((y_train - y_mu) / y_sigma).astype(np.float32)
    return {
        "split": split,
        "train": train,
        "test": test,
        "x_train": x_train,
        "x_test": x_test,
        "y_train_z": y_train_z,
        "y_test": y_test,
        "y_mu": y_mu,
        "y_sigma": y_sigma,
    }


def train_once(frame: pd.DataFrame, features: list[str], label: str, seed: int) -> dict[str, Any]:
    set_seed(seed)
    data = prepare(frame, features, label)
    x = torch.from_numpy(data["x_train"])
    y = torch.from_numpy(data["y_train_z"])
    model = MLP(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    loss_fn = nn.SmoothL1Loss()
    best_loss = math.inf
    best_state = None
    patience = 0
    # Small full-batch training is enough for this smoke test and avoids dataloader noise.
    for _ in range(220):
        model.train()
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        loss_value = float(loss.detach())
        if loss_value + 1e-6 < best_loss:
            best_loss = loss_value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 35:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_z = model(torch.from_numpy(data["x_test"])).numpy()
    pred = pred_z * data["y_sigma"] + data["y_mu"]
    scored = data["test"][["date", "symbol", label]].copy()
    scored["pred"] = pred
    spread = top_bottom_spread(scored, label)
    return {
        "seed": seed,
        "split_date": str(data["split"]),
        "train_rows": int(len(data["train"])),
        "test_rows": int(len(data["test"])),
        "test_spearman": spearman(pred, data["y_test"]),
        "test_top_bottom_spread_mean": spread["mean"],
        "test_top_bottom_spread_hit_rate": spread["hit_rate"],
        "test_days": spread["n"],
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["test_spearman", "test_top_bottom_spread_mean", "test_top_bottom_spread_hit_rate"]
    out = {"runs": runs, "n_runs": len(runs)}
    for key in keys:
        arr = np.asarray([run[key] for run in runs], dtype=float)
        out[key + "_mean"] = float(arr.mean())
        out[key + "_std"] = float(arr.std())
        out[key + "_min"] = float(arr.min())
        out[key + "_max"] = float(arr.max())
    out["train_rows"] = runs[0]["train_rows"]
    out["test_rows"] = runs[0]["test_rows"]
    out["test_days"] = runs[0]["test_days"]
    out["split_date"] = runs[0]["split_date"]
    return out


def main() -> None:
    frame = pd.read_csv(PANEL)
    seeds = [20260714, 20260715, 20260716]
    results: dict[str, Any] = {}
    for label in LABELS:
        for name, features in {
            "daily_only": FEATURES_DAILY,
            "daily_plus_minute": FEATURES_DAILY + FEATURES_MINUTE,
        }.items():
            runs = [train_once(frame, features, label, seed) for seed in seeds]
            results[f"{name}_{label}"] = summarize_runs(runs)
    comparisons = {}
    for label in LABELS:
        base = results[f"daily_only_{label}"]
        plus = results[f"daily_plus_minute_{label}"]
        comparisons[label] = {
            "spearman_delta": plus["test_spearman_mean"] - base["test_spearman_mean"],
            "spread_delta": plus["test_top_bottom_spread_mean_mean"] - base["test_top_bottom_spread_mean_mean"],
            "hit_rate_delta": plus["test_top_bottom_spread_hit_rate_mean"] - base["test_top_bottom_spread_hit_rate_mean"],
        }
    verdict = "mlp_no_clear_minute_increment"
    if comparisons["next_open_to_high_ret"]["spearman_delta"] > 0.02 and comparisons["next_open_to_high_ret"]["spread_delta"] > 0.002:
        verdict = "mlp_minute_increment_for_next_day_intraday_high_only"
    out = {
        "experiment": "tiny_mlp_cached_5m_signal_probe_v1",
        "panel": str(PANEL),
        "features_daily": FEATURES_DAILY,
        "features_minute": FEATURES_MINUTE,
        "labels": LABELS,
        "results": results,
        "comparisons": comparisons,
        "verdict": verdict,
        "caveat": "This is a small chronological smoke test, not a deployable deep learning result. The sample has cached 5m bars only, no historical bid/ask queue or true Level2 data.",
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT_JSON), "verdict": verdict, "comparisons": comparisons}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
