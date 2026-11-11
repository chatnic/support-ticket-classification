"""Confidence-based routing: send confident predictions straight to a queue, the rest to triage."""

from __future__ import annotations

import numpy as np
import pandas as pd


def routing_curve(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray,
                  thresholds: np.ndarray | None = None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.arange(0.30, 0.991, 0.01), 2)
    conf = proba.max(axis=1)
    pred = classes[proba.argmax(axis=1)]
    rows = []
    for t in thresholds:
        auto = conf >= t
        rows.append({
            "threshold": t,
            "automation_rate": float(auto.mean()),
            "auto_routed_accuracy": float((pred[auto] == y_true[auto]).mean()) if auto.any() else np.nan,
        })
    return pd.DataFrame(rows)


def pick_threshold(curve: pd.DataFrame, target_accuracy: float = 0.97) -> pd.Series:
    """Lowest threshold (most automation) whose auto-routed accuracy meets the target."""
    ok = curve[curve["auto_routed_accuracy"] >= target_accuracy]
    return (ok.iloc[0] if len(ok) else curve.iloc[-1])
