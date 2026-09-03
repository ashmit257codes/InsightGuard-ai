"""
severity_scoring.py -- Assigns a 0-100 severity score and LOW/MEDIUM/HIGH/
CRITICAL label to flagged anomalies, combining:
  1. Deviation magnitude -- how far the value is from its group's normal
     baseline, as a percentage.
  2. Persistence -- how many CONSECUTIVE anomalous days precede this one
     within the same group's chronological sequence.
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class SeverityThresholds:
    low_max: float = 30
    medium_max: float = 60
    high_max: float = 80


def _label_from_score(score: float, thresholds: SeverityThresholds) -> str:
    if score <= thresholds.low_max:
        return "LOW"
    elif score <= thresholds.medium_max:
        return "MEDIUM"
    elif score <= thresholds.high_max:
        return "HIGH"
    return "CRITICAL"


def compute_persistence(
    df: pd.DataFrame, date_col: str, group_cols: list, anomaly_flags: pd.Series
) -> pd.Series:
    """
    For each row, counts consecutive True values preceding it (inclusive)
    within its OWN group's chronological sequence.

    IMPORTANT: must sort and iterate WITHIN each group separately. Real
    datasets interleave groups row-by-row (e.g. one row per region+category
    for every date), so consecutive rows in the raw dataframe usually
    belong to DIFFERENT groups. Iterating naively over df.index resets the
    streak on almost every row -- a bug that doesn't crash, it just quietly
    produces wrong (always-1) persistence counts.
    """
    persistence = pd.Series(0, index=df.index)
    group_key = df[group_cols].astype(str).agg("_".join, axis=1)
    sorted_df = df.sort_values(date_col)

    for _, group_df in sorted_df.groupby(group_key.loc[sorted_df.index]):
        streak = 0
        for idx in group_df.index:
            if anomaly_flags.loc[idx]:
                streak += 1
            else:
                streak = 0
            persistence.loc[idx] = streak

    return persistence


def compute_severity_scores(
    df: pd.DataFrame,
    value_col: str,
    date_col: str,
    group_cols: list,
    anomaly_flags: pd.Series,
    thresholds: SeverityThresholds | None = None,
) -> pd.DataFrame:
    """
    Computes deviation %, persistence, severity score, and severity label
    for every row where anomaly_flags is True.

    severity_score formula (0-100, capped):
        abs(deviation_pct) * 0.6 + persistence_days * 15
    This weights magnitude and persistence together -- a huge one-day spike
    and a moderate multi-day dip can both reach HIGH/CRITICAL.
    """
    thresholds = thresholds or SeverityThresholds()

    group_key = df[group_cols].astype(str).agg("_".join, axis=1)
    group_means = df.groupby(group_key)[value_col].transform("mean")

    deviation_pct = ((df[value_col] - group_means) / group_means.replace(0, np.nan)) * 100
    deviation_pct = deviation_pct.fillna(0)

    persistence = compute_persistence(df, date_col, group_cols, anomaly_flags)

    severity_score = (deviation_pct.abs() * 0.6 + persistence * 15).clip(upper=100)
    severity_label = severity_score.apply(lambda s: _label_from_score(s, thresholds))

    result = pd.DataFrame(
        {
            "deviation_pct": deviation_pct.round(1),
            "persistence_days": persistence,
            "severity_score": severity_score.round(1),
            "severity_label": severity_label,
        },
        index=df.index,
    )

    return result[anomaly_flags]