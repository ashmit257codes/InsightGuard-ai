"""
kpi_engine.py -- Deterministic KPI aggregation and trend analysis.
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np


def build_kpi_timeseries(
    df: pd.DataFrame,
    date_col: str,
    kpi_col: str,
    freq: str = "D",
    agg: str = "sum",
) -> pd.Series:
    """
    Aggregates `kpi_col` into a single time series indexed by date.
    """
    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not found in dataframe")
    if kpi_col not in df.columns:
        raise ValueError(f"kpi_col '{kpi_col}' not found in dataframe")

    working = df[[date_col, kpi_col]].copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce", format="mixed")
    working = working.dropna(subset=[date_col])

    grouped = working.groupby(date_col)[kpi_col]
    daily = grouped.sum() if agg == "sum" else grouped.mean()

    if freq != "D":
        daily = daily.resample(freq).sum() if agg == "sum" else daily.resample(freq).mean()

    return daily.sort_index()


@dataclass
class TrendMetrics:
    kpi_name: str
    current_value: float
    previous_value: float
    absolute_change: float
    pct_change: float
    moving_average: float
    rolling_volatility: float
    trend_direction: str
    window_size: int


def compute_trend_metrics(
    series: pd.Series,
    kpi_name: str = "value",
    window: int = 7,
    stable_threshold_pct: float = 5.0,
) -> TrendMetrics:
    """
    Computes trend metrics comparing the most recent value against the
    previous period, plus a rolling moving average and volatility measure.
    """
    clean = series.dropna()
    if len(clean) < 2:
        raise ValueError(
            f"Need at least 2 data points to compute trend metrics, got {len(clean)}"
        )

    current_value = float(clean.iloc[-1])
    previous_value = float(clean.iloc[-2])

    absolute_change = current_value - previous_value

    if previous_value == 0:
        pct_change = 100.0 if current_value > 0 else 0.0
    else:
        pct_change = (absolute_change / abs(previous_value)) * 100

    effective_window = min(window, len(clean))
    moving_average = float(clean.tail(effective_window).mean())

    window_std = float(clean.tail(effective_window).std())
    window_mean = float(clean.tail(effective_window).mean())
    rolling_volatility = (window_std / window_mean * 100) if window_mean != 0 else 0.0

    if abs(pct_change) < stable_threshold_pct:
        trend_direction = "stable"
    elif pct_change > 0:
        trend_direction = "increasing"
    else:
        trend_direction = "decreasing"

    return TrendMetrics(
        kpi_name=kpi_name,
        current_value=current_value,
        previous_value=previous_value,
        absolute_change=absolute_change,
        pct_change=pct_change,
        moving_average=moving_average,
        rolling_volatility=rolling_volatility,
        trend_direction=trend_direction,
        window_size=effective_window,
    )


def compute_full_rolling_stats(series: pd.Series, window: int = 7) -> pd.DataFrame:
    """
    Returns the FULL rolling moving-average and rolling-std as time series.
    """
    clean = series.dropna()
    return pd.DataFrame(
        {
            "actual": clean,
            "moving_average": clean.rolling(window=window, min_periods=1).mean(),
            "rolling_std": clean.rolling(window=window, min_periods=1).std(),
        }
    )


TREND_ICONS = {
    "increasing": "📈",
    "decreasing": "📉",
    "stable": "➡️",
}