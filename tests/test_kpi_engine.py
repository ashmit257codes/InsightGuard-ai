"""
tests/test_kpi_engine.py -- Run with: pytest tests/test_kpi_engine.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kpi_engine import build_kpi_timeseries, compute_trend_metrics, compute_full_rolling_stats


@pytest.fixture
def multi_row_df():
    return pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"],
            "region": ["North", "South", "North", "South"],
            "revenue": [100.0, 50.0, 120.0, 60.0],
        }
    )


class TestBuildKpiTimeseries:
    def test_aggregates_multiple_rows_per_date(self, multi_row_df):
        ts = build_kpi_timeseries(multi_row_df, "date", "revenue", freq="D", agg="sum")
        assert len(ts) == 2
        assert ts.iloc[0] == 150.0
        assert ts.iloc[1] == 180.0

    def test_mean_aggregation(self, multi_row_df):
        ts = build_kpi_timeseries(multi_row_df, "date", "revenue", freq="D", agg="mean")
        assert ts.iloc[0] == 75.0

    def test_missing_date_col_raises(self, multi_row_df):
        with pytest.raises(ValueError):
            build_kpi_timeseries(multi_row_df, "not_a_col", "revenue")

    def test_missing_kpi_col_raises(self, multi_row_df):
        with pytest.raises(ValueError):
            build_kpi_timeseries(multi_row_df, "date", "not_a_col")

    def test_output_is_sorted_chronologically(self):
        df = pd.DataFrame(
            {"date": ["2025-01-03", "2025-01-01", "2025-01-02"], "value": [3, 1, 2]}
        )
        ts = build_kpi_timeseries(df, "date", "value")
        assert list(ts.values) == [1, 2, 3]


class TestComputeTrendMetrics:
    def test_basic_increasing_trend(self):
        s = pd.Series([100.0, 110.0, 120.0, 150.0])
        metrics = compute_trend_metrics(s, kpi_name="Revenue", stable_threshold_pct=5.0)
        assert metrics.current_value == 150.0
        assert metrics.previous_value == 120.0
        assert metrics.trend_direction == "increasing"

    def test_basic_decreasing_trend(self):
        s = pd.Series([100.0, 90.0, 80.0, 50.0])
        metrics = compute_trend_metrics(s, kpi_name="Revenue")
        assert metrics.trend_direction == "decreasing"

    def test_small_change_is_stable(self):
        s = pd.Series([100.0, 101.0, 102.0])
        metrics = compute_trend_metrics(s, kpi_name="Revenue", stable_threshold_pct=5.0)
        assert metrics.trend_direction == "stable"

    def test_zero_previous_value_does_not_crash(self):
        s = pd.Series([0.0, 0.0, 50.0])
        metrics = compute_trend_metrics(s, kpi_name="Revenue")
        assert metrics.pct_change == 100.0

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            compute_trend_metrics(pd.Series([100.0]), kpi_name="Revenue")

    def test_empty_series_raises(self):
        with pytest.raises(ValueError):
            compute_trend_metrics(pd.Series([], dtype=float), kpi_name="Revenue")

    def test_window_size_capped_by_series_length(self):
        s = pd.Series([10.0, 20.0, 30.0])
        metrics = compute_trend_metrics(s, kpi_name="Revenue", window=7)
        assert metrics.window_size == 3


class TestRollingStats:
    def test_returns_expected_columns(self):
        s = pd.Series([10.0, 20.0, 15.0, 25.0, 30.0])
        result = compute_full_rolling_stats(s, window=3)
        assert list(result.columns) == ["actual", "moving_average", "rolling_std"]
        assert len(result) == 5