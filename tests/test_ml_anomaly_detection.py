"""
tests/test_ml_anomaly_detection.py
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ml_anomaly_detection import isolation_forest_flags, lof_flags, detect_ml_anomalies, compare_all_methods
from severity_scoring import compute_persistence, compute_severity_scores, SeverityThresholds


class TestIsolationForestFlags:
    def test_flags_obvious_outlier(self):
        values = pd.Series([10.0, 11, 9, 10, 12, 10, 11, 9, 10, 500])
        flags = isolation_forest_flags(values, contamination=0.1)
        assert flags.iloc[-1] == True

    def test_too_few_points_returns_no_flags(self):
        values = pd.Series([1.0, 2.0, 3.0])
        flags = isolation_forest_flags(values)
        assert flags.sum() == 0


class TestLofFlags:
    def test_too_few_points_returns_no_flags(self):
        values = pd.Series([1.0, 2.0])
        flags = lof_flags(values, n_neighbors=20)
        assert flags.sum() == 0


class TestComputePersistence:
    def test_counts_consecutive_streaks_and_resets(self):
        df = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=6),
            "region": ["A"] * 6,
        })
        flags = pd.Series([False, True, True, True, False, True])
        result = compute_persistence(df, "date", ["region"], flags)
        assert list(result) == [0, 1, 2, 3, 0, 1]

    def test_handles_interleaved_groups_correctly(self):
        # Regression test: rows for the same group are NOT adjacent in the
        # dataframe (mirrors real data where each date has one row per
        # group). Persistence must still track each group's own streak
        # correctly despite the interleaving.
        df = pd.DataFrame({
            "date": ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"],
            "region": ["A", "B", "A", "B", "A", "B"],
        })
        # Group A flagged on all 3 days; Group B flagged on none
        flags = pd.Series([True, False, True, False, True, False])
        result = compute_persistence(df, "date", ["region"], flags)
        # Group A rows (indices 0, 2, 4) should show streak 1, 2, 3
        assert list(result.loc[[0, 2, 4]]) == [1, 2, 3]
        # Group B rows (indices 1, 3, 5) should show streak 0, 0, 0
        assert list(result.loc[[1, 3, 5]]) == [0, 0, 0]


class TestSeverityScoring:
    def test_only_flagged_rows_returned(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=5),
                "region": ["North"] * 5,
                "revenue": [100.0, 100.0, 100.0, 100.0, 500.0],
            }
        )
        flags = pd.Series([False, False, False, False, True])
        result = compute_severity_scores(df, "revenue", "date", ["region"], flags)
        assert len(result) == 1
        assert result["severity_score"].iloc[0] > 0

    def test_severity_escalates_with_persistence(self):
        # A sustained multi-day anomaly should show INCREASING severity as
        # persistence climbs -- this was broken before the interleaved-group
        # fix (persistence silently stayed at 1 for every flagged row).
        df = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=5),
                "region": ["North"] * 5,
                "revenue": [100.0, 50.0, 50.0, 50.0, 100.0],
            }
        )
        flags = pd.Series([False, True, True, True, False])
        result = compute_severity_scores(df, "revenue", "date", ["region"], flags)
        scores = result["severity_score"].tolist()
        assert scores[0] < scores[1] < scores[2]  # strictly increasing