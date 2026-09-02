"""
tests/test_anomaly_detection.py -- Run with: pytest tests/test_anomaly_detection.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anomaly_detection import (
    z_score_flags,
    iqr_flags,
    rolling_baseline_flags,
    detect_anomalies,
    evaluate_detector,
    compare_methods,
)


class TestZScoreFlags:
    def test_flags_obvious_outlier(self):
        values = pd.Series([10, 11, 9, 10, 12, 10, 100])
        flags = z_score_flags(values, threshold=2.0)
        assert flags.iloc[-1] == True
        assert flags.iloc[:-1].sum() == 0

    def test_constant_series_flags_nothing(self):
        values = pd.Series([5, 5, 5, 5, 5])
        flags = z_score_flags(values)
        assert flags.sum() == 0


class TestIqrFlags:
    def test_flags_obvious_outlier(self):
        values = pd.Series([10, 11, 9, 10, 12, 10, 500])
        flags = iqr_flags(values)
        assert flags.iloc[-1] == True

    def test_constant_series_flags_nothing(self):
        values = pd.Series([5, 5, 5, 5, 5])
        flags = iqr_flags(values)
        assert flags.sum() == 0


class TestRollingBaselineFlags:
    def test_does_not_use_current_point_in_its_own_baseline(self):
        # Regression test for look-ahead bias: if the rolling window
        # included the anomalous point itself, a huge spike could inflate
        # its own baseline and mask itself.
        values = pd.Series([10.0] * 10 + [1000.0])
        flags = rolling_baseline_flags(values, window=7, threshold_std=3.0)
        assert flags.iloc[-1] == True

    def test_insufficient_history_does_not_crash(self):
        values = pd.Series([10.0, 12.0])
        flags = rolling_baseline_flags(values, window=7)
        assert flags.sum() == 0
        assert not flags.isna().any()


class TestDetectAnomaliesPerGroup:
    def test_runs_independently_per_group(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=10).tolist() * 2,
                "group": ["A"] * 10 + ["B"] * 10,
                "value": [10.0] * 9 + [500.0] + [10.0] * 10,
            }
        )
        flags = detect_anomalies(df, "value", "date", ["group"], method="z_score", threshold=2.0)
        assert flags.iloc[9] == True
        assert flags.iloc[10:].sum() == 0

    def test_invalid_method_raises(self):
        df = pd.DataFrame({"date": [1, 2], "group": ["A", "A"], "value": [1.0, 2.0]})
        with pytest.raises(ValueError):
            detect_anomalies(df, "value", "date", ["group"], method="not_a_real_method")


class TestEvaluateDetector:
    def test_perfect_detection(self):
        y_true = pd.Series([False, True, False, True])
        y_pred = pd.Series([False, True, False, True])
        result = evaluate_detector(y_true, y_pred)
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_no_predictions_gives_zero_precision_not_crash(self):
        y_true = pd.Series([False, True, False])
        y_pred = pd.Series([False, False, False])
        result = evaluate_detector(y_true, y_pred)
        assert result.precision == 0.0
        assert result.recall == 0.0

    def test_false_positives_reduce_precision(self):
        y_true = pd.Series([False, False, True])
        y_pred = pd.Series([True, True, True])
        result = evaluate_detector(y_true, y_pred)
        assert result.true_positives == 1
        assert result.false_positives == 2
        assert result.precision == pytest.approx(1 / 3)


class TestCompareMethods:
    def test_returns_all_methods_sorted_by_f1(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=20).tolist(),
                "group": ["A"] * 20,
                "value": [10.0] * 19 + [500.0],
                "is_anomaly": [False] * 19 + [True],
            }
        )
        result = compare_methods(df, "value", "date", ["group"], "is_anomaly")
        assert len(result) == 3
        assert result["f1"].is_monotonic_decreasing