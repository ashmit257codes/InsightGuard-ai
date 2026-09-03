"""
tests/test_root_cause_analysis.py -- Run with: pytest tests/test_root_cause_analysis.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from root_cause_analysis import analyze_drivers, format_driver_summary


def make_two_period_df():
    """
    14 days of data, two segments (A, B). Segment A drops sharply in the
    second (current) 7-day period; Segment B stays flat. This lets us
    verify that A is correctly identified as the dominant contributor.
    """
    dates = pd.date_range("2025-01-01", periods=14)
    rows = []
    for i, d in enumerate(dates):
        is_current_period = i >= 7
        a_value = 50.0 if is_current_period else 100.0  # A drops 50%
        b_value = 30.0  # B stays flat throughout
        rows.append({"date": d, "segment": "A", "value": a_value})
        rows.append({"date": d, "segment": "B", "value": b_value})
    return pd.DataFrame(rows)


class TestAnalyzeDrivers:
    def test_identifies_dominant_contributor(self):
        df = make_two_period_df()
        result = analyze_drivers(df, "date", "value", ["segment"], period_days=7)
        top_contributor = result.drivers.iloc[0]
        assert top_contributor["segment"] == "A"
        assert top_contributor["absolute_change"] < 0  # A declined

    def test_flat_segment_has_near_zero_contribution(self):
        df = make_two_period_df()
        result = analyze_drivers(df, "date", "value", ["segment"], period_days=7)
        b_row = result.drivers[result.drivers["segment"] == "B"].iloc[0]
        assert b_row["absolute_change"] == pytest.approx(0.0, abs=0.01)

    def test_overall_change_matches_sum_of_segment_changes(self):
        df = make_two_period_df()
        result = analyze_drivers(df, "date", "value", ["segment"], period_days=7)
        # Total change should equal the sum of all segments' absolute changes
        assert result.overall_current - result.overall_baseline == pytest.approx(
            result.drivers["absolute_change"].sum(), abs=0.01
        )

    def test_insufficient_history_raises_clear_error(self):
        df = make_two_period_df()
        with pytest.raises(ValueError, match="Not enough historical data"):
            analyze_drivers(df, "date", "value", ["segment"], period_days=30)

    def test_empty_group_cols_raises(self):
        df = make_two_period_df()
        with pytest.raises(ValueError, match="group_cols"):
            analyze_drivers(df, "date", "value", [], period_days=7)

    def test_segment_disappearing_is_treated_as_dropping_to_zero(self):
        # Segment "C" exists only in the baseline period (all 7 days, at
        # value 40.0/day), not the current period at all.
        dates = pd.date_range("2025-01-01", periods=14)
        rows = []
        for i, d in enumerate(dates):
            is_current = i >= 7
            rows.append({"date": d, "segment": "A", "value": 100.0})
            if not is_current:
                rows.append({"date": d, "segment": "C", "value": 40.0})
        df = pd.DataFrame(rows)

        result = analyze_drivers(df, "date", "value", ["segment"], period_days=7)
        c_row = result.drivers[result.drivers["segment"] == "C"].iloc[0]
        assert c_row["current_value"] == 0.0
        # Baseline is SUMMED across all 7 days at 40.0/day -> 280.0, not a
        # single day's value -- since agg="sum" is the default.
        assert c_row["baseline_value"] == pytest.approx(280.0)
        assert c_row["absolute_change"] == pytest.approx(-280.0)

    def test_timestamps_with_time_of_day_are_handled(self):
        # Same class of bug as the Day 5 kpi_engine fix -- dates with time
        # components must still group correctly by calendar day.
        dates = pd.date_range("2025-01-01", periods=14, freq="D") + pd.Timedelta(hours=8)
        rows = [{"date": d, "segment": "A", "value": 100.0} for d in dates]
        df = pd.DataFrame(rows)
        result = analyze_drivers(df, "date", "value", ["segment"], period_days=7)
        assert result.overall_current == pytest.approx(700.0)


class TestFormatDriverSummary:
    def test_returns_expected_number_of_summaries(self):
        df = make_two_period_df()
        result = analyze_drivers(df, "date", "value", ["segment"], period_days=7)
        summaries = format_driver_summary(result, top_n=2)
        assert len(summaries) == 2

    def test_uses_non_causal_language(self):
        df = make_two_period_df()
        result = analyze_drivers(df, "date", "value", ["segment"], period_days=7)
        summaries = format_driver_summary(result, top_n=1)
        assert "caused" not in summaries[0].lower()
        assert "contributor" in summaries[0].lower() or "contribut" in summaries[0].lower()