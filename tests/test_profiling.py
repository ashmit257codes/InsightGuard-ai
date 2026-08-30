"""
tests/test_profiling.py -- Run with: pytest tests/test_profiling.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from profiling import classify_columns, generate_quality_report


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "region": ["North", "South", "North"],
            "revenue": [1000.50, 2000.75, 1500.25],
            "orders": [10, 20, 15],
            "is_anomaly": [False, True, False],
            "customer_id": [101, 102, 103],
        }
    )


class TestClassifyColumns:
    def test_datetime_column_stored_as_text_is_detected(self, sample_df):
        types = classify_columns(sample_df)
        assert types["date"] == "datetime"

    def test_numeric_float_column_is_numeric_not_identifier(self, sample_df):
        types = classify_columns(sample_df)
        assert types["revenue"] == "numeric"

    def test_boolean_column_is_categorical(self, sample_df):
        types = classify_columns(sample_df)
        assert types["is_anomaly"] == "categorical"

    def test_low_cardinality_text_is_categorical(self, sample_df):
        types = classify_columns(sample_df)
        assert types["region"] == "categorical"

    def test_integer_column_is_numeric(self, sample_df):
        types = classify_columns(sample_df)
        assert types["orders"] == "numeric"


class TestQualityReport:
    def test_no_issues_on_clean_data(self, sample_df):
        report = generate_quality_report(sample_df)
        assert report.summary_counts()["critical"] == 0

    def test_missing_values_detected(self):
        df = pd.DataFrame({"a": [1, 2, None, 4], "b": [1, 2, 3, 4]})
        report = generate_quality_report(df)
        missing_issues = [i for i in report.issues if i.category == "missing_values"]
        assert len(missing_issues) == 1
        assert missing_issues[0].column == "a"

    def test_duplicates_detected(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
        report = generate_quality_report(df)
        dup_issues = [i for i in report.issues if i.category == "duplicates"]
        assert len(dup_issues) == 1

    def test_constant_column_detected(self):
        df = pd.DataFrame({"a": [5, 5, 5, 5], "b": [1, 2, 3, 4]})
        report = generate_quality_report(df)
        const_issues = [i for i in report.issues if i.category == "constant_column"]
        assert any(i.column == "a" for i in const_issues)

    def test_empty_dataframe_does_not_crash(self):
        df = pd.DataFrame()
        report = generate_quality_report(df)
        assert report.n_rows == 0

    def test_tiny_dataframe_does_not_crash(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        report = generate_quality_report(df)
        assert report.n_rows == 2

    def test_high_missing_pct_is_critical_severity(self):
        df = pd.DataFrame({"a": [None] * 9 + [1], "b": range(10)})
        report = generate_quality_report(df)
        missing_issue = next(i for i in report.issues if i.column == "a")
        assert missing_issue.severity == "critical"