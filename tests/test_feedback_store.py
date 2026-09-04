"""
tests/test_feedback_store.py -- Uses a temporary DB path (via monkeypatch)
so tests never touch your real feedback.db.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import feedback_store


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """
    Redirects DB_PATH to a temporary file for every test, so tests never
    read/write your actual data/feedback.db.
    """
    test_db_path = tmp_path / "test_feedback.db"
    monkeypatch.setattr(feedback_store, "DB_PATH", test_db_path)
    yield


class TestSubmitAndGetFeedback:
    def test_submit_and_retrieve(self):
        feedback_store.submit_feedback("2025-09-12", "East + Groceries", "revenue", "valid")
        records = feedback_store.get_feedback()
        assert len(records) == 1
        assert records[0]["feedback"] == "valid"

    def test_invalid_feedback_value_raises(self):
        with pytest.raises(ValueError):
            feedback_store.submit_feedback("2025-09-12", "East + Groceries", "revenue", "maybe")

    def test_resubmitting_same_anomaly_updates_not_duplicates(self):
        feedback_store.submit_feedback("2025-09-12", "East + Groceries", "revenue", "valid")
        feedback_store.submit_feedback("2025-09-12", "East + Groceries", "revenue", "false_positive")
        records = feedback_store.get_feedback()
        assert len(records) == 1  # still just one row, not two
        assert records[0]["feedback"] == "false_positive"  # updated to the latest

    def test_filter_by_kpi_name(self):
        feedback_store.submit_feedback("2025-09-12", "East + Groceries", "revenue", "valid")
        feedback_store.submit_feedback("2025-09-12", "East + Groceries", "orders", "false_positive")
        revenue_only = feedback_store.get_feedback(kpi_name="revenue")
        assert len(revenue_only) == 1
        assert revenue_only[0]["kpi_name"] == "revenue"


class TestFeedbackLookup:
    def test_lookup_dict_shape(self):
        feedback_store.submit_feedback("2025-09-12", "East + Groceries", "revenue", "valid")
        lookup = feedback_store.get_feedback_lookup()
        assert lookup[("2025-09-12", "East + Groceries", "revenue")] == "valid"


class TestFeedbackSummary:
    def test_counts_by_type(self):
        feedback_store.submit_feedback("2025-09-11", "East + Groceries", "revenue", "valid")
        feedback_store.submit_feedback("2025-09-12", "East + Groceries", "revenue", "valid")
        feedback_store.submit_feedback("2025-09-13", "East + Groceries", "revenue", "false_positive")
        summary = feedback_store.get_feedback_summary()
        assert summary["valid"] == 2
        assert summary["false_positive"] == 1

    def test_empty_db_returns_zero_counts(self):
        summary = feedback_store.get_feedback_summary()
        assert summary == {"valid": 0, "false_positive": 0}