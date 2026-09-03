"""
tests/test_llm_insights.py -- Tests structured data building and error
handling WITHOUT calling the real Groq API (no cost, runs without a key).
"""

import sys
from pathlib import Path
import os

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_insights import build_structured_summary, generate_business_insight
from kpi_engine import compute_trend_metrics
import pandas as pd


class TestBuildStructuredSummary:
    def test_includes_trend_data_when_provided(self):
        series = pd.Series([100.0, 110.0, 120.0, 150.0])
        metrics = compute_trend_metrics(series, kpi_name="Revenue")
        summary = build_structured_summary("Revenue", trend_metrics=metrics)
        assert summary["kpi"] == "Revenue"
        assert "trend" in summary
        assert summary["trend"]["current_value"] == 150.0

    def test_omits_trend_key_when_not_provided(self):
        summary = build_structured_summary("Revenue")
        assert "trend" not in summary

    def test_only_includes_provided_sections(self):
        summary = build_structured_summary("Revenue", severity_summary={"critical_count": 2})
        assert "trend" not in summary
        assert "top_contributors" not in summary
        assert summary["anomalies"]["critical_count"] == 2


class TestGenerateBusinessInsightErrorHandling:
    def test_raises_clear_error_without_api_key(self, monkeypatch):
        # Temporarily remove the API key to confirm we get a helpful error,
        # not a confusing library-level exception.
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GROQ_API_KEY not found"):
            generate_business_insight({"kpi": "Revenue"})