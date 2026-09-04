"""
tests/test_chat_agent.py -- Tests tool schema building and tool dispatch
logic WITHOUT calling the real Groq API.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chat_agent import build_tool_schemas, build_tool_dispatch


@pytest.fixture
def sample_df():
    dates = pd.date_range("2025-01-01", periods=20).tolist() * 2
    return pd.DataFrame({
        "date": dates,
        "region": ["North"] * 20 + ["South"] * 20,
        "revenue": [100.0 + i for i in range(20)] + [50.0 + i for i in range(20)],
    })


class TestBuildToolSchemas:
    def test_returns_four_tools(self):
        schemas = build_tool_schemas(["revenue"], ["region"])
        assert len(schemas) == 4

    def test_kpi_col_enum_matches_input(self):
        schemas = build_tool_schemas(["revenue", "orders"], ["region"])
        trend_tool = next(s for s in schemas if s["function"]["name"] == "get_kpi_trend")
        assert trend_tool["function"]["parameters"]["properties"]["kpi_col"]["enum"] == ["revenue", "orders"]


class TestToolDispatch:
    def test_get_kpi_trend_returns_expected_keys(self, sample_df):
        dispatch = build_tool_dispatch(sample_df, date_col="date", group_cols=["region"])
        result = dispatch["get_kpi_trend"]({"kpi_col": "revenue"})
        assert "current_value" in result
        assert "trend_direction" in result

    def test_get_data_quality_summary_returns_counts(self, sample_df):
        dispatch = build_tool_dispatch(sample_df, date_col="date", group_cols=["region"])
        result = dispatch["get_data_quality_summary"]({})
        assert "critical" in result
        assert "warning" in result

    def test_get_top_drivers_handles_insufficient_history_gracefully(self, sample_df):
        dispatch = build_tool_dispatch(sample_df, date_col="date", group_cols=["region"])
        result = dispatch["get_top_drivers"]({"kpi_col": "revenue", "period_days": 100})
        assert "error" in result  # should return an error dict, not raise/crash

    def test_get_anomaly_summary_returns_count(self, sample_df):
        dispatch = build_tool_dispatch(sample_df, date_col="date", group_cols=["region"])
        result = dispatch["get_anomaly_summary"]({"kpi_col": "revenue"})
        assert "anomaly_count" in result