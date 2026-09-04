"""
tests/test_email_alerts.py -- Tests email CONTENT building only, never
actually sends an email (no network, no credentials needed to run these).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from email_alerts import build_alert_email, send_alert_email


@pytest.fixture
def sample_critical_anomalies():
    return pd.DataFrame({
        "date": ["2025-09-11", "2025-09-12"],
        "region": ["East", "East"],
        "category": ["Groceries", "Groceries"],
        "deviation_pct": [-43.3, -46.9],
        "severity_score": [86.0, 100.0],
    })


class TestBuildAlertEmail:
    def test_subject_includes_correct_count(self, sample_critical_anomalies):
        message = build_alert_email(
            sample_critical_anomalies, kpi_name="revenue", date_col="date",
            group_cols=["region", "category"], recipient_email="test@example.com",
        )
        assert "2 Critical Anomalies" in message["Subject"]

    def test_singular_count_uses_singular_wording(self):
        single = pd.DataFrame({
            "date": ["2025-09-11"], "region": ["East"], "category": ["Groceries"],
            "deviation_pct": [-43.3], "severity_score": [86.0],
        })
        message = build_alert_email(
            single, kpi_name="revenue", date_col="date",
            group_cols=["region", "category"], recipient_email="test@example.com",
        )
        assert "1 Critical Anomaly" in message["Subject"]  # not "Anomalies"

    def test_recipient_is_set(self, sample_critical_anomalies):
        message = build_alert_email(
            sample_critical_anomalies, kpi_name="revenue", date_col="date",
            group_cols=["region", "category"], recipient_email="someone@example.com",
        )
        assert message["To"] == "someone@example.com"


class TestSendAlertEmailErrorHandling:
    def test_missing_credentials_returns_clean_error(self, sample_critical_anomalies, monkeypatch):
        monkeypatch.delenv("GMAIL_ADDRESS", raising=False)
        monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)

        message = build_alert_email(
            sample_critical_anomalies, kpi_name="revenue", date_col="date",
            group_cols=["region", "category"], recipient_email="test@example.com",
        )
        success, error = send_alert_email(message)
        assert success is False
        assert "GMAIL_ADDRESS" in error