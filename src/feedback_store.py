"""
feedback_store.py -- Persists user feedback on flagged anomalies (valid /
false positive) to a local SQLite database, so feedback survives across
app restarts (unlike st.session_state, which resets every session).

Design choice: anomalies are identified by (date, segment, kpi_name) rather
than a dataframe row index, since row position isn't a stable identity --
the same anomaly should be recognized as "the same one" even if the
dataframe is re-sorted or re-filtered differently next time.
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "feedback.db"


def init_db() -> None:
    """
    Creates the feedback table if it doesn't already exist. Safe to call
    every time the app starts -- CREATE TABLE IF NOT EXISTS is a no-op if
    the table is already there.
    """
    DB_PATH.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                anomaly_date TEXT NOT NULL,
                segment TEXT NOT NULL,
                kpi_name TEXT NOT NULL,
                feedback TEXT NOT NULL CHECK(feedback IN ('valid', 'false_positive')),
                submitted_at TEXT NOT NULL,
                UNIQUE(anomaly_date, segment, kpi_name)
            )
        """)


def submit_feedback(anomaly_date: str, segment: str, kpi_name: str, feedback: str) -> None:
    """
    Records (or updates) feedback for a specific anomaly. Uses
    INSERT ... ON CONFLICT to overwrite if the user changes their mind
    about the same anomaly later, rather than creating duplicate rows.
    """
    if feedback not in ("valid", "false_positive"):
        raise ValueError("feedback must be 'valid' or 'false_positive'")

    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO anomaly_feedback (anomaly_date, segment, kpi_name, feedback, submitted_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(anomaly_date, segment, kpi_name)
            DO UPDATE SET feedback = excluded.feedback, submitted_at = excluded.submitted_at
        """, (anomaly_date, segment, kpi_name, feedback, datetime.now(timezone.utc).isoformat()))


def get_feedback(kpi_name: str | None = None) -> list:
    """
    Retrieves all stored feedback, optionally filtered to one KPI.
    Returns a list of dicts (easy to convert to a DataFrame in the UI).
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        if kpi_name:
            rows = conn.execute(
                "SELECT * FROM anomaly_feedback WHERE kpi_name = ? ORDER BY submitted_at DESC",
                (kpi_name,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM anomaly_feedback ORDER BY submitted_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]


def get_feedback_lookup(kpi_name: str | None = None) -> dict:
    """
    Returns feedback as a lookup dict keyed by (anomaly_date, segment,
    kpi_name) -> feedback string. Convenient for quickly checking "has this
    specific anomaly already been reviewed?" when rendering a table.
    """
    records = get_feedback(kpi_name)
    return {(r["anomaly_date"], r["segment"], r["kpi_name"]): r["feedback"] for r in records}


def get_feedback_summary() -> dict:
    """
    Returns counts of valid vs false_positive feedback -- useful for a
    quick "how's detection quality tracking according to users?" metric.
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT feedback, COUNT(*) as count FROM anomaly_feedback GROUP BY feedback"
        ).fetchall()
        summary = {"valid": 0, "false_positive": 0}
        for feedback, count in rows:
            summary[feedback] = count
        return summary