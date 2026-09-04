"""
email_alerts.py -- Sends email alerts for CRITICAL-severity anomalies using
Gmail's SMTP server. Uses only Python's built-in smtplib -- no paid service,
no third-party API key required beyond a free Gmail App Password.

Design choice: only CRITICAL severity anomalies trigger an email. Sending
an email for every LOW/MEDIUM anomaly would create alert fatigue -- the
whole point of severity scoring (Day 5) is to filter down to what actually
deserves attention.
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465  # SSL port


def build_alert_email(
    critical_anomalies: pd.DataFrame,
    kpi_name: str,
    date_col: str,
    group_cols: list,
    recipient_email: str,
) -> MIMEMultipart:
    """
    Builds the email message object (subject + HTML body) for a batch of
    critical anomalies. Does NOT send anything -- fully testable without
    network access or credentials.

    Args:
        critical_anomalies: dataframe already filtered to severity_label == "CRITICAL",
                             with deviation_pct and severity_score columns (from severity_scoring.py)
        kpi_name: the KPI being monitored, e.g. "revenue"
        date_col: date column name
        group_cols: segment columns, e.g. ["region", "category"]
        recipient_email: who receives the alert
    """
    n = len(critical_anomalies)
    subject = f"🔴 {n} Critical Anomal{'y' if n == 1 else 'ies'} Detected — {kpi_name}"

    rows_html = ""
    for _, row in critical_anomalies.iterrows():
        segment = " + ".join(str(row[c]) for c in group_cols)
        rows_html += f"""
        <tr>
            <td style="padding:6px 10px;border-bottom:1px solid #eee;">{row[date_col]}</td>
            <td style="padding:6px 10px;border-bottom:1px solid #eee;">{segment}</td>
            <td style="padding:6px 10px;border-bottom:1px solid #eee;">{row['deviation_pct']:+.1f}%</td>
            <td style="padding:6px 10px;border-bottom:1px solid #eee;">{row['severity_score']:.0f}</td>
        </tr>"""

    html_body = f"""
    <html>
    <body style="font-family:Arial,sans-serif;color:#222;">
        <h2 style="color:#c0392b;">🔴 Critical Anomaly Alert — {kpi_name}</h2>
        <p>{n} critical-severity anomal{'y was' if n == 1 else 'ies were'} detected in <b>{kpi_name}</b>.</p>
        <table style="border-collapse:collapse;width:100%;max-width:600px;">
            <tr style="background:#f4f4f4;text-align:left;">
                <th style="padding:6px 10px;">Date</th>
                <th style="padding:6px 10px;">Segment</th>
                <th style="padding:6px 10px;">Deviation</th>
                <th style="padding:6px 10px;">Severity Score</th>
            </tr>
            {rows_html}
        </table>
        <p style="color:#777;font-size:12px;margin-top:20px;">
            This is an automated alert from InsightGuard AI. Only CRITICAL-severity
            anomalies trigger an email, to avoid alert fatigue on lower-priority items.
        </p>
    </body>
    </html>
    """

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["To"] = recipient_email
    message.attach(MIMEText(html_body, "html"))
    return message


def send_alert_email(message: MIMEMultipart) -> tuple:
    """
    Actually sends the pre-built email via Gmail's SMTP server.

    Returns (success: bool, error_message: str | None) rather than raising,
    so calling UI code can show a clean error instead of crashing.
    """
    sender_email = os.getenv("GMAIL_ADDRESS")
    app_password = os.getenv("GMAIL_APP_PASSWORD")

    if not sender_email or not app_password:
        return False, (
            "GMAIL_ADDRESS or GMAIL_APP_PASSWORD not found in .env. "
            "See README for setup instructions (requires a free Gmail App Password)."
        )

    message["From"] = sender_email

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, message["To"], message.as_string())
        return True, None
    except smtplib.SMTPAuthenticationError:
        return False, "Gmail rejected the credentials. Check GMAIL_ADDRESS/GMAIL_APP_PASSWORD in .env."
    except Exception as e:
        return False, f"Failed to send email: {e}"