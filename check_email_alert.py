import sys
sys.path.insert(0, 'src')

import pandas as pd
from email_alerts import build_alert_email, send_alert_email

# Fake critical anomaly data, just to test the send mechanism
sample = pd.DataFrame({
    "date": ["2025-09-11", "2025-09-12"],
    "region": ["East", "East"],
    "category": ["Groceries", "Groceries"],
    "deviation_pct": [-43.3, -46.9],
    "severity_score": [86.0, 100.0],
})

# Sends to yourself -- replace with your own email if testing a different inbox
recipient = input("Enter email address to send the test alert to: ").strip()

message = build_alert_email(
    sample, kpi_name="revenue", date_col="date",
    group_cols=["region", "category"], recipient_email=recipient,
)

success, error = send_alert_email(message)

if success:
    print(f"✅ Email sent successfully to {recipient}. Check your inbox (and spam folder).")
else:
    print(f"❌ Failed: {error}")