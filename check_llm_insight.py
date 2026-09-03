import sys
sys.path.insert(0, 'src')

from llm_insights import build_structured_summary, generate_business_insight
from kpi_engine import build_kpi_timeseries, compute_trend_metrics
from root_cause_analysis import analyze_drivers
import pandas as pd

df = pd.read_csv('data/labeled_sales_data.csv')

# Build real trend metrics
ts = build_kpi_timeseries(df, date_col='date', kpi_col='revenue', freq='D', agg='sum')
trend = compute_trend_metrics(ts, kpi_name='Revenue')

# Build real driver analysis
drivers = analyze_drivers(df, date_col='date', value_col='revenue', group_cols=['region', 'category'], period_days=7)

# Assemble structured summary -- this is EXACTLY what the LLM will see
summary = build_structured_summary('Revenue', trend_metrics=trend, driver_result=drivers)

print("=== Structured data sent to LLM ===")
import json
print(json.dumps(summary, indent=2))
print()

print("=== LLM-generated insight ===")
insight = generate_business_insight(summary)
print(insight)