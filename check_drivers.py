import pandas as pd
import sys
sys.path.insert(0, 'src')
from root_cause_analysis import analyze_drivers, format_driver_summary

df = pd.read_csv('data/labeled_sales_data.csv')

result = analyze_drivers(df, date_col='date', value_col='revenue', group_cols=['region', 'category'], period_days=7)

print(f"Current period:  {result.current_period}")
print(f"Baseline period: {result.baseline_period}")
print(f"Overall: {result.overall_baseline:.0f} -> {result.overall_current:.0f} ({result.overall_change_pct:+.1f}%)")
print()
print("Top 5 contributors:")
print(result.drivers.head(5).to_string(index=False))
print()
print("Summary:")
for line in format_driver_summary(result, top_n=3):
    print(f"  - {line}")