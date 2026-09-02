# save as check_anomaly_results.py in project root, then run: python check_anomaly_results.py
import pandas as pd
import sys
sys.path.insert(0, 'src')
from anomaly_detection import compare_methods

df = pd.read_csv('data/labeled_sales_data.csv')
comparison = compare_methods(
    df, value_col='revenue', date_col='date',
    group_cols=['region', 'category'], ground_truth_col='is_anomaly',
)
print(comparison.to_string(index=False))