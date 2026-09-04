import sys
sys.path.insert(0, 'src')

import pandas as pd
from chat_agent import build_tool_schemas, build_tool_dispatch, run_agent_turn
from profiling import classify_columns

df = pd.read_csv('data/labeled_sales_data.csv')

col_types = classify_columns(df)
numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
categorical_cols = [c for c, t in col_types.items() if t == "categorical"]

tool_schemas = build_tool_schemas(numeric_cols, categorical_cols)
tool_dispatch = build_tool_dispatch(df, date_col="date", group_cols=["region", "category"])

questions = [
    "What's the current revenue trend?",
    "Why did revenue change recently? Which segments were the biggest contributors?",
    "Are there any anomalies in revenue?",
]

history = []
for q in questions:
    print(f"\n{'='*60}")
    print(f"USER: {q}")
    print('='*60)
    answer, history = run_agent_turn(q, tool_schemas, tool_dispatch, chat_history=history)
    print(f"AGENT: {answer}")