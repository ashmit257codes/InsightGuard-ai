"""
app.py -- InsightGuard AI (Day 1: upload + preview only)

This is the entry point for the Streamlit app. Run it with:
    streamlit run src/app.py

Today's scope is deliberately small: let the user upload a CSV/Excel file
and see a basic preview + summary. Every later phase (profiling, KPI
detection, anomaly detection, LLM insights, chat) will build on top of this
file, adding new tabs/sections -- we are NOT building everything today.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from profiling import classify_columns, generate_quality_report
from kpi_engine import (
    build_kpi_timeseries,
    compute_trend_metrics,
    compute_full_rolling_stats,
    TREND_ICONS,
)
st.set_page_config(
    page_title="InsightGuard AI",
    page_icon="📊",
    layout="wide",
)

st.title("📊 InsightGuard AI")
st.caption("AI-powered business intelligence & anomaly detection")

st.markdown(
    """
    Upload a business dataset (CSV or Excel) to get started.
    You can also try the included **synthetic sample data**
    (`data/labeled_sales_data.csv`) to explore the app.
    """
)

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"],
    help="Max recommended size: a few thousand rows for now.",
)


def load_dataset(file) -> pd.DataFrame | None:
    """
    Reads an uploaded file into a DataFrame, handling both CSV and Excel.
    Returns None (and shows an error) if the file can't be parsed.
    """
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None


if uploaded_file is not None:
    df = load_dataset(uploaded_file)

    if df is not None:
        if df.empty:
            st.warning("The uploaded file is empty.")
        else:
            st.success(f"Loaded **{uploaded_file.name}** — {df.shape[0]} rows × {df.shape[1]} columns")

            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", f"{df.shape[0]:,}")
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing values", int(df.isna().sum().sum()))

                        # Stash in session state so later phases (KPI engine, anomaly
            # detection) can access the same dataframe without re-uploading.
            st.session_state["current_df"] = df

            tab_preview, tab_types, tab_quality, tab_kpi = st.tabs(
                ["📋 Preview", "🏷️ Column Types", "🩺 Data Quality", "📈 KPI Trends"]
            )

            with tab_preview:
                st.dataframe(df.head(20), use_container_width=True)

            with tab_types:
                st.caption(
                    "Automatic classification used later to detect KPIs and "
                    "dimensions. Datetime detection also catches dates stored "
                    "as plain text."
                )
                col_types = classify_columns(df)
                type_df = pd.DataFrame(
                    {
                        "column": list(col_types.keys()),
                        "detected_type": list(col_types.values()),
                        "pandas_dtype": [str(df[c].dtype) for c in col_types.keys()],
                    }
                )
                st.dataframe(type_df, use_container_width=True, hide_index=True)

            with tab_quality:
                report = generate_quality_report(df)
                counts = report.summary_counts()

                c1, c2, c3 = st.columns(3)
                c1.metric("🔴 Critical", counts["critical"])
                c2.metric("🟡 Warning", counts["warning"])
                c3.metric("🔵 Info", counts["info"])

                if not report.issues:
                    st.success("No data quality issues detected.")
                else:
                    severity_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
                    order = {"critical": 0, "warning": 1, "info": 2}
                    sorted_issues = sorted(report.issues, key=lambda i: order[i.severity])

                    for issue in sorted_issues:
                        icon = severity_icon[issue.severity]
                        col_label = f"**{issue.column}**" if issue.column else "**Dataset-wide**"
                        st.markdown(f"{icon} {col_label} — {issue.detail}")

            with tab_kpi:
                col_types = classify_columns(df)
                datetime_cols = [c for c, t in col_types.items() if t == "datetime"]
                numeric_cols = [c for c, t in col_types.items() if t == "numeric"]

                if not datetime_cols:
                    st.warning("No datetime column detected — KPI trend analysis needs a date column.")
                elif not numeric_cols:
                    st.warning("No numeric columns detected to use as KPIs.")
                else:
                    col_a, col_b, col_c = st.columns(3)
                    date_col = col_a.selectbox("Date column", datetime_cols)
                    kpi_col = col_b.selectbox("KPI to analyze", numeric_cols)
                    freq_label = col_c.selectbox(
                        "Aggregation", ["Daily", "Weekly", "Monthly"], index=0
                    )
                    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
                    agg = "sum" if kpi_col.lower() in ("revenue", "orders", "sales", "profit", "quantity") else "mean"

                    try:
                        ts = build_kpi_timeseries(df, date_col, kpi_col, freq=freq_map[freq_label], agg=agg)
                        metrics = compute_trend_metrics(ts, kpi_name=kpi_col)

                        icon = TREND_ICONS[metrics.trend_direction]
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric(
                            f"Current {kpi_col}",
                            f"{metrics.current_value:,.2f}",
                            f"{metrics.pct_change:+.1f}%",
                        )
                        m2.metric(f"{7 if freq_label=='Daily' else metrics.window_size}-period Moving Avg", f"{metrics.moving_average:,.2f}")
                        m3.metric("Volatility (CoV)", f"{metrics.rolling_volatility:.1f}%")
                        m4.metric("Trend", f"{icon} {metrics.trend_direction.capitalize()}")

                        rolling_df = compute_full_rolling_stats(ts)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=rolling_df.index, y=rolling_df["actual"], name=kpi_col, line=dict(color="#636EFA")))
                        fig.add_trace(go.Scatter(x=rolling_df.index, y=rolling_df["moving_average"], name="Moving Average", line=dict(color="#EF553B", dash="dash")))
                        fig.update_layout(
                            title=f"{kpi_col} over time ({freq_label.lower()})",
                            xaxis_title="Date",
                            yaxis_title=kpi_col,
                            hovermode="x unified",
                            height=450,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    except ValueError as e:
                        st.error(f"Couldn't compute trend: {e}")
else:
    st.info("👆 Upload a file to see a preview here.")
