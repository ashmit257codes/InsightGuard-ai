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

from profiling import classify_columns, generate_quality_report

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

            tab_preview, tab_types, tab_quality = st.tabs(
                ["📋 Preview", "🏷️ Column Types", "🩺 Data Quality"]
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
else:
    st.info("👆 Upload a file to see a preview here.")
