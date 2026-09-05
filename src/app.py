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
from anomaly_detection import detect_anomalies, DETECTOR_REGISTRY
from ml_anomaly_detection import detect_ml_anomalies, ML_DETECTOR_REGISTRY
from severity_scoring import compute_severity_scores
from root_cause_analysis import analyze_drivers, format_driver_summary
from llm_insights import build_structured_summary, generate_business_insight
from chat_agent import build_tool_schemas, build_tool_dispatch, run_agent_turn
from email_alerts import build_alert_email, send_alert_email
from feedback_store import submit_feedback, get_feedback_lookup, get_feedback_summary

st.set_page_config(
    page_title="InsightGuard AI",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
    .hero-section { text-align: center; padding: 30px 20px 10px 20px; }
    .hero-logo svg { width: 60px; height: 60px; }
    .hero-title { font-size: 2.2rem; font-weight: 800; color: #1E293B; margin: 10px 0 4px 0; }
    .hero-tagline { font-size: 1.05rem; color: #2563EB; font-weight: 600; margin-bottom: 12px; }
    .hero-description { max-width: 650px; margin: 0 auto; color: #475569; font-size: 0.95rem; line-height: 1.6; }
    .feature-grid { display: flex; flex-wrap: wrap; gap: 16px; justify-content: center; padding: 25px 10px 10px 10px; }
    .feature-card {
        background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
        padding: 18px; width: 250px; box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .feature-title { font-weight: 700; color: #1E293B; margin-bottom: 6px; font-size: 1rem; }
    .feature-desc { color: #64748B; font-size: 0.85rem; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

LOGO_SVG = """<svg viewBox='0 0 48 48' fill='none' xmlns='http://www.w3.org/2000/svg'>
<path d='M24 4L40 10V22C40 32.5 33.5 41 24 44C14.5 41 8 32.5 8 22V10L24 4Z' fill='#2563EB' fill-opacity='0.1' stroke='#2563EB' stroke-width='2.5' stroke-linejoin='round'/>
<path d='M15 26L20 20L25 25L33 15' stroke='#2563EB' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'/>
<circle cx='33' cy='15' r='2.2' fill='#2563EB'/>
</svg>"""

FEATURES = [
    {"key": "profiling", "title": "Automatic Data Profiling", "desc": "Detects column types, missing values, and data quality issues instantly."},
    {"key": "kpi", "title": "KPI Trend Analysis", "desc": "Tracks revenue, orders, or any metric over time with moving averages and trend direction."},
    {"key": "stat_anomaly", "title": "Statistical Anomaly Detection", "desc": "Z-score, IQR, and rolling-baseline methods flag unusual points."},
    {"key": "ml_anomaly", "title": "ML-Powered Detection", "desc": "Isolation Forest and Local Outlier Factor catch complex anomalies."},
    {"key": "drivers", "title": "Root Cause Analysis", "desc": "Breaks down KPI changes by segment to find the biggest contributors."},
    {"key": "chat", "title": "AI Insights & Chat", "desc": "A grounded LLM explains your metrics and answers follow-up questions."},
    {"key": "alerts", "title": "Alerts & Feedback", "desc": "Get emailed on critical anomalies; mark detections valid or false positive."},
]

if "landing_detail" not in st.session_state:
    st.session_state["landing_detail"] = None

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"],
    help="Max recommended size: a few thousand rows for now.",
)


def load_dataset(file) -> pd.DataFrame | None:
    """
    Reads an uploaded file into a DataFrame, handling both CSV and Excel.
    Larger Excel files can take a noticeable number of seconds to parse --
    without a spinner, the app looks frozen during that time.
    """
    try:
        with st.spinner(f"Reading {file.name}... this can take a while for large files."):
            if file.name.endswith(".csv"):
                return pd.read_csv(file)
            else:
                return pd.read_excel(file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None

def render_landing_page():
    st.markdown(f"""
    <div class='hero-section'>
        <div class='hero-logo'>{LOGO_SVG}</div>
        <h1 class='hero-title'>InsightGuard AI</h1>
        <p class='hero-tagline'>AI-Powered Business Intelligence &amp; Anomaly Detection</p>
        <p class='hero-description'>
            Upload a business dataset (CSV or Excel) and instantly get automatic
            data profiling, KPI trend tracking, statistical and ML-based anomaly
            detection, root-cause analysis, and AI-generated business insights —
            all backed by deterministic, tested analytics code.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    for i, feature in enumerate(FEATURES):
        with cols[i % 4]:
            st.markdown(f"""
            <div class='feature-card'>
                <div class='feature-title'>{feature['title']}</div>
                <div class='feature-desc'>{feature['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Learn more", key=f"learn_{feature['key']}"):
                st.session_state["landing_detail"] = (
                    feature['key'] if st.session_state["landing_detail"] != feature['key'] else None
                )

    if st.session_state["landing_detail"]:
        detail = next(f for f in FEATURES if f["key"] == st.session_state["landing_detail"])
        st.divider()
        st.subheader(detail["title"])
        st.write(detail["desc"])
        st.caption("Upload a dataset above to try this feature live.")
if uploaded_file is not None:
    # Cache the parsed dataframe per uploaded file so Streamlit's constant
    # re-running on every UI interaction doesn't re-parse a large file each
    # time -- only re-parses when a genuinely different file is uploaded.
    cache_key = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("_cache_key") != cache_key:
        df = load_dataset(uploaded_file)
        st.session_state["_cache_key"] = cache_key
        st.session_state["_cached_df"] = df
    else:
        df = st.session_state.get("_cached_df")

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

            tab_preview, tab_types, tab_quality, tab_kpi, tab_anomaly, tab_drivers, tab_chat = st.tabs(
                ["📋 Preview", "🏷️ Column Types", "🩺 Data Quality", "📈 KPI Trends", "🚨 Anomaly Detection", "🔍 Root Cause", "💬 Chat"]
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

            with tab_anomaly:
                col_types = classify_columns(df)
                datetime_cols = [c for c, t in col_types.items() if t == "datetime"]
                numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
                categorical_cols = [c for c, t in col_types.items() if t == "categorical"]

                if not datetime_cols or not numeric_cols:
                    st.warning("Need at least one datetime and one numeric column for anomaly detection.")
                else:
                    a1, a2, a3 = st.columns(3)
                    ad_date_col = a1.selectbox("Date column", datetime_cols, key="ad_date")
                    ad_value_col = a2.selectbox("Value to check", numeric_cols, key="ad_value")

                    all_methods = {**{k: "statistical" for k in DETECTOR_REGISTRY}, **{k: "ml" for k in ML_DETECTOR_REGISTRY}}
                    ad_method = a3.selectbox("Method", list(all_methods.keys()), key="ad_method")
                    is_ml_method = all_methods[ad_method] == "ml"

                    group_cols = st.multiselect(
                        "Group by (detect separately within each group)",
                        categorical_cols,
                        default=categorical_cols[:2] if len(categorical_cols) >= 2 else categorical_cols,
                        help="Anomalies localized to one segment (e.g. one region) get diluted if not grouped.",
                    )

                    method_kwargs = {}
                    if is_ml_method:
                        st.caption(
                            "ML methods require an assumed anomaly rate (contamination). "
                            "Too high → many false alarms. Too low → real anomalies missed."
                        )
                        contamination = st.slider(
                            "Assumed anomaly rate (contamination)",
                            min_value=0.005, max_value=0.10, value=0.01, step=0.005,
                            format="%.3f",
                        )
                        method_kwargs["contamination"] = contamination

                    if group_cols:
                        if is_ml_method:
                            flags = detect_ml_anomalies(df, ad_value_col, ad_date_col, group_cols, method=ad_method, **method_kwargs)
                        else:
                            flags = detect_anomalies(df, ad_value_col, ad_date_col, group_cols, method=ad_method)

                        n_flagged = int(flags.sum())
                        st.metric("Anomalies flagged", n_flagged)

                        if n_flagged > 0:
                            severity = compute_severity_scores(df, ad_value_col, ad_date_col, group_cols, flags)
                            flagged_rows = df.loc[flags, [ad_date_col, ad_value_col] + group_cols].join(severity)
                            flagged_rows = flagged_rows.sort_values("severity_score", ascending=False)

                            severity_color = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}
                            flagged_rows.insert(0, "severity", flagged_rows["severity_label"].map(severity_color) + " " + flagged_rows["severity_label"])

                            feedback_lookup = get_feedback_lookup(kpi_name=ad_value_col)

                            st.caption("Review each flagged anomaly — your feedback is saved and persists across sessions.")
                            header_cols = st.columns([2, 2, 1.5, 1.5, 1, 1.5, 1.5])
                            for col, label in zip(header_cols, [ad_date_col, "Segment", ad_value_col, "Deviation", "Persist.", "Severity", "Feedback"]):
                                col.markdown(f"**{label}**")

                            for idx, row in flagged_rows.iterrows():
                                segment_label = " + ".join(str(row[c]) for c in group_cols)
                                date_str = str(row[ad_date_col])
                                key_tuple = (date_str, segment_label, ad_value_col)
                                existing = feedback_lookup.get(key_tuple)

                                cols = st.columns([2, 2, 1.5, 1.5, 1, 1.5, 1.5])
                                cols[0].write(date_str)
                                cols[1].write(segment_label)
                                cols[2].write(f"{row[ad_value_col]:,.1f}")
                                cols[3].write(f"{row['deviation_pct']:+.1f}%")
                                cols[4].write(int(row["persistence_days"]))
                                cols[5].write(row["severity"])

                                with cols[6]:
                                    if existing == "valid":
                                        st.markdown("✅ Valid")
                                    elif existing == "false_positive":
                                        st.markdown("❌ False +")
                                    else:
                                        btn_col1, btn_col2 = st.columns(2)
                                        if btn_col1.button("✓", key=f"valid_{idx}", help="Mark as valid anomaly"):
                                            submit_feedback(date_str, segment_label, ad_value_col, "valid")
                                            st.rerun()
                                        if btn_col2.button("✗", key=f"fp_{idx}", help="Mark as false positive"):
                                            submit_feedback(date_str, segment_label, ad_value_col, "false_positive")
                                            st.rerun()

                            fb_summary = get_feedback_summary()
                            if fb_summary["valid"] + fb_summary["false_positive"] > 0:
                                st.divider()
                                s1, s2 = st.columns(2)
                                s1.metric("✅ Confirmed valid (all-time)", fb_summary["valid"])
                                s2.metric("❌ Marked false positive (all-time)", fb_summary["false_positive"])

                            critical_rows = flagged_rows[flagged_rows["severity_label"] == "CRITICAL"]
                            if len(critical_rows) > 0:
                                st.divider()
                                st.subheader("📧 Email Alert")
                                st.caption(f"{len(critical_rows)} CRITICAL-severity anomal{'y' if len(critical_rows)==1 else 'ies'} found. Only CRITICAL items trigger an alert — avoids alert fatigue on lower-priority anomalies.")

                                alert_email = st.text_input("Send alert to:", key="alert_email")
                                if st.button("Send Email Alert", key="send_alert"):
                                    if not alert_email:
                                        st.warning("Enter a recipient email address first.")
                                    else:
                                        message = build_alert_email(
                                            critical_rows.reset_index(drop=True), kpi_name=ad_value_col,
                                            date_col=ad_date_col, group_cols=group_cols, recipient_email=alert_email,
                                        )
                                        success, error = send_alert_email(message)
                                        if success:
                                            st.success(f"Alert sent to {alert_email}")
                                        else:
                                            st.error(error)
                        else:
                            st.info("No anomalies flagged with this method/grouping.")
                    else:
                        st.info("Select at least one column to group by.")
            with tab_drivers:
                col_types = classify_columns(df)
                datetime_cols = [c for c, t in col_types.items() if t == "datetime"]
                numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
                categorical_cols = [c for c, t in col_types.items() if t == "categorical"]

                if not datetime_cols or not numeric_cols or not categorical_cols:
                    st.warning("Need a datetime column, a numeric KPI, and at least one categorical column to break down.")
                else:
                    d1, d2, d3 = st.columns(3)
                    rc_date_col = d1.selectbox("Date column", datetime_cols, key="rc_date")
                    rc_value_col = d2.selectbox("KPI to analyze", numeric_cols, key="rc_value")
                    rc_period_days = d3.number_input("Period length (days)", min_value=1, max_value=90, value=7, key="rc_period")

                    rc_group_cols = st.multiselect(
                        "Break down by",
                        categorical_cols,
                        default=categorical_cols[:2] if len(categorical_cols) >= 2 else categorical_cols,
                        key="rc_groups",
                    )
                    rc_agg = "sum" if rc_value_col.lower() in ("revenue", "orders", "sales", "profit", "quantity") else "mean"

                    if rc_group_cols:
                        try:
                            result = analyze_drivers(
                                df, date_col=rc_date_col, value_col=rc_value_col,
                                group_cols=rc_group_cols, period_days=int(rc_period_days), agg=rc_agg,
                            )

                            icon = "📈" if result.overall_change_pct >= 0 else "📉"
                            st.metric(
                                f"{rc_value_col}: {result.baseline_period[0]}–{result.baseline_period[1]} → {result.current_period[0]}–{result.current_period[1]}",
                                f"{result.overall_current:,.0f}",
                                f"{result.overall_change_pct:+.1f}% {icon}",
                            )

                            st.caption(
                                "Contribution shows WHERE a change concentrated across segments, "
                                "not WHY it happened — treat these as leads to investigate, not root causes."
                            )

                            st.subheader("Top contributors")
                            top_n = st.slider("Show top N segments", 3, 20, 5, key="rc_topn")
                            display_df = result.drivers.head(top_n).copy()
                            display_df["contribution_pct"] = display_df["contribution_pct"].round(1)
                            display_df["pct_change"] = display_df["pct_change"].round(1)
                            st.dataframe(display_df, use_container_width=True, hide_index=True)

                            st.subheader("Summary")
                            for line in format_driver_summary(result, top_n=3):
                                st.markdown(f"- {line}")

                            st.divider()
                            st.subheader("🤖 AI Business Insight")
                            st.caption("Generated by an LLM from the computed numbers above only — nothing here is invented.")

                            cache_key = f"insight_{rc_value_col}_{rc_period_days}_{'_'.join(rc_group_cols)}_{result.overall_change_pct:.2f}"
                            if st.button("Generate AI insight", key="gen_insight"):
                                with st.spinner("Asking the AI analyst..."):
                                    try:
                                        summary = build_structured_summary(rc_value_col, driver_result=result)
                                        insight = generate_business_insight(summary)
                                        st.session_state[cache_key] = insight
                                    except RuntimeError as e:
                                        st.error(str(e))

                            if cache_key in st.session_state:
                                st.markdown(st.session_state[cache_key])

                        except ValueError as e:
                            st.error(str(e))
                    else:
                        st.info("Select at least one column to break down by.") 
            with tab_chat:
                col_types = classify_columns(df)
                datetime_cols = [c for c, t in col_types.items() if t == "datetime"]
                numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
                categorical_cols = [c for c, t in col_types.items() if t == "categorical"]

                if not datetime_cols or not numeric_cols or not categorical_cols:
                    st.warning("Chat needs a datetime column, a numeric KPI, and at least one categorical column.")
                else:
                    st.caption(
                        "Ask questions about your data — the AI picks the right analysis tool "
                        "(trend, drivers, or anomalies) and answers using only computed results."
                    )

                    chat_date_col = datetime_cols[0]
                    chat_group_cols = categorical_cols[:2]

                    if "chat_messages" not in st.session_state:
                        st.session_state.chat_messages = []

                    for msg in st.session_state.chat_messages:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])

                    user_question = st.chat_input("Ask about your data...")
                    if user_question:
                        st.session_state.chat_messages.append({"role": "user", "content": user_question})
                        with st.chat_message("user"):
                            st.markdown(user_question)

                        tool_schemas = build_tool_schemas(numeric_cols, categorical_cols)
                        tool_dispatch = build_tool_dispatch(df, date_col=chat_date_col, group_cols=chat_group_cols)

                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                try:
                                    # Reconstruct simple history (user/assistant text only)
                                    # for context -- tool-call internals aren't replayed.
                                    simple_history = [
                                        {"role": m["role"], "content": m["content"]}
                                        for m in st.session_state.chat_messages[:-1]
                                    ]
                                    answer, _ = run_agent_turn(
                                        user_question, tool_schemas, tool_dispatch, chat_history=simple_history
                                    )
                                    st.markdown(answer)
                                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                                except RuntimeError as e:
                                    st.error(str(e))

                    if st.session_state.chat_messages:
                        if st.button("Clear chat"):
                            st.session_state.chat_messages = []
                            st.rerun()      
else:
    render_landing_page()