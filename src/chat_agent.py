"""
chat_agent.py -- A tool-calling chat agent: the LLM chooses which of our
already-built, already-tested analytics functions to call based on the
user's natural-language question, and answers using only the results those
functions return.

Design choice (per project principles): ONE agent with MULTIPLE
deterministic tools, not a multi-agent system. The LLM's only job is to
pick the right tool(s) and phrase a grounded answer -- it never computes
numbers itself and never sees the raw dataframe directly.
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq

from kpi_engine import build_kpi_timeseries, compute_trend_metrics
from root_cause_analysis import analyze_drivers, format_driver_summary
from anomaly_detection import detect_anomalies
from severity_scoring import compute_severity_scores
from profiling import generate_quality_report

load_dotenv()
MODEL_NAME = "openai/gpt-oss-20b"

AGENT_SYSTEM_PROMPT = """You are a business data analyst assistant with access
to tools that compute real metrics from the user's uploaded dataset.

RULES:
- ALWAYS use a tool to get numbers before answering any question about the
  data. Never guess or estimate a metric yourself.
- If a tool returns an error, tell the user plainly rather than making up
  a plausible-sounding answer.
- Use "contributor" / "accounted for" language for driver analysis, never
  "caused."
- Keep answers concise and to the point -- a few sentences, not an essay.
"""


def build_tool_schemas(numeric_cols: list, categorical_cols: list) -> list:
    """
    Builds the JSON tool schemas the LLM sees, with `enum` populated from
    THIS dataset's actual columns -- so the model can only ever request
    columns that genuinely exist, rather than guessing free-form strings.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "get_kpi_trend",
                "description": "Get the current trend for a KPI: current value, % change, moving average, and trend direction.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kpi_col": {"type": "string", "enum": numeric_cols},
                    },
                    "required": ["kpi_col"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_top_drivers",
                "description": "Get the top contributing segments (e.g. region/category) behind a recent change in a KPI.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kpi_col": {"type": "string", "enum": numeric_cols},
                        "period_days": {"type": "integer", "description": "Comparison window length in days, default 7"},
                    },
                    "required": ["kpi_col"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_anomaly_summary",
                "description": "Get a count and list of detected anomalies for a KPI using statistical detection.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kpi_col": {"type": "string", "enum": numeric_cols},
                    },
                    "required": ["kpi_col"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_data_quality_summary",
                "description": "Get a summary of data quality issues (missing values, duplicates, outliers) in the dataset.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def build_tool_dispatch(df, date_col: str, group_cols: list) -> dict:
    """
    Returns {tool_name: callable(args_dict) -> dict}. Each callable wraps
    an already-tested analytics function, binding the current dataframe and
    key columns via closure so the LLM only needs to supply the parts it's
    actually deciding (like which KPI to look at).
    """

    def _get_kpi_trend(args: dict) -> dict:
        kpi_col = args["kpi_col"]
        ts = build_kpi_timeseries(df, date_col=date_col, kpi_col=kpi_col, freq="D", agg="sum")
        metrics = compute_trend_metrics(ts, kpi_name=kpi_col)
        return {
            "current_value": round(metrics.current_value, 2),
            "previous_value": round(metrics.previous_value, 2),
            "pct_change": round(metrics.pct_change, 2),
            "trend_direction": metrics.trend_direction,
            "moving_average": round(metrics.moving_average, 2),
            "volatility_pct": round(metrics.rolling_volatility, 2),
        }

    def _get_top_drivers(args: dict) -> dict:
        kpi_col = args["kpi_col"]
        period_days = args.get("period_days", 7)
        try:
            result = analyze_drivers(df, date_col=date_col, value_col=kpi_col, group_cols=group_cols, period_days=period_days)
        except ValueError as e:
            return {"error": str(e)}
        top = result.drivers.head(5)
        cols = [c for c in top.columns if c not in ("baseline_value", "current_value", "absolute_change", "contribution_pct", "pct_change")]
        return {
            "overall_change_pct": round(result.overall_change_pct, 2),
            "top_contributors": [
                {
                    "segment": " + ".join(str(row[c]) for c in cols),
                    "contribution_pct": round(row["contribution_pct"], 1),
                    "segment_pct_change": round(row["pct_change"], 1),
                }
                for _, row in top.iterrows()
            ],
        }

    def _get_anomaly_summary(args: dict) -> dict:
        kpi_col = args["kpi_col"]
        flags = detect_anomalies(df, kpi_col, date_col, group_cols, method="z_score")
        n_flagged = int(flags.sum())
        if n_flagged == 0:
            return {"anomaly_count": 0, "details": []}
        severity = compute_severity_scores(df, kpi_col, date_col, group_cols, flags)
        top_severity = severity.sort_values("severity_score", ascending=False).head(5)
        details_df = df.loc[top_severity.index, [date_col] + group_cols].join(top_severity)
        return {
            "anomaly_count": n_flagged,
            "top_anomalies": [
                {
                    "date": str(row[date_col]),
                    "segment": " + ".join(str(row[c]) for c in group_cols),
                    "severity": row["severity_label"],
                    "deviation_pct": row["deviation_pct"],
                }
                for _, row in details_df.iterrows()
            ],
        }

    def _get_data_quality_summary(args: dict) -> dict:
        report = generate_quality_report(df)
        return report.summary_counts()

    return {
        "get_kpi_trend": _get_kpi_trend,
        "get_top_drivers": _get_top_drivers,
        "get_anomaly_summary": _get_anomaly_summary,
        "get_data_quality_summary": _get_data_quality_summary,
    }


def run_agent_turn(user_message: str, tool_schemas: list, tool_dispatch: dict, chat_history: list | None = None) -> tuple:
    """
    Runs one full agent turn: sends the user's message (plus history) to
    the LLM with tools available, executes any tool calls the LLM
    requests, feeds results back, and returns the final grounded answer.

    Returns (answer_text, updated_chat_history).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found. Check your .env file.")

    client = Groq(api_key=api_key)
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    messages += (chat_history or [])
    messages.append({"role": "user", "content": user_message})

    MAX_TOOL_ROUNDS = 3
    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tool_schemas,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=600,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            # No more tools needed -- this is the final answer.
            messages.append({"role": "assistant", "content": message.content})
            return message.content, messages[1:]  # exclude system prompt from returned history

        # The model wants to call one or more tools -- execute each and
        # feed the results back before asking for the next step.
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [tc.model_dump() for tc in message.tool_calls],
        })

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            try:
                fn_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            if fn_name in tool_dispatch:
                try:
                    result = tool_dispatch[fn_name](fn_args)
                except Exception as e:
                    result = {"error": f"Tool execution failed: {e}"}
            else:
                result = {"error": f"Unknown tool: {fn_name}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

    # Safety net: if we somehow exceed MAX_TOOL_ROUNDS without a final answer
    return "I wasn't able to fully answer that -- could you rephrase your question?", messages[1:]