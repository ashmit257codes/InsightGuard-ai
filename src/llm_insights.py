"""
llm_insights.py -- Turns structured, already-computed analytics into a
plain-language business explanation using Groq's LLM API.

CRITICAL DESIGN RULE: the LLM receives ONLY structured numbers we've
already calculated (KPI change, drivers, severity) -- never raw data rows.
It is not allowed to invent metrics; its job is strictly to explain numbers
it's handed, using "contributor" language rather than causal claims.
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """You are a business analyst assistant. You will be given
structured JSON data containing pre-computed KPI metrics, anomaly details,
and driver/contribution analysis for a business dataset.

STRICT RULES:
- Only reference numbers that appear in the provided JSON. Never invent,
  estimate, or guess any figure not explicitly given.
- Use "contributor" / "accounted for" language. Never say a segment
  "caused," "drove," or "resulted in" a change -- contribution analysis
  shows correlation/concentration, not proven causation. When suggesting
  investigation steps, phrase them as open questions ("investigate what
  may explain...") rather than assuming a specific factor is responsible.
- Be concise: 3-5 sentences of explanation, then 2-3 bullet-point
  recommended investigation steps (phrased as suggestions, not certainties).
- If the data shows no significant change or no anomalies, say so plainly
  rather than manufacturing a narrative.
"""


def build_structured_summary(
    kpi_name: str,
    trend_metrics=None,
    driver_result=None,
    severity_summary: dict | None = None,
) -> dict:
    """
    Assembles the structured JSON payload sent to the LLM, pulling only
    from already-computed results (TrendMetrics, DriverAnalysisResult,
    severity info) -- never from raw dataframes.
    """
    summary = {"kpi": kpi_name}

    if trend_metrics is not None:
        summary["trend"] = {
            "current_value": round(trend_metrics.current_value, 2),
            "previous_value": round(trend_metrics.previous_value, 2),
            "pct_change": round(trend_metrics.pct_change, 2),
            "trend_direction": trend_metrics.trend_direction,
            "volatility_pct": round(trend_metrics.rolling_volatility, 2),
        }

    if driver_result is not None:
        top_drivers = driver_result.drivers.head(5)
        group_cols = [c for c in top_drivers.columns if c not in (
            "baseline_value", "current_value", "absolute_change", "contribution_pct", "pct_change"
        )]
        summary["overall_change_pct"] = round(driver_result.overall_change_pct, 2)
        summary["top_contributors"] = [
            {
                "segment": " + ".join(str(row[c]) for c in group_cols),
                "contribution_pct": round(row["contribution_pct"], 1),
                "segment_pct_change": round(row["pct_change"], 1),
            }
            for _, row in top_drivers.iterrows()
        ]

    if severity_summary is not None:
        summary["anomalies"] = severity_summary

    return summary


def generate_business_insight(structured_data: dict) -> str:
    """
    Sends the structured summary to Groq and returns a plain-language
    business explanation. Raises a clear error if no API key is configured,
    rather than failing with a confusing library-level exception.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Create a .env file in the project root "
            "with GROQ_API_KEY=your_key_here (get a free key at console.groq.com)."
        )

    client = Groq(api_key=api_key)

    user_message = (
        "Here is the structured analytics data:\n\n"
        f"{json.dumps(structured_data, indent=2)}\n\n"
        "Write the business explanation and recommended investigation steps."
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,  # low temperature -- we want grounded, consistent
                           # explanations, not creative variation, since this
                           # is meant to be trustworthy business reporting
        max_tokens=500,
    )

    return response.choices[0].message.content