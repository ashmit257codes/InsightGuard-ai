"""
root_cause_analysis.py -- Driver / contribution analysis: decomposes an
aggregate KPI change into per-segment contributions.

Answers: "Revenue changed by X% -- which specific segments (region,
category, etc.) actually drove that change, and how much did each
contribute?"

IMPORTANT LANGUAGE NOTE: this module identifies "contributors," never
"causes." A segment can be the primary driver of a change in the data
without us knowing WHY that segment changed -- that requires business
context this module doesn't have. All output should be phrased as
contribution, not causation.
"""

from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from datetime import date


@dataclass
class DriverAnalysisResult:
    current_period: tuple
    baseline_period: tuple
    overall_current: float
    overall_baseline: float
    overall_change_pct: float
    drivers: pd.DataFrame  # sorted by |absolute_change| descending


def analyze_drivers(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    group_cols: list,
    period_days: int = 7,
    agg: str = "sum",
) -> DriverAnalysisResult:
    """
    Compares the most recent `period_days` of data against the equally-sized
    period immediately before it, broken down by `group_cols`, and returns
    each segment's contribution to the overall change.

    Args:
        df: raw dataframe (can have multiple rows per date/segment)
        date_col: datetime column name
        value_col: numeric KPI column (e.g. "revenue")
        group_cols: columns to break down by (e.g. ["region", "category"])
        period_days: length of each comparison window in days
        agg: "sum" (typical for revenue/orders) or "mean" (typical for rates)

    Raises:
        ValueError if there isn't enough historical data for two full
        non-overlapping periods, or if group_cols is empty.
    """
    if not group_cols:
        raise ValueError("group_cols must contain at least one column")
    if agg not in ("sum", "mean"):
        raise ValueError("agg must be 'sum' or 'mean'")

    working = df.copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce", format="mixed")
    working = working.dropna(subset=[date_col])
    # Normalize to date-only, same reasoning as kpi_engine.py: raw timestamps
    # would otherwise fragment a single day into many sub-groups.
    working[date_col] = working[date_col].dt.floor("D")

    max_date = working[date_col].max()
    min_date = working[date_col].min()

    current_start = max_date - pd.Timedelta(days=period_days - 1)
    baseline_end = current_start - pd.Timedelta(days=1)
    baseline_start = baseline_end - pd.Timedelta(days=period_days - 1)

    if baseline_start < min_date:
        raise ValueError(
            f"Not enough historical data for a {period_days}-day baseline "
            f"comparison. Earliest available date is {min_date.date()}, but "
            f"a full comparison needs data from {baseline_start.date()} onward. "
            f"Try a smaller period_days."
        )

    current_mask = (working[date_col] >= current_start) & (working[date_col] <= max_date)
    baseline_mask = (working[date_col] >= baseline_start) & (working[date_col] <= baseline_end)

    current_df = working[current_mask]
    baseline_df = working[baseline_mask]

    def agg_apply(series: pd.Series) -> float:
        return float(series.sum()) if agg == "sum" else float(series.mean())

    current_agg = current_df.groupby(group_cols)[value_col].agg(agg)
    baseline_agg = baseline_df.groupby(group_cols)[value_col].agg(agg)

    # Constructing a DataFrame from two Series with potentially different
    # indices automatically performs an outer join, aligning on the union
    # of segment combinations. Segments present in only one period get NaN
    # for the other -- fillna(0) below turns "segment didn't exist here"
    # into "segment's value was 0 here," which is the correct interpretation
    # for a segment appearing/disappearing entirely.
    combined = pd.DataFrame({"baseline_value": baseline_agg, "current_value": current_agg}).fillna(0)
    combined["absolute_change"] = combined["current_value"] - combined["baseline_value"]

    total_change = combined["absolute_change"].sum()
    if total_change != 0:
        combined["contribution_pct"] = combined["absolute_change"] / total_change * 100
    else:
        combined["contribution_pct"] = 0.0

    def segment_pct_change(row) -> float:
        if row["baseline_value"] == 0:
            return 100.0 if row["current_value"] > 0 else 0.0
        return row["absolute_change"] / abs(row["baseline_value"]) * 100

    combined["pct_change"] = combined.apply(segment_pct_change, axis=1)

    combined = combined.reset_index()
    combined = combined.sort_values(
        by="absolute_change", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)

    overall_current = agg_apply(current_df[value_col])
    overall_baseline = agg_apply(baseline_df[value_col])
    overall_change = overall_current - overall_baseline
    if overall_baseline != 0:
        overall_change_pct = overall_change / abs(overall_baseline) * 100
    else:
        overall_change_pct = 100.0 if overall_current > 0 else 0.0

    return DriverAnalysisResult(
        current_period=(current_start.date(), max_date.date()),
        baseline_period=(baseline_start.date(), baseline_end.date()),
        overall_current=overall_current,
        overall_baseline=overall_baseline,
        overall_change_pct=overall_change_pct,
        drivers=combined,
    )


def format_driver_summary(result: DriverAnalysisResult, top_n: int = 3) -> list:
    """
    Produces plain-language summary strings for the top N contributors,
    deliberately using "contributor" / "accounted for" language rather than
    causal wording like "caused" -- contribution analysis shows WHERE a
    change concentrated, not WHY it happened.
    """
    top = result.drivers.head(top_n)
    group_cols = [c for c in top.columns if c not in (
        "baseline_value", "current_value", "absolute_change", "contribution_pct", "pct_change"
    )]

    summaries = []
    for _, row in top.iterrows():
        segment_label = " + ".join(str(row[c]) for c in group_cols)
        direction = "increase" if row["absolute_change"] > 0 else "decline"
        summaries.append(
            f"{segment_label}: primary contributor to the {direction}, "
            f"accounting for {abs(row['contribution_pct']):.1f}% of the total change "
            f"(segment itself changed {row['pct_change']:+.1f}%)"
        )
    return summaries