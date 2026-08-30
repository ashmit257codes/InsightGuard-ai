"""
profiling.py -- Automatic data profiling & data-quality reporting.
"""

from dataclasses import dataclass, field
import pandas as pd
import numpy as np


def classify_columns(df: pd.DataFrame) -> dict:
    """
    Classifies every column into one of:
      - "datetime"    : parseable as a date/time
      - "identifier"  : very high cardinality relative to row count (likely an ID)
      - "numeric"     : numeric dtype, not an identifier -> candidate KPI/measure
      - "categorical" : everything else
    """
    classifications = {}
    n_rows = len(df)

    for col in df.columns:
        series = df[col]

        if _looks_like_datetime(series):
            classifications[col] = "datetime"
            continue

        if pd.api.types.is_bool_dtype(series):
            classifications[col] = "categorical"
            continue

        if pd.api.types.is_numeric_dtype(series):
            cardinality_ratio = series.nunique(dropna=True) / max(n_rows, 1)
            is_integer_like = pd.api.types.is_integer_dtype(series) or (
                series.dropna() % 1 == 0
            ).all()
            name_suggests_id = any(
                token in col.lower() for token in ["_id", "id_", "customer_id", "order_id"]
            ) or col.lower() == "id"

            if name_suggests_id and cardinality_ratio > 0.5:
                classifications[col] = "identifier"
            elif is_integer_like and cardinality_ratio > 0.95 and n_rows > 20:
                classifications[col] = "identifier"
            else:
                classifications[col] = "numeric"
            continue

        cardinality_ratio = series.nunique(dropna=True) / max(n_rows, 1)
        if cardinality_ratio > 0.95 and n_rows > 20:
            classifications[col] = "identifier"
        else:
            classifications[col] = "categorical"

    return classifications


def _looks_like_datetime(series: pd.Series, sample_size: int = 50, success_threshold: float = 0.9) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False

    sample = series.dropna().astype(str).head(sample_size)
    if len(sample) == 0:
        return False

    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    success_rate = parsed.notna().mean()
    return success_rate >= success_threshold


@dataclass
class QualityIssue:
    category: str
    column: str | None
    detail: str
    severity: str


@dataclass
class DataQualityReport:
    n_rows: int
    n_columns: int
    column_types: dict
    issues: list = field(default_factory=list)

    def issues_by_severity(self, severity: str) -> list:
        return [i for i in self.issues if i.severity == severity]

    def summary_counts(self) -> dict:
        return {
            "critical": len(self.issues_by_severity("critical")),
            "warning": len(self.issues_by_severity("warning")),
            "info": len(self.issues_by_severity("info")),
        }


def generate_quality_report(df: pd.DataFrame) -> DataQualityReport:
    issues = []
    n_rows, n_cols = df.shape
    column_types = classify_columns(df)

    missing_pct = df.isna().mean() * 100
    for col, pct in missing_pct.items():
        if pct == 0:
            continue
        severity = "critical" if pct > 30 else "warning" if pct > 5 else "info"
        issues.append(
            QualityIssue(
                category="missing_values",
                column=col,
                detail=f"{pct:.1f}% missing ({int(df[col].isna().sum())} rows)",
                severity=severity,
            )
        )

    n_duplicates = int(df.duplicated().sum())
    if n_duplicates > 0:
        pct = n_duplicates / n_rows * 100
        issues.append(
            QualityIssue(
                category="duplicates",
                column=None,
                detail=f"{n_duplicates} duplicate rows ({pct:.1f}% of dataset)",
                severity="warning" if pct > 1 else "info",
            )
        )

    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            issues.append(
                QualityIssue(
                    category="constant_column",
                    column=col,
                    detail="Column has only one unique value (no analytical value)",
                    severity="info",
                )
            )

    for col, ctype in column_types.items():
        if ctype != "numeric":
            continue
        series = df[col].dropna()
        if len(series) < 10:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = int(((series < lower) | (series > upper)).sum())
        if n_outliers > 0:
            pct = n_outliers / len(series) * 100
            issues.append(
                QualityIssue(
                    category="potential_outliers",
                    column=col,
                    detail=f"{n_outliers} values ({pct:.1f}%) outside IQR bounds [{lower:.2f}, {upper:.2f}]",
                    severity="info",
                )
            )

    for col, ctype in column_types.items():
        if ctype != "datetime":
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        n_invalid = int(parsed.isna().sum() - df[col].isna().sum())
        if n_invalid > 0:
            issues.append(
                QualityIssue(
                    category="invalid_dates",
                    column=col,
                    detail=f"{n_invalid} values could not be parsed as dates",
                    severity="warning",
                )
            )

    return DataQualityReport(
        n_rows=n_rows,
        n_columns=n_cols,
        column_types=column_types,
        issues=issues,
    )