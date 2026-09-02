"""
anomaly_detection.py -- Statistical anomaly detection (Z-score, IQR, rolling
baseline) plus evaluation against ground-truth labels.

Detection runs PER GROUP (e.g. per region+category combination), not on
aggregated totals. This matters: an anomaly localized to one segment gets
diluted into invisibility if you only look at the grand total. Each
detector returns a boolean flag per row, aligned with the input's index, so
results can be merged straight back onto the original dataframe.
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np


def z_score_flags(values: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Flags points whose Z-score (distance from the mean, in standard
    deviations) exceeds `threshold`.

    Weakness: mean and std are themselves distorted by large anomalies
    (a single huge spike inflates std, potentially hiding smaller real
    anomalies) -- this is why we also implement IQR and rolling baseline
    to compare.
    """
    mean = values.mean()
    std = values.std()
    if std == 0 or pd.isna(std):
        return pd.Series(False, index=values.index)
    z_scores = (values - mean) / std
    return z_scores.abs() > threshold


def iqr_flags(values: pd.Series, k: float = 1.5) -> pd.Series:
    """
    Flags points outside [Q1 - k*IQR, Q3 + k*IQR].

    More robust than Z-score to extreme values, since quartiles aren't
    pulled around by a single huge outlier the way mean/std are.
    """
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=values.index)
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (values < lower) | (values > upper)


def rolling_baseline_flags(
    values: pd.Series, window: int = 14, threshold_std: float = 3.5
) -> pd.Series:
    """
    Flags points that deviate from a ROLLING baseline (recent mean/std)
    rather than the whole series' stats -- better suited to data with
    trend or seasonality, where "normal" shifts over time.

    IMPORTANT: the rolling window is shifted by 1 so each point is compared
    only against PRIOR data, never including itself. Without this shift, an
    anomalous point would pull the rolling mean/std toward itself, partly
    masking its own deviation -- a subtle bug that silently weakens
    detection without raising an error.

    Tuning note: a small window (e.g. 7) makes the rolling std ESTIMATE
    itself noisy -- computed from only 7 points, it swings erratically and
    caused ~6x more false positives in testing than a 14-day window with the
    same detection rate. Defaults here (window=14, threshold_std=3.5) were
    chosen after comparing several settings against labeled ground truth.
    """
    baseline_mean = values.shift(1).rolling(window=window, min_periods=3).mean()
    baseline_std = values.shift(1).rolling(window=window, min_periods=3).std()

    deviation = (values - baseline_mean).abs()
    threshold = threshold_std * baseline_std

    flags = deviation > threshold
    flags = flags.fillna(False)
    return flags


DETECTOR_REGISTRY = {
    "z_score": z_score_flags,
    "iqr": iqr_flags,
    "rolling_baseline": rolling_baseline_flags,
}


def detect_anomalies(
    df: pd.DataFrame,
    value_col: str,
    date_col: str,
    group_cols: list,
    method: str = "z_score",
    **kwargs,
) -> pd.Series:
    """
    Runs the chosen detector independently on each (group_cols) combination's
    time series, and returns a boolean Series aligned with df's original
    index -- True where that method flags an anomaly.
    """
    if method not in DETECTOR_REGISTRY:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(DETECTOR_REGISTRY)}")

    detector_fn = DETECTOR_REGISTRY[method]
    result = pd.Series(False, index=df.index)

    sorted_df = df.sort_values(date_col)

    for _, group_df in sorted_df.groupby(group_cols):
        flags = detector_fn(group_df[value_col], **kwargs)
        result.loc[flags.index] = flags

    return result


@dataclass
class EvaluationResult:
    method: str
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int


def evaluate_detector(y_true: pd.Series, y_pred: pd.Series, method_name: str = "") -> EvaluationResult:
    """
    Computes precision, recall, and F1 for a detector's flags against
    ground-truth labels.

    precision = TP / (TP + FP)  -- "of what I flagged, how much was real?"
    recall    = TP / (TP + FN)  -- "of all real anomalies, how many did I catch?"
    f1        = harmonic mean of precision and recall.
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return EvaluationResult(
        method=method_name,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
    )


def compare_methods(
    df: pd.DataFrame,
    value_col: str,
    date_col: str,
    group_cols: list,
    ground_truth_col: str,
    methods_config: dict | None = None,
) -> pd.DataFrame:
    """
    Runs every statistical method and returns a comparison table (as a
    DataFrame) of precision/recall/F1 -- this is the table that goes
    straight into the README.
    """
    methods_config = methods_config or {}
    rows = []

    for method in DETECTOR_REGISTRY:
        kwargs = methods_config.get(method, {})
        predicted = detect_anomalies(df, value_col, date_col, group_cols, method=method, **kwargs)
        result = evaluate_detector(df[ground_truth_col], predicted, method_name=method)
        rows.append(
            {
                "method": result.method,
                "precision": round(result.precision, 3),
                "recall": round(result.recall, 3),
                "f1": round(result.f1, 3),
                "true_positives": result.true_positives,
                "false_positives": result.false_positives,
                "false_negatives": result.false_negatives,
            }
        )

    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)