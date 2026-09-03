"""
ml_anomaly_detection.py -- ML-based anomaly detection (Isolation Forest,
Local Outlier Factor), plus a combined comparison against the statistical
methods from anomaly_detection.py.

Unlike Z-score/IQR (which derive thresholds automatically from the data's
own spread), both ML methods here require an upfront `contamination`
estimate -- what fraction of the data you expect to be anomalous. This is a
real trade-off: these methods can capture more complex patterns, but they
need a human-set assumption that statistical methods don't.

IMPORTANT CALIBRATION NOTE: contamination must roughly match your actual
expected anomaly rate. Setting it too high (e.g. the sklearn default
region of 5%) forces the model to flag ~5% of every group regardless of
whether real anomalies exist there -- this was tested and produced 370
false positives against only 10 true anomalies (0.14% real rate) in our
evaluation. Lower values (0.01-0.02) performed far better here.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from anomaly_detection import DETECTOR_REGISTRY, detect_anomalies, evaluate_detector


def isolation_forest_flags(
    values: pd.Series, contamination: float = 0.01, random_state: int = 42
) -> pd.Series:
    """
    Flags anomalies using Isolation Forest: builds random decision trees
    that split the data repeatedly; points that get isolated (separated
    into their own leaf) in FEWER splits than average are flagged as
    anomalies, since outliers are -- by definition -- easier to isolate
    than points buried in a dense normal cluster.
    """
    if len(values) < 10:
        return pd.Series(False, index=values.index)

    X = values.values.reshape(-1, 1)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    predictions = model.fit_predict(X)  # -1 = anomaly, 1 = normal
    return pd.Series(predictions == -1, index=values.index)


def lof_flags(
    values: pd.Series, n_neighbors: int = 20, contamination: float = 0.01
) -> pd.Series:
    """
    Flags anomalies using Local Outlier Factor: compares each point's local
    density to its neighbors' density. A point sitting in a much sparser
    neighborhood than its surroundings gets flagged.
    """
    effective_neighbors = min(n_neighbors, len(values) - 1)
    if effective_neighbors < 2:
        return pd.Series(False, index=values.index)

    X = values.values.reshape(-1, 1)
    model = LocalOutlierFactor(n_neighbors=effective_neighbors, contamination=contamination)
    predictions = model.fit_predict(X)
    return pd.Series(predictions == -1, index=values.index)


ML_DETECTOR_REGISTRY = {
    "isolation_forest": isolation_forest_flags,
    "lof": lof_flags,
}


def detect_ml_anomalies(
    df: pd.DataFrame,
    value_col: str,
    date_col: str,
    group_cols: list,
    method: str = "isolation_forest",
    **kwargs,
) -> pd.Series:
    """
    Same pattern as detect_anomalies() in anomaly_detection.py, but for ML
    methods -- runs the chosen ML detector independently per group.
    """
    if method not in ML_DETECTOR_REGISTRY:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(ML_DETECTOR_REGISTRY)}")

    detector_fn = ML_DETECTOR_REGISTRY[method]
    result = pd.Series(False, index=df.index)
    sorted_df = df.sort_values(date_col)

    for _, group_df in sorted_df.groupby(group_cols):
        flags = detector_fn(group_df[value_col], **kwargs)
        result.loc[flags.index] = flags

    return result


def compare_all_methods(
    df: pd.DataFrame,
    value_col: str,
    date_col: str,
    group_cols: list,
    ground_truth_col: str,
    stat_config: dict | None = None,
    ml_config: dict | None = None,
) -> pd.DataFrame:
    """
    Runs EVERY method -- statistical (Z-score, IQR, rolling baseline) AND
    ML (Isolation Forest, LOF) -- and returns one combined comparison table,
    sorted by F1. This is the final table for the README.
    """
    stat_config = stat_config or {}
    ml_config = ml_config or {}
    rows = []

    for method in DETECTOR_REGISTRY:
        kwargs = stat_config.get(method, {})
        predicted = detect_anomalies(df, value_col, date_col, group_cols, method=method, **kwargs)
        result = evaluate_detector(df[ground_truth_col], predicted, method_name=method)
        rows.append(_result_to_row(result, category="statistical"))

    for method in ML_DETECTOR_REGISTRY:
        kwargs = ml_config.get(method, {})
        predicted = detect_ml_anomalies(df, value_col, date_col, group_cols, method=method, **kwargs)
        result = evaluate_detector(df[ground_truth_col], predicted, method_name=method)
        rows.append(_result_to_row(result, category="ml"))

    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def _result_to_row(result, category: str) -> dict:
    return {
        "method": result.method,
        "category": category,
        "precision": round(result.precision, 3),
        "recall": round(result.recall, 3),
        "f1": round(result.f1, 3),
        "true_positives": result.true_positives,
        "false_positives": result.false_positives,
        "false_negatives": result.false_negatives,
    }