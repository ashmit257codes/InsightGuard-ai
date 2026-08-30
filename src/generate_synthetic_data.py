"""
generate_synthetic_data.py

Generates a realistic-looking daily business dataset (revenue, orders, region,
category) AND a version with deliberately injected anomalies + a ground-truth
label column.

WHY THIS FILE MATTERS:
Real business CSVs never come with a "this row was an anomaly" label, so you
can't compute precision/recall against them. By generating our own data with
KNOWN injected anomalies, we can later measure how well Z-score, IQR, and
Isolation Forest actually detect them -- turning "I built anomaly detection"
into "I built and evaluated three anomaly detection methods, here are the
precision/recall numbers."

Run:
    python src/generate_synthetic_data.py

Produces:
    data/clean_sales_data.csv          (no anomalies -- for early-phase testing)
    data/labeled_sales_data.csv        (anomalies injected + is_anomaly column)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Reproducibility: same "random" data every run, so your results are stable
# and repeatable -- important if you want to re-run and compare methods later.
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

REGIONS = ["North", "South", "East", "West"]
CATEGORIES = ["Electronics", "Apparel", "Home & Kitchen", "Groceries", "Sports"]

N_DAYS = 365  # one full year of daily data
START_DATE = "2025-01-01"


def generate_base_timeseries(n_days: int) -> pd.DataFrame:
    """
    Builds one row per (day, region, category) combination with a realistic
    revenue and order count.

    The revenue signal has three components layered together, which mirrors
    real business data:
      1. Trend      - slow overall growth over the year
      2. Seasonality - weekly pattern (weekends busier for some categories)
      3. Noise      - random day-to-day fluctuation

    This is a deliberately simple model. The goal isn't to perfectly simulate
    a business -- it's to have data with a KNOWN underlying pattern, so that
    when we inject anomalies, we know exactly what "normal" looks like and can
    check whether our detection methods find the deviations.
    """
    dates = pd.date_range(start=START_DATE, periods=n_days, freq="D")

    rows = []
    for region in REGIONS:
        # give each region a different baseline scale, like a real business
        region_base = rng.uniform(8000, 15000)

        for category in CATEGORIES:
            category_multiplier = rng.uniform(0.6, 1.4)

            for i, date in enumerate(dates):
                # 1. Trend: gentle linear growth over the year (~20% by year end)
                trend = 1 + (0.20 * i / n_days)

                # 2. Weekly seasonality: weekends get a lift
                is_weekend = date.dayofweek >= 5
                seasonal = 1.15 if is_weekend else 1.0

                # 3. Noise: normal random fluctuation (this is what makes
                #    Z-score/IQR necessary -- there's always some natural spread)
                noise = rng.normal(loc=1.0, scale=0.08)

                base_revenue = region_base * category_multiplier
                revenue = base_revenue * trend * seasonal * noise
                revenue = max(revenue, 0)  # revenue can't be negative

                # orders roughly track revenue but with their own noise
                orders = max(int((revenue / rng.uniform(40, 60)) * rng.normal(1.0, 0.1)), 0)

                rows.append(
                    {
                        "date": date,
                        "region": region,
                        "category": category,
                        "revenue": round(revenue, 2),
                        "orders": orders,
                    }
                )

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def inject_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deliberately corrupts specific, known rows to simulate real business
    anomalies, and records which rows were touched in `is_anomaly`.

    We inject three DIFFERENT KINDS of anomalies on purpose, because real
    detection methods behave differently on each:
      - Sudden drop   (e.g. system outage, stockout)
      - Sudden spike  (e.g. flash sale, data entry error)
      - Sustained dip (e.g. multi-day regional problem -- harder to catch
                        with single-point methods like Z-score)

    Having a mix is what makes the later comparison table interesting: no
    single method wins on all three types, and explaining that trade-off is
    exactly the kind of insight that stands out on a resume project.
    """
    df = df.copy()
    df["is_anomaly"] = False
    df["anomaly_type"] = None

    dates = df["date"].unique()

    # --- Anomaly 1: Sudden drop in North / Electronics -------------------
    drop_date = pd.Timestamp(dates[100])
    mask = (df["date"] == drop_date) & (df["region"] == "North") & (df["category"] == "Electronics")
    df.loc[mask, "revenue"] *= 0.35  # ~65% drop
    df.loc[mask, "is_anomaly"] = True
    df.loc[mask, "anomaly_type"] = "sudden_drop"

    # --- Anomaly 2: Sudden spike in South / Apparel -----------------------
    spike_date = pd.Timestamp(dates[180])
    mask = (df["date"] == spike_date) & (df["region"] == "South") & (df["category"] == "Apparel")
    df.loc[mask, "revenue"] *= 3.2  # 3x spike
    df.loc[mask, "is_anomaly"] = True
    df.loc[mask, "anomaly_type"] = "sudden_spike"

    # --- Anomaly 3: Sustained multi-day dip in East / Groceries -----------
    dip_start = pd.Timestamp(dates[250])
    dip_end = pd.Timestamp(dates[256])  # 7-day sustained dip
    mask = (
        (df["date"] >= dip_start)
        & (df["date"] <= dip_end)
        & (df["region"] == "East")
        & (df["category"] == "Groceries")
    )
    df.loc[mask, "revenue"] *= 0.55
    df.loc[mask, "is_anomaly"] = True
    df.loc[mask, "anomaly_type"] = "sustained_dip"

    # --- Anomaly 4: Sudden spike in West / Sports (different day) --------
    spike_date_2 = pd.Timestamp(dates[310])
    mask = (df["date"] == spike_date_2) & (df["region"] == "West") & (df["category"] == "Sports")
    df.loc[mask, "revenue"] *= 2.6
    df.loc[mask, "is_anomaly"] = True
    df.loc[mask, "anomaly_type"] = "sudden_spike"

    df["revenue"] = df["revenue"].round(2)
    return df


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    print("Generating base time series...")
    clean_df = generate_base_timeseries(N_DAYS)

    print("Injecting labeled anomalies...")
    labeled_df = inject_anomalies(clean_df)

    clean_path = data_dir / "clean_sales_data.csv"
    labeled_path = data_dir / "labeled_sales_data.csv"

    clean_df.to_csv(clean_path, index=False)
    labeled_df.to_csv(labeled_path, index=False)

    n_anomalies = labeled_df["is_anomaly"].sum()
    print(f"\nDone.")
    print(f"  Clean dataset:   {clean_path}  ({len(clean_df)} rows)")
    print(f"  Labeled dataset: {labeled_path}  ({len(labeled_df)} rows, {n_anomalies} labeled anomalies)")
    print(f"\nAnomaly breakdown:")
    print(labeled_df[labeled_df["is_anomaly"]]["anomaly_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
