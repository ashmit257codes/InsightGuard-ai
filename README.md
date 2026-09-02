# InsightGuard AI

AI-powered business intelligence & anomaly detection platform. Upload business
data (CSV/Excel), and the app automatically profiles the dataset, tracks KPI
trends, detects statistical and ML-based anomalies, identifies likely drivers,
and generates grounded business explanations using an LLM.

> 🚧 **Status: In active development.** Core data pipeline (upload → profiling
> → KPI trend analysis) is working end-to-end. Anomaly detection and LLM
> insight layers are next. See commit history for build progress.

## Why this project

Most "AI dashboard" projects let an LLM read a spreadsheet and guess at
insights. InsightGuard AI does the opposite: **all numbers come from
deterministic Python/statistics/ML code**, and the LLM's only job is to
explain and contextualize numbers it's handed — never to invent them. This is
closer to how real analytics/BI systems that use AI are actually built.

## Current data pipeline

```text
Upload (CSV/Excel)
   ↓
Column classification (datetime / numeric / categorical / identifier)
   ↓
Data quality report (missing values, duplicates, constant columns, outlier flags)
   ↓
KPI aggregation (multi-row-per-day → clean daily/weekly/monthly time series)
   ↓
Trend metrics (% change, moving average, volatility, trend direction)
```

Each stage is implemented as a standalone, independently-tested module
(`src/profiling.py`, `src/kpi_engine.py`) rather than logic embedded directly
in the UI — this keeps the analytics layer reusable once the anomaly
detection and LLM layers are added on top.

## Planned features


- [x] Dataset upload (CSV/Excel) + preview
- [x] Automatic data profiling & data-quality report
- [x] KPI trend analysis (% change, moving average, volatility, trend classification)
- [x] Statistical anomaly detection (Z-score, IQR, rolling baseline) with labeled-data evaluation
- [ ] ML anomaly detection (Isolation Forest, LOF) with precision/recall comparison
- [ ] Anomaly severity scoring
- [ ] Root-cause / driver analysis
- [ ] LLM-grounded business insight generation
- [ ] Chat-with-your-data agent (tool-calling)
- [ ] Anomaly history + feedback loop
- [ ] (Stretch) Email alerts for critical anomalies

## Tech stack

- **App**: Streamlit (Python-only, single deployable app)
- **Data**: Pandas, NumPy
- **ML**: scikit-learn (Isolation Forest, Local Outlier Factor)
- **LLM**: Groq API (Llama models, free tier)
- **Storage**: SQLite (anomaly history/feedback)
- **Deployment**: Streamlit Community Cloud (free)

## Evaluation approach

Real business data has no ground-truth "this was an anomaly" labels. To
actually measure detection quality (not just eyeball it), this project
generates a synthetic dataset with **deliberately injected, labeled
anomalies** (`src/generate_synthetic_data.py`) covering sudden drops, sudden
spikes, and sustained multi-day dips. Each detection method is evaluated
against these ground-truth labels using precision/recall/F1 — see
`docs/evaluation.md` (added in Phase 5/6) for results.

### Statistical method results (Day 4)

Evaluated against 10 ground-truth injected anomalies across 7,300 rows:

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Z-score | 0.290 | 0.900 | **0.439** |
| IQR | 0.122 | 0.900 | 0.214 |
| Rolling baseline (14-day) | 0.093 | 0.400 | 0.151 |

**Key finding:** Z-score outperformed IQR despite both catching 9/10 anomalies,
because IQR's fixed quartile thresholds triggered more false positives on
normal day-to-day noise. Rolling-baseline underperformed on sustained
multi-day anomalies specifically — its adaptive window began treating the
ongoing dip as "the new normal" after the first day, a known limitation of
adaptive-baseline approaches. ML-based detection (Isolation Forest) is
compared against these same baselines in the next phase.


## Getting started

```bash
git clone <your-repo-url>
cd insightguard-ai
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Generate synthetic sample data
python src/generate_synthetic_data.py


# Run the app
streamlit run src/app.py

# Run the test suite
pytest tests/ -v


## Project structure

```text
insightguard-ai/
├── data/           # sample/synthetic datasets
├── src/            # application + analytics code
├── tests/          # unit tests
├── notebooks/      # exploratory analysis
├── docs/           # architecture, evaluation, design notes
└── requirements.txt
```

## License

MIT
