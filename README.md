# InsightGuard AI

AI-powered business intelligence & anomaly detection platform. Upload business
data (CSV/Excel), and the app automatically profiles the dataset, tracks KPI
trends, detects statistical and ML-based anomalies, identifies likely drivers,
and generates grounded business explanations using an LLM.

> 🚧 **Status: In active development.** This README is updated as each phase
> is completed — see commit history for build progress.

## Why this project

Most "AI dashboard" projects let an LLM read a spreadsheet and guess at
insights. InsightGuard AI does the opposite: **all numbers come from
deterministic Python/statistics/ML code**, and the LLM's only job is to
explain and contextualize numbers it's handed — never to invent them. This is
closer to how real analytics/BI systems that use AI are actually built.

## Planned features

- [x] Dataset upload (CSV/Excel) + preview
- [ ] Automatic data profiling & data-quality report
- [ ] KPI detection & trend analysis
- [ ] Statistical anomaly detection (Z-score, IQR) with labeled-data evaluation
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
```

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
