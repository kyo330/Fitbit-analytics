# Longitudinal Health & Behavior Intelligence Platform
### Dataset: FitBit Fitness Tracker Data (Kaggle · arashnic/fitbit)

A production grade Streamlit dashboard for longitudinal behavioral analysis of real wearable data from 30 FitBit users across 31 days (March–April 2016).

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at **https://fitbit-analytics-01.streamlit.app**

---

## 📦 Project Structure

```
fitbit_platform/
├── app.py                      # Main Streamlit app (all 6 modules)
├── requirements.txt            # Python dependencies
├── README.md
└── data/
    ├── dailyActivity_merged.csv   # 859 rows · 30 users · activity data
    └── sleepDay_merged.csv        # 561 rows · 24 users · sleep data
```

---

## Dataset — Kaggle FitBit Fitness Tracker Data

**Source**: [kaggle.com/datasets/arashnic/fitbit](https://www.kaggle.com/datasets/arashnic/fitbit)

- **30 users** who consented to share FitBit tracker data
- **31 days** of data: March 12 – April 11, 2016
- Collected via Amazon Mechanical Turk distributed survey
- **Two files used**:

| File | Rows | Key Columns |
|------|------|-------------|
| `dailyActivity_merged.csv` | 859 | TotalSteps, Calories, VeryActiveMinutes, FairlyActiveMinutes, LightlyActiveMinutes, SedentaryMinutes, TotalDistance |
| `sleepDay_merged.csv` | 561 | TotalMinutesAsleep, TotalTimeInBed, TotalSleepRecords |

**Engineered columns**:
- `sleep_hours` — TotalMinutesAsleep ÷ 60
- `sleep_efficiency` — TotalMinutesAsleep ÷ TotalTimeInBed × 100
- `active_minutes` — sum of all active minute categories

---

## Modules

| # | Page | What it does |
|---|------|-------------|
| 1 | 📌 Overview | KPI cards, data preview, per-user step distribution boxplots |
| 2 | 📈 Trends | Per-user + population trend lines with 7/14-day rolling averages, anomaly markers |
| 3 | 🔍 Patterns | Day-of-week analysis, per-user profiles, correlation heatmap, activity mix pie chart |
| 4 | 🚨 Anomalies | Isolation Forest per-user, anomaly score timeline, flagged records table |
| 5 | 🔮 Prediction | Random Forest regression for Calories/Steps/Sedentary/Active mins — Actual vs Predicted |
| 6 | 🧾 Insights | Auto-generated natural language interpretations with WHO benchmarks |

---

## ML Pipeline

### Anomaly Detection
- **Model**: `sklearn.ensemble.IsolationForest`
- **Scope**: Per-user (each user's baseline is independent)
- **Contamination**: 7%
- **Features**: TotalSteps, Calories, SedentaryMinutes, VeryActiveMinutes

### Prediction
- **Model**: `sklearn.ensemble.RandomForestRegressor` (150 trees, max_depth=6)
- **Targets** (selectable): Calories, TotalSteps, SedentaryMinutes, VeryActiveMinutes
- **Features**: All activity columns + sleep minutes
- **Split**: 80/20 chronological train/test
- **Metrics shown**: MAE and R² on test set

---

## Sidebar Controls

| Control | What it does |
|---------|-------------|
| User selector | Filter to a single user or all 30 |
| Metric selector | Choose which signals appear in Trends |
| Anomaly toggle | Show/hide red anomaly markers |
| Rolling avg toggle | Show/hide 7-day and 14-day trend lines |

---

## Key Findings (from real data)

- Average daily steps: ~8,500 (below 10K WHO recommendation)
- Average sedentary time: ~15.7 hours/day (high)
- Sleep efficiency: ~94% average
- Strong steps ↔ calories correlation (r ≈ 0.73)
- Negative steps ↔ sedentary time correlation (r ≈ −0.35)

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Streamlit | Web UI |
| Plotly | Interactive charts |
| scikit-learn | Isolation Forest, Random Forest |
| pandas / numpy | Data wrangling |
| scipy | Statistical tests (Pearson r) |
