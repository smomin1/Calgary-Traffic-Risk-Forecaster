# Calgary Traffic Risk Forecaster

A machine learning system that predicts which Calgary communities are at elevated risk of severe traffic incidents in the following month. Built on a panel dataset of 198 communities spanning December 2016 to February 2026.

---

## Project Structure

```
├── Calgary_Traffic_Risk_Forecaster.ipynb     # Data pipeline, feature engineering & EDA
├── modeling_high_risk_prediction.ipynb       # Modeling, evaluation & forecasting
├── calgary_traffic_panel_dataset.csv         # Engineered panel dataset (output of notebook 1)
├── calgary_risk_model.pkl                    # Full model — 25 features (with weather)
├── calgary_risk_model_lean.pkl               # Lean model — 17 features (no weather)
└── README.md
```

---

## Notebooks

### 1. `Calgary_Traffic_Risk_Forecaster.ipynb` — Data Pipeline & EDA

Covers the full data pipeline from raw sources to analysis-ready panel:

**Data Integration**
- Traffic incidents from City of Calgary open data (`Traffic_Incidents_20260216.csv`)
- Daily climate data from Environment Canada (`calgary_climate_daily_2016_2026.csv`)
- Community boundaries via spatial join (GeoPandas + Shapely)
- Census population data merged by community code

**Feature Engineering**
- Severity rates: `severe_per_1000`, `incidents_per_1000`, `incidents_per_km2`
- Lag features: `lag_1`, `lag_3_avg`, `rolling_std_6`, `lag_severe`
- Weather interactions: `snow_squared`, `wind_snow_interaction`, `cold_days_flag`
- Cyclical temporal encoding: `sin_month`, `cos_month`
- Binary target: `high_risk` (top 25% of communities by `severe_per_1000` per month)

**Exploratory Data Analysis**
- City-level time series (2016–2026) — COVID structural break identified
- Bimodal seasonality — winter (Feb) and late-summer (Aug–Sep) risk peaks
- Community risk profiling — chronic vs. spike-driven risk separation
- Climate correlation analysis — near-zero linear weather associations
- Outlier detection — small-population rate distortion (BVD, CNS)
- High-risk label stability diagnostics

---

### 2. `modeling_high_risk_prediction.ipynb` — Modeling & Forecasting

**Baseline Models**
- Logistic Regression (AUC: 0.736)
- Random Forest baseline (AUC: 0.748)

**Main Modeling Pipeline**

| Step | Detail |
|------|--------|
| Feature set | 25 features including `community_mean_risk` (expanding historical mean) |
| Outlier handling | Winsorization of rate columns at 99th percentile |
| Cross-validation | Panel-aware `TimeSeriesSplit` on month boundaries (not row indices) |
| Scaler | `StandardScaler` fit inside each CV fold |
| Class imbalance | `class_weight="balanced"` (RF) / `scale_pos_weight` (XGBoost) |
| Model selection | Random Forest (CV AUC: 0.767) outperformed XGBoost (CV AUC: 0.734) |
| Calibration | Isotonic regression on temporally held-out calibration set |
| Threshold | Optimized via precision-recall curve, set at 0.30 for recall priority |

**Experiments & Ablations**

- **COVID Split Test** : confirmed pre-2020 data hurts performance; post-COVID training (2021+) improves AUC by ~3.5 points
- **Weather Ablation** : 8 weather features contribute negligible AUC improvement (+0.002); lean 17-feature model saved as alternative
- **Per-community sanity check** : BRD, DNC, WIN, ALB, VIS all achieve recall of 1.00 across 36 test months

**Forecasting**
- Predicts next month's high-risk communities using the most recent available panel data
- Weather inputs replaced with historical climate averages for the forecast month
- Lag and rolling features updated to reflect most recent observed months

---

## Model Performance

### Final Model (Post-COVID, Calibrated Random Forest)

| Metric | Full Model (25 features) | Lean Model (17 features) |
|--------|--------------------------|--------------------------|
| ROC-AUC | 0.7748 | 0.7728 |
| Recall (high-risk) | 0.55 | 0.57 |
| Precision (high-risk) | 0.53 | 0.52 |
| Classification threshold | 0.30 | 0.30 |

---

## Key Findings

**`community_mean_risk` is the dominant feature (~24% importance)** : encoding each community's historical high-risk rate as an expanding mean (no future leakage) outperforms all weather, incident count, and lag features.

**Weather is not a primary driver** : snow, temperature, and wind show near-zero linear correlation with severity rates at monthly community aggregation. Weather ablation confirms removing all 8 weather features costs only 0.002 AUC.

**A COVID structural break exists** : pre-2020 data reflects a different incident regime (higher volumes, different commute patterns). Training exclusively on post-2021 data improves generalization on recent test data.

**Seasonality is bimodal** : risk peaks in both February (winter) and August–September (summer pedestrian activity), not just winter as commonly assumed.

**Small-population communities distort rate metrics** : communities like Belvedere (BVD) and Cornerstone (CNS) appear as extreme outliers due to low population denominators (3–4 incidents → 80+ per 1,000). Winsorization at the 99th percentile is applied before modeling.

---

## Known Limitations

- Communities near the decision boundary (probability 0.25–0.35) are inherently uncertain and should be reviewed manually
- The high-risk label uses a relative top-25% threshold per month, which becomes unstable in low-activity months
- Population data is static, no year-over-year updates per community
- Weather forecast inputs use historical climate averages; actual forecast data would improve predictions
- The lean model (no weather) requires no weather data source in production but loses marginal predictive signal

---

## Data Sources

| Source | Description |
|--------|-------------|
| [Traffic Incidents in Communities](https://data.calgary.ca/Transportation-Transit/Traffic-incidents-in-Communities/5vyw-zf4x) | City of Calgary traffic incident records |
| [Community District Boundaries](https://data.calgary.ca/Base-Maps/Community-District-Boundaries/surr-xmvs/data_preview) | Community polygon boundaries for spatial join |
| [Historical Community Populations](https://data.calgary.ca/Demographics/Historical-Community-Populations/jtpc-xgsh/data_preview) | Annual population estimates by community |
| [Environment Canada — Calgary Airport](https://climate.weather.gc.ca/climate_data/daily_data_e.html?StationID=50430&Prov=AB) | Daily climate data (Station ID: 50430, 2016–2026) |

---

## Requirements

```
pandas
numpy
geopandas
shapely
scikit-learn
xgboost
matplotlib
seaborn
joblib
```

Install with:
```bash
pip install pandas numpy geopandas shapely scikit-learn xgboost matplotlib seaborn joblib
```

---

## Reproducing the Models
The saved model files are not included in this repository due to size.
To regenerate them, run the notebooks in order:
1. `Calgary_Traffic_Risk_Forecaster.ipynb` , produces `calgary_traffic_panel_dataset.csv`
2. `modeling_high_risk_prediction.ipynb` , produces `calgary_risk_model.pkl` and `calgary_risk_model_lean.pkl`

---

## Loading the Saved Models

```python
import joblib
import pandas as pd

# Load full model
bundle = joblib.load("calgary_risk_model.pkl")
model      = bundle["model"]
scaler     = bundle["scaler"]
features   = bundle["feature_cols"]
threshold  = bundle["threshold"]

# Load lean model (no weather features)
lean = joblib.load("calgary_risk_model_lean.pkl")

# Predict
X_new = pd.DataFrame(...)  # your feature data
X_scaled = pd.DataFrame(scaler.transform(X_new[features]), columns=features)
probs = model.predict_proba(X_scaled)[:, 1]
predictions = (probs >= threshold).astype(int)
```
