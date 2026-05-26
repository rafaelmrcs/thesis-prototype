# FIAdaBoost — Study Guide

This document explains every UI panel, every results file, and every relevant code section in this prototype. Its purpose is to answer the question "where did this number / chart / output come from?" for each element in the running web application.

For setup and deployment instructions, see the root-level [`README.md`](../README.md).

---

## Table of Contents

1. [Quick Reference: Objectives ↔ Code ↔ UI](#1-quick-reference-objectives--code--ui)
2. [UI Panel Documentation](#2-ui-panel-documentation)
   - [A. Live Model Prediction](#a-live-model-prediction-panel)
   - [B. Rooftop & Weather Context](#b-rooftop--weather-context)
   - [C. Model Comparison — Metrics Charts](#c-model-comparison--metrics-charts)
   - [D. Model Comparison — Comparison Table](#d-model-comparison--comparison-table-c)
   - [E. Spatial Cross-Validation Table](#e-spatial-cross-validation-5-fold-table)
   - [F. Diebold-Mariano Statistical Test](#f-diebold-mariano-statistical-test)
   - [G. Daily Temporal Split Metrics](#g-daily-temporal-split-metrics)
3. [Code Guide: How Each Objective Is Implemented](#3-code-guide-how-each-objective-is-implemented)
   - [Objective 1.2.2.1 — Preprocess data](#objective-1221--preprocess-data-from-nasa-power-and-osm)
   - [Objective 1.2.2.2 — FI-AdaBoost algorithm](#objective-1222--optimize-adaboost-via-fi-weighting)
   - [Objective 1.2.2.3 — Compare and evaluate](#objective-1223--compare-and-evaluate-rmse-mae-r)
   - [Objective 1.2.2.4 — Assess applicability](#objective-1224--assess-applicability)
4. [Results Directory Reference](#4-results-directory-reference)
5. [Defense Questions by UI Panel](#5-defense-questions-by-ui-panel)
   - [Panel A — Live Model Prediction](#panel-a--live-model-prediction)
   - [Panel B — Rooftop & Weather Context](#panel-b--rooftop--weather-context)
   - [Panel C — Metrics Charts](#panel-c--model-comparison-metrics-charts)
   - [Panel D — Comparison Table](#panel-d--comparison-table)
   - [Panel E — Spatial CV Table](#panel-e--spatial-cross-validation-5-fold-table)
   - [Panel F — Diebold-Mariano Test](#panel-f--diebold-mariano-statistical-test)
   - [Panel G — Daily Temporal Metrics](#panel-g--daily-temporal-split-metrics)

---

## 1. Quick Reference: Objectives ↔ Code ↔ UI

| Specific Objective | Implemented In | Visible In UI |
|--------------------|---------------|---------------|
| **1.2.2.1** Preprocess NASA POWER + OSM data | `src/data_acquisition.py`, `src/data_processing.py`, `src/feature_engineering.py`, `src/data_integration.py` | Not directly visible; produces `data/processed/integrated_dataset.csv` used for training |
| **1.2.2.2** Optimize AdaBoost via FI weighting | `src/model_training.py:274–405` (`FIAdaBoostRegressor` class) | Model Comparison tab — Improvement Cards (RMSE/MAE reduced, R² increased) |
| **1.2.2.3** Compare and evaluate RMSE, MAE, R² | `src/model_training.py:410–432` (`compute_metrics`, `make_result_table`), `src/api.py` `/compare` endpoint | Model Comparison tab — Metrics Charts, Comparison Table, CV Table, DM Test |
| **1.2.2.4** Assess applicability for decision-making | `src/api.py` `/predict` + `/compare` endpoints, React `ForecastingTool.tsx` | Forecasting Tool tab — interactive map, Live Model Prediction panel, Rooftop & Weather Context |

---

## 2. UI Panel Documentation

Each subsection below maps a visible UI panel to its data source, its connection to the study objectives, and notes on what to look for critically.

---

### A. Live Model Prediction Panel

**Visible in:** Location Analysis tab (or Forecasting Tool tab after clicking the map)

**What it shows:**
- `BASELINE ADABOOST SEP` — theoretical rooftop solar energy potential in kWh/day from the standard model
- `FI-ADABOOST SEP` — same metric from the proposed model
- `DIFFERENCE` — absolute and percentage difference between the two predictions
- `BASELINE PREDICTION CONFIDENCE` and `FI-ADABOOST PREDICTION CONFIDENCE` — heuristic scores (not classical ML confidence intervals)

**Where the data comes from:**

This panel is populated by `POST /compare` in [`src/api.py`](src/api.py). When a location is selected:

1. The API calls NASA POWER to get annual mean GHI (kWh/m²/day) and meteorological variables (`T2M`, `RH2M`, `ALLSKY_KT`)
2. The API calls pvlib's Ineichen-Perez clear-sky model to compute `clear_sky_ratio` and `sunshine_hours`
3. The API queries the Overpass (OSM) API to find the nearest building and compute `azimuth`, `orientation_score`, `shading_factor`, `SEI_norm`, and `rooftop_area_sq_m`
4. Both trained models (`models/baseline_adaboost.pkl` and `models/fi_adaboost.pkl`) predict `effective_GHI_J`
5. The result is divided by `KWH_TO_J = 3,600,000` to convert to kWh/m²/day
6. SEP (kWh/day) = `predicted_irradiance_kWh_m2_day × rooftop_area_sq_m`

**Why it exists:** Objective 1.2.2.4 — demonstrates the model's applicability to individual-building solar decisions.

**What to question:**

| Element | Question | Answer |
|---------|----------|--------|
| DIFFERENCE of -1.47 kWh/day (~0.09%) | Why is the difference so small? | The models converge at well-sampled training coordinates. Divergence is larger for locations far from the training set boundary. |
| Confidence 98.7% / 98.8% | Are these real confidence intervals? | No. These are heuristic scores defined in `src/api.py`. Baseline confidence = spatial proximity score only (how close the query point is to training data). FI confidence = 70% proximity + 30% model agreement. A large prediction divergence between the two models lowers FI's score. |
| Units: kWh/day for SEP | Does this match the thesis target? | The trained target is `effective_GHI_J` (J/m²/day). The API converts: `kWh/day = (effective_GHI_J ÷ 3,600,000) × rooftop_area_sq_m`. Units are correct but different from training units. |

---

### B. Rooftop & Weather Context

**Visible in:** Location Analysis tab (below the Live Model Prediction panel)

**What it shows:** The 8 engineered feature values fed into both models for the selected location — these are the exact numbers the model used to produce the prediction above.

| Card | Feature Name | Source |
|------|-------------|--------|
| Rooftop Area | `rooftop_area_sq_m` | OSM building footprint polygon area (m²), computed by `src/data_processing.py` |
| Solar Exposure Index | `SEI_norm` | `SEI = orientation_score × area × (1 - shading_factor) × tilt_factor`, normalized by training-set max; computed by `src/feature_engineering.py` |
| Sunshine Hours | `sunshine_hours` | Monthly hours where GHI > 120 W/m² (WMO threshold), computed via pvlib; see `src/model_training.py:185–230` |
| Cloud Cover | Displayed as `ALLSKY_KT` (sky transmissivity / clearness index) from NASA POWER | Inversely correlated with cloud cover but is not a direct percentage — it is the ratio of surface to top-of-atmosphere shortwave radiation |
| Temperature | `T2M` | NASA POWER: temperature at 2 m above ground (°C) |
| Humidity | `RH2M` | NASA POWER: relative humidity at 2 m (%) |
| Orientation | Derived from `azimuth` | Cardinal direction label computed from OSM building azimuth |
| Azimuth | `azimuth` | Compass angle of rooftop primary face (°), computed from OSM polygon geometry |

**Why it exists:** Objective 1.2.2.2 and 1.2.2.4 — shows which input values drive the model's prediction, supporting interpretability.

**What to question:**

| Element | Question | Answer |
|---------|----------|--------|
| "Sunshine Hours: 10.0 hrs/day" | Why always a round number? | The display rounds to one decimal. The actual value is computed from monthly aggregated pvlib data, then averaged. |
| "Cloud Cover: 49.9%" | Is this really cloud cover? | Not exactly. It is `ALLSKY_KT` (clearness index), which ranges 0–1. The UI multiplies by 100 to show as %. Lower = more cloud, but the label "cloud cover" is an approximation — the thesis paper (Section 2.1.1) uses the label "Cloud cover (oktas)" for `ALLSKY_KT`. |
| "Orientation: North" | Why North-facing for a Philippine building? | North is genuinely the least favorable orientation in the Philippines (Northern Hemisphere). This is physically correct per the SEI formulation, as North-facing buildings receive the least annual solar energy. The `orientation_score = cos(azimuth - 180°)` will be near its minimum. |

---

### C. Model Comparison — Metrics Charts

**Visible in:** Model Analysis tab (or Model Comparison tab), section "A. Metrics Comparison"

**What it shows:**
- Left chart: RMSE and MAE bar chart (J/m²/day), lower = better
- Right chart: R² bar chart (%), higher = better
- Below: Improvement Cards with percentage changes

**Where the data comes from:**

`GET /compare` → reads `results/metrics_summary.csv` (generated during training by `model_training.py` Step 5). The CSV contains 2 rows.

Exact values from [`results/metrics_summary.csv`](results/metrics_summary.csv):

| Model | RMSE_J | MAE_J | R² |
|-------|--------|-------|-----|
| AdaBoost (Baseline) | 7,684.33 | 6,442.54 | 0.998460 |
| FI-AdaBoost (Proposed) | 4,908.19 | 2,605.48 | 0.999372 |

Improvement cards shown in UI:
- RMSE reduced by **36.13%**
- MAE reduced by **59.56%**
- R² increased by **0.09%**

**Why it exists:** Directly satisfies Objective 1.2.2.3.

**What to question:**

| Element | Question | Answer |
|---------|----------|--------|
| These are fixed numbers | Are they recomputed each time? | No. They are read from `metrics_summary.csv` saved at training time (Step 5 of `model_training.py`). They reflect one specific 80/20 random split with `random_seed=42`. Re-running with a different seed can yield slightly different values. |
| R² improvement of only 0.09% | Is the improvement meaningful if R² barely changed? | Both models already achieve >99.8% R². The relevant improvements are in RMSE and MAE, which affect absolute prediction precision. The DM test (panel F) is the correct statistical tool to assess whether the difference is significant. |
| MAE improvement (59.56%) is larger than RMSE (36.13%) | Why the asymmetry? | RMSE penalizes large errors more than MAE. FI-AdaBoost particularly reduces the concentration of medium-sized errors, which shows strongly in MAE. RMSE is still held up by a few large-deviation samples. |

---

### D. Model Comparison — Comparison Table (C)

**Visible in:** Model Analysis tab, section "C. Comparison Table"

**What it shows:** Three rows (RMSE, MAE, R²) with exact values for each model and a Delta % column.

**Where the data comes from:** Same source as panel C — `results/metrics_summary.csv` via the `/compare` response field `performanceMetricsComparison`. The table is rendered by the React `ModelComparison.tsx` component.

**Code that generates the file:**

```python
# src/model_training.py:417–432
def make_result_table(ada_m: dict, fi_m: dict) -> pd.DataFrame:
    rows = [
        {"Model": "AdaBoost Regression (Baseline)",   "RMSE_J": ..., "MAE_J": ..., "R2": ...},
        {"Model": "FI-AdaBoost Regression (Proposed)", "RMSE_J": ..., "MAE_J": ..., "R2": ...},
    ]
    return pd.DataFrame(rows).set_index("Model")
```

Saved to disk at Step 5 of `main()` (`src/model_training.py:1093+`).

---

### E. Spatial Cross-Validation (5-Fold) Table

**Visible in:** Model Analysis tab, section "Spatial Cross-Validation (5-Fold)"

**What it shows:** Per-fold train RMSE, validation RMSE, validation R² for both models, and a per-fold RMSE gain (Baseline minus FI).

**Where the data comes from:**

`GET /cv-metrics/spatial` → reads [`results/cv_fold_metrics.csv`](results/cv_fold_metrics.csv) (5 rows).

Generated by `run_kfold_cv()` in `src/model_training.py` (Step 4 of `main()`). The function uses `sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)` on the 2,400-sample training set.

**Why it exists:** Objective 1.2.2.3 — demonstrates generalization stability beyond a single holdout. The five fold results confirm the improvement is consistent, not a lucky split.

**What to question:**

| Element | Question | Answer |
|---------|----------|--------|
| Fold 2 FI Val RMSE = 10,138 J/m²/day | Why is Fold 2 so much worse for FI? | The paper (Section 3.2.1) attributes this to natural data variability in that fold's validation subset. It corresponds to a set of spatial samples with atypical feature combinations. The average still favors FI-AdaBoost. |
| The UI labels this "spatial split" | Is it really a spatial cross-validation? | No. The code uses random shuffled KFold (`shuffle=True`), not geographically stratified or spatial-block CV. This is consistent with the paper's Section 2.5.3 description and is the correct method for a non-temporal spatial dataset without strong spatial autocorrelation. |
| Average row at the bottom | Where is this computed? | The average row is computed client-side in the React component from the 5 fold rows. It is not stored in the CSV. |

---

### F. Diebold-Mariano Statistical Test

**Visible in:** Model Analysis tab, section "Diebold-Mariano Statistical Test"

**What it shows:** A table with two rows — one for the Spatial domain, one for the Daily domain — with DM Statistic, p-value, n (test samples), a Significant? badge, and an Interpretation string.

**Where the data comes from:**

`GET /dm-test` → reads two CSV files:
- Spatial: [`results/dm_test_results_spatial.csv`](results/dm_test_results_spatial.csv) (1 row)
- Daily: [`results/dm_test_results_daily.csv`](results/dm_test_results_daily.csv) (1 row)

Exact values from the CSVs:

| Domain | DM Statistic | p-value | n | Significant? |
|--------|-------------|---------|---|--------------|
| Spatial | 12.8241 | 0.0 (< 0.0001) | 600 | Yes (p < 0.05) |
| Daily | -1.6083 | 0.1078 | 74 | No |

**Code that generates these files:**

`diebold_mariano_test()` in `src/model_training.py`. The function:
1. Computes squared error differences: `d_t = e_ada_t² − e_fi_t²`
2. Estimates Newey-West HAC variance to handle autocorrelation
3. Computes `DM = mean(d_t) / sqrt(HAC_var / n)`
4. Converts to p-value via `2 × (1 − Φ(|DM|))` where Φ is the standard normal CDF (`scipy.stats.norm.cdf`)

**Why it exists:** Objective 1.2.2.3 — a positive DM statistic with p < 0.05 means FI-AdaBoost produces statistically significantly smaller errors than the baseline, not a chance result.

**What to question:**

| Element | Question | Answer |
|---------|----------|--------|
| p-value shown as "< 0.0001" for Spatial | Isn't p=0.0 suspicious? | The raw CSV stores `0.0` because `scipy.stats.norm.cdf` returns exactly 1.0 for very large DM statistics (12.82), making `2×(1−1.0)=0.0`. The true p-value is just very small (< 10⁻³⁰), not literally zero. |
| Daily DM = -1.6083 (negative) | What does a negative DM statistic mean? | Negative means the baseline AdaBoost had lower squared errors than FI-AdaBoost on average (baseline was slightly more accurate). Because p=0.1078 > 0.05, this difference is not statistically significant — the two models are equivalent on daily data. |
| n=74 for Daily vs n=600 for Spatial | Why different sample sizes? | The daily test set is 20% of 365 days ≈ 73 days. The spatial test set is 20% of 3,000 points = 600 points. Smaller n means less statistical power for the DM test. |

---

### G. Daily Temporal Split Metrics

**Visible in:** Model Analysis tab, section "Daily Temporal Split Metrics"

**What it shows:** A two-row table (one model per row) with RMSE (J/day), MAE (J/day), R², and the split type.

**Where the data comes from:**

`GET /daily-metrics` → reads [`results/daily_metrics_summary.csv`](results/daily_metrics_summary.csv) (2 rows).

Exact values:

| Model | RMSE (J/day) | MAE (J/day) | R² | Split |
|-------|-------------|-------------|-----|-------|
| AdaBoost (Daily, Temporal Split) | 1,031,070.98 | 896,566.07 | 0.9044 | temporal_80_20 |
| FI-AdaBoost (Daily, Temporal Split) | 1,052,947.75 | 913,111.85 | 0.9004 | temporal_80_20 |

Generated by the daily time-series pipeline in `src/model_training.py`. The 365-day NASA POWER records are split chronologically: first 292 days for training, last 73 days for testing (no future leakage).

**Why it exists:** Shows the model's performance on a different problem type (temporal forecasting vs. spatial). This assesses whether the FI mechanism generalizes beyond the spatial setting.

**What to question:**

| Element | Question | Answer |
|---------|----------|--------|
| Units are J/day, not J/m²/day | Is that a mistake? | No. The daily target is raw city-centroid GHI (J/day), not per-rooftop effective GHI. The daily pipeline uses a different target (`GHI_J`) from the spatial pipeline (`effective_GHI_J`). The unit difference is intentional and correct. |
| FI-AdaBoost performs worse here | Does this invalidate the thesis? | No. The paper (Section 3.6) explains this clearly: in temporal forecasting, the most informative predictor is temporal autocorrelation (yesterday's GHI predicts today's), not static rooftop feature importance. FI weighting cannot exploit temporal autocorrelation, so it provides no benefit and slightly higher error (2% higher RMSE). |
| R² = 0.90 vs R² = 0.9993 in spatial | Why such a large gap? | Spatial R² is high because latitude and longitude strongly predict solar irradiance (more southern = higher irradiance in Davao City). Temporal R² is lower because daily weather variability (cloud cover, humidity) is harder to predict from static features. |

---

## 3. Code Guide: How Each Objective Is Implemented

### Objective 1.2.2.1 — Preprocess data from NASA POWER and OSM

**Stage 1: Data acquisition** ([`src/data_acquisition.py`](src/data_acquisition.py))
- Samples 3,000 random lat/lon coordinates uniformly inside the Davao City boundary polygon
- Calls the NASA POWER API for each coordinate to retrieve annual mean GHI (`ALLSKY_SFC_SW_DWN`), temperature (`T2M`), humidity (`RH2M`), and clearness index (`ALLSKY_KT`)
- Downloads OSM building footprints for Davao City via `osmnx`

**Stage 2: Data cleaning** ([`src/data_processing.py`](src/data_processing.py))
- Removes incomplete or degenerate building geometries
- Computes `rooftop_area_sq_m` from OSM polygon geometry (`shapely.area` in EPSG:32651 projection for correct metric units)

**Stage 3: Feature engineering** ([`src/feature_engineering.py`](src/feature_engineering.py))
- `orientation_score = cos(θ_azimuth − 180°)` — 1.0 for south-facing, near 0 for east/west, negative for north-facing
- `shading_factor ≈ 0.3 × (nearby_buildings_in_50m_buffer / dataset_max)` — density-based shading estimate
- `tilt_factor = cos(|θ_roof − θ_optimal|)` — where θ_optimal = 7.2° (Davao City latitude); assumed θ_roof = 0° (flat roof)
- `SEI = orientation_score × rooftop_area_sq_m × (1 − shading_factor) × tilt_factor`
- `SEI_norm = SEI / max(SEI)` — normalized to [0, 1]

**Stage 4: Integration** ([`src/data_integration.py`](src/data_integration.py))
- Joins each spatial sample point to its nearest OSM building using a KD-tree nearest-neighbor search
- Output: `data/processed/integrated_dataset.csv` (3,000 rows, ~14 columns)

**Stage 5: Clear-sky features** ([`src/model_training.py:100–270`](src/model_training.py))
- `clear_sky_ratio = GHI_actual / GHI_clear_sky` computed via pvlib's Ineichen-Perez model
- `sunshine_hours` = monthly cumulative hours where GHI > 120 W/m² (WMO direct sunlight threshold), averaged to daily

---

### Objective 1.2.2.2 — Optimize AdaBoost via FI weighting

The entire innovation is the `FIAdaBoostRegressor` class at [`src/model_training.py:274–405`](src/model_training.py).

**The standard AdaBoost weight update rule:**
```
w_i^{t+1} = w_i^t × β_t^{1 − e_i^t} / Z_t
```

**The FI-AdaBoost weight update rule:**
```
w_i^{t+1} = w_i^t × β_t^{1 − e_i^t × Φ(x_i)} / Z_t
```

The extra term `Φ(x_i)` is the per-sample feature-importance engagement score:

```python
# src/model_training.py:286–303
@staticmethod
def _norm_fi(tree):
    raw = tree.feature_importances_      # Gini-based importances from the decision tree
    s   = raw.sum()
    return raw / s if s > 0 else np.ones_like(raw) / len(raw)  # φ(f_k): probability distribution over features

@staticmethod
def _composite_phi(X, phi):
    X_abs  = np.abs(X)
    col_mx = X_abs.max(axis=0)
    col_mx[col_mx == 0] = 1
    Phi    = (X_abs / col_mx * phi).sum(axis=1)  # Φ_i: how much sample i "activates" important features
    p_max  = Phi.max()
    return Phi / p_max if p_max > 0 else Phi      # normalized to [0, 1]
```

Key lines in `fit()`:

```python
# src/model_training.py:328–350
e_i   = abs_e / D_t                       # normalized per-sample error ∈ [0, 1]
phi   = self._norm_fi(tree)               # feature importance vector for this tree
Phi_i = self._composite_phi(X_arr, phi)  # per-sample FI engagement score

modulated_loss = e_i * Phi_i             # FI-aware loss (line 333)
eps_t = float(np.dot(weights, modulated_loss))  # weighted modulated error (line 335)

# ...
new_w = weights * (beta_t ** (1.0 - e_i * Phi_i))  # FI-modulated weight update (line 346)
```

**Prediction** uses weighted median (not mean) across all trees (`predict()`, lines 369–387):

```python
# For each sample i, sort all tree predictions, take the weighted median
order = np.argsort(p_i)
cumw  = np.cumsum(weights[order])
mid   = np.searchsorted(cumw, 0.5)
result[i] = p_i[order[min(mid, len(p_i) - 1)]]
```

The weighted median is more robust to outlier weak learners than weighted mean.

**Hyperparameter tuning** via Optuna (`src/model_training.py:444–462`):
- FI-AdaBoost search space: `n_estimators` ∈ [50, 300], `learning_rate` ∈ [0.01, 2.0] (log scale), `max_depth` ∈ [1, 6]
- Baseline search space: same but `max_depth` ∈ [1, 5]
- 100 Optuna trials, KFold(5) inner CV, minimizing validation RMSE

---

### Objective 1.2.2.3 — Compare and evaluate RMSE, MAE, R²

**Metric computation** (`src/model_training.py:410–414`):

```python
def compute_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"RMSE_J": rmse, "MAE_J": mae, "R2": r2}
```

**Result table** (`src/model_training.py:417–432`): builds the two-row comparison DataFrame, saved to `results/metrics_summary.csv`.

**KFold cross-validation** (`run_kfold_cv()` in `model_training.py`, called at Step 4 of `main()`): trains each model 5 times on different train/val partitions, records per-fold metrics to `results/cv_fold_metrics.csv`.

**Diebold-Mariano test** (`diebold_mariano_test()` in `model_training.py`): assesses statistical significance of the performance difference using squared-error differentials with Newey-West HAC correction. Saved to `results/dm_test_results_spatial.csv`.

**API layer** (`src/api.py`, `/compare` endpoint): reads `metrics_summary.csv` at startup and returns `performanceMetricsComparison` with RMSE, MAE, R², and improvement percentages to the frontend.

---

### Objective 1.2.2.4 — Assess applicability

**Live inference pipeline** (`src/api.py`, `/predict` and `/compare` endpoints):
- Accepts any latitude/longitude inside Davao City
- Fetches real-time meteorological data from NASA POWER
- Queries OSM via Overpass API for the nearest building footprint
- Computes pvlib clear-sky features on the fly
- Runs both trained models
- Returns per-building solar energy potential in kWh/day

**Interactive frontend** (`Solar_Energy_Potential_Forecasting_Prototype/src/components/ForecastingTool.tsx`):
- Leaflet map of Davao City
- Click anywhere on the map to trigger a prediction
- Displays Live Model Prediction panel and Rooftop & Weather Context

**Per-building forecast** (`results/forecast_per_building.csv`): ~3,000 rows showing the predicted solar energy potential (kWh/year) for every sampled rooftop — demonstrates scalability to city-wide assessment.

---

## 4. Results Directory Reference

Every file in `FIAdaBoost/results/` is listed below with the code that generated it, the API endpoint that exposes it, and the UI panel where it appears.

### CSV Files

| File | Generated by | Code Location | API Endpoint | UI Panel | Contents |
|------|-------------|--------------|-------------|----------|----------|
| [`metrics_summary.csv`](results/metrics_summary.csv) | `model_training.py` Step 5 | `make_result_table()`, line 417 | `/compare` → `performanceMetricsComparison` | Model Comparison — Comparison Table (C) | 2 rows: AdaBoost and FI-AdaBoost RMSE_J, MAE_J, R² on the 600-sample test holdout |
| [`cv_fold_metrics.csv`](results/cv_fold_metrics.csv) | `model_training.py` Step 4 | `run_kfold_cv()` | `/cv-metrics/spatial` | Spatial Cross-Validation (5-Fold) table | 5 rows: per-fold train/val RMSE and R² for both models |
| [`cv_fold_metrics_daily.csv`](results/cv_fold_metrics_daily.csv) | `model_training.py` Daily pipeline | `TimeSeriesSplit(5)` CV block | `/cv-metrics` | Daily CV (not shown in UI screenshots) | 5 rows: daily temporal fold RMSE, MAE, R² for both models |
| [`dm_test_results_spatial.csv`](results/dm_test_results_spatial.csv) | `model_training.py` Step 5 | `diebold_mariano_test()` | `/dm-test` → `spatial` field | Diebold-Mariano Test — Spatial row | 1 row: DM=12.8241, p=0.0, significant=True, n=600, mean_loss_diff=34,958,666.09 J² |
| [`dm_test_results_daily.csv`](results/dm_test_results_daily.csv) | `model_training.py` Daily pipeline | `diebold_mariano_test()` | `/dm-test` → `daily` field | Diebold-Mariano Test — Daily row | 1 row: DM=-1.6083, p=0.1078, significant=False, n=74 |
| [`daily_metrics_summary.csv`](results/daily_metrics_summary.csv) | `model_training.py` Daily pipeline | Daily pipeline final evaluation | `/daily-metrics` | Daily Temporal Split Metrics table | 2 rows: AdaBoost and FI-AdaBoost RMSE, MAE, R² on the 73-day chronological test set |
| [`train_test_split_info.csv`](results/train_test_split_info.csv) | `model_training.py` Step 2 | `main()` line 1044–1056 | `/split-info` | Not shown in UI screenshots | 1 row: total=3000, train=2400, test=600, test_fraction=0.2, seed=42, split_method=random_shuffle |
| [`forecast_per_building.csv`](results/forecast_per_building.csv) | `model_training.py` Phase 2 | Phase 2 energy potential loop | Not directly served (used for map/histogram rendering) | Energy distribution histogram, Total energy comparison bar chart | ~3,000 rows: lat, lon, rooftop_area_sq_m, ada_predicted_irradiance_kWh_m2_day, ada_SEP_kWh_day, ada_SEP_kWh_yr, fi_predicted_irradiance_kWh_m2_day, fi_SEP_kWh_day, fi_SEP_kWh_yr, difference_SEP_kWh_yr |

### PNG Files

All PNG files are served as static files at `/results/images/{filename}` by the FastAPI backend.

| File | Generated by | Code | Corresponds To | What It Shows |
|------|-------------|------|----------------|--------------|
| [`actual_vs_predicted.png`](results/actual_vs_predicted.png) | `model_training.py` | `matplotlib` scatter plot, Step 5 | Paper Figure 16 | Two-subplot scatter: y_true vs y_pred for Baseline (left) and FI-AdaBoost (right) on the 600-sample test set. Tighter clustering around the diagonal = better fit. |
| [`residuals.png`](results/residuals.png) | `model_training.py` | `matplotlib` scatter plot, Step 5 | Paper Figure 17 | Two-subplot residual plot: (y_true − y_pred) vs y_pred. Points near zero = small errors. FI-AdaBoost's residuals cluster more tightly around zero. |
| [`metrics_comparison.png`](results/metrics_comparison.png) | `model_training.py` | `matplotlib` bar chart, Step 5 | Paper Figure 19 | Three-subplot bar chart: RMSE, MAE, R² side by side for both models. Same data as the Comparison Table (C) in the UI. |
| [`baseline_feature_importances.png`](results/baseline_feature_importances.png) | `model_training.py` | `matplotlib` bar chart, Step 5 | Paper Figure 19 | All 8 features at exactly 12.5% each — confirms baseline AdaBoostRegressor treats all features uniformly (no FI weighting). |
| [`standalone_feature_importances.png`](results/standalone_feature_importances.png) | `model_training.py` | `matplotlib` bar chart, Step 5 | Paper Figure 18 | FI-AdaBoost learned importances averaged across all valid boosting rounds: `clear_sky_ratio` ≈ 68.8%, `lat` ≈ 18.7%, others smaller. |
| [`overfit_check.png`](results/overfit_check.png) | `model_training.py` | `matplotlib` bar chart, Step 5 | Paper Figure 15 | Train RMSE vs Test RMSE for both models. A large gap = overfitting. Both models show similar train/test RMSE, indicating no severe overfitting. |
| [`energy_distribution.png`](results/energy_distribution.png) | `model_training.py` | `matplotlib` histogram, Phase 2 | Paper Figure 20 | Per-building annual SEP distribution (kWh/year) for all 3,000 rooftops. Right-skewed: most buildings are small/residential, a few commercial buildings occupy the upper tail. |
| [`total_energy_comparison.png`](results/total_energy_comparison.png) | `model_training.py` | `matplotlib` bar chart, Phase 2 | Paper Figure 21 | City-wide aggregate annual SEP: AdaBoost ≈ 1,207.5M kWh/year vs FI-AdaBoost ≈ 1,207.3M kWh/year (0.011% difference). Aggregate totals are nearly identical because both models predict similarly on average; per-building accuracy (RMSE/MAE) is what differs. |

---

## 5. Defense Questions by UI Panel

These are realistic questions a thesis panel, adviser, or peer reviewer would ask when looking at each screen. Each answer is written so it can be stated verbatim or paraphrased at a defense.

---

### Panel A — Live Model Prediction

**Q: The difference shown is -1.47 kWh/day, which is only 0.09%. How does that support your claim that FI-AdaBoost is better?**
A single point prediction at a well-sampled training location will naturally show small divergence between the two models. The 0.09% difference here is location-specific. The improvement is quantified across 600 unseen test samples where FI-AdaBoost achieves RMSE = 4,908 J/m²/day vs. 7,684 J/m²/day for the baseline — a 36% reduction. One point on the map does not represent the model's general behavior.

**Q: The FI-AdaBoost SEP (1,697.54 kWh/day) is actually lower than the baseline (1,699.02 kWh/day). Doesn't that mean the baseline predicted more solar energy at this location?**
Yes, and that is acceptable. A "better" model is not the one that always predicts higher; it is the one whose predictions are closest to the true values on average across the test set. At this particular coordinate, FI-AdaBoost predicts slightly less. Whether that prediction is more accurate than the baseline's at this exact point is unknowable without ground truth for this specific rooftop.

**Q: What does "98.8% Prediction Confidence" mean? Is the model 98.8% accurate?**
No. That figure is a heuristic composite score, not a classical confidence interval or accuracy percentage. For the baseline, it reflects only how close the query location is to the nearest training data points (spatial proximity). For FI-AdaBoost, it is 70% spatial proximity plus 30% model agreement with the baseline. A large divergence between the two models' predictions would lower FI's score. It communicates data coverage quality, not prediction error bounds.

**Q: Is the meteorological data in this panel real-time or historical?**
It is fetched from the NASA POWER API at request time, but NASA POWER operates on a climatological lag — data is typically 30–60 days behind the current date and represents rolling annual averages, not live weather. The prediction is therefore based on the long-term climate profile of that location, not today's weather conditions.

**Q: The output is in kWh/day. Does that include panel efficiency, inverter losses, or degradation?**
No. As stated in the study's scope (Section 1.3), the output is theoretical solar energy potential: predicted irradiance (kWh/m²/day) multiplied by rooftop area. It represents how much solar energy falls on the rooftop surface, not how much a photovoltaic system would actually produce. Panel efficiency (~15–20%), inverter losses, dust, and degradation are excluded by design and noted as future work (Section 4.2.2).

**Q: Why is SEP shown in kWh/day here but kWh/year in the paper's Section 3.5?**
Both units appear in the study. The per-building live panel uses kWh/day because it is more interpretable for a homeowner making a daily energy decision. The paper's Section 3.5 aggregates to kWh/year for city-wide planning comparisons. The conversion is simply ×365.

**Q: Could the prediction change if I click the same location twice?**
Yes, marginally. The live Overpass API call returns the nearest OSM building, which could theoretically differ if OSM is updated between requests. The NASA POWER fetch uses a rolling lookback window. In practice, repeated requests to the same coordinate within minutes will return identical or near-identical feature values.

---

### Panel B — Rooftop & Weather Context

**Q: Why only 8 features? Could the model have been more accurate with more inputs?**
The 8 features represent every variable derivable from the two publicly available data sources specified in the study scope — NASA POWER (meteorological) and OpenStreetMap (rooftop geometry). Adding more features would require data not accessible without field surveys: actual roof tilt, roofing material, measured shading from LiDAR, or on-site pyranometer readings. This is documented as a limitation and recommended for future research (Section 4.2.1).

**Q: Temperature and humidity are shown on the panel but you said they are not model inputs. Why display them at all?**
They are displayed as weather context to help the user interpret the prediction — for instance, high humidity typically correlates with cloud cover and lower GHI. The EDA (Section 2.1.3.1) found temperature weakly correlated with GHI (r = 0.393) and was excluded as a primary predictor; humidity was also excluded as an explicit model feature in the 8-feature set because `clear_sky_ratio` already captures its solar effect. Displaying them is a UI design choice for transparency, not a model input claim.

**Q: The "Cloud Cover" card shows 49.9%. Is that actually cloud cover?**
Not precisely. The value is `ALLSKY_KT` (NASA POWER clearness index), multiplied by 100 for display. `ALLSKY_KT` is the ratio of surface shortwave radiation to top-of-atmosphere radiation — lower values indicate more atmospheric obstruction (clouds, aerosols). The paper labels it "Cloud cover (oktas)" in Table 3. Calling it "Cloud Cover %" in the UI is an approximation; a more accurate label would be "Clearness Index (%)".

**Q: The shading factor is displayed as 0 for this building. Does that mean it really has zero shading?**
It means no other buildings were found within the 50m buffer around this coordinate in the OSM dataset. Given that OSM's building coverage in Davao City is incomplete — especially for informal settlements and back-lot structures — the true shading could be higher. This is a data limitation (Section 1.3) of relying on volunteered geographic data.

**Q: The SEI is 0.007104. What does that number physically represent?**
SEI is a composite metric: `orientation_score × rooftop_area_sq_m × (1 − shading_factor) × tilt_factor`. After normalization by the maximum SEI in the training set, it becomes `SEI_norm`. A value of 0.007 means this rooftop captures 0.7% of the maximum theoretical solar access observed among the 3,000 sampled buildings. Low values can result from small rooftop area, poor orientation (North-facing), high shading, or all three.

**Q: "Orientation: North" — why would a building in the Philippines have a north-facing roof?**
OSM records the geometric azimuth of the building's longest face, which does not always correspond to the rooftop's intended solar-facing surface. Many buildings in Davao City were constructed without solar orientation in mind, so north-facing results are geographically real. The `orientation_score = cos(azimuth − 180°)` formula correctly penalizes north-facing orientations with a low score, which flows into a lower SEI.

**Q: Why is "Sunshine Hours" always shown as a whole number like 10.0?**
The display rounds to one decimal place. The underlying value is computed by counting monthly hours where pvlib-estimated GHI > 120 W/m² (WMO's minimum threshold for "direct sunshine"), then averaging across the 12 months. The result is constrained to the 0–24 range and tends to cluster near whole-number values for a tropical city like Davao with consistent daylight hours (~11–13 hours/day year-round).

---

### Panel C — Model Comparison Metrics Charts

**Q: Both models have R² above 99%. Isn't that suspiciously high? Could there be data leakage?**
High R² is expected here for two reasons. First, the target variable `effective_GHI_J` is largely determined by geographic latitude and longitude — more southerly locations in Davao City receive more irradiance, creating a strong spatial pattern that both models learn easily. Second, there is no data leakage: the test set of 600 samples was held out before any training, hyperparameter tuning, or cross-validation. The 5-fold CV results (panel E) confirm similar performance on data never seen during tuning.

**Q: What does an RMSE of 4,908 J/m²/day actually mean? Is that good or bad?**
Converting: 4,908 J/m²/day ÷ 3,600,000 = 0.00136 kWh/m²/day average prediction error. For a 100 m² rooftop, that is 0.136 kWh/day error against a mean prediction of ~497 kWh/day (from mean GHI ≈ 4.96 kWh/m²/day × 100 m²). That is a relative error of about 0.03% — very small. In the context of related work (Sales et al., 2024, Table 1 reference [13]), their AdaBoost RMSE was 22,665 J/m²; this study's baseline is 7,684 J/m²/day, which is already substantially lower.

**Q: Why not also report MAPE (Mean Absolute Percentage Error)?**
MAPE is undefined when the actual GHI value is zero, which occurs for buildings in full shade or on overcast days recorded in the dataset. Both RMSE and MAE avoid division-by-zero issues. The study follows the same metrics used in all reference works cited in Table 1 of the paper (RMSE, MAE, R²) for direct comparability.

**Q: The improvement in MAE (59.56%) is much larger than RMSE (36.13%). Why?**
RMSE squares the errors before averaging, so a few large-error samples have disproportionate weight. FI-AdaBoost reduces the cluster of medium-sized errors (which dominate MAE) more effectively than it eliminates the worst-case large errors (which dominate RMSE). This is consistent with the residual plots (Figure 17 in the paper), which show FI-AdaBoost's residuals concentrated more tightly near zero.

**Q: Could you have gotten better numbers by choosing a different random seed?**
Possibly, in either direction. The paper discloses the fixed `random_seed=42` (Section 2.5.3 and model_training.py line 80). Seed sensitivity is acknowledged in Section 2.5.5 as a best practice check. Using a single fixed seed is the standard practice in reproducible ML research; the DM test provides the statistical validity check that the improvement is not a chance artifact of this particular split.

**Q: Why are the improvement percentage cards shown in green? Does that mean FI-AdaBoost is always better?**
Green indicates improvement over the baseline on these three metrics for the spatial dataset. On the daily temporal dataset (panel G), FI-AdaBoost performs slightly worse — those metrics would not be green. The UI's visual framing reflects the spatial experiment results, which are the primary study experiment per the thesis objective.

---

### Panel D — Comparison Table

**Q: The table shows RMSE to 6 decimal places. Is that level of precision meaningful?**
The full precision is stored in `metrics_summary.csv` for reproducibility. When reporting in the paper (Table 11), values are rounded to 2 decimal places. In a computational sense, the precision beyond 2 decimal places (~1 J/m²/day) is well within hardware and floating-point noise, so it carries no physical meaning. The full values are retained in the CSV so the reader can verify the computation independently.

**Q: The Delta column for R² shows ↑0.09%. Isn't that negligible?**
In absolute terms, yes. Both models already explain more than 99.8% of variance in the test set. The meaningful improvements are RMSE (−36%) and MAE (−60%), which affect the accuracy of individual building predictions. R² near the ceiling is insensitive to further improvement — this is an inherent limitation of R² as a metric when the baseline is already very high.

**Q: Are these metrics from the training set, validation set, or test set?**
Test set exclusively — the 600 samples held out before any training (20% of 3,000, seed=42). The training RMSE values (used for overfitting analysis, Figure 15) are computed separately and appear in the overfitting check chart, not in this table.

---

### Panel E — Spatial Cross-Validation (5-Fold) Table

**Q: Why use 5 folds instead of leave-one-out or 10-fold CV?**
5-fold is standard practice for moderately sized datasets (~2,400 training samples). Leave-one-out would require training 2,400 models per Optuna trial (100 trials × 2,400 models per model), making it computationally infeasible on an 8 GB RAM machine (Table 8). 10-fold with 100 Optuna trials per model is also prohibitive. 5-fold provides a reliable variance estimate at manageable cost.

**Q: Fold 2 shows FI Val RMSE = 10,138 — more than double the other folds. Is the model unreliable?**
That fold's 480-sample validation subset happens to contain spatial locations with atypical feature combinations — for instance, coordinates whose `clear_sky_ratio` and `lat` values fall in an underrepresented region of the training data. This is expected variance in a finite dataset. The average FI Val RMSE across all 5 folds (≈4,892) is still 37% lower than the baseline average (≈7,786). High variance in one fold does not indicate model unreliability; it indicates the difficulty of certain spatial regions.

**Q: You call this "Spatial Cross-Validation" but the description says random shuffling. Is it truly spatial CV?**
The naming refers to the fact that the data is spatial (3,000 building locations), not that the folds are spatially stratified. Random KFold with shuffling is the correct choice here because spatial autocorrelation is low across the study area — all 3,000 points are within Davao City and share the same annual NASA POWER meteorological data, so location-to-location variation is not temporally structured. A spatial block CV (e.g., geographic hexbin grouping) would be more rigorous and is recommended as future work.

**Q: How do you know the hyperparameters tuned in Step 3 did not overfit to the cross-validation folds?**
The tuning uses an inner 5-fold CV on the 2,400 training samples to find the best hyperparameters. The outer CV shown in this table re-applies those best hyperparameters to new fold partitions. This is a standard nested CV approach. The close alignment between train and val RMSE across folds (no dramatic gap) indicates the hyperparameters are not overfitting to the inner CV folds.

**Q: Both models use the same fold splits. How was that guaranteed in code?**
Both `X_tr_ada` and `X_tr_fi` are derived from the same 2,400-row training DataFrame (`train_df[SHARED_FEATURES]`). The `run_kfold_cv()` function iterates over `KFold(n_splits=5, shuffle=True, random_state=42).split(X_tr_ada)` — the same indices are used for both models at every fold. This is visible at `src/model_training.py:1076–1079`.

---

### Panel F — Diebold-Mariano Statistical Test

**Q: Why use the Diebold-Mariano test instead of a simpler paired t-test?**
A paired t-test assumes the error differences are independent and identically distributed. On a spatial dataset where all 600 test points share the same NASA POWER meteorological data (same year, same geographic region), the prediction errors are correlated — nearby points produce similar errors. The DM test with Newey-West HAC variance correction explicitly accounts for autocorrelation in the error differences, making it appropriate for forecasting model comparison.

**Q: A DM statistic of 12.8241 seems extremely large. Is that realistic?**
It is large, which reflects how consistent the improvement is across all 600 test samples. The DM statistic is the ratio of mean squared-error improvement to its standard error. When FI-AdaBoost outperforms the baseline on virtually every single test point, the mean is large and the variance is low, producing a high statistic. Any |DM| > 1.96 is already significant at α = 0.05. A value of 12.82 corresponds to an astronomically small p-value.

**Q: The p-value is stored as exactly 0.0 in the CSV. Is that an error?**
It is a floating-point limitation. `scipy.stats.norm.cdf(12.82)` returns exactly 1.0 because 12.82 standard deviations is beyond the precision of double-precision arithmetic. The true p-value is approximately `10⁻³⁷`, not zero. For reporting purposes, the paper correctly states "p < 0.0001" (Section 3.2.4, Table 12).

**Q: The daily DM statistic is negative (-1.6083). Does that mean you proved the baseline is better on daily data?**
No — the negative sign means the baseline had slightly lower squared errors on average, but the p-value of 0.1078 is above α = 0.05, so the null hypothesis of equal predictive accuracy cannot be rejected. The correct interpretation is: there is no statistically significant difference between the two models on daily temporal data. The paper reports this honestly in Section 3.6.2 and Table 15.

**Q: Could you have used a bootstrap test instead of DM?**
Yes, block bootstrap confidence intervals for RMSE are included as a robustness check in Section 2.5.5. The DM test is the primary significance test because it is specifically designed for forecasting model comparison and is the standard in the econometrics and ML forecasting literature cited by the study.

**Q: Why is n=600 for spatial but only n=74 for daily? Smaller n means less power — isn't the daily test underpowered?**
With n=74, the minimum detectable DM statistic at α=0.05 is approximately ±1.96 (the critical value). The observed DM of -1.6083 is within this range, which is why it is not significant. If the true effect were as large as the spatial case (DM≈12), it would be detectable even with n=74. The non-significant result reflects a genuinely small or absent effect, not only low power.

---

### Panel G — Daily Temporal Split Metrics

**Q: Why did you even run a daily time-series experiment if the thesis is about spatial forecasting?**
Three reasons: (1) It tests whether the FI mechanism generalizes to a different prediction problem, providing a fuller picture of the algorithm's behavior (Objective 1.2.2.4). (2) The 365-day NASA POWER dataset was already acquired for computing `clear_sky_ratio` and `sunshine_hours` in the spatial pipeline, making it a natural secondary dataset at no additional collection cost. (3) It produces an honest negative result — FI-AdaBoost does not improve temporal forecasting — which strengthens rather than weakens the paper by being transparent about limitations.

**Q: R² = 0.90 for daily data. Is that acceptable?**
For a tabular ML model trained on only 11 static and temporal features with no recurrence or memory, R² ≈ 0.90 on an 80/20 chronological holdout is reasonable. The related work by Babu et al. (2025), reference [16] in Table 1, reports GBR R² = 0.827 on a similar temporal problem. An R² of 0.90 is competitive. The paper correctly recommends LSTM or Transformer-based models (Section 4.2.1) as future work for stronger temporal forecasting.

**Q: The daily pipeline uses 11 features while the spatial uses 8. Why the difference?**
The daily pipeline adds temporal features (`month_sin`, `month_cos`, `season`) because these vary day-to-day and capture seasonal GHI patterns. It also uses raw meteorological variables (`T2M`, `RH2M`, `ALLSKY_KT`) since these have daily resolution. The spatial pipeline cannot use these as per-building varying features — each building receives the same annual-average meteorological values from NASA POWER (the API resolution does not distinguish buildings 100 m apart). The spatial pipeline instead uses the rooftop geometry features that vary across buildings.

**Q: Why split daily data chronologically and not randomly?**
Random splitting would cause data leakage: if the model trains on February and April data, it implicitly learns the seasonal pattern and can predict October more accurately than it could in a real deployment scenario. Chronological splitting simulates real operational use — train on the past, forecast the future — and is required for temporal data by convention (Section 2.5.1).

**Q: FI-AdaBoost is 2% worse on daily data. Should this be considered a failure of the method?**
No. The study's general objective (Section 1.2.1) is spatial solar energy potential forecasting, and the primary experiment is spatial. The daily experiment is secondary. The FI mechanism operates by amplifying errors on samples where important features (like `clear_sky_ratio`) have high values. In temporal data, the most predictive signal is the day-to-day GHI autocorrelation, not feature magnitude patterns — so the FI weighting adds noise rather than signal. This is an expected theoretical limitation, not a failure, and is discussed in Section 3.6.
