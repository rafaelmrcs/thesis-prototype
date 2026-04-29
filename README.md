# Solar Energy Potential Forecasting Thesis Prototype

This repository contains the code, trained artifacts, and prototype application for a Davao City solar energy potential forecasting study. It combines a spatial machine learning pipeline, a live FastAPI inference service, and a React frontend for interactive map-based prediction and analytics.

For a fuller explanation of the research purpose, methodology, system scope, and current implementation status, see:

- [`ABOUT_THIS_STUDY.md`](./ABOUT_THIS_STUDY.md)

The project now reflects the current study implementation:

- a 3,000-point spatial dataset sampled inside Davao City
- OSM-derived rooftop feature engineering
- a fair AdaBoost vs FI-AdaBoost comparison using the same target and the same 8 input features
- saved trained models and evaluation outputs in `FIAdaBoost/models` and `FIAdaBoost/results`
- a live backend that predicts from trained models plus live NASA POWER, Overpass, and pvlib features
- a frontend that calls the backend instead of using mock prediction values

## Study Scope

The study focuses on forecasting effective solar irradiance for rooftop suitability analysis in Davao City, Philippines.

The current modeling design is:

1. Acquire annual GHI and building data.
2. Engineer rooftop features from OSM building footprints.
3. Join each sampled coordinate to its nearest building.
4. Train and compare Baseline AdaBoost and FI-AdaBoost on the same target and features.
5. Serve predictions through a live API and interactive web UI.

## What Was Implemented

### Spatial pipeline

- `src/data_acquisition.py`
  - samples 3,000 random coordinates inside the Davao City boundary
  - fetches annual NASA POWER GHI values
  - downloads OSM building footprints
- `src/data_processing.py`
  - cleans the spatial dataset
  - computes rooftop area from OSM geometry
- `src/feature_engineering.py`
  - computes `azimuth`
  - computes `orientation_score`
  - computes `shading_factor`
  - computes `tilt_factor`
  - computes `solar_exposure_index`
  - normalizes SEI to `SEI_norm`
- `src/data_integration.py`
  - attaches the nearest building features to each spatial point
  - writes `integrated_dataset.csv`
- `src/model_training.py`
  - trains Baseline AdaBoost and FI-AdaBoost on the same 8-feature input space
  - predicts the same target for both models: pvlib-adjusted effective GHI in `J/m^2/day`
  - saves trained models and result plots/tables

### Daily auxiliary pipeline

`src/model_training.py` also contains a secondary daily time-series evaluation pipeline using NASA POWER centroid data. This produces `daily_metrics_summary.csv` and fold metrics for additional analysis, but the main thesis comparison is the spatial rooftop forecasting pipeline.

### Live application

- `FIAdaBoost/src/api.py`
  - loads the saved models from `models/`
  - exposes `/predict`, `/compare`, `/cv-metrics`, and `/training-analytics`
  - uses live NASA POWER meteorology
  - uses live Overpass building lookup for the selected coordinate
  - uses pvlib to compute clear-sky reference features
- `Solar_Energy_Potential_Forecasting_Prototype`
  - React + Vite frontend
  - map-based location picker
  - live prediction request to the FastAPI backend
  - saved-model analytics and comparison views

## Current Input Features and Target

Both machine learning models use the same 8 features:

1. `lat`
2. `lon`
3. `azimuth`
4. `orientation_score`
5. `shading_factor`
6. `SEI_norm`
7. `clear_sky_ratio`
8. `sunshine_hours`

Both models predict the same target:

- `effective_GHI_J`
- pvlib POA-adjusted effective irradiance in `J/m^2/day`

The only intended algorithmic difference is the FI-AdaBoost weighting strategy.

## Saved Artifacts

### Models

- `FIAdaBoost/models/baseline_adaboost.pkl`
- `FIAdaBoost/models/fi_adaboost.pkl`

### Evaluation files

- `FIAdaBoost/results/metrics_summary.csv`
- `FIAdaBoost/results/cv_fold_metrics.csv`
- `FIAdaBoost/results/daily_metrics_summary.csv`
- `FIAdaBoost/results/dm_test_results_spatial.csv`
- `FIAdaBoost/results/dm_test_results_daily.csv`
- `FIAdaBoost/results/forecast_per_building.csv`

### Main saved spatial metrics

From `FIAdaBoost/results/metrics_summary.csv`:

| Model | RMSE_J | MAE_J | R2 |
| --- | ---: | ---: | ---: |
| AdaBoost (Baseline) | 4978.419973 | 3970.179255 | 0.999381 |
| FI-AdaBoost (Proposed) | 4791.060750 | 3858.021327 | 0.999427 |

### Saved daily temporal metrics

From `FIAdaBoost/results/daily_metrics_summary.csv`:

| Model | RMSE_J | MAE_J | R2 |
| --- | ---: | ---: | ---: |
| AdaBoost (Daily, Temporal Split) | 1031071.000000 | 896566.072084 | 0.904448 |
| FI-AdaBoost (Daily, Temporal Split) | 1075806.000000 | 933027.948853 | 0.895977 |

## Repository Layout

```text
thesis-prototype/
├── README.md
├── docker-compose.yml
├── FIAdaBoost/
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   ├── models/
│   ├── results/
│   ├── src/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── requirements.runtime.txt
└── Solar_Energy_Potential_Forecasting_Prototype/
    ├── src/
    ├── nginx/
    ├── Dockerfile
    └── package.json
```

## Prerequisites

- Python 3.11 recommended
- Node.js 20 recommended
- npm
- Internet access for live predictions

Live predictions depend on:

- NASA POWER
- Overpass / OpenStreetMap
- pvlib clear-sky calculations
- frontend geocoding through Nominatim

## How To Run

### Option 1: Run the existing saved system locally

This is the fastest way if you only want to use the app with the already trained models and results.

#### Backend

```bash
cd thesis-prototype/FIAdaBoost
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.runtime.txt
uvicorn src.api:app --host 0.0.0.0 --port 8501
```

#### Frontend

Open a second terminal:

```bash
cd thesis-prototype/Solar_Energy_Potential_Forecasting_Prototype
npm ci
VITE_BACKEND_URL=http://localhost:8501 npm run dev
```

Then open `http://localhost:5173`.

### Option 2: Re-run the full study pipeline

Use this if you want to regenerate the processed datasets, models, and result files.

```bash
cd thesis-prototype/FIAdaBoost
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the study stages in order:

```bash
python src/data_acquisition.py
python src/data_processing.py
python src/feature_engineering.py
python src/data_integration.py
python src/model_training.py
```

After training completes, start the API:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8501
```

Then start the frontend:

```bash
cd ../Solar_Energy_Potential_Forecasting_Prototype
npm ci
VITE_BACKEND_URL=http://localhost:8501 npm run dev
```

### Option 3: Run with Docker Compose

From the repository root:

```bash
cd thesis-prototype
docker compose up --build
```

This starts:

- backend at `http://localhost:8501`
- frontend at `http://localhost:5173`

## API Endpoints

### Core endpoints

- `GET /health`
- `POST /predict`
- `POST /compare`
- `GET /cv-metrics`
- `GET /training-analytics`

### Example request

```bash
curl -X POST http://localhost:8501/predict \
  -H "Content-Type: application/json" \
  -d '{"lat": 7.0731, "lng": 125.6128}'
```

## Environment Variables

### Backend

- `CORS_ORIGINS`
- `CORS_ORIGIN_REGEX`
- `NASA_LOOKBACK_DAYS`
- `NASA_LAG_DAYS`
- `OVERPASS_API_URLS`

`OVERPASS_API_URLS` accepts a comma-separated list of Overpass API endpoints. If unset, the backend
tries multiple public mirrors in order for live rooftop lookup.

### Frontend

- `VITE_BACKEND_URL`

For local development, `VITE_BACKEND_URL=http://localhost:8501` is the safest default.

## Deployment Notes

### Railway

The current Dockerfiles are set up for a monorepo-style Railway deployment where:

- the build context is the `thesis-prototype` directory
- the backend service uses `FIAdaBoost/Dockerfile`
- the frontend service uses `Solar_Energy_Potential_Forecasting_Prototype/Dockerfile`

Recommended service configuration:

- backend source root: `thesis-prototype`
- backend Dockerfile path: `FIAdaBoost/Dockerfile`
- frontend source root: `thesis-prototype`
- frontend Dockerfile path: `Solar_Energy_Potential_Forecasting_Prototype/Dockerfile`
- frontend variable: `VITE_BACKEND_URL=https://<your-backend-domain>`
- backend variable: `CORS_ORIGINS=https://<your-frontend-domain>`

### Render

`render.yaml` is included for a non-Docker Render setup using:

- a Python web service for the backend
- a static site for the frontend

## Important Notes and Limitations

- Live prediction requires network access.
- OSM building lookup may fail for coordinates with no mapped nearby building footprint.
- NASA POWER availability directly affects prediction success.
- Live rooftop lookup now fails over across multiple public Overpass API mirrors, but prediction can still fail if all configured Overpass endpoints are unavailable.
- The study is centered on Davao City and the saved training artifacts reflect that coverage.
- The live API uses the trained models, but prediction quality still depends on upstream live data quality and map coverage.

## Authors and Academic Context

This repository was developed as a thesis prototype for solar energy potential forecasting in Davao City, Philippines.

Researchers:

- Mercado
- Retardo
- Verzosa

Academic year:

- 2026
