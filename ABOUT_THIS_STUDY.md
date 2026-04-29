# About This Study

## Title

**Solar Energy Potential Forecasting Thesis Prototype**

## What this project is about

This project is an academic study and working software prototype for estimating solar energy potential in **Davao City, Philippines**. The study combines spatial data, rooftop-related topographical features, meteorological indicators, and machine learning to forecast effective solar irradiance for location-based solar suitability analysis.

The goal is not just to build a user interface, but to support a research workflow that:

- gathers solar and building-related data
- transforms those data into engineered spatial features
- compares a conventional machine learning baseline against a proposed FI-AdaBoost model
- serves the trained model through a live web application
- lets users select a real location and inspect predicted solar potential and model analytics

In short, this repository contains both:

- the **research pipeline** used in the study
- the **interactive prototype system** used to demonstrate the study results

## Main research objective

The main objective of the study is to evaluate whether a **Feature-Importance AdaBoost Regression model (FI-AdaBoost)** can improve solar irradiance forecasting for rooftop suitability analysis compared with a standard **AdaBoost Regression** baseline.

The work focuses on a spatial forecasting setup where each prediction point represents a location in Davao City and is associated with the nearest building’s rooftop characteristics.

## Study area

The geographic focus of the study is:

- **Davao City, Philippines**

The dataset and trained models in this repository were prepared specifically around Davao City spatial coverage, so the current saved models are best interpreted within that area.

## Problem the study addresses

Planning solar installations requires understanding how suitable a location is for solar energy harvesting. Raw irradiance alone is not enough. Real-world suitability is affected by:

- rooftop orientation
- rooftop area
- nearby building shading
- solar exposure
- local clear-sky conditions
- sunlight availability

This study attempts to represent those influences in a structured machine learning workflow so that solar potential can be estimated more meaningfully than by using latitude and longitude alone.

## Core idea of the methodology

The study uses a **spatial machine learning pipeline** built around 3,000 sampled coordinates across Davao City.

At a high level, the workflow is:

1. Sample random spatial points inside the Davao City boundary.
2. Fetch annual solar irradiance values for those points from NASA POWER.
3. Download OSM building footprints for the city.
4. Compute rooftop-related topographical features from the building geometries.
5. Match each spatial point to its nearest building.
6. Build a unified training dataset.
7. Train and compare Baseline AdaBoost and FI-AdaBoost using the same target and the same feature set.
8. Save trained models, metrics, and plots.
9. Serve the trained models through a live API and frontend prototype.

## Data sources used

The current implementation uses these external sources and tools:

- **NASA POWER**
  - annual and daily solar and meteorological variables
- **OpenStreetMap / Overpass**
  - building footprints near a target location
- **OSMnx**
  - place boundary and building extraction
- **pvlib**
  - clear-sky reference calculations and POA-related solar adjustment logic

## What the dataset represents

The main spatial dataset consists of **3,000 random coordinates** sampled within Davao City.

Each point is associated with:

- annual average GHI
- nearest building features from OSM
- engineered rooftop descriptors
- derived meteorological indicators

This produces an integrated spatial dataset used for model training and evaluation.

## Feature engineering used in the study

The active model input features are:

1. `lat`
2. `lon`
3. `azimuth`
4. `orientation_score`
5. `shading_factor`
6. `SEI_norm`
7. `clear_sky_ratio`
8. `sunshine_hours`

These features combine:

- raw geographic coordinates
- rooftop orientation behavior
- rooftop shading behavior
- normalized solar exposure
- clear-sky comparison
- expected sunshine availability

### Rooftop-oriented engineered features

The rooftop-related components include:

- **Azimuth**
  - derived from building bounding-box orientation
- **Orientation score**
  - a south-facing preference score derived from azimuth
- **Shading factor**
  - a heuristic derived from nearby building density
- **Tilt factor**
  - based on an assumed flat-roof condition relative to Davao’s optimal tilt
- **Solar Exposure Index**
  - composite rooftop suitability measure
- **SEI_norm**
  - normalized SEI used directly as a model feature

## Target variable used

Both machine learning models predict the same target:

- `effective_GHI_J`

This is the **pvlib-adjusted effective irradiance target** expressed in:

- **J/m^2/day**

Using the same target for both models is important because it makes the comparison fair. The study’s intent is for the algorithm to be the main difference, not the label definition.

## Models compared in the study

The research compares two models:

### 1. Baseline AdaBoost

A standard AdaBoost regression setup used as the comparison baseline.

### 2. FI-AdaBoost

A proposed variation that incorporates feature-importance information into the boosting update behavior.

### Important fairness rule in the current study design

Both models:

- use the same 8 features
- predict the same target
- are evaluated on the same study dataset

This makes the comparison focused on the algorithmic difference rather than on different feature spaces or different targets.

## What FI-AdaBoost is meant to contribute

The proposed FI-AdaBoost model is intended to make the boosting process more sensitive to feature importance when updating instance weights. In the context of the thesis, this is the main proposed methodological contribution.

The practical question behind the comparison is:

- can a feature-importance-aware boosting strategy improve solar potential forecasting performance over standard AdaBoost?

## Outputs produced by the study

The study produces several kinds of outputs:

### Trained model artifacts

- `FIAdaBoost/models/baseline_adaboost.pkl`
- `FIAdaBoost/models/fi_adaboost.pkl`

### Result tables

- `FIAdaBoost/results/metrics_summary.csv`
- `FIAdaBoost/results/cv_fold_metrics.csv`
- `FIAdaBoost/results/daily_metrics_summary.csv`
- `FIAdaBoost/results/dm_test_results_spatial.csv`
- `FIAdaBoost/results/dm_test_results_daily.csv`
- `FIAdaBoost/results/forecast_per_building.csv`

### Visual outputs

- feature importance charts
- residual plots
- actual vs predicted plots
- energy comparison plots
- metrics comparison plots

## Current saved results in this repository

The repository already includes saved trained artifacts and saved evaluation outputs.

From the main spatial summary:

- Baseline AdaBoost:
  - `RMSE_J = 4978.419973`
  - `MAE_J = 3970.179255`
  - `R2 = 0.999381`
- FI-AdaBoost:
  - `RMSE_J = 4791.060750`
  - `MAE_J = 3858.021327`
  - `R2 = 0.999427`

These values indicate that the saved FI-AdaBoost run slightly outperformed the saved baseline run on the primary spatial evaluation summary included in the repository.

## What the web application does

The prototype application is the interactive layer of the study.

It allows a user to:

- search for a place
- click a point on the map
- submit a prediction request
- retrieve live building and meteorological context
- inspect predicted solar potential
- compare baseline and FI-AdaBoost outputs
- view saved training analytics from the study results

## How the live prediction system works now

The current backend is no longer a mock prediction service.

The live workflow is:

1. The frontend sends `lat` and `lng` to the backend.
2. The backend queries live building footprints near that location.
3. The backend computes rooftop features from the live building geometry.
4. The backend fetches live NASA POWER meteorological values.
5. The backend computes pvlib clear-sky support features.
6. The backend assembles the exact 8-feature input vector.
7. The backend runs the saved trained model from `models/`.
8. The backend returns prediction and context values to the frontend.

That means the interactive system now reflects the trained study model much more directly than the earlier mock prototype version.

## Repository components

### `FIAdaBoost/`

This directory contains the research and backend side of the study:

- data acquisition
- preprocessing
- feature engineering
- data integration
- model training
- trained model files
- result artifacts
- live FastAPI inference service

### `Solar_Energy_Potential_Forecasting_Prototype/`

This directory contains the interactive frontend:

- map UI
- location search
- coordinate input
- live prediction requests
- analytics views
- deployment-ready frontend build

## What this repository can be used for

This repository can support several purposes:

- thesis demonstration
- academic presentation
- methodology review
- prototype validation
- model comparison analysis
- future extension into a more operational solar suitability tool

## Limitations

This study and prototype still have important limitations:

- the training focus is Davao City, not a general nationwide model
- live prediction depends on external services remaining available
- OSM building coverage may be incomplete in some locations
- rooftop tilt is simplified rather than measured per building
- shading is heuristic rather than based on detailed 3D urban geometry
- prediction quality is influenced by both training data quality and live upstream data quality

## What changed from the earlier prototype direction

The repository has moved beyond a purely mock or illustrative frontend prototype.

The current state now includes:

- trained saved models
- saved evaluation outputs
- live model-backed prediction
- live API integration
- working deployment configuration for backend and frontend

So this project should now be understood as a **thesis prototype backed by an implemented study pipeline**, not just a UI demo.

## Recommended way to read this repository

If someone is trying to understand the full project, the best order is:

1. Read `README.md`
2. Read this file: `ABOUT_THIS_STUDY.md`
3. Review `FIAdaBoost/src/model_training.py`
4. Review `FIAdaBoost/src/api.py`
5. Review the frontend in `Solar_Energy_Potential_Forecasting_Prototype/src`

## Related documentation

- Main repository guide: [`README.md`](./README.md)
- Frontend note: [`Solar_Energy_Potential_Forecasting_Prototype/README.md`](./Solar_Energy_Potential_Forecasting_Prototype/README.md)

## Summary

This study is about building and evaluating a rooftop-oriented solar potential forecasting system for Davao City using a fair comparison between Baseline AdaBoost and FI-AdaBoost, then turning that research output into a live, interactive prototype application.

The repository therefore represents:

- a data pipeline
- a machine learning study
- saved research outputs
- a backend inference system
- a deployable frontend prototype
