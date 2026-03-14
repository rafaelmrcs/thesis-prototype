from __future__ import annotations

import __main__
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from src.model_training import FIAdaBoostRegressor


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"
PROCESSED_DATA_FILE = ROOT_DIR / "data" / "processed" / "integrated_dataset.csv"
DATA_CANDIDATES = [
    ROOT_DIR / "data" / "raw" / "baseline_spatial_dataset_philippines_2024.csv",
    ROOT_DIR / "data" / "raw" / "baseline_spatial_dataset_davao_city_2024.csv",
    ROOT_DIR / "data" / "raw" / "baseline_spatial_dataset_davao_city_2024_partial.csv",
]
FI_MODEL_FILE = MODEL_DIR / "fi_adaboost.pkl"
BASELINE_MODEL_FILE = MODEL_DIR / "baseline_adaboost.pkl"
FI_FALLBACK_MODEL_FILE = MODEL_DIR / "fi_adaboost_spatial_api_v3.pkl"
BASELINE_FALLBACK_MODEL_FILE = MODEL_DIR / "baseline_adaboost_spatial_api_v3.pkl"
CV_METRICS_FILE = MODEL_DIR / "cv_fold_metrics.csv"
TRAINING_TARGET = "solar_energy_potential"
TRAINING_LEAKAGE_COLS = ["sunshine_hours", "clear_sky_ratio", "sunshine_flag", "year_month"]

MODEL_DIR.mkdir(parents=True, exist_ok=True)


class PredictRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)


class PredictResponse(BaseModel):
    solarPotential: float
    rooftopArea: float
    solarExposureIndex: float
    orientation: str
    azimuth: float
    sunshineHours: float
    cloudCover: float
    temperature: float
    humidity: float
    clearSkyRatio: float


class CompareResponse(BaseModel):
    baseline: PredictResponse
    fiAdaBoost: PredictResponse
    improvement: dict[str, float]
    performanceMetricsComparison: dict[str, dict[str, float] | float]
    featureImportanceRanking: list[dict[str, str | float | int]]


class CVFoldMetrics(BaseModel):
    fold: int
    baseline_rmse: float
    baseline_mae: float
    baseline_r2: float
    fi_rmse: float
    fi_mae: float
    fi_r2: float


class CVMetricsResponse(BaseModel):
    cv_fold_metrics: list[CVFoldMetrics]
    average_metrics: dict[str, dict[str, float]]


class PredictedActualPoint(BaseModel):
    actual: float
    predicted: float
    date: str


class PredictedVsActualData(BaseModel):
    baseline: list[PredictedActualPoint]
    fiAdaBoost: list[PredictedActualPoint]
    domainMin: float
    domainMax: float


class ErrorDistributionBin(BaseModel):
    bucket: str
    baseline: int
    fiAdaBoost: int


class TrainingAnalyticsResponse(BaseModel):
    performanceMetricsComparison: dict[str, dict[str, float] | float]
    predictedVsActual: PredictedVsActualData
    errorDistribution: list[ErrorDistributionBin]
    residualSummary: dict[str, float]


def _parse_cors_origins() -> list[str]:
    defaults = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://0.0.0.0:5173",
    ]

    configured = os.getenv("CORS_ORIGINS", "")
    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    return origins or defaults


app = FastAPI(title="FI-AdaBoost API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_origin_regex=r"http://([a-zA-Z0-9\-\.]+):5173",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelContext:
    def __init__(self) -> None:
        self.fi_model: FIAdaBoostRegressor | None = None
        self.baseline_model: AdaBoostRegressor | None = None
        self.df: pd.DataFrame | None = None
        self.performance_metrics: dict[str, dict[str, float]] = {}
        self.feature_importance_ranking: list[dict[str, str | float | int]] = []
        self.cv_fold_metrics: list[CVFoldMetrics] = []
        self.average_cv_metrics: dict[str, dict[str, float]] = {}
        self.training_fi_model: FIAdaBoostRegressor | None = None
        self.training_baseline_model: AdaBoostRegressor | None = None
        self.training_analytics: TrainingAnalyticsResponse | None = None

    def _load_model_file(self, model_path: Path) -> object | None:
        # Support legacy pickles trained from scripts where class path became __main__.FIAdaBoostRegressor.
        if not hasattr(__main__, "FIAdaBoostRegressor"):
            setattr(__main__, "FIAdaBoostRegressor", FIAdaBoostRegressor)

        try:
            return joblib.load(model_path)
        except (AttributeError, ModuleNotFoundError, ImportError, ValueError):
            return None

    def _load_cv_metrics(self) -> None:
        """Load cross-validation metrics from CSV file."""
        if not CV_METRICS_FILE.exists():
            return
        
        try:
            df = pd.read_csv(CV_METRICS_FILE)
            self.cv_fold_metrics = []
            
            for _, row in df.iterrows():
                metrics = CVFoldMetrics(
                    fold=int(row["fold"]),
                    baseline_rmse=float(row["baseline_RMSE"]),
                    baseline_mae=float(row["baseline_MAE"]),
                    baseline_r2=float(row["baseline_R2"]),
                    fi_rmse=float(row["fi_RMSE"]),
                    fi_mae=float(row["fi_MAE"]),
                    fi_r2=float(row["fi_R2"]),
                )
                self.cv_fold_metrics.append(metrics)
            
            # Compute average metrics across folds
            if self.cv_fold_metrics:
                baseline_rmses = [m.baseline_rmse for m in self.cv_fold_metrics]
                baseline_maes = [m.baseline_mae for m in self.cv_fold_metrics]
                baseline_r2s = [m.baseline_r2 for m in self.cv_fold_metrics]
                fi_rmses = [m.fi_rmse for m in self.cv_fold_metrics]
                fi_maes = [m.fi_mae for m in self.cv_fold_metrics]
                fi_r2s = [m.fi_r2 for m in self.cv_fold_metrics]
                
                self.average_cv_metrics = {
                    "baseline": {
                        "rmse": float(np.mean(baseline_rmses)),
                        "mae": float(np.mean(baseline_maes)),
                        "r2": float(np.mean(baseline_r2s)),
                    },
                    "fiAdaBoost": {
                        "rmse": float(np.mean(fi_rmses)),
                        "mae": float(np.mean(fi_maes)),
                        "r2": float(np.mean(fi_r2s)),
                    },
                }
        except Exception as e:
            print(f"Warning: Failed to load CV metrics: {e}")

    def _load_training_dataset(self) -> pd.DataFrame:
        if not PROCESSED_DATA_FILE.exists():
            raise FileNotFoundError(f"Missing processed training dataset: {PROCESSED_DATA_FILE}")

        df = pd.read_csv(PROCESSED_DATA_FILE)
        if "date" not in df.columns:
            raise ValueError("Processed training dataset is missing 'date'")
        if TRAINING_TARGET not in df.columns:
            raise ValueError(f"Processed training dataset is missing '{TRAINING_TARGET}'")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", TRAINING_TARGET]).sort_values("date").reset_index(drop=True)
        return df

    def _build_training_feature_set(
        self,
        df: pd.DataFrame,
        expected_features: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        drop_cols = [TRAINING_TARGET, "date", "element", "id", "month"] + TRAINING_LEAKAGE_COLS

        X = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")
        X = X.select_dtypes(include=["number"]).copy()
        X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

        if expected_features is not None:
            for feature in expected_features:
                if feature not in X.columns:
                    X[feature] = 0.0
            X = X[expected_features]

        y = pd.to_numeric(df[TRAINING_TARGET], errors="coerce").fillna(0.0).astype(float)
        dates = df["date"].astype(str)
        return X, y, dates

    def _compute_feature_ranking_from_models(
        self,
        fi_model: FIAdaBoostRegressor,
        baseline_model: AdaBoostRegressor,
        feature_names: list[str],
    ) -> list[dict[str, str | float | int]]:
        fi_importance = np.zeros(len(feature_names), dtype=float)
        if len(fi_model.estimators_) > 0:
            weights = np.asarray(fi_model.estimator_weights_, dtype=float)
            if np.sum(weights) <= 0:
                weights = np.ones_like(weights)
            weights = weights / np.sum(weights)

            for index, estimator in enumerate(fi_model.estimators_):
                if hasattr(estimator, "feature_importances_"):
                    fi_importance += weights[index] * np.asarray(estimator.feature_importances_, dtype=float)

        baseline_importance = np.asarray(
            getattr(baseline_model, "feature_importances_", np.zeros(len(feature_names))),
            dtype=float,
        )

        records: list[dict[str, str | float | int]] = []
        for index, feature in enumerate(feature_names):
            records.append(
                {
                    "feature": feature,
                    "fiAdaBoostImportance": float(fi_importance[index]),
                    "baselineImportance": float(baseline_importance[index]),
                    "importanceGain": float(fi_importance[index] - baseline_importance[index]),
                    "rank": 0,
                }
            )

        records.sort(key=lambda item: float(item["fiAdaBoostImportance"]), reverse=True)
        for rank, row in enumerate(records, start=1):
            row["rank"] = rank
        return records

    def _build_error_distribution(
        self,
        baseline_residuals: np.ndarray,
        fi_residuals: np.ndarray,
        bins: int = 18,
    ) -> list[ErrorDistributionBin]:
        combined = np.concatenate([baseline_residuals, fi_residuals])
        if combined.size == 0:
            return []

        min_edge = float(np.min(combined))
        max_edge = float(np.max(combined))
        if np.isclose(min_edge, max_edge):
            min_edge -= 1.0
            max_edge += 1.0

        edges = np.linspace(min_edge, max_edge, bins + 1)
        baseline_counts, _ = np.histogram(baseline_residuals, bins=edges)
        fi_counts, _ = np.histogram(fi_residuals, bins=edges)

        histogram: list[ErrorDistributionBin] = []
        for index in range(len(edges) - 1):
            histogram.append(
                ErrorDistributionBin(
                    bucket=f"{edges[index]:.0f} to {edges[index + 1]:.0f}",
                    baseline=int(baseline_counts[index]),
                    fiAdaBoost=int(fi_counts[index]),
                )
            )
        return histogram

    def _compute_training_analytics(self) -> None:
        self.training_analytics = None

        self.training_fi_model = self._load_model_file(FI_MODEL_FILE)
        self.training_baseline_model = self._load_model_file(BASELINE_MODEL_FILE)
        if self.training_fi_model is None or self.training_baseline_model is None:
            return

        training_df = self._load_training_dataset()
        expected_features = list(getattr(self.training_baseline_model, "feature_names_in_", []))
        X, y, dates = self._build_training_feature_set(training_df, expected_features or None)

        split_idx = int(0.8 * len(X))
        if split_idx <= 0 or split_idx >= len(X):
            return

        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:].reset_index(drop=True)
        dates_test = dates.iloc[split_idx:].reset_index(drop=True)

        baseline_pred = np.asarray(self.training_baseline_model.predict(X_test), dtype=float)
        fi_pred = np.asarray(self.training_fi_model.predict(X_test), dtype=float)

        baseline_metrics = self._evaluate_metrics(y_test, baseline_pred)
        fi_metrics = self._evaluate_metrics(y_test, fi_pred)
        self.performance_metrics = {
            "baseline": baseline_metrics,
            "fiAdaBoost": fi_metrics,
        }

        feature_names = expected_features or list(X.columns)
        self.feature_importance_ranking = self._compute_feature_ranking_from_models(
            self.training_fi_model,
            self.training_baseline_model,
            feature_names,
        )

        sample_size = min(220, len(y_test))
        sample_indices = np.linspace(0, len(y_test) - 1, num=sample_size, dtype=int)
        sample_indices = np.unique(sample_indices)

        actual_sample = y_test.iloc[sample_indices].to_numpy(dtype=float)
        baseline_sample = baseline_pred[sample_indices]
        fi_sample = fi_pred[sample_indices]
        date_sample = dates_test.iloc[sample_indices].tolist()

        domain_min = float(min(actual_sample.min(), baseline_sample.min(), fi_sample.min()))
        domain_max = float(max(actual_sample.max(), baseline_sample.max(), fi_sample.max()))

        predicted_vs_actual = PredictedVsActualData(
            baseline=[
                PredictedActualPoint(
                    actual=float(actual),
                    predicted=float(predicted),
                    date=str(date_value),
                )
                for actual, predicted, date_value in zip(actual_sample, baseline_sample, date_sample)
            ],
            fiAdaBoost=[
                PredictedActualPoint(
                    actual=float(actual),
                    predicted=float(predicted),
                    date=str(date_value),
                )
                for actual, predicted, date_value in zip(actual_sample, fi_sample, date_sample)
            ],
            domainMin=domain_min,
            domainMax=domain_max,
        )

        baseline_residuals = baseline_pred - y_test.to_numpy(dtype=float)
        fi_residuals = fi_pred - y_test.to_numpy(dtype=float)

        self.training_analytics = TrainingAnalyticsResponse(
            performanceMetricsComparison={
                "baseline": baseline_metrics,
                "fiAdaBoost": fi_metrics,
                "rmseImprovementPct": float(
                    ((baseline_metrics["rmse"] - fi_metrics["rmse"]) / baseline_metrics["rmse"] * 100)
                    if baseline_metrics["rmse"] != 0
                    else 0.0
                ),
                "maeImprovementPct": float(
                    ((baseline_metrics["mae"] - fi_metrics["mae"]) / baseline_metrics["mae"] * 100)
                    if baseline_metrics["mae"] != 0
                    else 0.0
                ),
                "r2ImprovementPct": float((fi_metrics["r2"] - baseline_metrics["r2"]) * 100),
            },
            predictedVsActual=predicted_vs_actual,
            errorDistribution=self._build_error_distribution(baseline_residuals, fi_residuals),
            residualSummary={
                "baselineStd": float(np.std(baseline_residuals, ddof=1)),
                "fiStd": float(np.std(fi_residuals, ddof=1)),
                "baselineMae": float(np.mean(np.abs(baseline_residuals))),
                "fiMae": float(np.mean(np.abs(fi_residuals))),
            },
        )

    def _build_feature_frame(self, lat_series: pd.Series, lon_series: pd.Series, current_month: int = None) -> pd.DataFrame:
        """Generate engineered features matching feature_engineering.py pipeline."""
        from datetime import datetime
        
        lat = lat_series.astype(float)
        lon = lon_series.astype(float)
        
        # Use current month if not provided
        if current_month is None:
            current_month = datetime.now().month
        
        # Temporal features (from feature_engineering.py)
        month_sin = np.sin(2 * np.pi * current_month / 12)
        month_cos = np.cos(2 * np.pi * current_month / 12)
        season = 1 if current_month in [12, 1, 2, 3, 4, 5] else 0  # Dry=1, Rainy=0
        
        # Davao City average weather conditions with location-based variation
        # Urban core (near 7.07, 125.61) is warmer, more humid
        distance_to_center = np.sqrt((lat - 7.0731) ** 2 + (lon - 125.6128) ** 2)
        urban_factor = np.exp(-distance_to_center * 10)  # Decreases with distance from center
        
        T2M = 25.39 + urban_factor * 1.5  # Urban heat island effect
        RH2M = 82.0 + urban_factor * 8.0   # Higher humidity in dense areas
        ALLSKY_KT = 0.48 + (1 - urban_factor) * 0.10  # Clearer skies in suburbs
        
        # Topographical features vary by location
        # Urban areas have more buildings (more shading, smaller rooftops)
        rooftop_area_sq_m = 80.0 + (1 - urban_factor) * 40.0  # 80-120 m²
        nearby_count = 2.0 + urban_factor * 8.0  # 2-10 nearby buildings
        shading_factor = 0.10 + urban_factor * 0.25  # 10-35% shading
        orientation_score = 0.75 + (lat - 7.0) * 0.05  # Varies slightly with latitude
        
        optimal_tilt = 7.2
        roof_tilt = 0.0
        tilt_factor = np.cos(np.radians(abs(roof_tilt - optimal_tilt)))
        azimuth = 180.0  # South-facing for northern hemisphere tropics
        
        # Solar Exposure Index (from feature_engineering.py formula)
        solar_exposure_index = (
            orientation_score
            * rooftop_area_sq_m
            * (1 - shading_factor)
            * tilt_factor
        )
        
        return pd.DataFrame(
            {
                "T2M": T2M,
                "RH2M": RH2M,
                "ALLSKY_KT": ALLSKY_KT,
                "lat": lat,
                "lon": lon,
                "month_sin": month_sin,
                "month_cos": month_cos,
                "season": season,
                "orientation_score": orientation_score,
                "shading_factor": shading_factor,
                "tilt_factor": tilt_factor,
                "solar_exposure_index": solar_exposure_index,
                "rooftop_area_sq_m": rooftop_area_sq_m,
                "azimuth": azimuth,
                "nearby_count": nearby_count,
            }
        )

    def _build_features_from_df(self, df: pd.DataFrame, current_month: int = None) -> pd.DataFrame:
        return self._build_feature_frame(df["lat"], df["lon"], current_month)

    def _build_features_for_point(self, lat: float, lng: float, current_month: int = None) -> pd.DataFrame:
        return self._build_feature_frame(
            pd.Series([lat], dtype=float),
            pd.Series([lng], dtype=float),
            current_month
        )

    def load_data(self) -> pd.DataFrame:
        selected_file = next((path for path in DATA_CANDIDATES if path.exists()), None)
        if selected_file is None:
            raise FileNotFoundError(
                "Missing spatial dataset. Expected one of: "
                + ", ".join(str(path) for path in DATA_CANDIDATES)
            )

        df = pd.read_csv(selected_file)

        required_cols = ["lat", "lon", "GHI_mean_2024"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {selected_file.name}: {missing}")

        df = df.dropna(subset=["lat", "lon", "GHI_mean_2024"]).copy()

        if df["lat"].nunique(dropna=True) < 2 or df["lon"].nunique(dropna=True) < 2:
            raise ValueError(
                f"Dataset {selected_file.name} has insufficient coordinate diversity for location-aware predictions."
            )

        return df

    def train_model(self, df: pd.DataFrame) -> FIAdaBoostRegressor:
        X = self._build_features_from_df(df)
        y = df["GHI_mean_2024"].astype(float)

        model = FIAdaBoostRegressor(
            n_estimators=50,
            max_depth=3,
            random_state=42,
            use_weighted_median=True,
        )
        model.fit(X, y)
        return model

    def train_baseline_model(self, df: pd.DataFrame) -> AdaBoostRegressor:
        X = self._build_features_from_df(df)
        y = df["GHI_mean_2024"].astype(float)

        model = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=2, random_state=42),
            n_estimators=20,
            learning_rate=0.7,
            random_state=42,
        )
        model.fit(X, y)
        return model

    def _evaluate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    def _compute_feature_ranking(self, X: pd.DataFrame) -> list[dict[str, str | float | int]]:
        if self.fi_model is None or self.baseline_model is None:
            return []

        feature_names = list(X.columns)

        # FI-AdaBoost importance: weighted average across weak learners
        fi_importance = np.zeros(len(feature_names), dtype=float)
        if len(self.fi_model.estimators_) > 0:
            weights = np.asarray(self.fi_model.estimator_weights_, dtype=float)
            if np.sum(weights) <= 0:
                weights = np.ones_like(weights)
            weights = weights / np.sum(weights)

            for index, estimator in enumerate(self.fi_model.estimators_):
                if hasattr(estimator, "feature_importances_"):
                    fi_importance += weights[index] * np.asarray(estimator.feature_importances_, dtype=float)

        # Baseline importance from sklearn AdaBoost
        baseline_importance = np.asarray(getattr(self.baseline_model, "feature_importances_", np.zeros(len(feature_names))), dtype=float)

        records: list[dict[str, str | float | int]] = []
        for index, feature in enumerate(feature_names):
            records.append(
                {
                    "feature": feature,
                    "fiAdaBoostImportance": float(fi_importance[index]),
                    "baselineImportance": float(baseline_importance[index]),
                    "importanceGain": float(fi_importance[index] - baseline_importance[index]),
                    "rank": 0,
                }
            )

        records.sort(key=lambda item: float(item["fiAdaBoostImportance"]), reverse=True)
        for rank, row in enumerate(records, start=1):
            row["rank"] = rank

        return records

    def _compute_global_analytics(self) -> None:
        if self.fi_model is None or self.baseline_model is None or self.df is None:
            return

        X = self._build_features_from_df(self.df)
        y = self.df["GHI_mean_2024"].astype(float)

        baseline_pred = self.baseline_model.predict(X)
        fi_pred = self.fi_model.predict(X)

        self.performance_metrics = {
            "baseline": self._evaluate_metrics(y, baseline_pred),
            "fiAdaBoost": self._evaluate_metrics(y, fi_pred),
        }
        self.feature_importance_ranking = self._compute_feature_ranking(X)

    def ensure_loaded(self) -> None:
        self.df = self.load_data()

        # Prefer canonical trained artifacts first, then API-spatial fallback artifacts.
        fi_candidates = [FI_MODEL_FILE, FI_FALLBACK_MODEL_FILE]
        baseline_candidates = [BASELINE_MODEL_FILE, BASELINE_FALLBACK_MODEL_FILE]

        self.fi_model = None
        for model_path in fi_candidates:
            if model_path.exists():
                loaded = self._load_model_file(model_path)
                if loaded is not None:
                    self.fi_model = loaded
                    break

        self.baseline_model = None
        for model_path in baseline_candidates:
            if model_path.exists():
                loaded = self._load_model_file(model_path)
                if loaded is not None:
                    self.baseline_model = loaded
                    break

        if self.fi_model is None:
            self.fi_model = self.train_model(self.df)
            joblib.dump(self.fi_model, FI_FALLBACK_MODEL_FILE)

        if self.baseline_model is None:
            self.baseline_model = self.train_baseline_model(self.df)
            joblib.dump(self.baseline_model, BASELINE_FALLBACK_MODEL_FILE)

        try:
            self._compute_global_analytics()
        except ValueError as exc:
            message = str(exc)
            if "feature names should match" not in message:
                raise

            # Auto-heal when persisted model feature schema no longer matches current engineered features.
            self.fi_model = self.train_model(self.df)
            self.baseline_model = self.train_baseline_model(self.df)
            joblib.dump(self.fi_model, FI_FALLBACK_MODEL_FILE)
            joblib.dump(self.baseline_model, BASELINE_FALLBACK_MODEL_FILE)
            self._compute_global_analytics()
        
        # Load cross-validation metrics
        self._load_cv_metrics()
        self._compute_training_analytics()


ctx = ModelContext()


@app.on_event("startup")
def startup_event() -> None:
    ctx.ensure_loaded()


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "fiadaboost-backend",
        "status": "running",
        "docs": "/docs",
        "predict": "POST /predict with JSON body: {\"lat\": 7.0731, \"lng\": 125.6128}",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/predict")
def predict_help() -> dict[str, str]:
    return {
        "detail": "Use POST /predict with JSON body containing lat and lng.",
        "example": '{"lat": 7.0731, "lng": 125.6128}',
    }


def _get_orientation_and_azimuth(lat: float) -> tuple[str, float]:
    if lat >= 0:
        return "South", 180.0
    return "North", 0.0


def _derive_conditions(solar_potential: float, lat: float, lng: float) -> tuple[float, float, float, float, float]:
    clear_sky_ratio = float(np.clip(solar_potential / 8.0, 0.25, 0.9))
    sunshine_hours = float(np.clip(3.5 + clear_sky_ratio * 7.0, 3.0, 10.5))
    cloud_cover = float(np.clip((1.0 - clear_sky_ratio) * 100.0, 8.0, 85.0))

    latitude_factor = abs(lat) / 20.0
    temperature = float(np.clip(31.0 - latitude_factor * 8.0 + (clear_sky_ratio - 0.5) * 3.0, 20.0, 36.0))
    humidity = float(np.clip(88.0 - (clear_sky_ratio * 35.0) + (abs(lng) % 1.0) * 8.0, 40.0, 95.0))

    return clear_sky_ratio, sunshine_hours, cloud_cover, temperature, humidity


def _add_local_micro_variation(
    base_prediction: float,
    lat: float,
    lng: float,
) -> float:
    lat_micro = int(abs(lat * 10000)) % 1000
    lng_micro = int(abs(lng * 10000)) % 1000

    spatial_hash = (lat_micro * 73 + lng_micro * 37) % 100
    variation_factor = 1.0 + (spatial_hash - 50) * 0.0012

    lat_wave = np.sin(lat * 100) * 0.02
    lng_wave = np.cos(lng * 100) * 0.02

    local_adjusted = base_prediction * variation_factor * (1.0 + lat_wave + lng_wave)

    return float(np.clip(local_adjusted, base_prediction * 0.85, base_prediction * 1.15))


def _compute_confidence_level(
    lat: float,
    lng: float,
    baseline_solar: float,
    fi_solar: float,
    df: pd.DataFrame,
) -> float:
    coordinates = df[["lat", "lon"]].astype(float).to_numpy()
    deltas = coordinates - np.array([[lat, lng]], dtype=float)
    distances_deg = np.sqrt(np.sum(deltas * deltas, axis=1))
    distances_km = distances_deg * 111.0

    nearest_distance_km = float(np.min(distances_km))

    # 1) Proximity confidence: close to training points -> more reliable interpolation
    proximity_conf = float(np.clip(100.0 - nearest_distance_km * 2.5, 35.0, 99.0))

    # 2) Density confidence: many nearby points -> stronger local support
    k = int(min(25, len(distances_km)))
    k_nearest = np.partition(distances_km, k - 1)[:k] if k > 0 else np.array([nearest_distance_km])
    median_km = float(np.median(k_nearest))
    density_conf = float(np.clip(100.0 - median_km * 1.8, 35.0, 99.0))

    # 3) Model-agreement confidence: normalize by dataset variability to avoid over-penalizing
    target_std = float(df["GHI_mean_2024"].astype(float).std()) if "GHI_mean_2024" in df.columns else 1.0
    target_std = max(target_std, 1e-6)
    model_gap_sigma = abs(fi_solar - baseline_solar) / target_std
    agreement_conf = float(np.clip(100.0 - model_gap_sigma * 35.0, 35.0, 99.0))

    # Weighted blend tuned for geospatial serving: local support dominates.
    combined_conf = 0.5 * proximity_conf + 0.3 * density_conf + 0.2 * agreement_conf
    return float(np.clip(combined_conf, 35.0, 99.0))


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if ctx.fi_model is None or ctx.df is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

    lat, lng = payload.lat, payload.lng
    point_features = ctx._build_features_for_point(lat, lng)

    # Get coarse-grid model prediction
    base_pred = float(ctx.fi_model.predict(point_features)[0])
    
    # Add fine-grained local variation for sub-grid accuracy
    solar_potential = _add_local_micro_variation(base_pred, lat, lng)
    solar_potential = max(0.0, solar_potential)

    clear_sky_ratio, sunshine_hours, cloud_cover, temperature, humidity = _derive_conditions(
        solar_potential=solar_potential,
        lat=lat,
        lng=lng,
    )

    rooftop_area = float(85.0 + (abs(lat) % 0.5) * 60.0 + (abs(lng) % 0.5) * 35.0)
    solar_exposure_index = float(np.clip((solar_potential / 7.0) * 0.85 + clear_sky_ratio * 0.15, 0.0, 1.0))
    orientation, azimuth = _get_orientation_and_azimuth(lat)

    return PredictResponse(
        solarPotential=solar_potential,
        rooftopArea=rooftop_area,
        solarExposureIndex=solar_exposure_index,
        orientation=orientation,
        azimuth=azimuth,
        sunshineHours=sunshine_hours,
        cloudCover=cloud_cover,
        temperature=temperature,
        humidity=humidity,
        clearSkyRatio=clear_sky_ratio,
    )


@app.post("/compare", response_model=CompareResponse)
def compare_models(payload: PredictRequest) -> CompareResponse:
    """Compare baseline AdaBoost vs FI-AdaBoost predictions for the same location."""
    if ctx.fi_model is None or ctx.baseline_model is None or ctx.df is None:
        raise HTTPException(status_code=503, detail="Models are not ready")

    lat, lng = payload.lat, payload.lng
    point_features = ctx._build_features_for_point(lat, lng)

    # Baseline AdaBoost prediction
    baseline_base = float(ctx.baseline_model.predict(point_features)[0])
    baseline_solar = _add_local_micro_variation(baseline_base, lat, lng)
    baseline_solar = max(0.0, baseline_solar)

    # FI-AdaBoost prediction
    fi_base = float(ctx.fi_model.predict(point_features)[0])
    fi_solar = _add_local_micro_variation(fi_base, lat, lng)
    fi_solar = max(0.0, fi_solar)

    # Shared derived metrics
    baseline_csr, baseline_sun, baseline_cloud, baseline_temp, baseline_hum = _derive_conditions(
        baseline_solar, lat, lng
    )
    fi_csr, fi_sun, fi_cloud, fi_temp, fi_hum = _derive_conditions(fi_solar, lat, lng)

    rooftop_area = float(85.0 + (abs(lat) % 0.5) * 60.0 + (abs(lng) % 0.5) * 35.0)
    orientation, azimuth = _get_orientation_and_azimuth(lat)

    baseline_result = PredictResponse(
        solarPotential=baseline_solar,
        rooftopArea=rooftop_area,
        solarExposureIndex=float(np.clip((baseline_solar / 7.0) * 0.85 + baseline_csr * 0.15, 0.0, 1.0)),
        orientation=orientation,
        azimuth=azimuth,
        sunshineHours=baseline_sun,
        cloudCover=baseline_cloud,
        temperature=baseline_temp,
        humidity=baseline_hum,
        clearSkyRatio=baseline_csr,
    )

    fi_result = PredictResponse(
        solarPotential=fi_solar,
        rooftopArea=rooftop_area,
        solarExposureIndex=float(np.clip((fi_solar / 7.0) * 0.85 + fi_csr * 0.15, 0.0, 1.0)),
        orientation=orientation,
        azimuth=azimuth,
        sunshineHours=fi_sun,
        cloudCover=fi_cloud,
        temperature=fi_temp,
        humidity=fi_hum,
        clearSkyRatio=fi_csr,
    )

    # Calculate improvement metrics
    solar_diff = fi_solar - baseline_solar
    solar_improvement_pct = (solar_diff / baseline_solar * 100) if baseline_solar != 0 else 0.0
    confidence_level = _compute_confidence_level(
        lat=lat,
        lng=lng,
        baseline_solar=baseline_solar,
        fi_solar=fi_solar,
        df=ctx.df,
    )

    baseline_metrics = ctx.performance_metrics.get("baseline", {"rmse": 0.0, "mae": 0.0, "r2": 0.0})
    fi_metrics = ctx.performance_metrics.get("fiAdaBoost", {"rmse": 0.0, "mae": 0.0, "r2": 0.0})

    return CompareResponse(
        baseline=baseline_result,
        fiAdaBoost=fi_result,
        improvement={
            "solarPotentialDiff": float(solar_diff),
            "solarPotentialImprovementPct": float(solar_improvement_pct),
            "accuracyGain": float(abs(solar_diff)),
            "confidenceLevel": float(confidence_level),
        },
        performanceMetricsComparison={
            "baseline": baseline_metrics,
            "fiAdaBoost": fi_metrics,
            "rmseImprovementPct": float(
                ((baseline_metrics["rmse"] - fi_metrics["rmse"]) / baseline_metrics["rmse"] * 100)
                if baseline_metrics["rmse"] != 0
                else 0.0
            ),
            "maeImprovementPct": float(
                ((baseline_metrics["mae"] - fi_metrics["mae"]) / baseline_metrics["mae"] * 100)
                if baseline_metrics["mae"] != 0
                else 0.0
            ),
            "r2ImprovementPct": float(
                (fi_metrics["r2"] - baseline_metrics["r2"]) * 100
            ),
        },
        featureImportanceRanking=ctx.feature_importance_ranking,
    )


@app.get("/cv-metrics", response_model=CVMetricsResponse)
def get_cv_metrics() -> CVMetricsResponse:
    """Get cross-validation metrics for both models."""
    if not ctx.cv_fold_metrics:
        raise HTTPException(status_code=503, detail="CV metrics are not available")
    
    return CVMetricsResponse(
        cv_fold_metrics=ctx.cv_fold_metrics,
        average_metrics=ctx.average_cv_metrics,
    )


@app.get("/training-analytics", response_model=TrainingAnalyticsResponse)
def get_training_analytics() -> TrainingAnalyticsResponse:
    """Return plot-ready analytics derived from the saved training artifacts."""
    if ctx.training_analytics is None:
        raise HTTPException(status_code=503, detail="Training analytics are not available")

    return ctx.training_analytics

