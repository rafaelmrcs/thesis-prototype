from __future__ import annotations

import __main__
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.model_training2 import (
    BASELINE_FEATURES,
    FI_FEATURES,
    TARGET_J,
    BaselineAdaBoost,
    FIAdaBoostRegressor,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
INTEGRATED_DATA_FILE = ROOT_DIR / "data" / "processed" / "integrated_dataset.csv"
CV_METRICS_FILE = MODEL_DIR / "cv_fold_metrics.csv"
FI_MODEL_FILE = MODEL_DIR / "fi_adaboost.pkl"
BASELINE_MODEL_FILE = MODEL_DIR / "baseline_adaboost.pkl"
TABLE1_FILE = RESULTS_DIR / "table1_training.csv"
TABLE2_FILE = RESULTS_DIR / "table2_test.csv"
ADA_FORECAST_FILE = RESULTS_DIR / "ada_solar_forecast.csv"
FI_FORECAST_FILE = RESULTS_DIR / "fi_solar_forecast.csv"

# Must match model_training2.py exactly
KWH_TO_J = 3_600_000
RANDOM_SEED = 42

NASA_POWER_URL = os.getenv(
    "NASA_POWER_URL",
    "https://power.larc.nasa.gov/api/temporal/daily/point",
)
OVERPASS_API_URL = os.getenv(
    "OVERPASS_API_URL",
    "https://overpass-api.de/api/interpreter",
)
LIVE_FETCH_TIMEOUT_SECONDS = float(os.getenv("LIVE_FETCH_TIMEOUT_SECONDS", "25"))
LIVE_FEATURE_CACHE_TTL_SECONDS = int(os.getenv("LIVE_FEATURE_CACHE_TTL_SECONDS", "1800"))
OVERPASS_RADIUS_METERS = int(os.getenv("OVERPASS_RADIUS_METERS", "150"))
OVERPASS_SHADE_RADIUS_METERS = float(os.getenv("OVERPASS_SHADE_RADIUS_METERS", "50"))

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
    raw_origins = [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]
    origins: list[str] = []

    for origin in raw_origins:
        if origin.startswith(("http://", "https://")):
            origins.append(origin)
            continue

        origins.append(f"https://{origin}")

        if origin.startswith(("localhost", "127.0.0.1", "0.0.0.0")):
            origins.append(f"http://{origin}")

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
        self.baseline_model: BaselineAdaBoost | None = None
        self.df: pd.DataFrame | None = None
        self._kd_tree: cKDTree | None = None
        self._live_cache: dict[str, tuple[float, dict[str, float | str]]] = {}
        self._live_cache_lock = threading.Lock()
        self.performance_metrics: dict[str, dict[str, float]] = {}
        self.feature_importance_ranking: list[dict[str, str | float | int]] = []
        self.cv_fold_metrics: list[CVFoldMetrics] = []
        self.average_cv_metrics: dict[str, dict[str, float]] = {}
        self.training_analytics: TrainingAnalyticsResponse | None = None
        self.training_table: pd.DataFrame | None = None
        self.test_table: pd.DataFrame | None = None
        self.ada_forecast_df: pd.DataFrame | None = None
        self.fi_forecast_df: pd.DataFrame | None = None

    # ── Dataset ───────────────────────────────────────────────────────────

    def _load_dataset(self) -> pd.DataFrame:
        if not INTEGRATED_DATA_FILE.exists():
            raise FileNotFoundError(
                f"Missing integrated dataset: {INTEGRATED_DATA_FILE}\n"
                "Run the full data pipeline first: data_acquisition → data_processing → "
                "feature_engineering → data_integration"
            )

        df = pd.read_csv(INTEGRATED_DATA_FILE)

        required = [TARGET_J, "rooftop_area_sq_m"] + FI_FEATURES
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"integrated_dataset.csv is missing columns: {missing}")

        df = df.dropna(subset=required).reset_index(drop=True)
        return df

    def _load_result_artifacts(self) -> None:
        self.training_table = pd.read_csv(TABLE1_FILE) if TABLE1_FILE.exists() else None
        self.test_table = pd.read_csv(TABLE2_FILE) if TABLE2_FILE.exists() else None
        self.ada_forecast_df = pd.read_csv(ADA_FORECAST_FILE) if ADA_FORECAST_FILE.exists() else None
        self.fi_forecast_df = pd.read_csv(FI_FORECAST_FILE) if FI_FORECAST_FILE.exists() else None

    @staticmethod
    def _metric_row(table: pd.DataFrame | None, model_name: str) -> pd.Series | None:
        if table is None or "Model" not in table.columns:
            return None
        rows = table.loc[table["Model"].astype(str).str.contains(model_name, case=False, na=False)]
        if rows.empty:
            return None
        return rows.iloc[0]

    @staticmethod
    def _to_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _load_metrics_from_results(self) -> bool:
        baseline_row = self._metric_row(self.test_table, "Baseline")
        fi_row = self._metric_row(self.test_table, "FI-AdaBoost")
        if baseline_row is None or fi_row is None:
            return False

        self.performance_metrics = {
            "baseline": {
                "rmse": self._to_float(baseline_row.get("RMSE (kWh/m²/day)")),
                "mae": self._to_float(baseline_row.get("MAE  (kWh/m²/day)")),
                "r2": self._to_float(baseline_row.get("R²")),
            },
            "fiAdaBoost": {
                "rmse": self._to_float(fi_row.get("RMSE (kWh/m²/day)")),
                "mae": self._to_float(fi_row.get("MAE  (kWh/m²/day)")),
                "r2": self._to_float(fi_row.get("R²")),
            },
        }
        return True

    def _build_kd_tree(self) -> None:
        assert self.df is not None
        self._kd_tree = cKDTree(self.df[["lat", "lon"]].values)

    def _get_nearest_features(self, lat: float, lng: float) -> dict[str, float]:
        """Return OSM-derived features from the nearest training point."""
        assert self._kd_tree is not None and self.df is not None
        _, idx = self._kd_tree.query([lat, lng])
        row = self.df.iloc[idx]
        return {
            "orientation_score": float(row["orientation_score"]),
            "shading_factor": float(row["shading_factor"]),
            "tilt_factor": float(row["tilt_factor"]),
            "SEI_norm": float(row["SEI_norm"]),
            "rooftop_area_sq_m": float(row["rooftop_area_sq_m"]),
        }

    # ── Training targets (replicate model_training2.py logic) ─────────────

    def _target_j(self, df: pd.DataFrame) -> np.ndarray:
        return df[TARGET_J].values.astype(float)

    def _effective_target_j(self, df: pd.DataFrame) -> np.ndarray:
        """Target_Effective_J = GHI_mean_J * safe_shade * safe_orient (same as model_training2.py)."""
        s_min, s_max = df["shading_factor"].min(), df["shading_factor"].max()
        o_min, o_max = df["orientation_score"].min(), df["orientation_score"].max()
        safe_shade  = 0.85 + 0.15 * ((df["shading_factor"]  - s_min) / (s_max - s_min + 1e-9))
        safe_orient = 0.85 + 0.15 * ((df["orientation_score"] - o_min) / (o_max - o_min + 1e-9))
        return (df[TARGET_J] * safe_shade * safe_orient).values.astype(float)

    # ── Model loading / training ──────────────────────────────────────────

    def _load_model_file(self, model_path: Path) -> object | None:
        if not hasattr(__main__, "FIAdaBoostRegressor"):
            setattr(__main__, "FIAdaBoostRegressor", FIAdaBoostRegressor)
        if not hasattr(__main__, "BaselineAdaBoost"):
            setattr(__main__, "BaselineAdaBoost", BaselineAdaBoost)
        try:
            return joblib.load(model_path)
        except (FileNotFoundError, AttributeError, ModuleNotFoundError, ImportError, ValueError):
            return None

    def _model_supports_feature_count(self, model: object, expected_count: int) -> bool:
        wrapped = getattr(model, "_model", model)
        feature_count = getattr(wrapped, "n_features_in_", None)
        if feature_count is not None:
            return int(feature_count) == expected_count

        estimators = getattr(model, "estimators_", None)
        if estimators:
            first_feature_count = getattr(estimators[0], "n_features_in_", None)
            if first_feature_count is not None:
                return int(first_feature_count) == expected_count

        return True

    def _train_fi_model(self) -> FIAdaBoostRegressor:
        assert self.df is not None
        X = self.df[FI_FEATURES].values.astype(float)
        y = self._effective_target_j(self.df)
        model = FIAdaBoostRegressor()
        model.fit(X, y)
        return model

    def _train_baseline_model(self) -> BaselineAdaBoost:
        assert self.df is not None
        X = self.df[BASELINE_FEATURES].values.astype(float)
        y = self._target_j(self.df)
        model = BaselineAdaBoost()
        model.fit(X, y)
        return model

    # ── Analytics ─────────────────────────────────────────────────────────

    def _evaluate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    def _analytics_test_df(self) -> pd.DataFrame:
        assert self.df is not None
        idx = np.arange(len(self.df))
        _, test_idx = train_test_split(
            idx,
            test_size=0.20,
            random_state=RANDOM_SEED,
            shuffle=True,
        )
        return self.df.iloc[test_idx].reset_index(drop=True)

    def _compute_feature_importance_ranking(self) -> list[dict[str, str | float | int]]:
        """Build per-feature ranking for FI_FEATURES. Baseline importances are
        padded with zeros for features it was not trained on."""
        assert self.fi_model is not None and self.baseline_model is not None

        # FI model: weighted average over weak learners
        fi_importance = np.zeros(len(FI_FEATURES), dtype=float)
        if len(self.fi_model.estimators_) > 0:
            weights = np.asarray(self.fi_model.estimator_weights_, dtype=float)
            if np.sum(weights) <= 0:
                weights = np.ones_like(weights)
            weights = weights / np.sum(weights)
            for i, estimator in enumerate(self.fi_model.estimators_):
                if hasattr(estimator, "feature_importances_"):
                    fi_importance += weights[i] * np.asarray(
                        estimator.feature_importances_, dtype=float
                    )

        # Baseline model: only trained on BASELINE_FEATURES (lat, lon)
        # Pad zeros for the spatial features it doesn't use
        base_raw = np.asarray(
            getattr(self.baseline_model, "feature_importances_", np.zeros(len(BASELINE_FEATURES))),
            dtype=float,
        )
        baseline_importance = np.zeros(len(FI_FEATURES), dtype=float)
        for i, feat in enumerate(BASELINE_FEATURES):
            if feat in FI_FEATURES:
                baseline_importance[FI_FEATURES.index(feat)] = base_raw[i]

        records: list[dict[str, str | float | int]] = []
        for i, feature in enumerate(FI_FEATURES):
            records.append(
                {
                    "feature": feature,
                    "fiAdaBoostImportance": float(fi_importance[i]),
                    "baselineImportance": float(baseline_importance[i]),
                    "importanceGain": float(fi_importance[i] - baseline_importance[i]),
                    "rank": 0,
                }
            )

        records.sort(key=lambda r: float(r["fiAdaBoostImportance"]), reverse=True)
        for rank, row in enumerate(records, start=1):
            row["rank"] = rank
        return records

    def _compute_global_analytics(self) -> None:
        assert self.df is not None and self.fi_model is not None and self.baseline_model is not None

        # Prefer persisted evaluation metrics from results artifacts so frontend and reports stay consistent.
        loaded_from_results = self._load_metrics_from_results()

        if not loaded_from_results:
            test_df = self._analytics_test_df()

            X_fi_test = test_df[FI_FEATURES].values.astype(float)
            X_base_test = test_df[BASELINE_FEATURES].values.astype(float)
            y_base_test = self._target_j(test_df)
            y_fi_test = self._effective_target_j(test_df)

            baseline_pred = np.asarray(self.baseline_model.predict(X_base_test), dtype=float)
            fi_pred = np.asarray(self.fi_model.predict(X_fi_test), dtype=float)

            # Convert fallback metrics to kWh/m²/day for UI consistency.
            base_metrics = self._evaluate_metrics(y_base_test / KWH_TO_J, baseline_pred / KWH_TO_J)
            fi_metrics = self._evaluate_metrics(y_fi_test / KWH_TO_J, fi_pred / KWH_TO_J)
            self.performance_metrics = {
                "baseline": base_metrics,
                "fiAdaBoost": fi_metrics,
            }

        self.feature_importance_ranking = self._compute_feature_importance_ranking()

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

        # Pick decimal places based on the range so labels are readable
        span = max_edge - min_edge
        fmt = ".3f" if span < 1 else ".1f"
        return [
            ErrorDistributionBin(
                bucket=f"{edges[i]:{fmt}} to {edges[i + 1]:{fmt}}",
                baseline=int(baseline_counts[i]),
                fiAdaBoost=int(fi_counts[i]),
            )
            for i in range(len(edges) - 1)
        ]

    def _compute_training_analytics(self) -> None:
        self.training_analytics = None
        if self.df is None or self.fi_model is None or self.baseline_model is None:
            return

        baseline_metrics = self.performance_metrics.get("baseline", {"rmse": 0.0, "mae": 0.0, "r2": 0.0})
        fi_metrics = self.performance_metrics.get("fiAdaBoost", {"rmse": 0.0, "mae": 0.0, "r2": 0.0})

        if self.ada_forecast_df is not None and self.fi_forecast_df is not None:
            ada_df = self.ada_forecast_df.copy()
            fi_df = self.fi_forecast_df.copy()

            if "GHI_mean_2024" not in ada_df.columns or "predicted_GHI_kWh" not in ada_df.columns:
                return

            base_actual = ada_df["GHI_mean_2024"].astype(float).to_numpy()
            base_pred = ada_df["predicted_GHI_kWh"].astype(float).to_numpy()
            fi_actual = fi_df["GHI_mean_2024"].astype(float).to_numpy()
            fi_pred = fi_df["predicted_GHI_kWh"].astype(float).to_numpy()

            sample_size = min(220, len(ada_df), len(fi_df))
            sample_idx = np.unique(np.linspace(0, sample_size - 1, num=sample_size, dtype=int))
            date_labels = [str(i) for i in sample_idx]

            actual_base_sample = base_actual[sample_idx]
            base_pred_sample = base_pred[sample_idx]
            actual_fi_sample = fi_actual[sample_idx]
            fi_pred_sample = fi_pred[sample_idx]

            all_vals = np.concatenate([actual_base_sample, actual_fi_sample, base_pred_sample, fi_pred_sample])
            domain_min = float(np.min(all_vals))
            domain_max = float(np.max(all_vals))

            predicted_vs_actual = PredictedVsActualData(
                baseline=[
                    PredictedActualPoint(actual=float(a), predicted=float(p), date=d)
                    for a, p, d in zip(actual_base_sample, base_pred_sample, date_labels)
                ],
                fiAdaBoost=[
                    PredictedActualPoint(actual=float(a), predicted=float(p), date=d)
                    for a, p, d in zip(actual_fi_sample, fi_pred_sample, date_labels)
                ],
                domainMin=domain_min,
                domainMax=domain_max,
            )

            baseline_residuals = base_pred - base_actual
            fi_residuals = fi_pred - fi_actual
        else:
            if len(self.df) < 2:
                return

            # Fallback: compute analytics directly from held-out split if saved result files are missing.
            test_df = self._analytics_test_df()
            X_fi_test = test_df[FI_FEATURES].values.astype(float)
            X_base_test = test_df[BASELINE_FEATURES].values.astype(float)
            y_base_kwh = self._target_j(test_df) / KWH_TO_J
            y_fi_kwh = self._effective_target_j(test_df) / KWH_TO_J
            base_pred = np.asarray(self.baseline_model.predict(X_base_test), dtype=float) / KWH_TO_J
            fi_pred = np.asarray(self.fi_model.predict(X_fi_test), dtype=float) / KWH_TO_J

            sample_size = min(220, len(test_df))
            sample_idx = np.unique(np.linspace(0, len(test_df) - 1, num=sample_size, dtype=int))
            date_labels = [str(test_df.index[i]) for i in sample_idx]

            actual_base_sample = y_base_kwh[sample_idx]
            base_pred_sample = base_pred[sample_idx]
            actual_fi_sample = y_fi_kwh[sample_idx]
            fi_pred_sample = fi_pred[sample_idx]

            all_vals = np.concatenate([actual_base_sample, actual_fi_sample, base_pred_sample, fi_pred_sample])
            domain_min = float(np.min(all_vals))
            domain_max = float(np.max(all_vals))

            predicted_vs_actual = PredictedVsActualData(
                baseline=[
                    PredictedActualPoint(actual=float(a), predicted=float(p), date=d)
                    for a, p, d in zip(actual_base_sample, base_pred_sample, date_labels)
                ],
                fiAdaBoost=[
                    PredictedActualPoint(actual=float(a), predicted=float(p), date=d)
                    for a, p, d in zip(actual_fi_sample, fi_pred_sample, date_labels)
                ],
                domainMin=domain_min,
                domainMax=domain_max,
            )

            baseline_residuals = base_pred - y_base_kwh
            fi_residuals = fi_pred - y_fi_kwh

            baseline_metrics = self._evaluate_metrics(y_base_kwh, base_pred)
            fi_metrics = self._evaluate_metrics(y_fi_kwh, fi_pred)
            self.performance_metrics = {
                "baseline": baseline_metrics,
                "fiAdaBoost": fi_metrics,
            }

        self.training_analytics = TrainingAnalyticsResponse(
            performanceMetricsComparison={
                "baseline": baseline_metrics,
                "fiAdaBoost": fi_metrics,
                "rmseImprovementPct": float(
                    ((baseline_metrics["rmse"] - fi_metrics["rmse"]) / baseline_metrics["rmse"] * 100)
                    if baseline_metrics["rmse"] != 0 else 0.0
                ),
                "maeImprovementPct": float(
                    ((baseline_metrics["mae"] - fi_metrics["mae"]) / baseline_metrics["mae"] * 100)
                    if baseline_metrics["mae"] != 0 else 0.0
                ),
                "r2ImprovementPct": float((fi_metrics["r2"] - baseline_metrics["r2"]) * 100),
            },
            predictedVsActual=predicted_vs_actual,
            errorDistribution=self._build_error_distribution(baseline_residuals, fi_residuals),
            residualSummary={
                "baselineStd": float(np.std(baseline_residuals, ddof=1)) if len(baseline_residuals) > 1 else 0.0,
                "fiStd": float(np.std(fi_residuals, ddof=1)) if len(fi_residuals) > 1 else 0.0,
                "baselineMae": float(np.mean(np.abs(baseline_residuals))),
                "fiMae": float(np.mean(np.abs(fi_residuals))),
            },
        )

    def _load_cv_metrics(self) -> None:
        self.cv_fold_metrics = []
        self.average_cv_metrics = {}

        if CV_METRICS_FILE.exists():
            try:
                df = pd.read_csv(CV_METRICS_FILE)
                for _, row in df.iterrows():
                    self.cv_fold_metrics.append(
                        CVFoldMetrics(
                            fold=int(row["fold"]),
                            baseline_rmse=float(row["baseline_RMSE"]),
                            baseline_mae=float(row["baseline_MAE"]),
                            baseline_r2=float(row["baseline_R2"]),
                            fi_rmse=float(row["fi_RMSE"]),
                            fi_mae=float(row["fi_MAE"]),
                            fi_r2=float(row["fi_R2"]),
                        )
                    )
            except Exception as exc:
                print(f"Warning: Failed to load CV metrics file: {exc}")

        # Fallback to saved test table when explicit CV file is unavailable.
        if not self.cv_fold_metrics:
            baseline_row = self._metric_row(self.test_table, "Baseline")
            fi_row = self._metric_row(self.test_table, "FI-AdaBoost")
            if baseline_row is not None and fi_row is not None:
                self.cv_fold_metrics = [
                    CVFoldMetrics(
                        fold=1,
                        baseline_rmse=self._to_float(baseline_row.get("RMSE (J/m²/day)")),
                        baseline_mae=self._to_float(baseline_row.get("MAE  (J/m²/day)")),
                        baseline_r2=self._to_float(baseline_row.get("R²")),
                        fi_rmse=self._to_float(fi_row.get("RMSE (J/m²/day)")),
                        fi_mae=self._to_float(fi_row.get("MAE  (J/m²/day)")),
                        fi_r2=self._to_float(fi_row.get("R²")),
                    )
                ]

        if self.cv_fold_metrics:
            self.average_cv_metrics = {
                "baseline": {
                    "rmse": float(np.mean([m.baseline_rmse for m in self.cv_fold_metrics])),
                    "mae": float(np.mean([m.baseline_mae for m in self.cv_fold_metrics])),
                    "r2": float(np.mean([m.baseline_r2 for m in self.cv_fold_metrics])),
                },
                "fiAdaBoost": {
                    "rmse": float(np.mean([m.fi_rmse for m in self.cv_fold_metrics])),
                    "mae": float(np.mean([m.fi_mae for m in self.cv_fold_metrics])),
                    "r2": float(np.mean([m.fi_r2 for m in self.cv_fold_metrics])),
                },
            }

    # ── Startup ───────────────────────────────────────────────────────────

    def ensure_loaded(self) -> None:
        self.df = self._load_dataset()
        self._load_result_artifacts()
        self._build_kd_tree()

        # Try loading canonical persisted models; fall back to training fresh.
        fi_candidates = [FI_MODEL_FILE]
        baseline_candidates = [BASELINE_MODEL_FILE]

        self.fi_model = None
        for path in fi_candidates:
            if path.exists():
                loaded = self._load_model_file(path)
                if loaded is not None and self._model_supports_feature_count(loaded, len(FI_FEATURES)):
                    self.fi_model = loaded
                    break

        self.baseline_model = None
        for path in baseline_candidates:
            if path.exists():
                loaded = self._load_model_file(path)
                if loaded is not None and self._model_supports_feature_count(loaded, len(BASELINE_FEATURES)):
                    self.baseline_model = loaded
                    break

        if self.fi_model is None:
            print("Training FI-AdaBoost model from integrated_dataset.csv …")
            self.fi_model = self._train_fi_model()
            joblib.dump(self.fi_model, FI_MODEL_FILE)

        if self.baseline_model is None:
            print("Training baseline AdaBoost model from integrated_dataset.csv …")
            self.baseline_model = self._train_baseline_model()
            joblib.dump(self.baseline_model, BASELINE_MODEL_FILE)

        self._compute_global_analytics()
        self._load_cv_metrics()
        self._compute_training_analytics()


ctx = ModelContext()


@app.on_event("startup")
def startup_event() -> None:
    ctx.ensure_loaded()


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _cache_key(prefix: str, lat: float, lng: float) -> str:
    return f"{prefix}:{lat:.5f}:{lng:.5f}"


def _cache_get(ctx_obj: ModelContext, key: str) -> dict[str, float | str] | None:
    now = time.time()
    with ctx_obj._live_cache_lock:
        item = ctx_obj._live_cache.get(key)
        if item is None:
            return None
        expires_at, payload = item
        if expires_at < now:
            ctx_obj._live_cache.pop(key, None)
            return None
        return payload


def _cache_set(ctx_obj: ModelContext, key: str, payload: dict[str, float | str]) -> None:
    expires_at = time.time() + LIVE_FEATURE_CACHE_TTL_SECONDS
    with ctx_obj._live_cache_lock:
        ctx_obj._live_cache[key] = (expires_at, payload)


def _safe_mean(values: dict[str, float] | None) -> float:
    if values is None:
        return float("nan")
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    series = series[(series > -900)]
    if series.empty:
        return float("nan")
    return float(series.mean())


def _haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)
    a = np.sin(d_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2.0) ** 2
    return float(2.0 * r * np.arcsin(np.sqrt(a)))


def _latlon_to_xy_m(points: list[tuple[float, float]], ref_lat: float) -> list[tuple[float, float]]:
    r = 6_371_000.0
    cos_lat = np.cos(np.radians(ref_lat))
    out: list[tuple[float, float]] = []
    for lat, lon in points:
        x = float(np.radians(lon) * r * cos_lat)
        y = float(np.radians(lat) * r)
        out.append((x, y))
    return out


def _polygon_area_m2(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0

    if points[0] != points[-1]:
        ring = points + [points[0]]
    else:
        ring = points

    ref_lat = float(np.mean([p[0] for p in ring]))
    xy = _latlon_to_xy_m(ring, ref_lat)
    x = np.array([p[0] for p in xy], dtype=float)
    y = np.array([p[1] for p in xy], dtype=float)
    area = 0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))
    return float(area)


def _bearing_to_cardinal(azimuth: float) -> str:
    sectors = ["North", "Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest"]
    idx = int(((azimuth + 22.5) % 360) // 45)
    return sectors[idx]


def _derive_weather_fallback(
    solar_potential_kwh: float,
    lat: float,
    lng: float,
) -> dict[str, float]:
    clear_sky_ratio = float(np.clip(solar_potential_kwh / 8.0, 0.25, 0.9))
    sunshine_hours = float(np.clip(3.5 + clear_sky_ratio * 7.0, 3.0, 10.5))
    cloud_cover = float(np.clip((1.0 - clear_sky_ratio) * 100.0, 8.0, 85.0))
    latitude_factor = abs(lat) / 20.0
    temperature = float(np.clip(31.0 - latitude_factor * 8.0 + (clear_sky_ratio - 0.5) * 3.0, 20.0, 36.0))
    humidity = float(np.clip(88.0 - (clear_sky_ratio * 35.0) + (abs(lng) % 1.0) * 8.0, 40.0, 95.0))
    return {
        "clear_sky_ratio": clear_sky_ratio,
        "sunshine_hours": sunshine_hours,
        "cloud_cover": cloud_cover,
        "temperature": temperature,
        "humidity": humidity,
    }


def _fetch_nasa_weather(lat: float, lng: float) -> dict[str, float]:
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=45)

    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M,RH2M",
        "community": "RE",
        "latitude": lat,
        "longitude": lng,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON",
    }

    response = requests.get(NASA_POWER_URL, params=params, timeout=LIVE_FETCH_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    parameters = payload["properties"]["parameter"]

    allsky = _safe_mean(parameters.get("ALLSKY_SFC_SW_DWN"))
    clearsky = _safe_mean(parameters.get("CLRSKY_SFC_SW_DWN"))
    temperature = _safe_mean(parameters.get("T2M"))
    humidity = _safe_mean(parameters.get("RH2M"))

    if not np.isfinite(allsky) or not np.isfinite(clearsky) or clearsky <= 0:
        raise ValueError("NASA POWER response does not contain valid ALLSKY/CLRSKY irradiance values")

    clear_sky_ratio = float(np.clip(allsky / clearsky, 0.05, 1.0))
    cloud_cover = float(np.clip((1.0 - clear_sky_ratio) * 100.0, 0.0, 100.0))
    sunshine_hours = float(np.clip(4.0 + (clear_sky_ratio * 8.0), 2.0, 12.0))

    if not np.isfinite(temperature):
        temperature = 28.0
    if not np.isfinite(humidity):
        humidity = 75.0

    return {
        "clear_sky_ratio": clear_sky_ratio,
        "sunshine_hours": sunshine_hours,
        "cloud_cover": cloud_cover,
        "temperature": float(np.clip(temperature, -20.0, 55.0)),
        "humidity": float(np.clip(humidity, 0.0, 100.0)),
    }


def _fetch_overpass_geometries(lat: float, lng: float) -> list[list[tuple[float, float]]]:
    query = f"""
    [out:json][timeout:25];
    (
      way(around:{OVERPASS_RADIUS_METERS},{lat},{lng})[\"building\"];
      relation(around:{OVERPASS_RADIUS_METERS},{lat},{lng})[\"building\"];
    );
    out geom;
    """.strip()

    response = requests.post(
        OVERPASS_API_URL,
        data=query,
        timeout=LIVE_FETCH_TIMEOUT_SECONDS,
        headers={"Content-Type": "text/plain"},
    )
    response.raise_for_status()

    payload = response.json()
    elements = payload.get("elements", [])
    geometries: list[list[tuple[float, float]]] = []

    for element in elements:
        geometry = element.get("geometry")
        if not isinstance(geometry, list) or len(geometry) < 3:
            continue

        points: list[tuple[float, float]] = []
        for g in geometry:
            g_lat = g.get("lat")
            g_lon = g.get("lon")
            if g_lat is None or g_lon is None:
                continue
            points.append((float(g_lat), float(g_lon)))

        if len(points) >= 3:
            geometries.append(points)

    return geometries


def _compute_live_rooftop_features(lat: float, lng: float) -> dict[str, float | str]:
    geometries = _fetch_overpass_geometries(lat, lng)
    if not geometries:
        raise ValueError("No nearby OSM building geometry found")

    buildings: list[dict[str, float | str]] = []
    centroids: list[tuple[float, float]] = []
    max_area = 0.0

    for points in geometries:
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        centroid_lat = float(np.mean(lats))
        centroid_lon = float(np.mean(lons))
        area_m2 = _polygon_area_m2(points)
        if area_m2 <= 5.0:
            continue

        max_lat = max(lats)
        min_lat = min(lats)
        max_lon = max(lons)
        min_lon = min(lons)

        meters_per_deg_lat = 111_132.0
        meters_per_deg_lon = 111_320.0 * np.cos(np.radians(centroid_lat))
        dx = (max_lon - min_lon) * meters_per_deg_lon
        dy = (max_lat - min_lat) * meters_per_deg_lat
        azimuth = float((np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0)

        orientation_score = float((np.cos(np.radians(azimuth - 180.0)) + 1.0) / 2.0)
        distance_m = _haversine_distance_m(lat, lng, centroid_lat, centroid_lon)

        buildings.append(
            {
                "centroid_lat": centroid_lat,
                "centroid_lon": centroid_lon,
                "rooftop_area_sq_m": float(area_m2),
                "azimuth": azimuth,
                "orientation_score": orientation_score,
                "distance_m": distance_m,
            }
        )
        centroids.append((centroid_lat, centroid_lon))
        max_area = max(max_area, float(area_m2))

    if not buildings:
        raise ValueError("No valid rooftop polygon could be extracted from nearby OSM buildings")

    for i, building in enumerate(buildings):
        nearby_count = 0
        for j, (other_lat, other_lon) in enumerate(centroids):
            if i == j:
                continue
            d = _haversine_distance_m(
                float(building["centroid_lat"]),
                float(building["centroid_lon"]),
                other_lat,
                other_lon,
            )
            if d <= OVERPASS_SHADE_RADIUS_METERS:
                nearby_count += 1

        shade_norm = nearby_count / max(len(buildings) - 1, 1)
        building["shading_factor"] = float(np.clip(0.3 * shade_norm, 0.0, 1.0))

    optimal_tilt = abs(lat)
    tilt_factor = float(np.clip(np.cos(np.radians(abs(0.0 - optimal_tilt))), 0.7, 1.0))

    sei_values: list[float] = []
    for building in buildings:
        sei = (
            float(building["orientation_score"])
            * float(building["rooftop_area_sq_m"])
            * (1.0 - float(building["shading_factor"]))
            * tilt_factor
        )
        building["tilt_factor"] = tilt_factor
        building["solar_exposure_index"] = float(sei)
        sei_values.append(float(sei))

    max_sei = max(max(sei_values), 1e-9)
    for building in buildings:
        building["SEI_norm"] = float(np.clip(float(building["solar_exposure_index"]) / max_sei, 0.0, 1.0))

    nearest = min(buildings, key=lambda b: float(b["distance_m"]))
    orientation = _bearing_to_cardinal(float(nearest["azimuth"]))

    return {
        "rooftop_area_sq_m": float(nearest["rooftop_area_sq_m"]),
        "orientation_score": float(nearest["orientation_score"]),
        "shading_factor": float(nearest["shading_factor"]),
        "tilt_factor": float(nearest["tilt_factor"]),
        "SEI_norm": float(nearest["SEI_norm"]),
        "orientation": orientation,
        "azimuth": float(nearest["azimuth"]),
        "distance_m": float(nearest["distance_m"]),
        "source": "osm-live",
    }


def _get_live_rooftop_features(lat: float, lng: float, ctx_obj: ModelContext) -> dict[str, float | str]:
    key = _cache_key("rooftop", lat, lng)
    cached = _cache_get(ctx_obj, key)
    if cached is not None:
        return cached

    try:
        payload = _compute_live_rooftop_features(lat, lng)
        _cache_set(ctx_obj, key, payload)
        return payload
    except Exception:
        # Fallback to nearest precomputed spatial feature row.
        fallback = ctx_obj._get_nearest_features(lat, lng)
        orientation = "South" if lat >= 0 else "North"
        payload = {
            "rooftop_area_sq_m": float(fallback["rooftop_area_sq_m"]),
            "orientation_score": float(fallback["orientation_score"]),
            "shading_factor": float(fallback["shading_factor"]),
            "tilt_factor": float(fallback["tilt_factor"]),
            "SEI_norm": float(fallback["SEI_norm"]),
            "orientation": orientation,
            "azimuth": 180.0 if lat >= 0 else 0.0,
            "distance_m": 0.0,
            "source": "dataset-fallback",
        }
        _cache_set(ctx_obj, key, payload)
        return payload


def _get_live_weather(lat: float, lng: float, solar_potential_kwh: float, ctx_obj: ModelContext) -> dict[str, float]:
    key = _cache_key("weather", lat, lng)
    cached = _cache_get(ctx_obj, key)
    if cached is not None:
        return {
            "clear_sky_ratio": float(cached["clear_sky_ratio"]),
            "sunshine_hours": float(cached["sunshine_hours"]),
            "cloud_cover": float(cached["cloud_cover"]),
            "temperature": float(cached["temperature"]),
            "humidity": float(cached["humidity"]),
        }

    try:
        weather = _fetch_nasa_weather(lat, lng)
    except Exception:
        weather = _derive_weather_fallback(solar_potential_kwh, lat, lng)

    _cache_set(ctx_obj, key, {k: float(v) for k, v in weather.items()})
    return weather


def _compute_confidence_level(
    lat: float,
    lng: float,
    baseline_solar: float,
    fi_solar: float,
    df: pd.DataFrame,
) -> float:
    coords = df[["lat", "lon"]].astype(float).to_numpy()
    distances_km = np.sqrt(np.sum((coords - np.array([[lat, lng]])) ** 2, axis=1)) * 111.0

    nearest_km = float(np.min(distances_km))
    proximity_conf = float(np.clip(100.0 - nearest_km * 2.5, 35.0, 99.0))

    k = int(min(25, len(distances_km)))
    k_nearest = np.partition(distances_km, k - 1)[:k] if k > 0 else np.array([nearest_km])
    density_conf = float(np.clip(100.0 - float(np.median(k_nearest)) * 1.8, 35.0, 99.0))

    target_std = max(float(df[TARGET_J].astype(float).std() / KWH_TO_J), 1e-6)
    agreement_conf = float(np.clip(100.0 - abs(fi_solar - baseline_solar) / target_std * 35.0, 35.0, 99.0))

    return float(np.clip(0.5 * proximity_conf + 0.3 * density_conf + 0.2 * agreement_conf, 35.0, 99.0))


def _build_predict_response(
    lat: float,
    lng: float,
    solar_kwh: float,
    features: dict[str, float],
    weather: dict[str, float],
    rooftop_meta: dict[str, float | str],
) -> PredictResponse:
    return PredictResponse(
        solarPotential=solar_kwh,
        rooftopArea=features["rooftop_area_sq_m"],
        solarExposureIndex=features["SEI_norm"],
        orientation=str(rooftop_meta.get("orientation", "South")),
        azimuth=float(rooftop_meta.get("azimuth", 180.0 if lat >= 0 else 0.0)),
        sunshineHours=float(weather["sunshine_hours"]),
        cloudCover=float(weather["cloud_cover"]),
        temperature=float(weather["temperature"]),
        humidity=float(weather["humidity"]),
        clearSkyRatio=float(weather["clear_sky_ratio"]),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if ctx.fi_model is None or ctx._kd_tree is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

    lat, lng = payload.lat, payload.lng
    rooftop_payload = _get_live_rooftop_features(lat, lng, ctx)
    features = {
        "orientation_score": float(rooftop_payload["orientation_score"]),
        "shading_factor": float(rooftop_payload["shading_factor"]),
        "tilt_factor": float(rooftop_payload["tilt_factor"]),
        "SEI_norm": float(rooftop_payload["SEI_norm"]),
        "rooftop_area_sq_m": float(rooftop_payload["rooftop_area_sq_m"]),
    }

    X_fi = np.array([[lat, lng, features["orientation_score"], features["shading_factor"],
                      features["tilt_factor"], features["SEI_norm"]]])
    ghi_j = float(ctx.fi_model.predict(X_fi)[0])
    solar_kwh = max(0.0, ghi_j / KWH_TO_J)

    weather = _get_live_weather(lat, lng, solar_kwh, ctx)

    return _build_predict_response(lat, lng, solar_kwh, features, weather, rooftop_payload)


@app.post("/compare", response_model=CompareResponse)
def compare_models(payload: PredictRequest) -> CompareResponse:
    if ctx.fi_model is None or ctx.baseline_model is None or ctx._kd_tree is None or ctx.df is None:
        raise HTTPException(status_code=503, detail="Models are not ready")

    lat, lng = payload.lat, payload.lng
    rooftop_payload = _get_live_rooftop_features(lat, lng, ctx)
    features = {
        "orientation_score": float(rooftop_payload["orientation_score"]),
        "shading_factor": float(rooftop_payload["shading_factor"]),
        "tilt_factor": float(rooftop_payload["tilt_factor"]),
        "SEI_norm": float(rooftop_payload["SEI_norm"]),
        "rooftop_area_sq_m": float(rooftop_payload["rooftop_area_sq_m"]),
    }

    X_base = np.array([[lat, lng]])
    X_fi = np.array([[lat, lng, features["orientation_score"], features["shading_factor"],
                      features["tilt_factor"], features["SEI_norm"]]])

    baseline_kwh = max(0.0, float(ctx.baseline_model.predict(X_base)[0]) / KWH_TO_J)
    fi_kwh = max(0.0, float(ctx.fi_model.predict(X_fi)[0]) / KWH_TO_J)

    weather = _get_live_weather(lat, lng, fi_kwh, ctx)

    baseline_result = _build_predict_response(lat, lng, baseline_kwh, features, weather, rooftop_payload)
    fi_result = _build_predict_response(lat, lng, fi_kwh, features, weather, rooftop_payload)

    solar_diff = fi_kwh - baseline_kwh
    solar_improvement_pct = (solar_diff / baseline_kwh * 100) if baseline_kwh != 0 else 0.0
    confidence_level = _compute_confidence_level(lat, lng, baseline_kwh, fi_kwh, ctx.df)

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
                if baseline_metrics["rmse"] != 0 else 0.0
            ),
            "maeImprovementPct": float(
                ((baseline_metrics["mae"] - fi_metrics["mae"]) / baseline_metrics["mae"] * 100)
                if baseline_metrics["mae"] != 0 else 0.0
            ),
            "r2ImprovementPct": float((fi_metrics["r2"] - baseline_metrics["r2"]) * 100),
        },
        featureImportanceRanking=ctx.feature_importance_ranking,
    )


@app.get("/cv-metrics", response_model=CVMetricsResponse)
def get_cv_metrics() -> CVMetricsResponse:
    if not ctx.cv_fold_metrics:
        raise HTTPException(status_code=503, detail="CV metrics are not available")
    return CVMetricsResponse(
        cv_fold_metrics=ctx.cv_fold_metrics,
        average_metrics=ctx.average_cv_metrics,
    )


@app.get("/training-analytics", response_model=TrainingAnalyticsResponse)
def get_training_analytics() -> TrainingAnalyticsResponse:
    if ctx.training_analytics is None:
        raise HTTPException(status_code=503, detail="Training analytics are not available")
    return ctx.training_analytics
