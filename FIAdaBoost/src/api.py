"""
FastAPI backend for FI-AdaBoost solar potential forecasting.

Key runtime rules:
  - `/predict` and `/compare` build the exact 8-feature model vector live
    from the requested coordinates.
  - Model inference loads from `models/`, not from the training script.
  - `results/*.csv` remain the source for saved evaluation metrics.
  - Processed CSVs are optional at runtime and are only used for confidence
    scoring, analytics, or last-resort retraining if model files are missing.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import pvlib
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.model_runtime import (
    BASELINE_FEATURES,
    DEFAULT_LAT_RANGE,
    DEFAULT_LON_RANGE,
    DEFAULT_SEI_NORMALIZER,
    KWH_TO_J,
    RANDOM_SEED,
    SHARED_FEATURES,
    BaselineAdaBoost,
    FIAdaBoostRegressor,
    TARGET_COL,
    register_pickle_compat,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
INTEGRATED_DATA_FILE = PROCESSED_DIR / "integrated_dataset.csv"
FI_MODEL_FILE = MODEL_DIR / "fi_adaboost.pkl"
BASELINE_MODEL_FILE = MODEL_DIR / "baseline_adaboost.pkl"
METRICS_FILE = RESULTS_DIR / "metrics_summary.csv"
CV_METRICS_FILE = RESULTS_DIR / "cv_fold_metrics_daily.csv"
SPATIAL_CV_METRICS_FILE = RESULTS_DIR / "cv_fold_metrics.csv"
DM_SPATIAL_FILE = RESULTS_DIR / "dm_test_results_spatial.csv"
DM_DAILY_FILE = RESULTS_DIR / "dm_test_results_daily.csv"
DAILY_METRICS_FILE = RESULTS_DIR / "daily_metrics_summary.csv"
SPLIT_INFO_FILE = RESULTS_DIR / "train_test_split_info.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_TZ = "Asia/Manila"
TILT_FACTOR = 0.99211470131447788
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_TIMEOUT_SEC = 30
NASA_LOOKBACK_DAYS = int(os.getenv("NASA_LOOKBACK_DAYS", "365"))
NASA_LAG_DAYS = int(os.getenv("NASA_LAG_DAYS", "7"))
DEFAULT_OVERPASS_API_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
)
OVERPASS_TIMEOUT_SEC = 18
OVERPASS_RADIUS_SEQUENCE_METERS = (150, 300)
REQUEST_USER_AGENT = "FI-AdaBoost-Solar-Potential/2.1"
DEFAULT_CORS_ORIGIN_REGEX = (
    r"http://([a-zA-Z0-9\-\.]+):5173"
    r"|https://([a-zA-Z0-9\-]+\.)*(up\.)?railway\.app"
)

_pvlib_cache: dict[tuple[float, float, str, str], dict[str, float]] = {}
_nasa_cache: dict[tuple[float, float, str, str], dict[str, float | str]] = {}
logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)


class PredictResponse(BaseModel):
    solarPotential: float
    predictedIrradiance: float
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


class SpatialCVFold(BaseModel):
    fold: int
    baseline_train_rmse: float
    baseline_val_rmse: float
    baseline_train_r2: float
    baseline_val_r2: float
    fi_train_rmse: float
    fi_val_rmse: float
    fi_train_r2: float
    fi_val_r2: float


class SpatialCVResponse(BaseModel):
    folds: list[SpatialCVFold]
    average: dict[str, float]


class DMTestResult(BaseModel):
    dm_statistic: float
    p_value: float
    significant: bool
    interpretation: str
    n_samples: int
    mean_loss_diff_j2: float


class DMTestResponse(BaseModel):
    spatial: DMTestResult
    daily: DMTestResult


class DailyMetricsSummary(BaseModel):
    model: str
    split: str
    rmse_j: float
    mae_j: float
    r2: float


class DailyMetricsResponse(BaseModel):
    results: list[DailyMetricsSummary]


class SplitInfoResponse(BaseModel):
    total_samples: int
    train_samples: int
    test_samples: int
    test_fraction: float
    split_method: str
    random_seed: int
    target_col: str


class LiveFeatureError(RuntimeError):
    def __init__(self, detail: str, status_code: int = 502) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def _parse_cors_origins() -> list[str]:
    defaults = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://0.0.0.0:5173",
    ]
    configured = os.getenv("CORS_ORIGINS", "")
    raw_origins = [item.strip().rstrip("/") for item in configured.split(",") if item.strip()]
    origins: list[str] = []
    for origin in raw_origins:
        if origin.startswith(("http://", "https://")):
            origins.append(origin)
            continue
        origins.append(f"https://{origin}")
        if origin.startswith(("localhost", "127.0.0.1", "0.0.0.0")):
            origins.append(f"http://{origin}")
    return origins or defaults


def _parse_overpass_api_urls() -> tuple[str, ...]:
    configured = os.getenv("OVERPASS_API_URLS", "")
    raw_urls = [item.strip().rstrip("/") for item in configured.split(",") if item.strip()]
    if not raw_urls:
        return DEFAULT_OVERPASS_API_URLS

    urls: list[str] = []
    for url in raw_urls:
        if url.startswith(("http://", "https://")):
            urls.append(url)
        else:
            urls.append(f"https://{url}")
    return tuple(urls)


OVERPASS_API_URLS = _parse_overpass_api_urls()


app = FastAPI(title="FI-AdaBoost API", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_origin_regex=os.getenv("CORS_ORIGIN_REGEX", DEFAULT_CORS_ORIGIN_REGEX),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/results/images", StaticFiles(directory=str(RESULTS_DIR)), name="results_images")


def _rolling_nasa_window() -> tuple[date, date]:
    today_local = datetime.now(ZoneInfo(LOCAL_TZ)).date()
    end_date = today_local - timedelta(days=max(NASA_LAG_DAYS, 0))
    start_date = end_date - timedelta(days=max(NASA_LOOKBACK_DAYS - 1, 0))
    return start_date, end_date


def _pvlib_features(
    lat: float,
    lon: float,
    start_date: date = date(2024, 1, 1),
    end_date: date = date(2024, 12, 31),
) -> dict[str, float]:
    key = (round(lat, 2), round(lon, 2), start_date.isoformat(), end_date.isoformat())
    if key in _pvlib_cache:
        return _pvlib_cache[key]

    location = pvlib.location.Location(lat, lon, altitude=30, tz=LOCAL_TZ)
    times = pd.date_range(
        f"{start_date.isoformat()} 00:00",
        f"{end_date.isoformat()} 23:00",
        freq="h",
        tz=LOCAL_TZ,
    )
    clear_sky = location.get_clearsky(times, model="ineichen")
    ghi_clear = float((clear_sky["ghi"].resample("D").sum() / 1000).mean())
    sunshine_hours = float((clear_sky["ghi"] > 120).resample("D").sum().mean())

    result = {
        "ghi_clear_annual": ghi_clear,
        "sunshine_hours": sunshine_hours,
    }
    _pvlib_cache[key] = result
    return result


def fetch_nasa_live(lat: float, lng: float) -> dict[str, float | str]:
    start_date, end_date = _rolling_nasa_window()
    key = (
        round(lat, 2),
        round(lng, 2),
        start_date.isoformat(),
        end_date.isoformat(),
    )
    if key in _nasa_cache:
        return _nasa_cache[key]

    try:
        response = requests.get(
            NASA_POWER_URL,
            params={
                "parameters": "ALLSKY_SFC_SW_DWN,ALLSKY_KT,T2M,RH2M",
                "community": "RE",
                "latitude": lat,
                "longitude": lng,
                "start": start_date.strftime("%Y%m%d"),
                "end": end_date.strftime("%Y%m%d"),
                "format": "JSON",
            },
            headers={"User-Agent": REQUEST_USER_AGENT},
            timeout=NASA_TIMEOUT_SEC,
        )
        response.raise_for_status()
        data = response.json()["properties"]["parameter"]
    except Exception as exc:
        raise LiveFeatureError(
            f"NASA POWER live meteorology failed for {lat:.6f}, {lng:.6f}: {exc}",
            status_code=502,
        ) from exc

    def _mean(parameter: str) -> float:
        values = [value for value in data.get(parameter, {}).values() if value not in (None, -999.0)]
        if not values:
            raise LiveFeatureError(
                f"NASA POWER returned no usable {parameter} values for {lat:.6f}, {lng:.6f}.",
                status_code=502,
            )
        return float(np.mean(values))

    result = {
        "ghi_kwh": _mean("ALLSKY_SFC_SW_DWN"),
        "kt": _mean("ALLSKY_KT"),
        "temp_c": _mean("T2M"),
        "humidity_pct": _mean("RH2M"),
        "window_start": start_date.isoformat(),
        "window_end": end_date.isoformat(),
    }
    _nasa_cache[key] = result
    return result


def _overpass_buildings_for_radius(
    lat: float,
    lng: float,
    radius_m: int,
) -> list[dict[str, Any]]:
    query = (
        f"[out:json][timeout:{OVERPASS_TIMEOUT_SEC}];"
        f'(way["building"](around:{radius_m},{lat},{lng}););'
        "out body geom;"
    )
    for endpoint_url in OVERPASS_API_URLS:
        try:
            response = requests.post(
                endpoint_url,
                data={"data": query},
                headers={"User-Agent": REQUEST_USER_AGENT},
                timeout=OVERPASS_TIMEOUT_SEC + 5,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("unexpected response payload")
            elements = payload.get("elements", [])
            if not isinstance(elements, list):
                raise ValueError("missing elements array")
            return [element for element in elements if len(element.get("geometry", [])) >= 3]
        except Exception as exc:
            logger.warning(
                "Overpass endpoint failed for lat=%0.6f lng=%0.6f radius=%sm endpoint=%s: %s",
                lat,
                lng,
                radius_m,
                endpoint_url,
                exc,
            )

    raise LiveFeatureError(
        "Live rooftop lookup is temporarily unavailable because all configured Overpass "
        f"endpoints failed for {lat:.6f}, {lng:.6f}. Please try again shortly.",
        status_code=502,
    )


def _overpass_buildings(lat: float, lng: float) -> list[dict[str, Any]]:
    for radius_m in OVERPASS_RADIUS_SEQUENCE_METERS:
        buildings = _overpass_buildings_for_radius(lat, lng, radius_m)
        if buildings:
            return buildings

    final_radius_m = OVERPASS_RADIUS_SEQUENCE_METERS[-1]
    raise LiveFeatureError(
        "No mapped building footprint was found near "
        f"{lat:.6f}, {lng:.6f} after live Overpass searches up to {final_radius_m} m. "
        "The selected coordinate may be outside the study area or in an unmapped/sparse rooftop area.",
        status_code=404,
    )


def _centroid(nodes: list[dict[str, float]]) -> tuple[float, float]:
    return (
        sum(node["lat"] for node in nodes) / len(nodes),
        sum(node["lon"] for node in nodes) / len(nodes),
    )


def _dist_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = (lat2 - lat1) * 111_320.0
    dlon = (lon2 - lon1) * 111_320.0 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.sqrt(dlat ** 2 + dlon ** 2)


def _area_m2(nodes: list[dict[str, float]]) -> float:
    if len(nodes) < 3:
        return 0.0
    lat_center = sum(node["lat"] for node in nodes) / len(nodes)
    scale_x = 111_320.0 * math.cos(math.radians(lat_center))
    scale_y = 111_320.0
    xs = [node["lon"] * scale_x for node in nodes]
    ys = [node["lat"] * scale_y for node in nodes]
    total = 0.0
    for idx in range(len(xs)):
        next_idx = (idx + 1) % len(xs)
        total += xs[idx] * ys[next_idx] - xs[next_idx] * ys[idx]
    return abs(total) / 2.0


def _orientation_score(azimuth_deg: float) -> float:
    return float((math.cos(math.radians(azimuth_deg - 180.0)) + 1.0) / 2.0)


def compute_osm_features(lat: float, lng: float, sei_normalizer: float) -> dict[str, float]:
    elements = _overpass_buildings(lat, lng)

    centroids = [_centroid(element["geometry"]) for element in elements]
    nearby_counts: list[int] = []
    for idx, (center_lat, center_lng) in enumerate(centroids):
        count = 0
        for other_idx, (other_lat, other_lng) in enumerate(centroids):
            if idx == other_idx:
                continue
            if _dist_m(center_lat, center_lng, other_lat, other_lng) <= 50.0:
                count += 1
        nearby_counts.append(count)

    nearest_idx = min(
        range(len(elements)),
        key=lambda idx: _dist_m(lat, lng, centroids[idx][0], centroids[idx][1]),
    )
    nearest = elements[nearest_idx]
    nodes = nearest["geometry"]

    lats = [node["lat"] for node in nodes]
    lons = [node["lon"] for node in nodes]
    bbox_width = max(lons) - min(lons)
    bbox_height = max(lats) - min(lats)
    azimuth_deg = float(math.degrees(math.atan2(bbox_width, bbox_height)))
    orientation = _orientation_score(azimuth_deg)
    area = _area_m2(nodes)

    max_count = max(max(nearby_counts), 1)
    shading = float(np.clip(0.3 * nearby_counts[nearest_idx] / max_count, 0.0, 1.0))
    sei_raw = orientation * area * (1.0 - shading) * TILT_FACTOR
    sei_norm = float(np.clip(sei_raw / max(sei_normalizer, 1e-9), 0.0, 1.0))

    return {
        "orientation_score": orientation,
        "shading_factor": shading,
        "SEI_norm": sei_norm,
        "rooftop_area_sq_m": area,
        "azimuth_deg": azimuth_deg,
    }


def _orientation_label(azimuth_deg: float) -> str:
    azimuth = azimuth_deg % 360.0
    if 135.0 <= azimuth <= 225.0:
        return "South"
    if 45.0 <= azimuth < 135.0:
        return "East"
    if 225.0 < azimuth <= 315.0:
        return "West"
    return "North"


def _build_feature_frame(
    lat: float,
    lng: float,
    osm: dict[str, float],
    nasa: dict[str, float | str],
    pvlib_features: dict[str, float],
) -> pd.DataFrame:
    clear_sky_ratio = float(
        np.clip(
            float(nasa["ghi_kwh"]) / max(pvlib_features["ghi_clear_annual"], 1e-9),
            0.0,
            1.5,
        )
    )
    row = {
        "lat": float(lat),
        "lon": float(lng),
        "azimuth": float(osm["azimuth_deg"]),
        "orientation_score": float(osm["orientation_score"]),
        "shading_factor": float(osm["shading_factor"]),
        "SEI_norm": float(osm["SEI_norm"]),
        "clear_sky_ratio": clear_sky_ratio,
        "sunshine_hours": float(pvlib_features["sunshine_hours"]),
    }
    return pd.DataFrame([row], columns=SHARED_FEATURES)


def _build_response(
    ghi_j: float,
    osm: dict[str, float],
    nasa: dict[str, float | str],
    pvlib_features: dict[str, float],
) -> PredictResponse:
    ghi_kwh = float(ghi_j) / KWH_TO_J
    rooftop_area = float(osm["rooftop_area_sq_m"])
    theoretical_energy_kwh_day = ghi_kwh * rooftop_area
    kt = float(np.clip(float(nasa["kt"]), 0.0, 1.0))
    clear_sky_ratio = float(
        np.clip(
            float(nasa["ghi_kwh"]) / max(pvlib_features["ghi_clear_annual"], 1e-9),
            0.0,
            1.5,
        )
    )
    return PredictResponse(
        solarPotential=theoretical_energy_kwh_day,
        predictedIrradiance=ghi_kwh,
        rooftopArea=rooftop_area,
        solarExposureIndex=float(osm["SEI_norm"]),
        orientation=_orientation_label(float(osm["azimuth_deg"])),
        azimuth=float(osm["azimuth_deg"]),
        sunshineHours=float(pvlib_features["sunshine_hours"]),
        cloudCover=float(np.clip((1.0 - kt) * 100.0, 5.0, 95.0)),
        temperature=float(nasa["temp_c"]),
        humidity=float(nasa["humidity_pct"]),
        clearSkyRatio=clear_sky_ratio,
    )


class ModelContext:
    def __init__(self) -> None:
        self._core_lock = threading.Lock()
        self._location_lock = threading.Lock()
        self._analytics_lock = threading.Lock()

        self.fi_model: FIAdaBoostRegressor | None = None
        self.baseline_model: BaselineAdaBoost | None = None
        self.performance_metrics: dict[str, dict[str, float]] = {}
        self.feature_importance_ranking: list[dict[str, str | float | int]] = []
        self.cv_fold_metrics: list[CVFoldMetrics] = []
        self.average_cv_metrics: dict[str, dict[str, float]] = {}

        self.location_points: np.ndarray | None = None
        self._kd_tree: cKDTree | None = None
        self.lat_range = DEFAULT_LAT_RANGE
        self.lon_range = DEFAULT_LON_RANGE
        self.sei_normalizer = DEFAULT_SEI_NORMALIZER
        self.location_index_error: str | None = None

        self.training_df: pd.DataFrame | None = None
        self.training_analytics: TrainingAnalyticsResponse | None = None
        self.training_analytics_error: str | None = None

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _load_model_file(self, model_path: Path) -> object | None:
        register_pickle_compat()
        try:
            return joblib.load(model_path)
        except (FileNotFoundError, AttributeError, ImportError, ModuleNotFoundError, ValueError):
            return None

    def _model_supports_feature_count(self, model: object, expected_count: int) -> bool:
        wrapped = getattr(model, "_model", model)
        n_features = getattr(wrapped, "n_features_in_", None)
        if n_features is not None:
            return int(n_features) == expected_count
        estimators = getattr(model, "estimators_", None)
        if estimators:
            first = getattr(estimators[0], "n_features_in_", None)
            if first is not None:
                return int(first) == expected_count
        return True

    def _load_location_index(self) -> None:
        if not INTEGRATED_DATA_FILE.exists():
            self.location_index_error = (
                f"Missing integrated dataset: {INTEGRATED_DATA_FILE}"
            )
            return

        df = pd.read_csv(INTEGRATED_DATA_FILE, usecols=["lat", "lon"])
        df = df.dropna(subset=["lat", "lon"]).astype(float).reset_index(drop=True)
        if df.empty:
            self.location_index_error = "Integrated dataset has no valid lat/lon rows."
            return

        self.location_points = df[["lat", "lon"]].to_numpy(dtype=float)
        self._kd_tree = cKDTree(self.location_points)
        self.lat_range = (
            float(df["lat"].min()),
            float(df["lat"].max()),
        )
        self.lon_range = (
            float(df["lon"].min()),
            float(df["lon"].max()),
        )
        self.location_index_error = None

    def _load_training_dataset(self) -> pd.DataFrame:
        if not INTEGRATED_DATA_FILE.exists():
            raise FileNotFoundError(
                f"Missing integrated dataset: {INTEGRATED_DATA_FILE}"
            )

        df = pd.read_csv(INTEGRATED_DATA_FILE)
        if "effective_GHI_J" not in df.columns:
            if TARGET_COL in df.columns:
                df["effective_GHI_J"] = df[TARGET_COL]
            else:
                raise ValueError(
                    "integrated_dataset.csv is missing 'effective_GHI_J'."
                )

        if "clear_sky_ratio" not in df.columns or "sunshine_hours" not in df.columns:
            pvlib_features = df.apply(
                lambda row: pd.Series(
                    _pvlib_features(
                        float(row["lat"]),
                        float(row["lon"]),
                        start_date=date(2024, 1, 1),
                        end_date=date(2024, 12, 31),
                    )
                ),
                axis=1,
            )
            df["sunshine_hours"] = pvlib_features["sunshine_hours"]
            df["clear_sky_ratio"] = (
                df["GHI_mean_2024"] / pvlib_features["ghi_clear_annual"]
            ).clip(0.0, 1.5)

        required = SHARED_FEATURES + ["effective_GHI_J"]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"Dataset is missing columns: {missing}")

        df = df.dropna(subset=required).reset_index(drop=True)
        if df.empty:
            raise ValueError("Integrated dataset became empty after dropping NaNs.")
        return df

    def _train_model(self, model_cls, model_file: Path, feature_cols: list[str] = SHARED_FEATURES) -> object:
        if self.training_df is None:
            self.training_df = self._load_training_dataset()

        X = self.training_df[feature_cols].astype(float)
        y = self.training_df["effective_GHI_J"].astype(float)
        model = model_cls()
        model.fit(X, y)
        joblib.dump(model, model_file)
        return model

    def _load_or_train_model(self, model_cls, model_file: Path, feature_cols: list[str] = SHARED_FEATURES) -> object:
        expected_count = len(feature_cols)
        if model_file.exists():
            loaded = self._load_model_file(model_file)
            if loaded is not None and self._model_supports_feature_count(loaded, expected_count):
                return loaded

        return self._train_model(model_cls, model_file, feature_cols)

    def _load_metrics_from_results(self) -> bool:
        if not METRICS_FILE.exists():
            return False

        try:
            df = pd.read_csv(METRICS_FILE)
            base_row = df[df["model"].str.contains("Baseline", case=False, na=False)]
            fi_row = df[df["model"].str.contains("FI-AdaBoost", case=False, na=False)]
            if base_row.empty or fi_row.empty:
                return False

            baseline = base_row.iloc[0]
            fi = fi_row.iloc[0]
            self.performance_metrics = {
                "baseline": {
                    "rmse": float(baseline["RMSE_J"]) / KWH_TO_J,
                    "mae": float(baseline["MAE_J"]) / KWH_TO_J,
                    "r2": float(baseline["R2"]),
                },
                "fiAdaBoost": {
                    "rmse": float(fi["RMSE_J"]) / KWH_TO_J,
                    "mae": float(fi["MAE_J"]) / KWH_TO_J,
                    "r2": float(fi["R2"]),
                },
            }
            return True
        except Exception:
            return False

    def _load_cv_metrics(self) -> None:
        self.cv_fold_metrics = []
        self.average_cv_metrics = {}

        if CV_METRICS_FILE.exists():
            try:
                df = pd.read_csv(CV_METRICS_FILE)
                self.cv_fold_metrics = [
                    CVFoldMetrics(
                        fold=int(row["fold"]),
                        baseline_rmse=float(row["ada_RMSE_J"]),
                        baseline_mae=float(row["ada_MAE_J"]),
                        baseline_r2=float(row["ada_R2"]),
                        fi_rmse=float(row["fi_RMSE_J"]),
                        fi_mae=float(row["fi_MAE_J"]),
                        fi_r2=float(row["fi_R2"]),
                    )
                    for _, row in df.iterrows()
                ]
            except Exception:
                self.cv_fold_metrics = []

        if not self.cv_fold_metrics:
            return

        self.average_cv_metrics = {
            "baseline": {
                "rmse": float(np.mean([item.baseline_rmse for item in self.cv_fold_metrics])),
                "mae": float(np.mean([item.baseline_mae for item in self.cv_fold_metrics])),
                "r2": float(np.mean([item.baseline_r2 for item in self.cv_fold_metrics])),
            },
            "fiAdaBoost": {
                "rmse": float(np.mean([item.fi_rmse for item in self.cv_fold_metrics])),
                "mae": float(np.mean([item.fi_mae for item in self.cv_fold_metrics])),
                "r2": float(np.mean([item.fi_r2 for item in self.cv_fold_metrics])),
            },
        }

    def _compute_feature_importance_ranking(self) -> list[dict[str, str | float | int]]:
        if self.fi_model is None or self.baseline_model is None:
            return []

        fi_importance = np.zeros(len(SHARED_FEATURES), dtype=float)
        if getattr(self.fi_model, "estimators_", None):
            weights = np.asarray(self.fi_model.estimator_weights_, dtype=float)
            if weights.size > 0 and weights.sum() > 0:
                weights = weights / weights.sum()
                for idx, estimator in enumerate(self.fi_model.estimators_):
                    if hasattr(estimator, "feature_importances_"):
                        fi_importance += weights[idx] * np.asarray(
                            estimator.feature_importances_,
                            dtype=float,
                        )
        elif getattr(self.fi_model, "feature_importances_", None) is not None:
            fi_importance = np.asarray(self.fi_model.feature_importances_, dtype=float)

        baseline_imp_raw = np.asarray(
            getattr(self.baseline_model, "feature_importances_", np.zeros(len(BASELINE_FEATURES))),
            dtype=float,
        )
        baseline_imp_map = {
            feat: float(baseline_imp_raw[i]) if i < len(baseline_imp_raw) else 0.0
            for i, feat in enumerate(BASELINE_FEATURES)
        }

        ranking: list[dict[str, str | float | int]] = []
        for idx, feature in enumerate(SHARED_FEATURES):
            fi_value = float(fi_importance[idx] if idx < len(fi_importance) else 0.0)
            base_value = baseline_imp_map.get(feature, 0.0)
            ranking.append(
                {
                    "feature": feature,
                    "fiAdaBoostImportance": fi_value,
                    "baselineImportance": base_value,
                    "importanceGain": fi_value - base_value,
                    "rank": 0,
                }
            )

        ranking.sort(key=lambda item: float(item["fiAdaBoostImportance"]), reverse=True)
        for rank, item in enumerate(ranking, start=1):
            item["rank"] = rank
        return ranking

    def ensure_core_loaded(self) -> None:
        if self.fi_model is not None and self.baseline_model is not None:
            return

        with self._core_lock:
            if self.fi_model is not None and self.baseline_model is not None:
                return

            self.fi_model = self._load_or_train_model(FIAdaBoostRegressor, FI_MODEL_FILE, SHARED_FEATURES)
            self.baseline_model = self._load_or_train_model(
                BaselineAdaBoost,
                BASELINE_MODEL_FILE,
                BASELINE_FEATURES,
            )

            self._load_metrics_from_results()
            self._load_cv_metrics()
            self.feature_importance_ranking = self._compute_feature_importance_ranking()

    def ensure_location_index_loaded(self) -> None:
        if self._kd_tree is not None or self.location_points is not None:
            return

        with self._location_lock:
            if self._kd_tree is not None or self.location_points is not None:
                return
            self._load_location_index()

    def _evaluate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    def _compute_global_metrics_from_dataset(self) -> None:
        if self.training_df is None or self.fi_model is None or self.baseline_model is None:
            return

        idx = np.arange(len(self.training_df))
        _, test_idx = train_test_split(
            idx,
            test_size=0.20,
            random_state=RANDOM_SEED,
            shuffle=True,
        )
        test_df = self.training_df.iloc[test_idx].reset_index(drop=True)
        X_test_fi = test_df[SHARED_FEATURES].astype(float)
        X_test_baseline = test_df[BASELINE_FEATURES].astype(float)
        y_test = test_df["effective_GHI_J"].astype(float).to_numpy() / KWH_TO_J
        baseline_pred = np.asarray(self.baseline_model.predict(X_test_baseline), dtype=float) / KWH_TO_J
        fi_pred = np.asarray(self.fi_model.predict(X_test_fi), dtype=float) / KWH_TO_J

        self.performance_metrics = {
            "baseline": self._evaluate_metrics(y_test, baseline_pred),
            "fiAdaBoost": self._evaluate_metrics(y_test, fi_pred),
        }

    def _build_error_distribution(
        self,
        baseline_residuals: np.ndarray,
        fi_residuals: np.ndarray,
        bins: int = 18,
    ) -> list[ErrorDistributionBin]:
        combined = np.concatenate([baseline_residuals, fi_residuals])
        if combined.size == 0:
            return []

        mn = float(np.min(combined))
        mx = float(np.max(combined))
        if np.isclose(mn, mx):
            mn -= 1.0
            mx += 1.0

        edges = np.linspace(mn, mx, bins + 1)
        baseline_counts, _ = np.histogram(baseline_residuals, bins=edges)
        fi_counts, _ = np.histogram(fi_residuals, bins=edges)
        span = mx - mn
        fmt = ".3f" if span < 1 else ".1f"
        return [
            ErrorDistributionBin(
                bucket=f"{edges[idx]:{fmt}} to {edges[idx + 1]:{fmt}}",
                baseline=int(baseline_counts[idx]),
                fiAdaBoost=int(fi_counts[idx]),
            )
            for idx in range(len(edges) - 1)
        ]

    def _compute_training_analytics(self) -> None:
        if self.training_df is None or self.fi_model is None or self.baseline_model is None:
            return

        idx = np.arange(len(self.training_df))
        _, test_idx = train_test_split(
            idx,
            test_size=0.20,
            random_state=RANDOM_SEED,
            shuffle=True,
        )
        test_df = self.training_df.iloc[test_idx].reset_index(drop=True)
        X_test_fi = test_df[SHARED_FEATURES].astype(float)
        X_test_baseline = test_df[BASELINE_FEATURES].astype(float)
        y_kwh = test_df["effective_GHI_J"].astype(float).to_numpy() / KWH_TO_J
        baseline_pred = np.asarray(self.baseline_model.predict(X_test_baseline), dtype=float) / KWH_TO_J
        fi_pred = np.asarray(self.fi_model.predict(X_test_fi), dtype=float) / KWH_TO_J

        if not self.performance_metrics:
            self._compute_global_metrics_from_dataset()

        sample_size = min(220, len(test_df))
        sample_idx = np.unique(np.linspace(0, len(test_df) - 1, num=sample_size, dtype=int))
        labels = [str(index) for index in sample_idx]

        y_sample = y_kwh[sample_idx]
        baseline_sample = baseline_pred[sample_idx]
        fi_sample = fi_pred[sample_idx]
        combined = np.concatenate([y_sample, baseline_sample, fi_sample])

        predicted_vs_actual = PredictedVsActualData(
            baseline=[
                PredictedActualPoint(actual=float(actual), predicted=float(pred), date=label)
                for actual, pred, label in zip(y_sample, baseline_sample, labels)
            ],
            fiAdaBoost=[
                PredictedActualPoint(actual=float(actual), predicted=float(pred), date=label)
                for actual, pred, label in zip(y_sample, fi_sample, labels)
            ],
            domainMin=float(np.min(combined)),
            domainMax=float(np.max(combined)),
        )

        baseline_residuals = baseline_pred - y_kwh
        fi_residuals = fi_pred - y_kwh
        baseline_metrics = self.performance_metrics.get(
            "baseline",
            {"rmse": 0.0, "mae": 0.0, "r2": 0.0},
        )
        fi_metrics = self.performance_metrics.get(
            "fiAdaBoost",
            {"rmse": 0.0, "mae": 0.0, "r2": 0.0},
        )

        self.training_analytics = TrainingAnalyticsResponse(
            performanceMetricsComparison={
                "baseline": baseline_metrics,
                "fiAdaBoost": fi_metrics,
                "rmseImprovementPct": float(
                    (baseline_metrics["rmse"] - fi_metrics["rmse"])
                    / baseline_metrics["rmse"]
                    * 100
                    if baseline_metrics["rmse"] != 0
                    else 0.0
                ),
                "maeImprovementPct": float(
                    (baseline_metrics["mae"] - fi_metrics["mae"])
                    / baseline_metrics["mae"]
                    * 100
                    if baseline_metrics["mae"] != 0
                    else 0.0
                ),
                "r2ImprovementPct": float(
                    (fi_metrics["r2"] - baseline_metrics["r2"]) * 100
                ),
            },
            predictedVsActual=predicted_vs_actual,
            errorDistribution=self._build_error_distribution(
                baseline_residuals,
                fi_residuals,
            ),
            residualSummary={
                "baselineStd": float(np.std(baseline_residuals, ddof=1))
                if len(baseline_residuals) > 1
                else 0.0,
                "fiStd": float(np.std(fi_residuals, ddof=1))
                if len(fi_residuals) > 1
                else 0.0,
                "baselineMae": float(np.mean(np.abs(baseline_residuals))),
                "fiMae": float(np.mean(np.abs(fi_residuals))),
            },
        )

    def ensure_training_analytics_loaded(self) -> None:
        if self.training_analytics is not None:
            return

        with self._analytics_lock:
            if self.training_analytics is not None:
                return

            try:
                self.ensure_core_loaded()
                if self.training_df is None:
                    self.training_df = self._load_training_dataset()
                if not self.performance_metrics:
                    self._compute_global_metrics_from_dataset()
                self._compute_training_analytics()
                self.training_analytics_error = None
            except Exception as exc:
                self.training_analytics_error = str(exc)
                self.training_analytics = None

    def confidence_level(self, lat: float, lng: float, diff_kwh: float) -> float:
        self.ensure_location_index_loaded()

        agreement_score = float(np.clip(100.0 - abs(diff_kwh) * 20.0, 35.0, 99.0))
        if self._kd_tree is not None and self.location_points is not None:
            distance_deg, _ = self._kd_tree.query([lat, lng])
            distance_km = float(distance_deg) * 111.0
            coverage_score = float(np.clip(100.0 - distance_km * 2.5, 35.0, 99.0))
            return float(np.clip(0.7 * coverage_score + 0.3 * agreement_score, 35.0, 99.0))

        in_bounds = (
            self.lat_range[0] <= lat <= self.lat_range[1]
            and self.lon_range[0] <= lng <= self.lon_range[1]
        )
        coverage_score = 82.0 if in_bounds else 55.0
        return float(np.clip(0.7 * coverage_score + 0.3 * agreement_score, 35.0, 99.0))


ctx = ModelContext()


def _background_warm() -> None:
    try:
        ctx.ensure_core_loaded()
    except Exception as exc:
        print(f"Background model warmup failed: {exc}", flush=True)


threading.Thread(target=_background_warm, daemon=True).start()


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "fiadaboost-backend",
        "version": "2.1",
        "status": "running",
        "docs": "/docs",
        "predict": 'POST /predict  body: {"lat": 7.0731, "lng": 125.6128}',
    }


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        ctx.ensure_core_loaded()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Models are not ready: {exc}") from exc

    if ctx.fi_model is None or ctx.baseline_model is None:
        raise HTTPException(status_code=503, detail="Models are still warming up.")

    return {
        "status": "ok",
        "modelsLoaded": True,
        "resultsLoaded": bool(ctx.performance_metrics),
        "cvMetricsLoaded": bool(ctx.cv_fold_metrics),
    }


@app.get("/predict")
def predict_help() -> dict[str, str]:
    return {
        "detail": "Use POST /predict with JSON body containing lat and lng.",
        "example": '{"lat": 7.0731, "lng": 125.6128}',
    }


def _live_feature_bundle(lat: float, lng: float) -> tuple[dict[str, float], dict[str, float | str], dict[str, float]]:
    nasa = fetch_nasa_live(lat, lng)
    start_date = date.fromisoformat(str(nasa["window_start"]))
    end_date = date.fromisoformat(str(nasa["window_end"]))
    pvlib_features = _pvlib_features(lat, lng, start_date=start_date, end_date=end_date)
    osm = compute_osm_features(lat, lng, ctx.sei_normalizer)
    return osm, nasa, pvlib_features


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        ctx.ensure_core_loaded()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Models are not ready: {exc}") from exc

    if ctx.fi_model is None:
        raise HTTPException(status_code=503, detail="FI-AdaBoost model is not available.")

    try:
        osm, nasa, pvlib_features = _live_feature_bundle(payload.lat, payload.lng)
        X = _build_feature_frame(payload.lat, payload.lng, osm, nasa, pvlib_features)
        ghi_j = float(ctx.fi_model.predict(X)[0])
        return _build_response(ghi_j, osm, nasa, pvlib_features)
    except LiveFeatureError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@app.post("/compare", response_model=CompareResponse)
def compare_models(payload: PredictRequest) -> CompareResponse:
    try:
        ctx.ensure_core_loaded()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Models are not ready: {exc}") from exc

    if ctx.fi_model is None or ctx.baseline_model is None:
        raise HTTPException(status_code=503, detail="Comparison models are not available.")

    try:
        osm, nasa, pvlib_features = _live_feature_bundle(payload.lat, payload.lng)
        X_fi = _build_feature_frame(payload.lat, payload.lng, osm, nasa, pvlib_features)
        X_baseline = X_fi[BASELINE_FEATURES]

        baseline_ghi_j = float(ctx.baseline_model.predict(X_baseline)[0])
        fi_ghi_j = float(ctx.fi_model.predict(X_fi)[0])

        baseline_result = _build_response(baseline_ghi_j, osm, nasa, pvlib_features)
        fi_result = _build_response(fi_ghi_j, osm, nasa, pvlib_features)

        diff_kwh = fi_result.solarPotential - baseline_result.solarPotential
        irradiance_diff_kwh_m2 = (
            fi_result.predictedIrradiance - baseline_result.predictedIrradiance
        )
        improvement_pct = (
            diff_kwh / baseline_result.solarPotential * 100.0
            if baseline_result.solarPotential != 0
            else 0.0
        )
        confidence = ctx.confidence_level(payload.lat, payload.lng, irradiance_diff_kwh_m2)

        baseline_metrics = ctx.performance_metrics.get(
            "baseline",
            {"rmse": 0.0, "mae": 0.0, "r2": 0.0},
        )
        fi_metrics = ctx.performance_metrics.get(
            "fiAdaBoost",
            {"rmse": 0.0, "mae": 0.0, "r2": 0.0},
        )

        return CompareResponse(
            baseline=baseline_result,
            fiAdaBoost=fi_result,
            improvement={
                "solarPotentialDiff": float(diff_kwh),
                "solarPotentialImprovementPct": float(improvement_pct),
                "accuracyGain": float(abs(diff_kwh)),
                "confidenceLevel": confidence,
            },
            performanceMetricsComparison={
                "baseline": baseline_metrics,
                "fiAdaBoost": fi_metrics,
                "rmseImprovementPct": float(
                    (baseline_metrics["rmse"] - fi_metrics["rmse"])
                    / baseline_metrics["rmse"]
                    * 100
                    if baseline_metrics["rmse"] != 0
                    else 0.0
                ),
                "maeImprovementPct": float(
                    (baseline_metrics["mae"] - fi_metrics["mae"])
                    / baseline_metrics["mae"]
                    * 100
                    if baseline_metrics["mae"] != 0
                    else 0.0
                ),
                "r2ImprovementPct": float(
                    (fi_metrics["r2"] - baseline_metrics["r2"]) * 100
                ),
            },
            featureImportanceRanking=ctx.feature_importance_ranking,
        )
    except LiveFeatureError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@app.get("/cv-metrics", response_model=CVMetricsResponse)
def get_cv_metrics() -> CVMetricsResponse:
    try:
        ctx.ensure_core_loaded()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Models are not ready: {exc}") from exc

    if not ctx.cv_fold_metrics:
        raise HTTPException(status_code=503, detail="CV metrics are not available.")

    return CVMetricsResponse(
        cv_fold_metrics=ctx.cv_fold_metrics,
        average_metrics=ctx.average_cv_metrics,
    )


@app.get("/training-analytics", response_model=TrainingAnalyticsResponse)
def get_training_analytics() -> TrainingAnalyticsResponse:
    ctx.ensure_training_analytics_loaded()
    if ctx.training_analytics is None:
        detail = ctx.training_analytics_error or "Training analytics are not available."
        raise HTTPException(status_code=503, detail=detail)
    return ctx.training_analytics


@app.get("/cv-metrics/spatial", response_model=SpatialCVResponse)
def get_spatial_cv_metrics() -> SpatialCVResponse:
    if not SPATIAL_CV_METRICS_FILE.exists():
        raise HTTPException(status_code=503, detail="Spatial CV metrics file is not available.")
    try:
        df = pd.read_csv(SPATIAL_CV_METRICS_FILE)
        folds = [
            SpatialCVFold(
                fold=int(row["fold"]),
                baseline_train_rmse=float(row["ada_train_RMSE_J"]),
                baseline_val_rmse=float(row["ada_val_RMSE_J"]),
                baseline_train_r2=float(row["ada_train_R2"]),
                baseline_val_r2=float(row["ada_val_R2"]),
                fi_train_rmse=float(row["fi_train_RMSE_J"]),
                fi_val_rmse=float(row["fi_val_RMSE_J"]),
                fi_train_r2=float(row["fi_train_R2"]),
                fi_val_r2=float(row["fi_val_R2"]),
            )
            for _, row in df.iterrows()
        ]
        average = {
            "baseline_val_rmse": float(np.mean([f.baseline_val_rmse for f in folds])),
            "baseline_val_r2": float(np.mean([f.baseline_val_r2 for f in folds])),
            "fi_val_rmse": float(np.mean([f.fi_val_rmse for f in folds])),
            "fi_val_r2": float(np.mean([f.fi_val_r2 for f in folds])),
        }
        return SpatialCVResponse(folds=folds, average=average)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to load spatial CV metrics: {exc}") from exc


@app.get("/dm-test", response_model=DMTestResponse)
def get_dm_test() -> DMTestResponse:
    for path in (DM_SPATIAL_FILE, DM_DAILY_FILE):
        if not path.exists():
            raise HTTPException(status_code=503, detail=f"DM test file is not available: {path.name}")
    try:
        def _read_dm(path: Path) -> DMTestResult:
            row = pd.read_csv(path).iloc[0]
            return DMTestResult(
                dm_statistic=float(row["DM_statistic"]),
                p_value=float(row["p_value"]),
                significant=bool(row["significant_at_0.05"]),
                interpretation=str(row["interpretation"]),
                n_samples=int(row["n_test_samples"]),
                mean_loss_diff_j2=float(row["mean_loss_diff_J2"]),
            )
        return DMTestResponse(spatial=_read_dm(DM_SPATIAL_FILE), daily=_read_dm(DM_DAILY_FILE))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to load DM test results: {exc}") from exc


@app.get("/daily-metrics", response_model=DailyMetricsResponse)
def get_daily_metrics() -> DailyMetricsResponse:
    if not DAILY_METRICS_FILE.exists():
        raise HTTPException(status_code=503, detail="Daily metrics file is not available.")
    try:
        df = pd.read_csv(DAILY_METRICS_FILE)
        results = [
            DailyMetricsSummary(
                model=str(row["model"]),
                split=str(row["split"]),
                rmse_j=float(row["RMSE_J"]),
                mae_j=float(row["MAE_J"]),
                r2=float(row["R2"]),
            )
            for _, row in df.iterrows()
        ]
        return DailyMetricsResponse(results=results)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to load daily metrics: {exc}") from exc


@app.get("/split-info", response_model=SplitInfoResponse)
def get_split_info() -> SplitInfoResponse:
    if not SPLIT_INFO_FILE.exists():
        raise HTTPException(status_code=503, detail="Split info file is not available.")
    try:
        row = pd.read_csv(SPLIT_INFO_FILE).iloc[0]
        return SplitInfoResponse(
            total_samples=int(row["total_samples"]),
            train_samples=int(row["train_samples"]),
            test_samples=int(row["test_samples"]),
            test_fraction=float(row["test_fraction"]),
            split_method=str(row["split_method"]),
            random_seed=int(row["random_seed"]),
            target_col=str(row["target_col"]),
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to load split info: {exc}") from exc
