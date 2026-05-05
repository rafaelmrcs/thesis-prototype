"""
model_training.py
────────────────────────────────────────────────────────────────────────────
Solar Irradiance Forecasting — Davao City
AdaBoost Regression (Baseline)  vs  FI-AdaBoost Regression (Proposed)


METHODOLOGY (Two-Phase Pipeline)
──────────────────────────────────────────────────────────────────────────
  PHASE 1: MACHINE LEARNING (fair comparison)
  - Both models predict the SAME target: effective GHI (J/m²/day),
    pvlib POA-adjusted per building (Fix 1 — aligned targets).
  - Both models use the SAME 8 features (Fix 6 — no log transforms):
      lat, lon, azimuth, orientation_score, shading_factor, SEI_norm,
      clear_sky_ratio, sunshine_hours  (Fix 3 — meteo features active)
  - Hyperparameters tuned via Optuna + 5-fold KFold (Fix 4).
  - Statistical significance assessed via Diebold–Mariano test (Fix 5).
  - The ONLY algorithmic difference: standard AdaBoost vs FI-AdaBoost
    (feature-importance-weighted boosting update — the Φᵢ term).


  SECONDARY: DAILY TIME-SERIES PIPELINE (Fix 2)
  - Fetches 365 daily NASA POWER records for Davao City centroid.
  - Adds temporal features (month_sin/cos, season) and meteorological
    features (clear_sky_ratio, sunshine_hours) per day.
  - Chronological 80/20 split; Optuna + TimeSeriesSplit(5) CV.
  - Saves cv_fold_metrics.csv and dm_test_results.csv.


  PHASE 2: SOLAR ENERGY POTENTIAL FORECASTING
  - Converts predicted irradiance into theoretical rooftop solar energy
    potential: SEP = predicted GHI (kWh/m2/day) x OSM rooftop area (m2).
  - Panel efficiency and performance ratio are intentionally excluded because
    electrical PV output is outside this study scope.
────────────────────────────────────────────────────────────────────────────
"""


import os
import sys
import math
import warnings
warnings.filterwarnings("ignore")


import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy import stats
import pvlib


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
RESULTS_DIR   = os.path.join(ROOT_DIR, "results")
MODEL_DIR     = os.path.join(ROOT_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)


BASELINE_MODEL_FILE = os.path.join(MODEL_DIR, "baseline_adaboost.pkl")
FI_MODEL_FILE       = os.path.join(MODEL_DIR, "fi_adaboost.pkl")


# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_SEED   = 42
DAYS_PER_YEAR = 365
KWH_TO_J      = 3_600_000


np.random.seed(RANDOM_SEED)


# ── Feature sets — identical for fair algorithm comparison (Fix 1 + Fix 6) ───
# Both models predict Target_eff_J (pvlib POA-adjusted effective GHI).
# Both use the same 8 features. The only difference is the boosting algorithm.
# Log transforms removed (Fix 6): raw shading_factor and SEI_norm are used.
# clear_sky_ratio and sunshine_hours added from §2.2.2 (Fix 3).
SHARED_FEATURES   = [
    "lat", "lon", "azimuth", "orientation_score",
    "shading_factor", "SEI_norm", "clear_sky_ratio", "sunshine_hours",
]
BASELINE_FEATURES = SHARED_FEATURES
FI_FEATURES       = SHARED_FEATURES
TARGET_COL        = "Target_eff_J"


# Daily time-series feature set (§2.2.1 + §2.2.2 + §2.2.3 aggregates)
DAILY_FEATURES = [
    "month_sin", "month_cos", "season",
    "T2M", "RH2M", "ALLSKY_KT", "clear_sky_ratio", "sunshine_hours",
    "mean_orientation_score", "mean_shading_factor", "mean_SEI_norm",
]
DAILY_TARGET = "GHI_J"


C_ADA = "#E74C3C"
C_FI  = "#27AE60"


# =============================================================================
# pvlib CACHED HELPERS
# =============================================================================
_poa_cache:   dict = {}
_pvlib_cache: dict = {}


def _pvlib_features(lat: float, lon: float) -> dict:
    """
    Annual pvlib clear-sky scalars for a location (cached at 0.2° resolution).
    Returns ghi_clear_annual (kWh/m²/day) and sunshine_hours (hrs/day).
    ~9 unique grid cells cover Davao City — fast after first batch.
    """
    key = (round(lat, 2), round(lon, 2))
    if key in _pvlib_cache:
        return _pvlib_cache[key]

    loc   = pvlib.location.Location(lat, lon, altitude=30, tz="Asia/Manila")
    times = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h", tz="Asia/Manila")
    cs    = loc.get_clearsky(times, model="ineichen")

    ghi_clear_annual = float((cs["ghi"].resample("D").sum() / 1000).mean())
    sunshine_hours   = float((cs["ghi"] > 120).resample("D").sum().mean())

    result = {"ghi_clear_annual": ghi_clear_annual, "sunshine_hours": sunshine_hours}
    _pvlib_cache[key] = result
    return result


def _poa_ratio(lat: float, azimuth_deg: float, tilt_deg: float = 10.0) -> float:
    """Annual-average POA/GHI ratio for a surface at given azimuth and tilt."""
    key = (round(lat, 2), round(azimuth_deg, 1))
    if key in _poa_cache:
        return _poa_cache[key]
    loc       = pvlib.location.Location(lat, 125.6128, altitude=30, tz="Asia/Manila")
    times     = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h", tz="Asia/Manila")
    solar_pos = loc.get_solarposition(times)
    clearsky  = loc.get_clearsky(times, model="ineichen")
    ghi_mean  = float(clearsky["ghi"].mean())
    if ghi_mean == 0:
        _poa_cache[key] = 1.0
        return 1.0
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt_deg,
        surface_azimuth=azimuth_deg,
        solar_zenith=solar_pos["apparent_zenith"],
        solar_azimuth=solar_pos["azimuth"],
        dni=clearsky["dni"],
        ghi=clearsky["ghi"],
        dhi=clearsky["dhi"],
    )
    ratio = float(poa["poa_global"].mean()) / ghi_mean
    _poa_cache[key] = ratio
    return ratio


# =============================================================================
# DATA LOADING — SPATIAL DATASET (20,000 points)
# =============================================================================
def load_dataset() -> pd.DataFrame:
    """
    Load integrated_dataset.csv and compute all features for both models.

    Fix 1: Target_eff_J (pvlib POA-adjusted) used for BOTH models.
    Fix 3: clear_sky_ratio and sunshine_hours computed per spatial point.
    Fix 6: No log transforms — raw shading_factor and SEI_norm used.
    """
    path = os.path.join(PROCESSED_DIR, "integrated_dataset.csv")
    df   = pd.read_csv(path)

    # Target: reuse pre-computed effective_GHI_J if available, else compute.
    if "effective_GHI_J" in df.columns and df["effective_GHI_J"].notna().all():
        df["Target_eff_J"] = df["effective_GHI_J"]
        print("  Using pre-computed effective_GHI_J as target.")
    else:
        print("  Computing pvlib POA ratios for effective GHI target …", flush=True)
        df["Target_eff_J"] = df.apply(
            lambda r: (
                r["GHI_mean_J"]
                * _poa_ratio(r["lat"], r["azimuth"])
                * (1 - 0.03 * r["shading_factor"])
            ),
            axis=1,
        )
        df["effective_GHI_J"] = df["Target_eff_J"]
        df.to_csv(path, index=False)

    # §2.2.2 Meteorological features per spatial point (Fix 3)
    print("  Computing clear_sky_ratio and sunshine_hours …", flush=True)
    pvlib_feats = df.apply(
        lambda r: pd.Series(_pvlib_features(r["lat"], r["lon"])), axis=1
    )
    df["sunshine_hours"]  = pvlib_feats["sunshine_hours"]
    df["clear_sky_ratio"] = (
        df["GHI_mean_2024"] / pvlib_feats["ghi_clear_annual"]
    ).clip(0.0, 1.5)

    return df


def random_split(df: pd.DataFrame, test_size: float = 0.20):
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, random_state=RANDOM_SEED, shuffle=True
    )
    return df.iloc[idx_train].copy(), df.iloc[idx_test].copy()


# =============================================================================
# MODELS
# =============================================================================
class BaselineAdaBoost:
    def __init__(self, n_estimators=158, learning_rate=0.94, max_depth=3,
                 random_state=RANDOM_SEED):
        base = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        self._model = AdaBoostRegressor(
            estimator=base, n_estimators=n_estimators,
            learning_rate=learning_rate, random_state=random_state,
        )

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    @property
    def feature_importances_(self):
        return self._model.feature_importances_


class FIAdaBoostRegressor:
    def __init__(self, n_estimators=141, learning_rate=0.63, max_depth=4,
                 random_state=RANDOM_SEED):
        self.n_estimators         = n_estimators
        self.learning_rate        = learning_rate
        self.max_depth            = max_depth
        self.random_state         = random_state
        self.estimators_          = []
        self.estimator_weights_   = []
        self.feature_importances_ = None

    @staticmethod
    def _norm_fi(tree):
        raw = tree.feature_importances_
        s   = raw.sum()
        return raw / s if s > 0 else np.ones_like(raw) / len(raw)

    @staticmethod
    def _composite_phi(X, phi):
        X_abs  = np.abs(X)
        col_mx = X_abs.max(axis=0)
        col_mx[col_mx == 0] = 1
        Phi    = (X_abs / col_mx * phi).sum(axis=1)
        p_max  = Phi.max()
        return Phi / p_max if p_max > 0 else Phi

    def fit(self, X, y):
        X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        n     = len(y_arr)
        rng   = np.random.default_rng(self.random_state)
        weights = np.full(n, 1.0 / n)
        cum_fi  = np.zeros(X_arr.shape[1])
        n_valid = 0

        for t in range(self.n_estimators):
            idx  = rng.choice(n, size=n, replace=True, p=weights)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, random_state=self.random_state + t
            )
            tree.fit(X_arr[idx], y_arr[idx])
            y_pred = tree.predict(X_arr)
            abs_e  = np.abs(y_arr - y_pred)
            D_t    = abs_e.max()
            if D_t == 0:
                break
            e_i   = abs_e / D_t
            phi   = self._norm_fi(tree)                     # Calculate phi early
            Phi_i = self._composite_phi(X_arr, phi)         # Calculate Phi_i early

            modulated_loss = e_i * Phi_i                    # Modulate the error
            eps_t = float(np.dot(weights, modulated_loss))  # Calculate eps_t with modulated loss

            if eps_t >= 0.5:
                break

            beta_t = eps_t / (1.0 - eps_t + 1e-10)
            phi    = self._norm_fi(tree)
            Phi_i  = self._composite_phi(X_arr, phi)
            new_w  = weights * (beta_t ** (1.0 - e_i * Phi_i))
            Z_t    = new_w.sum()
            if Z_t == 0:
                break
            weights = new_w / Z_t

            est_w = max(
                self.learning_rate * math.log((1.0 - eps_t) / (eps_t + 1e-10)),
                1e-10,
            )
            self.estimators_.append(tree)
            self.estimator_weights_.append(est_w)
            cum_fi  += phi
            n_valid += 1

        self.feature_importances_ = (
            cum_fi / n_valid if n_valid > 0
            else np.ones(X_arr.shape[1]) / X_arr.shape[1]
        )
        return self

    def predict(self, X):
        X_arr   = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        preds   = np.array([e.predict(X_arr) for e in self.estimators_])
        weights = np.array(self.estimator_weights_)
        weights = weights / weights.sum()
        result  = np.zeros(X_arr.shape[0])
        for i in range(X_arr.shape[0]):
            p_i   = preds[:, i]
            order = np.argsort(p_i)
            cumw  = np.cumsum(weights[order])
            mid   = np.searchsorted(cumw, 0.5)
            result[i] = p_i[order[min(mid, len(p_i) - 1)]]
        return result


# =============================================================================
# EVALUATION METRICS
# =============================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"RMSE_J": rmse, "MAE_J": mae, "R2": r2}


def make_result_table(ada_m: dict, fi_m: dict) -> pd.DataFrame:
    rows = [
        {
            "Model":            "AdaBoost Regression (Baseline)",
            "RMSE (J/m²/day)":  round(ada_m["RMSE_J"], 2),
            "MAE  (J/m²/day)":  round(ada_m["MAE_J"],  2),
            "R²":               round(ada_m["R2"],      4),
        },
        {
            "Model":            "FI-AdaBoost Regression (Proposed)",
            "RMSE (J/m²/day)":  round(fi_m["RMSE_J"],  2),
            "MAE  (J/m²/day)":  round(fi_m["MAE_J"],   2),
            "R²":               round(fi_m["R2"],       4),
        },
    ]
    return pd.DataFrame(rows).set_index("Model")


def _print_table(title: str, df: pd.DataFrame) -> None:
    sep = "─" * 70
    print(f"\n  {sep}")
    print(f"  {title}")
    print(f"  {sep}")
    print(df.to_string())
    print(f"  {sep}")


# =============================================================================
# OPTUNA HYPERPARAMETER TUNING — Fix 4
# =============================================================================
def _baseline_space(trial) -> dict:
    return {
        "n_estimators":  trial.suggest_int("n_estimators",  50,   300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
        "max_depth":     trial.suggest_int("max_depth",      1,     5),
    }


def _fi_space(trial) -> dict:
    return {
        "n_estimators":  trial.suggest_int("n_estimators",  50,   300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
        "max_depth":     trial.suggest_int("max_depth",      1,     6),
    }


def _tune_kfold(model_cls, space_fn, X_train: np.ndarray, y_train: np.ndarray,
                n_trials: int = 100, n_splits: int = 5) -> dict:
    """Optuna + KFold tuning for spatial dataset. Returns best_params."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    def objective(trial):
        params = space_fn(trial)
        rmses  = []
        for tr_idx, val_idx in kf.split(X_train):
            m = model_cls(**params).fit(X_train[tr_idx], y_train[tr_idx])
            p = m.predict(X_train[val_idx])
            rmses.append(np.sqrt(mean_squared_error(y_train[val_idx], p)))
        return np.mean(rmses)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _tune_timeseries(model_cls, space_fn, X_train: np.ndarray, y_train: np.ndarray,
                     n_trials: int = 100, n_splits: int = 5) -> dict:
    """Optuna + TimeSeriesSplit tuning for daily dataset. Returns best_params."""
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        params = space_fn(trial)
        rmses  = []
        for tr_idx, val_idx in tscv.split(X_train):
            m = model_cls(**params).fit(X_train[tr_idx], y_train[tr_idx])
            p = m.predict(X_train[val_idx])
            rmses.append(np.sqrt(mean_squared_error(y_train[val_idx], p)))
        return np.mean(rmses)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# =============================================================================
# CROSS-VALIDATION METRICS — Fix 4
# =============================================================================
def run_kfold_cv(X_train_ada: np.ndarray, X_train_fi: np.ndarray,
                 y_train: np.ndarray,
                 ada_params: dict, fi_params: dict, n_splits: int = 5) -> pd.DataFrame:
    """KFold CV with tuned hyperparams — saves per-fold metrics including Train vs Val."""
    kf   = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(y_train), start=1):
        ada   = BaselineAdaBoost(**ada_params).fit(X_train_ada[tr_idx], y_train[tr_idx])
        fi    = FIAdaBoostRegressor(**fi_params).fit(X_train_fi[tr_idx], y_train[tr_idx])

        # Compute Training Metrics
        ada_train_m = compute_metrics(y_train[tr_idx], ada.predict(X_train_ada[tr_idx]))
        fi_train_m  = compute_metrics(y_train[tr_idx], fi.predict(X_train_fi[tr_idx]))

        # Compute Validation Metrics
        ada_val_m = compute_metrics(y_train[val_idx], ada.predict(X_train_ada[val_idx]))
        fi_val_m  = compute_metrics(y_train[val_idx], fi.predict(X_train_fi[val_idx]))

        rows.append({
            "fold":             fold,
            "ada_train_RMSE_J": ada_train_m["RMSE_J"],
            "ada_val_RMSE_J":   ada_val_m["RMSE_J"],
            "ada_train_R2":     ada_train_m["R2"],
            "ada_val_R2":       ada_val_m["R2"],
            "fi_train_RMSE_J":  fi_train_m["RMSE_J"],
            "fi_val_RMSE_J":    fi_val_m["RMSE_J"],
            "fi_train_R2":      fi_train_m["R2"],
            "fi_val_R2":        fi_val_m["R2"],
        })
    return pd.DataFrame(rows)


def save_cv_metrics(cv_df: pd.DataFrame) -> str:
    path = os.path.join(RESULTS_DIR, "cv_fold_metrics.csv")
    cv_df.to_csv(path, index=False)
    return path


# =============================================================================
# DIEBOLD–MARIANO TEST — Fix 5
# =============================================================================
def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray) -> dict:
    """
    Diebold–Mariano test with Newey–West variance estimator (h=1 lag).
    H₀: equal predictive accuracy.  H₁: FI-AdaBoost is more accurate.
    e1 = actual − predicted (Baseline); e2 = actual − predicted (FI).
    Positive DM_statistic ⇒ FI-AdaBoost has lower squared error.
    """
    d     = e1**2 - e2**2
    n     = len(d)
    d_bar = d.mean()
    g0    = ((d - d_bar)**2).sum() / (n - 1)
    g1    = ((d[1:] - d_bar) * (d[:-1] - d_bar)).sum() / (n - 1)
    sigma = max((g0 + 2 * g1) / n, 1e-20)
    dm    = d_bar / np.sqrt(sigma)
    pval  = float(2 * (1 - stats.norm.cdf(abs(dm))))
    return {
        "DM_statistic":        round(float(dm), 4),
        "p_value":             round(pval, 6),
        "significant_at_0.05": bool(pval < 0.05),
        "interpretation": (
            "FI-AdaBoost significantly more accurate (p<0.05)"   if dm > 0 and pval < 0.05
            else "AdaBoost baseline significantly more accurate (p<0.05)" if dm < 0 and pval < 0.05
            else "No statistically significant difference at α=0.05"
        ),
        "n_test_samples":      n,
        "mean_loss_diff_J2":   round(float(d_bar), 2),
    }


def save_dm_results(dm: dict, suffix: str = "") -> str:
    fname = f"dm_test_results{'_' + suffix if suffix else ''}.csv"
    path  = os.path.join(RESULTS_DIR, fname)
    pd.DataFrame([dm]).to_csv(path, index=False)
    return path


# =============================================================================
# PHASE 2: THEORETICAL SOLAR ENERGY POTENTIAL FORECAST
# =============================================================================
def forecast_solar_energy(
    df: pd.DataFrame,
    model,
    feature_cols: list,
    label: str,
) -> pd.DataFrame:
    X_all    = df[feature_cols].values.astype(float)
    y_pred_J = model.predict(X_all)
    H_daily  = y_pred_J / KWH_TO_J                        # predicted irradiance, kWh/m2/day
    result   = df[["lat", "lon", "rooftop_area_sq_m"]].copy()
    result[f"{label}_predicted_irradiance_kWh_m2_day"] = H_daily
    result[f"{label}_SEP_kWh_day"] = df["rooftop_area_sq_m"] * H_daily
    result[f"{label}_SEP_kWh_yr"]  = result[f"{label}_SEP_kWh_day"] * DAYS_PER_YEAR
    return result


# =============================================================================
# DAILY TIME-SERIES PIPELINE — Fix 2 + Fix 3
# =============================================================================
def load_daily_dataset() -> pd.DataFrame:
    """
    Load/fetch 365-day NASA POWER data and engineer all features.

    Features:
      Temporal  (§2.2.1): month_sin, month_cos, season
      Meteo     (§2.2.2): T2M, RH2M, ALLSKY_KT, clear_sky_ratio, sunshine_hours
      Topo agg  (§2.2.3): mean_orientation_score, mean_shading_factor, mean_SEI_norm
    Target: daily GHI in J/m²/day.
    """
    raw_path = os.path.join(RAW_DIR, "nasa_raw.csv")
    if not os.path.exists(raw_path):
        print("  [Daily] nasa_raw.csv not found — fetching from NASA POWER …")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from data_acquisition import fetch_nasa_timeseries
        fetch_nasa_timeseries()

    df = pd.read_csv(raw_path)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df.replace(-999, np.nan, inplace=True)
    df.interpolate(method="linear", inplace=True)
    df = df.dropna(
        subset=["ALLSKY_SFC_SW_DWN", "T2M", "RH2M", "ALLSKY_KT"]
    ).reset_index(drop=True)

    # §2.2.1 Temporal features — cyclical encoding
    df["month"]     = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["season"]    = df["month"].apply(lambda m: 1 if m in [12, 1, 2, 3, 4, 5] else 0)

    # pvlib clear-sky at city centroid
    lat = float(df["lat"].iloc[0])
    lon = float(df["lon"].iloc[0])
    loc     = pvlib.location.Location(lat, lon, altitude=30, tz="Asia/Manila")
    times_h = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h", tz="Asia/Manila")
    cs = loc.get_clearsky(times_h, model="ineichen")

    # §2.2.2.2 clear_sky_ratio = daily actual GHI / daily clear-sky GHI
    ghi_daily_series = cs["ghi"].resample("D").sum() / 1000  # kWh/m²/day
    # Use .date() to extract local Manila date (avoids UTC offset shift)
    daily_clear = pd.DataFrame({
        "date":          pd.to_datetime([ts.date() for ts in ghi_daily_series.index]),
        "GHI_clear_sky": ghi_daily_series.values,
    })
    df = df.merge(daily_clear, on="date", how="left")
    df["clear_sky_ratio"] = (
        df["ALLSKY_SFC_SW_DWN"] / df["GHI_clear_sky"].replace(0, np.nan)
    ).clip(0.0, 1.5)

    # §2.2.2.1 sunshine_hours = hours per day where clear-sky GHI > 120 W/m²
    sun_daily_series = (cs["ghi"] > 120).resample("D").sum()
    daily_sun = pd.DataFrame({
        "date":           pd.to_datetime([ts.date() for ts in sun_daily_series.index]),
        "sunshine_hours": sun_daily_series.values.astype(float),
    })
    df = df.merge(daily_sun, on="date", how="left")

    # §2.2.3 Aggregate topographical features (static across all days)
    osm_path = os.path.join(PROCESSED_DIR, "osm_features.geojson")
    if os.path.exists(osm_path):
        osm = gpd.read_file(osm_path)
        df["mean_orientation_score"] = float(osm["orientation_score"].mean())
        df["mean_shading_factor"]    = float(osm["shading_factor"].mean())
        df["mean_SEI_norm"]          = float(osm["SEI_norm"].mean())
    else:
        df["mean_orientation_score"] = 0.5
        df["mean_shading_factor"]    = 0.05
        df["mean_SEI_norm"]          = 0.1

    df["GHI_J"] = df["ALLSKY_SFC_SW_DWN"] * KWH_TO_J
    df = df.sort_values("date").reset_index(drop=True)

    # Drop any remaining NaN in feature or target columns
    required = DAILY_FEATURES + [DAILY_TARGET]
    before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) < before:
        print(f"  [Daily] Dropped {before - len(df)} rows with NaN in features/target.")
    return df


def temporal_split(df: pd.DataFrame, test_frac: float = 0.20):
    """Chronological 80/20 split — preserves time ordering (§2.5.1)."""
    split = int(len(df) * (1 - test_frac))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def run_daily_pipeline() -> None:
    """
    Secondary pipeline: 365-day time-series, temporal split, TimeSeriesSplit CV.
    Saves: cv_fold_metrics_daily.csv, dm_test_results_daily.csv, daily_metrics_summary.csv.
    """
    print("\n" + "─" * 70)
    print("  [Daily Pipeline] §2.5.1 Temporal split + §2.5.2 TimeSeriesSplit CV")
    print("─" * 70)

    df = load_daily_dataset()
    print(f"  Records: {len(df)} ({df['date'].min().date()} → {df['date'].max().date()})")

    train_df, test_df = temporal_split(df)
    print(f"  Train: {len(train_df)} days | Test: {len(test_df)} days")

    X_tr = train_df[DAILY_FEATURES].values.astype(float)
    y_tr = train_df[DAILY_TARGET].values.astype(float)
    X_te = test_df[DAILY_FEATURES].values.astype(float)
    y_te = test_df[DAILY_TARGET].values.astype(float)

    print("\n  Tuning AdaBoost (daily) — Optuna + TimeSeriesSplit(5), 100 trials …")
    ada_params = _tune_timeseries(BaselineAdaBoost, _baseline_space, X_tr, y_tr)
    print(f"  Best: {ada_params}")

    print("  Tuning FI-AdaBoost (daily) — Optuna + TimeSeriesSplit(5), 100 trials …")
    fi_params = _tune_timeseries(FIAdaBoostRegressor, _fi_space, X_tr, y_tr)
    print(f"  Best: {fi_params}")

    # CV fold metrics (Fix 4 — TimeSeriesSplit)
    tscv    = TimeSeriesSplit(n_splits=5)
    cv_rows = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_tr), start=1):
        ada   = BaselineAdaBoost(**ada_params).fit(X_tr[tr_idx], y_tr[tr_idx])
        fi    = FIAdaBoostRegressor(**fi_params).fit(X_tr[tr_idx], y_tr[tr_idx])
        ada_m = compute_metrics(y_tr[val_idx], ada.predict(X_tr[val_idx]))
        fi_m  = compute_metrics(y_tr[val_idx], fi.predict(X_tr[val_idx]))
        cv_rows.append({
            "fold":       fold,
            "ada_RMSE_J": ada_m["RMSE_J"], "ada_MAE_J": ada_m["MAE_J"], "ada_R2": ada_m["R2"],
            "fi_RMSE_J":  fi_m["RMSE_J"],  "fi_MAE_J":  fi_m["MAE_J"],  "fi_R2":  fi_m["R2"],
        })
    cv_df = pd.DataFrame(cv_rows)

    # Print CV summary
    print("\n  CV Fold Metrics (TimeSeriesSplit — daily):")
    for _, r in cv_df.iterrows():
        print(f"    Fold {int(r['fold'])}: Ada RMSE={r['ada_RMSE_J']:,.0f}  FI RMSE={r['fi_RMSE_J']:,.0f}  "
              f"FI R²={r['fi_R2']:.4f}")
    avg = cv_df.mean(numeric_only=True)
    print(f"    Avg:     Ada RMSE={avg['ada_RMSE_J']:,.0f}  FI RMSE={avg['fi_RMSE_J']:,.0f}  "
          f"FI R²={avg['fi_R2']:.4f}")

    # Final fit and test evaluation
    ada_final = BaselineAdaBoost(**ada_params).fit(X_tr, y_tr)
    fi_final  = FIAdaBoostRegressor(**fi_params).fit(X_tr, y_tr)

    # Train metrics
    ada_train_m = compute_metrics(y_tr, ada_final.predict(X_tr))
    fi_train_m  = compute_metrics(y_tr, fi_final.predict(X_tr))

    # Test metrics
    ada_pred = ada_final.predict(X_te)
    fi_pred  = fi_final.predict(X_te)
    ada_m    = compute_metrics(y_te, ada_pred)
    fi_m     = compute_metrics(y_te, fi_pred)

    print(f"\n  [Daily] Test Results (temporal holdout):")
    print(f"    AdaBoost   (Train): RMSE={ada_train_m['RMSE_J']:>12,.0f}  R²={ada_train_m['R2']:.4f}")
    print(f"    AdaBoost   (Test) : RMSE={ada_m['RMSE_J']:>12,.0f}  R²={ada_m['R2']:.4f}")
    print(f"    FI-AdaBoost(Train): RMSE={fi_train_m['RMSE_J']:>12,.0f}  R²={fi_train_m['R2']:.4f}")
    print(f"    FI-AdaBoost(Test) : RMSE={fi_m['RMSE_J']:>12,.0f}  R²={fi_m['R2']:.4f}")

    # Diebold–Mariano test (Fix 5)
    dm = diebold_mariano_test(y_te - ada_pred, y_te - fi_pred)

    # Save results
    daily_rows = [
        {"model": "AdaBoost (Daily, Temporal Split)",    "split": "temporal_80_20", **ada_m},
        {"model": "FI-AdaBoost (Daily, Temporal Split)", "split": "temporal_80_20", **fi_m},
    ]
    p_daily = os.path.join(RESULTS_DIR, "daily_metrics_summary.csv")
    pd.DataFrame(daily_rows).to_csv(p_daily, index=False)

    p_cv = os.path.join(RESULTS_DIR, "cv_fold_metrics_daily.csv")
    cv_df.to_csv(p_cv, index=False)
    p_dm = save_dm_results(dm, suffix="daily")

    print(f"\n  Test Results (temporal holdout):")
    print(f"    AdaBoost   : RMSE={ada_m['RMSE_J']:>12,.0f}  MAE={ada_m['MAE_J']:>12,.0f}  R²={ada_m['R2']:.4f}")
    print(f"    FI-AdaBoost: RMSE={fi_m['RMSE_J']:>12,.0f}  MAE={fi_m['MAE_J']:>12,.0f}  R²={fi_m['R2']:.4f}")
    print(f"\n  Diebold–Mariano Test (§2.5.4):")
    print(f"    DM statistic = {dm['DM_statistic']}   p-value = {dm['p_value']}")
    print(f"    {dm['interpretation']}")
    print(f"\n  Saved: {p_daily}")
    print(f"  Saved: {p_cv}")
    print(f"  Saved: {p_dm}")


# =============================================================================
# RESULT EXPORTS
# =============================================================================
def save_metrics_csv(ada_m: dict, fi_m: dict) -> str:
    rows = [
        {"model": "AdaBoost (Baseline)",    "target": "Effective GHI — pvlib POA-adjusted (J/m²/day)", **ada_m},
        {"model": "FI-AdaBoost (Proposed)", "target": "Effective GHI — pvlib POA-adjusted (J/m²/day)", **fi_m},
    ]
    path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_forecast_csv(ada_fcast: pd.DataFrame, fi_fcast: pd.DataFrame) -> str:
    merged = ada_fcast.merge(
        fi_fcast[
            [
                "lat",
                "lon",
                "fi_predicted_irradiance_kWh_m2_day",
                "fi_SEP_kWh_day",
                "fi_SEP_kWh_yr",
            ]
        ],
        on=["lat", "lon"],
        how="left",
    )
    merged["difference_SEP_kWh_yr"] = (
        merged["ada_SEP_kWh_yr"] - merged["fi_SEP_kWh_yr"]
    )
    path = os.path.join(RESULTS_DIR, "forecast_per_building.csv")
    merged.to_csv(path, index=False)
    return path


# =============================================================================
# PLOTS
# =============================================================================
def plot_standalone_feature_importance(fi_vals, feats):
    plt.figure(figsize=(10, 6))
    labels     = [f.replace("_", " ").title() for f in feats]
    idx        = np.argsort(fi_vals)
    sv         = fi_vals[idx]
    sl         = np.array(labels)[idx]
    bars       = plt.barh(sl, sv, color=C_FI, edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, sv):
        plt.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{val * 100:.4f}%", va="center", ha="left", fontsize=10, fontweight="bold")
    plt.title("FI-AdaBoost Feature Importance", fontsize=14, fontweight="bold")
    plt.xlabel("Relative Importance Weight", fontsize=12)
    plt.xlim(0, max(sv) + 0.1)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "standalone_feature_importances.png"), dpi=300)
    plt.close()


def plot_actual_vs_predicted(y_true_ada, y_pred_ada, y_true_fi, y_pred_fi):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, y_true, y_pred, color, title in [
        (axes[0], y_true_ada, y_pred_ada, C_ADA, "AdaBoost Baseline\n(Effective GHI)"),
        (axes[1], y_true_fi,  y_pred_fi,  C_FI,  "FI-AdaBoost Proposed\n(Effective GHI)"),
    ]:
        ax.scatter(y_true, y_pred, alpha=0.4, s=18, color=color, edgecolors="none")
        mn = min(y_true.min(), y_pred.min())
        mx = max(y_true.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.2, label="Perfect fit")
        ax.set_title(f"{title}\nR² = {r2_score(y_true, y_pred):.4f}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Actual (J/m²/day)", fontsize=10)
        ax.set_ylabel("Predicted (J/m²/day)", fontsize=10)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.suptitle("Actual vs Predicted — Effective GHI (same target, same features)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "actual_vs_predicted.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_residuals(y_true_ada, y_pred_ada, y_true_fi, y_pred_fi):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, y_true, y_pred, color, title in [
        (axes[0], y_true_ada, y_pred_ada, C_ADA, "AdaBoost Baseline"),
        (axes[1], y_true_fi,  y_pred_fi,  C_FI,  "FI-AdaBoost Proposed"),
    ]:
        res = y_true - y_pred
        ax.scatter(y_pred, res, alpha=0.4, s=18, color=color, edgecolors="none")
        ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
        ax.set_title(f"{title}\nResiduals", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted (J/m²/day)", fontsize=10)
        ax.set_ylabel("Residual (Actual − Predicted)", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.suptitle("Residual Plot — Phase 1 ML Validation", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "residuals.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_metrics_comparison(ada_m: dict, fi_m: dict):
    metrics  = ["RMSE_J", "MAE_J", "R2"]
    labels   = ["RMSE (J/m²/day)", "MAE (J/m²/day)", "R²"]
    ada_vals = [ada_m[k] for k in metrics]
    fi_vals  = [fi_m[k]  for k in metrics]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, label, av, fv in zip(axes, labels, ada_vals, fi_vals):
        bars = ax.bar(["AdaBoost\n(Baseline)", "FI-AdaBoost\n(Proposed)"], [av, fv],
                      color=[C_ADA, C_FI], edgecolor="black", alpha=0.85, width=0.5)
        for bar, val in zip(bars, [av, fv]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{val:,.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(av, fv) * 1.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.suptitle("Model Performance Comparison — Same Target & Features",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "metrics_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_energy_distribution(ada_fcast: pd.DataFrame, fi_fcast: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, fcast, col, color, title in [
        (axes[0], ada_fcast, "ada_SEP_kWh_yr", C_ADA, "AdaBoost Baseline"),
        (axes[1], fi_fcast,  "fi_SEP_kWh_yr",  C_FI,  "FI-AdaBoost Proposed"),
    ]:
        data = fcast[col].dropna()
        ax.hist(data, bins=40, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(), color="black", linestyle="--", linewidth=1.2,
                   label=f"Mean: {data.mean():,.0f}")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Solar Energy Potential (kWh/year)", fontsize=10)
        ax.set_ylabel("Number of Buildings", fontsize=10)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.suptitle("Per-Building Theoretical Solar Energy Potential Distribution — Phase 2 Forecast",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "energy_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_total_energy_comparison(ada_fcast: pd.DataFrame, fi_fcast: pd.DataFrame):
    ada_kwh = ada_fcast["ada_SEP_kWh_yr"].sum()
    fi_kwh  = fi_fcast["fi_SEP_kWh_yr"].sum()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["AdaBoost\n(Baseline)", "FI-AdaBoost\n(Proposed)"],
                  [ada_kwh, fi_kwh], color=[C_ADA, C_FI], edgecolor="black", alpha=0.85, width=0.45)
    for bar, val in zip(bars, [ada_kwh, fi_kwh]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val:,.0f} kWh", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Theoretical Solar Energy Potential (kWh/year)", fontsize=11)
    ax.set_title("Total Rooftop Solar Potential — Davao City\nPhase 2 Energy Forecast",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(ada_kwh, fi_kwh) * 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "total_energy_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_overfit_check(ada_train_m, ada_test_m, fi_train_m, fi_test_m):
    labels     = ["Baseline AdaBoost", "FI-AdaBoost Proposed"]
    train_rmse = [ada_train_m["RMSE_J"], fi_train_m["RMSE_J"]]
    test_rmse  = [ada_test_m["RMSE_J"],  fi_test_m["RMSE_J"]]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width / 2, train_rmse, width, label="Train RMSE", color="#3498DB", edgecolor="black")
    rects2 = ax.bar(x + width / 2, test_rmse,  width, label="Test RMSE",  color="#E74C3C", edgecolor="black")

    ax.set_ylabel("RMSE (J/m²/day)", fontsize=12)
    ax.set_title("Overfitting Check: Train vs Test Error\n(A massive gap indicates overfitting)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "overfit_check.png"), dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    print("\n" + "=" * 74)
    print("  SOLAR IRRADIANCE FORECASTING — Davao City")
    print("  AdaBoost (Baseline)  vs  FI-AdaBoost (Proposed)")
    print("  Fix 1: Same target | Fix 3: Meteo features | Fix 6: No log transforms")
    print("=" * 74)

    # ── [1] Load data ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading spatial dataset and computing features …")
    df = load_dataset()
    print(f"  Rows: {len(df)}")
    print(f"  Baseline features ({len(BASELINE_FEATURES)}): {BASELINE_FEATURES}")
    print(f"  FI-AdaBoost features ({len(FI_FEATURES)}): {FI_FEATURES}")

    # ── [2] Split ──────────────────────────────────────────────────────────────
    print("\n[2/6] 80/20 Random Split …")
    train_df, test_df = random_split(df)
    print(f"  Total: {len(df)}  |  Train: {len(train_df)}  |  Test: {len(test_df)}")

    split_info = [{
        "total_samples":  len(df),
        "train_samples":  len(train_df),
        "test_samples":   len(test_df),
        "test_fraction":  0.20,
        "split_method":   "random_shuffle",
        "random_seed":    RANDOM_SEED,
        "target_col":     TARGET_COL,
    }]
    pd.DataFrame(split_info).to_csv(
        os.path.join(RESULTS_DIR, "train_test_split_info.csv"), index=False
    )

    X_tr_ada = train_df[BASELINE_FEATURES].values.astype(float)
    X_te_ada = test_df[BASELINE_FEATURES].values.astype(float)
    X_tr_fi  = train_df[FI_FEATURES].values.astype(float)
    X_te_fi  = test_df[FI_FEATURES].values.astype(float)
    y_tr = train_df[TARGET_COL].values.astype(float)
    y_te = test_df[TARGET_COL].values.astype(float)

    # ── [3] Optuna hyperparameter tuning (Fix 4) ───────────────────────────────
    print("\n[3/6] Optuna Tuning — KFold(5), 100 trials per model …")
    print("  Tuning BaselineAdaBoost …")
    ada_params = _tune_kfold(BaselineAdaBoost, _baseline_space, X_tr_ada, y_tr)
    print(f"  Best AdaBoost params : {ada_params}")

    print("  Tuning FIAdaBoostRegressor …")
    fi_params = _tune_kfold(FIAdaBoostRegressor, _fi_space, X_tr_fi, y_tr)
    print(f"  Best FI-AdaBoost params: {fi_params}")

    # ── [4] KFold CV metrics (Fix 4) ───────────────────────────────────────────
    print("\n[4/6] KFold CV Metrics …")
    cv_df = run_kfold_cv(X_tr_ada, X_tr_fi, y_tr, ada_params, fi_params)
    save_cv_metrics(cv_df)

    print("  Fold metrics (Validation):")
    for _, r in cv_df.iterrows():
        print(f"    Fold {int(r['fold'])}: "
              f"Ada RMSE={r['ada_val_RMSE_J']:>12,.0f}  "
              f"FI  RMSE={r['fi_val_RMSE_J']:>12,.0f}  "
              f"FI  R²={r['fi_val_R2']:.4f}")
    avg = cv_df.mean(numeric_only=True)
    print(f"    Avg:     "
          f"Ada RMSE={avg['ada_val_RMSE_J']:>12,.0f}  "
          f"FI  RMSE={avg['fi_val_RMSE_J']:>12,.0f}  "
          f"FI  R²={avg['fi_val_R2']:.4f}")

    # ── [5] Train final models on full training set ────────────────────────────
    print("\n[5/6] Training final models …")
    ada = BaselineAdaBoost(**ada_params).fit(X_tr_ada, y_tr)
    fi  = FIAdaBoostRegressor(**fi_params).fit(X_tr_fi, y_tr)

    # Predict on Training Data for Overfit Check
    ada_train_m = compute_metrics(y_tr, ada.predict(X_tr_ada))
    fi_train_m  = compute_metrics(y_tr, fi.predict(X_tr_fi))

    # Predict on Test Data
    ada_te = ada.predict(X_te_ada)
    fi_te  = fi.predict(X_te_fi)

    ada_test_m = compute_metrics(y_te, ada_te)
    fi_test_m  = compute_metrics(y_te, fi_te)

    # Print the Overfit Check
    print("\n  OVERFITTING CHECK (Train vs Test RMSE):")
    print(f"    AdaBoost   : Train RMSE={ada_train_m['RMSE_J']:>12,.0f} | Test RMSE={ada_test_m['RMSE_J']:>12,.0f}")
    print(f"    FI-AdaBoost: Train RMSE={fi_train_m['RMSE_J']:>12,.0f} | Test RMSE={fi_test_m['RMSE_J']:>12,.0f}")
    if fi_train_m["RMSE_J"] < (fi_test_m["RMSE_J"] * 0.5):
        print("    WARNING: FI-AdaBoost Train error is less than half the Test error. Significant overfitting likely.")

    # Diebold–Mariano test (Fix 5)
    dm = diebold_mariano_test(y_te - ada_te, y_te - fi_te)

    print("\n  FI-AdaBoost Feature Importances (Fix 6 — raw, no log transform):")
    for feat, imp in sorted(zip(SHARED_FEATURES, fi.feature_importances_),
                            key=lambda x: -x[1]):
        print(f"    {feat:<25}: {imp * 100:.4f}%")

    t = make_result_table(ada_test_m, fi_test_m)
    _print_table("Phase 1: ML Validation (same target + same features)", t)

    print(f"\n  Diebold–Mariano Test (§2.5.4):")
    print(f"    DM statistic = {dm['DM_statistic']}   p-value = {dm['p_value']}")
    print(f"    {dm['interpretation']}")

    # ── [6] Phase 2 energy forecast + save results ────────────────────────────
    print("\n[6/6] Phase 2 Energy Forecast + Saving Results …")
    ada_fcast = forecast_solar_energy(df, ada, BASELINE_FEATURES, label="ada")
    fi_fcast  = forecast_solar_energy(df, fi,  FI_FEATURES,       label="fi")

    print(f"  Baseline   total: {ada_fcast['ada_SEP_kWh_yr'].sum():>18,.2f} kWh/year")
    print(f"  FI-AdaBoost total: {fi_fcast['fi_SEP_kWh_yr'].sum():>18,.2f} kWh/year")

    # Persist trained models
    joblib.dump(ada, BASELINE_MODEL_FILE)
    joblib.dump(fi,  FI_MODEL_FILE)

    # Save all spatial results
    p_metrics  = save_metrics_csv(ada_test_m, fi_test_m)
    p_forecast = save_forecast_csv(ada_fcast, fi_fcast)
    p_dm       = save_dm_results(dm, suffix="spatial")

    plot_standalone_feature_importance(fi.feature_importances_, SHARED_FEATURES)
    plot_actual_vs_predicted(y_te, ada_te, y_te, fi_te)
    plot_residuals(y_te, ada_te, y_te, fi_te)
    plot_metrics_comparison(ada_test_m, fi_test_m)
    plot_energy_distribution(ada_fcast, fi_fcast)
    plot_total_energy_comparison(ada_fcast, fi_fcast)
    plot_overfit_check(ada_train_m, ada_test_m, fi_train_m, fi_test_m)

    saved = [
        ("CSV",   os.path.join(RESULTS_DIR, "train_test_split_info.csv")),
        ("CSV",   p_metrics),
        ("CSV",   p_forecast),
        ("CSV",   p_dm),
        ("Plot",  os.path.join(RESULTS_DIR, "standalone_feature_importances.png")),
        ("Plot",  os.path.join(RESULTS_DIR, "actual_vs_predicted.png")),
        ("Plot",  os.path.join(RESULTS_DIR, "residuals.png")),
        ("Plot",  os.path.join(RESULTS_DIR, "metrics_comparison.png")),
        ("Plot",  os.path.join(RESULTS_DIR, "energy_distribution.png")),
        ("Plot",  os.path.join(RESULTS_DIR, "total_energy_comparison.png")),
        ("Plot",  os.path.join(RESULTS_DIR, "overfit_check.png")),
        ("Model", BASELINE_MODEL_FILE),
        ("Model", FI_MODEL_FILE),
    ]
    print(f"\n  {'Type':<8}  File")
    print(f"  {'─'*8}  {'─'*55}")
    for kind, p in saved:
        print(f"  {kind:<8}  {p}")

    # ── Daily pipeline (Fix 2 + Fix 3) ────────────────────────────────────────
    run_daily_pipeline()

    print("\n[Done] All results saved to 'results/' folder.\n")


if __name__ == "__main__":
    main()
