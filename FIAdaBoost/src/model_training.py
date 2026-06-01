"""
model_training.py
────────────────────────────────────────────────────────────────────────────
Solar Irradiance Forecasting — Davao City
AdaBoost Regression (Baseline)  vs  FI-AdaBoost Regression (Proposed)


METHODOLOGY (Two-Phase Pipeline)
──────────────────────────────────────────────────────────────────────────
  PHASE 1: MACHINE LEARNING (fair comparison)
  - Both models predict the SAME target: engineered effective_GHI_J
    (Fix 1 - aligned targets).
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
optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress per-trial output


import joblib                          # model serialization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd                # reads OSM GeoJSON for rooftop areas
from scipy import stats                # used for the Diebold–Mariano normal CDF
import pvlib                           # computes theoretical clear-sky irradiance


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
KWH_TO_J      = 3_600_000  # 1 kWh = 3.6 × 10⁶ J; converts irradiance units throughout


np.random.seed(RANDOM_SEED)


# ── Feature sets — identical for fair algorithm comparison (Fix 1 + Fix 6) ───
# Both models predict engineered effective_GHI_J.
# Both use the same 8 features. The only difference is the boosting algorithm.
# Log transforms removed (Fix 6): raw shading_factor and SEI_norm are used.
# clear_sky_ratio and sunshine_hours added from §2.2.2 (Fix 3).
SHARED_FEATURES   = [
    "lat", "lon", "azimuth", "orientation_score",
    "shading_factor", "SEI_norm", "clear_sky_ratio", "sunshine_hours",
]
BASELINE_FEATURES = SHARED_FEATURES   # alias — same 8 features for baseline
FI_FEATURES       = SHARED_FEATURES   # alias — same 8 features for proposed model
TARGET_COL        = "effective_GHI_J" # GHI adjusted for orientation and shading (J/m²/day)


# Daily time-series feature set (§2.2.1 + §2.2.2 + §2.2.3 aggregates)
DAILY_FEATURES = [
    "month_sin", "month_cos", "season",                            # §2.2.1 temporal
    "T2M", "RH2M", "ALLSKY_KT", "clear_sky_ratio", "sunshine_hours",  # §2.2.2 meteorological
    "mean_orientation_score", "mean_shading_factor", "mean_SEI_norm",  # §2.2.3 topographical aggregates
]
DAILY_TARGET = "GHI_J"  # raw daily GHI in J/m²/day (no orientation/shading adjustment)


C_ADA = "#3B82F6"  # blue — baseline AdaBoost color in all plots
C_FI  = "#F97316"  # orange — proposed FI-AdaBoost color in all plots


# =============================================================================
# pvlib CACHED HELPERS
# =============================================================================
_pvlib_cache: dict = {}  # keyed by (lat_rounded, lon_rounded) to avoid redundant pvlib calls


def _pvlib_features(lat: float, lon: float) -> dict:
    """
    Annual pvlib clear-sky scalars for a location (cached at 0.2° resolution).
    Returns ghi_clear_annual (kWh/m²/day) and sunshine_hours (hrs/day).
    ~9 unique grid cells cover Davao City — fast after first batch.
    """
    key = (round(lat, 2), round(lon, 2))  # 0.2° grid reduces unique pvlib calls from ~3000 to ~9
    if key in _pvlib_cache:
        return _pvlib_cache[key]

    loc   = pvlib.location.Location(lat, lon, altitude=30, tz="Asia/Manila")
    times = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h", tz="Asia/Manila")
    cs    = loc.get_clearsky(times, model="ineichen")  # Ineichen clear-sky model for tropical latitudes

    # Average daily clear-sky GHI across the year (kWh/m²/day)
    ghi_clear_annual = float((cs["ghi"].resample("D").sum() / 1000).mean())
    # Average number of hours per day where the sun is meaningfully strong (>120 W/m²)
    sunshine_hours   = float((cs["ghi"] > 120).resample("D").sum().mean())

    result = {"ghi_clear_annual": ghi_clear_annual, "sunshine_hours": sunshine_hours}
    _pvlib_cache[key] = result
    return result


def _ensure_spatial_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Generate required spatial columns for expanded datasets before training."""
    changed = False

    # Derive azimuth from orientation_score if missing: score ∈ [0,1] → azimuth ∈ [0°,180°]
    if "azimuth" not in df.columns and "orientation_score" in df.columns:
        score = pd.to_numeric(df["orientation_score"], errors="coerce").clip(0.0, 1.0)
        df["azimuth"] = 180.0 - np.degrees(np.arccos((2.0 * score) - 1.0))
        changed = True

    # Convert annual GHI from kWh/m²/day to J/m²/day
    if "GHI_mean_J" not in df.columns and "GHI_mean_2024" in df.columns:
        df["GHI_mean_J"] = pd.to_numeric(df["GHI_mean_2024"], errors="coerce") * KWH_TO_J
        changed = True

    orientation = pd.to_numeric(df["orientation_score"], errors="coerce").clip(0.0, 1.0)
    shading = pd.to_numeric(df["shading_factor"], errors="coerce").clip(0.0, 1.0)
    # Adjustment formula: better-oriented rooftops gain up to +3%; shading reduces up to -10%
    adjustment = (0.97 + 0.03 * orientation - 0.10 * shading).clip(0.85, 1.05)
    expected_effective = pd.to_numeric(df["GHI_mean_J"], errors="coerce") * adjustment

    # Recompute effective_GHI_J if absent, contains NaN, or diverges from the formula
    needs_effective = "effective_GHI_J" not in df.columns or df["effective_GHI_J"].isna().any()
    if not needs_effective:
        existing_effective = pd.to_numeric(df["effective_GHI_J"], errors="coerce")
        needs_effective = not np.allclose(
            existing_effective.to_numpy(),
            expected_effective.to_numpy(),
            rtol=1e-6,
            atol=1e-3,
            equal_nan=False,
        )
    if needs_effective:
        df["effective_GHI_J"] = expected_effective
        changed = True

    df["Target_eff_J"] = df["effective_GHI_J"]  # convenience alias for downstream code
    return df, changed


# =============================================================================
# DATA LOADING — SPATIAL DATASET (3,000 points)
# =============================================================================
def load_dataset() -> pd.DataFrame:
    """
    Load integrated_dataset.csv and compute all features for both models.

    Fix 1: effective_GHI_J used for BOTH models.
    Fix 2: Baseline and FI-AdaBoost use the SAME 8-feature vector.
    Fix 3: clear_sky_ratio and sunshine_hours computed per spatial point.
    Fix 6: No log transforms — raw shading_factor and SEI_norm used.
    """
    path = os.path.join(PROCESSED_DIR, "integrated_dataset.csv")
    df   = pd.read_csv(path)

    # Convert annual mean GHI from kWh to joules so all energy values share the same unit
    df["GHI_mean_J"] = pd.to_numeric(df["GHI_mean_2024"], errors="coerce") * KWH_TO_J
    df, generated_features = _ensure_spatial_columns(df)
    if generated_features:
        print("  Generated missing spatial target columns for expanded dataset.", flush=True)

    # Check whether pvlib-derived features need to be (re-)computed
    needs_clearsky = (
        "clear_sky_ratio" not in df.columns
        or "sunshine_hours" not in df.columns
        or df["clear_sky_ratio"].isna().any()
        or df["sunshine_hours"].isna().any()
    )

    if needs_clearsky:
        # Clear-sky/sunshine features only.
        print("  Computing clear_sky_ratio and sunshine_hours ...", flush=True)
        pvlib_feats = df.apply(
            lambda r: pd.Series(_pvlib_features(r["lat"], r["lon"])), axis=1
        )
        df["sunshine_hours"] = pvlib_feats["sunshine_hours"]
        # Ratio of actual GHI to theoretical clear-sky GHI; values >1 are capped at 1.5
        df["clear_sky_ratio"] = (
            pd.to_numeric(df["GHI_mean_2024"], errors="coerce") / pvlib_feats["ghi_clear_annual"]
        ).clip(0.0, 1.5)
    else:
        print("  Using pre-computed clear_sky_ratio and sunshine_hours.", flush=True)

    # Persist any newly generated columns back to CSV so subsequent runs skip recomputation
    if generated_features or needs_clearsky:
        df.to_csv(path, index=False)
        print("  Saved generated features back to CSV.", flush=True)

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    return df



def random_split(df: pd.DataFrame, test_size: float = 0.20):
    # Spatial data has no temporal ordering, so a random shuffle split is appropriate
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, random_state=RANDOM_SEED, shuffle=True
    )
    return df.iloc[idx_train].copy(), df.iloc[idx_test].copy()


# =============================================================================
# MODELS
# =============================================================================
class BaselineAdaBoost:
    # Thin wrapper around sklearn's AdaBoostRegressor — no algorithmic changes
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


# Objective 1.2.2.2: To optimize the AdaBoost Regression algorithm using a
# feature-importance-aware weighting mechanism for enhanced learning efficiency.
class FIAdaBoostRegressor:
    def __init__(self, n_estimators=141, learning_rate=0.63, max_depth=4,
                 random_state=RANDOM_SEED):
        self.n_estimators         = n_estimators
        self.learning_rate        = learning_rate
        self.max_depth            = max_depth
        self.random_state         = random_state
        self.estimators_          = []    # fitted DecisionTreeRegressor objects
        self.estimator_weights_   = []    # log-odds confidence weight for each tree
        self.feature_importances_ = None  # averaged phi across all valid rounds
        self.fallback_prediction_ = 0.0   # mean(y) used if no tree converges

    @staticmethod
    def _norm_fi(tree):
        # Normalize raw Gini importances so they sum to 1 (probability distribution over features)
        raw = tree.feature_importances_
        s   = raw.sum()
        return raw / s if s > 0 else np.ones_like(raw) / len(raw)  # uniform fallback if all zero

    @staticmethod
    def _composite_phi(X, phi):
        # Compute a per-sample importance score Φᵢ ∈ [0, 1]
        # Φᵢ is high when a sample has large values in the most important features,
        # meaning the model should have "seen" those features clearly.
        X_abs  = np.abs(X)
        col_mx = X_abs.max(axis=0)       # column-wise maximum for scaling
        col_mx[col_mx == 0] = 1          # avoid division by zero for constant features
        Phi    = (X_abs / col_mx * phi).sum(axis=1)  # weighted sum across features per sample
        p_max  = Phi.max()
        return Phi / p_max if p_max > 0 else Phi     # normalize to [0, 1]

    def fit(self, X, y):
        X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        self.fallback_prediction_ = float(np.mean(y_arr))  # safe default if boosting fails
        n     = len(y_arr)
        rng   = np.random.default_rng(self.random_state)
        weights = np.full(n, 1.0 / n)   # start with uniform sample weights
        cum_fi  = np.zeros(X_arr.shape[1])  # accumulates phi across rounds for final importances
        n_valid = 0                          # counts rounds that produced a usable tree

        for t in range(self.n_estimators):
            # Weighted bootstrap: draw n samples with replacement, biased toward hard samples
            idx  = rng.choice(n, size=n, replace=True, p=weights)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, random_state=self.random_state + t
            )
            tree.fit(X_arr[idx], y_arr[idx])   # train on the weighted bootstrap sample
            y_pred = tree.predict(X_arr)        # predict on the full training set
            abs_e  = np.abs(y_arr - y_pred)     # absolute residuals for all samples
            D_t    = abs_e.max()                # max error used to normalize residuals to [0, 1]
            if D_t == 0:
                break  # perfect fit — no further improvement possible

            e_i   = abs_e / D_t                          # normalized per-sample error ∈ [0, 1]
            phi   = self._norm_fi(tree)                  # feature importance vector for this tree
            Phi_i = self._composite_phi(X_arr, phi)      # per-sample FI engagement score ∈ [0, 1]

            # FI-aware loss: amplify errors on samples that involve important features
            modulated_loss = e_i * Phi_i
            # Weighted average modulated error — epsilon_t for this round
            eps_t = float(np.dot(weights, modulated_loss))

            if eps_t >= 0.5:
                break  # tree performs worse than random on important samples; discard it

            # beta_t ∈ (0, 1): smaller = more accurate tree, receives lower weight in update
            beta_t = eps_t / (1.0 - eps_t + 1e-10)
            phi    = self._norm_fi(tree)
            Phi_i  = self._composite_phi(X_arr, phi)
            # Weight update: samples with low FI-modulated error get down-weighted (already learned);
            # samples with high FI-modulated error get up-weighted (focus next round on them)
            new_w  = weights * (beta_t ** (1.0 - e_i * Phi_i))
            Z_t    = new_w.sum()   # normalization constant
            if Z_t == 0:
                break  # degenerate weights — stop early
            weights = new_w / Z_t  # re-normalize so weights form a valid probability distribution

            # Estimator weight: log-odds of accuracy; higher confidence trees vote more
            est_w = max(
                self.learning_rate * math.log((1.0 - eps_t) / (eps_t + 1e-10)),
                1e-10,  # floor to prevent zero or negative weights
            )
            self.estimators_.append(tree)
            self.estimator_weights_.append(est_w)
            cum_fi  += phi   # accumulate feature importances for final average
            n_valid += 1

        # Average feature importances across all valid boosting rounds
        self.feature_importances_ = (
            cum_fi / n_valid if n_valid > 0
            else np.ones(X_arr.shape[1]) / X_arr.shape[1]  # uniform fallback
        )
        return self

    def predict(self, X):
        X_arr   = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        if not self.estimators_:
            # No trees converged — return training mean for all samples
            return np.full(X_arr.shape[0], self.fallback_prediction_, dtype=float)
        preds   = np.array([e.predict(X_arr) for e in self.estimators_])  # shape: (n_trees, n_samples)
        weights = np.array(self.estimator_weights_)
        if weights.sum() <= 0:
            return np.full(X_arr.shape[0], self.fallback_prediction_, dtype=float)
        weights = weights / weights.sum()  # normalize voting weights to sum to 1
        result  = np.zeros(X_arr.shape[0])
        for i in range(X_arr.shape[0]):
            p_i   = preds[:, i]            # all trees' predictions for sample i
            order = np.argsort(p_i)        # sort predictions ascending
            cumw  = np.cumsum(weights[order])  # cumulative weight in sorted order
            # Find the weighted median: first index where cumulative weight crosses 0.5
            mid   = np.searchsorted(cumw, 0.5)
            result[i] = p_i[order[min(mid, len(p_i) - 1)]]
        return result


# How FIAdaBoostRegressor works: The model trains a sequence of shallow decision trees
# in a boosted ensemble, where each round begins by drawing a weighted bootstrap sample
# so that previously mishandled data points are seen more often. After each tree is fit,
# its Gini-based feature importances are normalized into a probability vector (phi), which
# is then used to compute a per-sample engagement score (Phi_i) — a measure of how much
# each sample "activates" the features that matter most. The raw prediction error for each
# sample is multiplied by this score to produce a feature-importance-aware (FI-modulated)
# loss, meaning errors on samples that strongly involve important features are penalized
# more heavily. This modulated loss drives both the weighted error rate (epsilon_t) used
# to assess the tree's quality and the sample weight update rule, which up-weights samples
# where the model failed on feature-critical regions. Trees that achieve a low modulated
# error rate receive a high estimator weight (log-odds), while trees that perform no better
# than chance are discarded. At prediction time, each tree casts a weighted vote and the
# final output is the weighted median across all trees — a choice that is more robust to
# outlier weak learners than a simple weighted mean.


# =============================================================================
# EVALUATION METRICS
# =============================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))  # penalizes large errors more than MAE
    mae  = float(mean_absolute_error(y_true, y_pred))          # average absolute error
    r2   = float(r2_score(y_true, y_pred))                     # proportion of variance explained
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
    # Search space for standard AdaBoost; max_depth capped at 5 (shallower than FI variant)
    return {
        "n_estimators":  trial.suggest_int("n_estimators",  50,   300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0, log=True),  # log scale explores small values more
        "max_depth":     trial.suggest_int("max_depth",      1,     5),
    }


def _fi_space(trial) -> dict:
    # FI-AdaBoost allows max_depth=6 since the FI weighting adds regularization pressure
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
        params = space_fn(trial)      # sample a hyperparameter configuration
        rmses  = []
        for tr_idx, val_idx in kf.split(X_train):
            m = model_cls(**params).fit(X_train[tr_idx], y_train[tr_idx])
            p = m.predict(X_train[val_idx])
            rmses.append(np.sqrt(mean_squared_error(y_train[val_idx], p)))
        return np.mean(rmses)         # minimize average RMSE across folds

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),  # TPE = Tree-structured Parzen Estimator
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _tune_timeseries(model_cls, space_fn, X_train: np.ndarray, y_train: np.ndarray,
                     n_trials: int = 100, n_splits: int = 5) -> dict:
    """Optuna + TimeSeriesSplit tuning for daily dataset. Returns best_params."""
    tscv = TimeSeriesSplit(n_splits=n_splits)  # respects temporal order — no future data leaks into training

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
    """KFold CV with tuned hyperparams — saves per-fold Train vs Val RMSE, MAE, and R²."""
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
            "ada_train_MAE_J":  ada_train_m["MAE_J"],
            "ada_val_RMSE_J":   ada_val_m["RMSE_J"],
            "ada_val_MAE_J":    ada_val_m["MAE_J"],
            "ada_train_R2":     ada_train_m["R2"],
            "ada_val_R2":       ada_val_m["R2"],
            "fi_train_RMSE_J":  fi_train_m["RMSE_J"],
            "fi_train_MAE_J":   fi_train_m["MAE_J"],
            "fi_val_RMSE_J":    fi_val_m["RMSE_J"],
            "fi_val_MAE_J":     fi_val_m["MAE_J"],
            "fi_train_R2":      fi_train_m["R2"],
            "fi_val_R2":        fi_val_m["R2"],
        })
    return pd.DataFrame(rows)


def save_cv_metrics(cv_df: pd.DataFrame) -> str:
    path = os.path.join(RESULTS_DIR, "cv_fold_metrics.csv")
    cv_df.to_csv(path, index=False)
    return path


def _hyperparameter_row(
    *,
    pipeline: str,
    model: str,
    params: dict,
    target: str,
    features: list,
    cv_method: str,
    n_trials: int = 100,
    n_splits: int = 5,
) -> dict:
    """Build one reproducibility row for the tuned hyperparameters CSV."""
    return {
        "pipeline":         pipeline,
        "model":            model,
        "tuning_method":    "Optuna TPE",
        "cv_method":        cv_method,
        "n_trials":         n_trials,
        "n_splits":         n_splits,
        "objective_metric": "mean_validation_RMSE_J",
        "target":           target,
        "target_unit":      "J/m2/day",
        "features":         ",".join(features),
        "random_seed":      RANDOM_SEED,
        "n_estimators":     params.get("n_estimators"),
        "learning_rate":    params.get("learning_rate"),
        "max_depth":        params.get("max_depth"),
    }


def save_hyperparameters_csv(rows: list[dict], append: bool = False) -> str:
    path = os.path.join(RESULTS_DIR, "hyperparameters_used.csv")
    df = pd.DataFrame(rows)
    if append and os.path.exists(path):
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(path, index=False)
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
    d     = e1**2 - e2**2  # loss differential: positive when baseline error > FI error
    n     = len(d)
    d_bar = d.mean()                                               # mean loss differential
    g0    = ((d - d_bar)**2).sum() / (n - 1)                      # lag-0 autocovariance
    g1    = ((d[1:] - d_bar) * (d[:-1] - d_bar)).sum() / (n - 1) # lag-1 autocovariance (Newey-West HAC)
    sigma = max((g0 + 2 * g1) / n, 1e-20)                         # HAC variance; floor avoids division by zero
    dm    = d_bar / np.sqrt(sigma)                                 # DM test statistic ~ N(0,1) under H₀
    pval  = float(2 * (1 - stats.norm.cdf(abs(dm))))               # two-sided p-value
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
        "mean_loss_diff_J2":   round(float(d_bar), 2),  # average squared-error advantage of FI model
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
    y_pred_J = model.predict(X_all)                        # predicted effective GHI in J/m²/day
    H_daily  = y_pred_J / KWH_TO_J                        # convert to kWh/m²/day for SEP formula
    result   = df[["lat", "lon", "rooftop_area_sq_m"]].copy()
    result[f"{label}_predicted_irradiance_kWh_m2_day"] = H_daily
    # SEP (Solar Energy Potential) = irradiance × rooftop area; daily then annualized
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
    df.replace(-999, np.nan, inplace=True)   # NASA POWER uses -999 as a missing-value sentinel
    df.interpolate(method="linear", inplace=True)  # fill isolated gaps with linear interpolation
    df = df.dropna(
        subset=["ALLSKY_SFC_SW_DWN", "T2M", "RH2M", "ALLSKY_KT"]
    ).reset_index(drop=True)

    # §2.2.1 Temporal features — cyclical encoding
    df["month"]     = df["date"].dt.month
    # Sine/cosine encoding preserves the circular continuity of months (Dec → Jan)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    # Dry season (1=Dec–May) vs wet season (0=Jun–Nov) for Davao City climate
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
    # Values >1 are possible (diffuse enhancement); cap at 1.5 to remove extreme outliers
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
        # City-wide averages serve as a single daily representative value
        df["mean_orientation_score"] = float(osm["orientation_score"].mean())
        df["mean_shading_factor"]    = float(osm["shading_factor"].mean())
        df["mean_SEI_norm"]          = float(osm["SEI_norm"].mean())
    else:
        # Sensible defaults when OSM file is unavailable (e.g., first run before data_integration)
        df["mean_orientation_score"] = 0.5
        df["mean_shading_factor"]    = 0.05
        df["mean_SEI_norm"]          = 0.1

    df["GHI_J"] = df["ALLSKY_SFC_SW_DWN"] * KWH_TO_J  # target: daily GHI in joules
    df = df.sort_values("date").reset_index(drop=True)  # ensure chronological order before temporal split

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
    # Slice rather than shuffle: earlier dates train, later dates test (no future leakage)
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

    p_hyper = save_hyperparameters_csv(
        [
            _hyperparameter_row(
                pipeline="daily_temporal",
                model="AdaBoost (Daily, Temporal Split)",
                params=ada_params,
                target=DAILY_TARGET,
                features=DAILY_FEATURES,
                cv_method="TimeSeriesSplit",
            ),
            _hyperparameter_row(
                pipeline="daily_temporal",
                model="FI-AdaBoost (Daily, Temporal Split)",
                params=fi_params,
                target=DAILY_TARGET,
                features=DAILY_FEATURES,
                cv_method="TimeSeriesSplit",
            ),
        ],
        append=True,
    )

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
    print(f"  Saved: {p_hyper}")


# =============================================================================
# RESULT EXPORTS
# =============================================================================
def save_metrics_csv(ada_m: dict, fi_m: dict) -> str:
    rows = [
        {"model": "AdaBoost (Baseline)",    "target": "Effective GHI - effective_GHI_J (J/m2/day)", **ada_m},
        {"model": "FI-AdaBoost (Proposed)", "target": "Effective GHI - effective_GHI_J (J/m2/day)", **fi_m},
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
    # Positive difference = baseline predicts more energy than FI model for that building
    merged["difference_SEP_kWh_yr"] = (
        merged["ada_SEP_kWh_yr"] - merged["fi_SEP_kWh_yr"]
    )
    path = os.path.join(RESULTS_DIR, "forecast_per_building.csv")
    merged.to_csv(path, index=False)
    return path


def save_feature_importance_csv(uniform_baseline_vals, actual_baseline_vals, fi_vals, feats) -> str:
    """Save exact feature-importance values used by the feature-importance PNGs."""
    rows = []
    for feat, uniform_val, actual_val, fi_val in zip(
        feats, uniform_baseline_vals, actual_baseline_vals, fi_vals
    ):
        rows.append({
            "feature":                            feat,
            "baseline_source":                    "uniform_no_feature_weighting_reference",
            "baseline_importance_weight":         float(uniform_val),
            "baseline_importance_percent":        float(uniform_val) * 100,
            "actual_baseline_source":             "ada_feature_importances_",
            "actual_baseline_importance_weight":  float(actual_val),
            "actual_baseline_importance_percent": float(actual_val) * 100,
            "fi_source":                          "fi_adaboost_feature_importances_",
            "fi_importance_weight":               float(fi_val),
            "fi_importance_percent":              float(fi_val) * 100,
        })

    df = pd.DataFrame(rows)
    df["actual_baseline_rank"] = df["actual_baseline_importance_weight"].rank(
        ascending=False, method="min"
    ).astype(int)
    df["fi_rank"] = df["fi_importance_weight"].rank(
        ascending=False, method="min"
    ).astype(int)
    path = os.path.join(RESULTS_DIR, "feature_importance_comparison.csv")
    df.to_csv(path, index=False, float_format="%.12g")
    return path


def save_feature_weight_csv(fi_vals, feats) -> str:
    """Save the exact FI-AdaBoost feature weights used as the weighting proof artifact."""
    order = np.argsort(fi_vals)[::-1]
    rows = []
    for rank, idx in enumerate(order, start=1):
        weight = float(fi_vals[idx])
        rows.append({
            "rank": rank,
            "feature": str(feats[idx]),
            "fi_feature_weight": weight,
            "fi_feature_weight_percent": weight * 100,
            "source": "fi_adaboost_feature_importances_",
        })

    path = os.path.join(RESULTS_DIR, "feature_weight_importance.csv")
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.12g")
    return path


# =============================================================================
# PLOTS
# =============================================================================
def plot_standalone_feature_importance(fi_vals, feats):
    plt.figure(figsize=(10, 6))
    labels     = [f.replace("_", " ").title() for f in feats]
    idx        = np.argsort(fi_vals)     # sort ascending so the longest bar is at the top
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


def plot_feature_weight_importance(fi_vals, feats):
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [f.replace("_", " ").title() for f in feats]
    idx = np.argsort(fi_vals)
    sv = fi_vals[idx]
    sl = np.array(labels)[idx]
    bars = ax.barh(sl, sv, color=C_FI, edgecolor="black", alpha=0.88)
    for bar, val in zip(bars, sv):
        ax.text(
            val + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val * 100:.4f}%",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title(
        "FI-AdaBoost Feature Weight Importance\nProof of Feature-Aware Weighting",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Feature Weight Used by FI-AdaBoost", fontsize=12)
    ax.set_xlim(0, max(sv) + 0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "feature_weight_importance.png"), dpi=300)
    plt.close(fig)


def plot_standalone_baseline_feature_importance(ada_vals, feats):
    plt.figure(figsize=(10, 6))
    labels     = [f.replace("_", " ").title() for f in feats]
    idx        = np.argsort(ada_vals)    # sort ascending so the longest bar is at the top
    sv         = ada_vals[idx]
    sl         = np.array(labels)[idx]
    bars       = plt.barh(sl, sv, color=C_ADA, edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, sv):
        plt.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{val * 100:.4f}%", va="center", ha="left", fontsize=10, fontweight="bold")
    plt.title("Baseline AdaBoost Feature Importance\n(No Feature Weighting — Uniform)", fontsize=14, fontweight="bold")
    plt.xlabel("Relative Importance Weight", fontsize=12)
    plt.xlim(0, max(sv) + 0.1)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "baseline_feature_importances.png"), dpi=300)
    plt.close()


def plot_actual_baseline_feature_importance(ada_vals, feats):
    plt.figure(figsize=(10, 6))
    labels     = [f.replace("_", " ").title() for f in feats]
    idx        = np.argsort(ada_vals)    # sort ascending so the longest bar is at the top
    sv         = ada_vals[idx]
    sl         = np.array(labels)[idx]
    bars       = plt.barh(sl, sv, color=C_ADA, edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, sv):
        plt.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{val * 100:.4f}%", va="center", ha="left", fontsize=10, fontweight="bold")
    plt.title("Actual Baseline AdaBoost Feature Importance\nGini-Based from Fitted AdaBoost Ensemble",
              fontsize=14, fontweight="bold")
    plt.xlabel("Relative Importance Weight", fontsize=12)
    plt.xlim(0, max(sv) + 0.1)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "actual_baseline_feature_importances.png"), dpi=300)
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
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.2, label="Perfect fit")  # 45° line = ideal predictions
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
        ax.axhline(0, color="black", linewidth=1.2, linestyle="--")  # zero-residual reference line
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
   for ax, label, key, av, fv in zip(axes, labels, metrics, ada_vals, fi_vals):
       bars = ax.bar(["AdaBoost\n(Baseline)", "FI-AdaBoost\n(Proposed)"], [av, fv],
                     color=[C_ADA, C_FI], edgecolor="black", alpha=0.85, width=0.5)
       for bar, val in zip(bars, [av, fv]):
           fmt = f"{val:.6f}" if key == "R2" else f"{val:,.2f}"  # more decimal places for R²
           ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                   fmt, ha="center", va="bottom", fontsize=10, fontweight="bold")
       ax.set_title(label, fontsize=12, fontweight="bold")
       ax.set_ylim(0, max(av, fv) * 1.18)  # 18% headroom so value labels don't overflow
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
                   label=f"Mean: {data.mean():,.0f}")  # vertical line at city-wide average SEP
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
    ada_kwh = ada_fcast["ada_SEP_kWh_yr"].sum()   # city-wide total for baseline
    fi_kwh  = fi_fcast["fi_SEP_kWh_yr"].sum()     # city-wide total for proposed model
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
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))  # comma-formatted y-axis
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
    width = 0.35  # bar width; two bars per model side by side

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width / 2, train_rmse, width, label="Train RMSE", color=C_ADA, edgecolor="black", alpha=0.85)
    rects2 = ax.bar(x + width / 2, test_rmse,  width, label="Test RMSE",  color=C_FI,  edgecolor="black", alpha=0.85)

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
    print(f"  Target: {TARGET_COL} ({df[TARGET_COL].nunique()} unique values)")
    print(f"  Baseline features ({len(BASELINE_FEATURES)}): {BASELINE_FEATURES}")
    print(f"  FI-AdaBoost features ({len(FI_FEATURES)}): {FI_FEATURES}")

    # ── [2] Split ──────────────────────────────────────────────────────────────
    print("\n[2/6] 80/20 Random Split …")
    train_df, test_df = random_split(df)
    print(f"  Total: {len(df)}  |  Train: {len(train_df)}  |  Test: {len(test_df)}")

    # Log split metadata for reproducibility and thesis appendix
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

    # Extract numpy arrays; both models use identical features for a fair comparison
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

    p_hyper = save_hyperparameters_csv(
        [
            _hyperparameter_row(
                pipeline="spatial_rooftop",
                model="AdaBoost (Baseline)",
                params=ada_params,
                target=TARGET_COL,
                features=BASELINE_FEATURES,
                cv_method="KFold",
            ),
            _hyperparameter_row(
                pipeline="spatial_rooftop",
                model="FI-AdaBoost (Proposed)",
                params=fi_params,
                target=TARGET_COL,
                features=FI_FEATURES,
                cv_method="KFold",
            ),
        ],
        append=False,
    )
    print(f"  Saved tuned hyperparameters: {p_hyper}")

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
    # Re-fit on the entire training set (not just one CV fold) using the tuned hyperparameters
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
    # Warn if train error is less than half the test error — suggests the model memorized training data
    if fi_train_m["RMSE_J"] < (fi_test_m["RMSE_J"] * 0.5):
        print("    WARNING: FI-AdaBoost Train error is less than half the Test error. Significant overfitting likely.")

    # Diebold–Mariano test (Fix 5)
    dm = diebold_mariano_test(y_te - ada_te, y_te - fi_te)

    print("\n  Baseline AdaBoost Feature Importances:")
    for feat, imp in sorted(zip(SHARED_FEATURES, ada.feature_importances_),
                            key=lambda x: -x[1]):
        print(f"    {feat:<25}: {imp * 100:.4f}%")

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
    # Apply each trained model to all 3,000 spatial points to produce per-building SEP estimates
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

    uniform_imp = np.ones(len(SHARED_FEATURES)) / len(SHARED_FEATURES)
    p_feature_importance = save_feature_importance_csv(
        uniform_imp, ada.feature_importances_, fi.feature_importances_, SHARED_FEATURES
    )
    p_feature_weight = save_feature_weight_csv(fi.feature_importances_, SHARED_FEATURES)
    plot_standalone_baseline_feature_importance(uniform_imp, SHARED_FEATURES)
    plot_actual_baseline_feature_importance(ada.feature_importances_, SHARED_FEATURES)
    plot_standalone_feature_importance(fi.feature_importances_, SHARED_FEATURES)
    plot_feature_weight_importance(fi.feature_importances_, SHARED_FEATURES)
    plot_actual_vs_predicted(y_te, ada_te, y_te, fi_te)
    plot_residuals(y_te, ada_te, y_te, fi_te)
    plot_metrics_comparison(ada_test_m, fi_test_m)
    plot_energy_distribution(ada_fcast, fi_fcast)
    plot_total_energy_comparison(ada_fcast, fi_fcast)
    plot_overfit_check(ada_train_m, ada_test_m, fi_train_m, fi_test_m)

    saved = [
        ("CSV",   os.path.join(RESULTS_DIR, "train_test_split_info.csv")),
        ("CSV",   p_hyper),
        ("CSV",   p_metrics),
        ("CSV",   p_forecast),
        ("CSV",   p_dm),
        ("CSV",   p_feature_importance),
        ("CSV",   p_feature_weight),
        ("Plot",  os.path.join(RESULTS_DIR, "baseline_feature_importances.png")),
        ("Plot",  os.path.join(RESULTS_DIR, "actual_baseline_feature_importances.png")),
        ("Plot",  os.path.join(RESULTS_DIR, "standalone_feature_importances.png")),
        ("Plot",  os.path.join(RESULTS_DIR, "feature_weight_importance.png")),
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
