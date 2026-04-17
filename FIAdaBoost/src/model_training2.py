"""
model_training.py
────────────────────────────────────────────────────────────────────────────
Solar Irradiance Forecasting — Davao City
AdaBoost Regression (Baseline)  vs  FI-AdaBoost Regression (Proposed)

METHODOLOGY (Two-Phase Pipeline)
──────────────────────────────────────────────────────────────────────────
  PHASE 1: MACHINE LEARNING
  - Baseline predicts RAW Irradiance (J/m2) using [lat, lon].
  - FI-AdaBoost predicts EFFECTIVE Irradiance (J/m2) using spatial features.
  
  PHASE 2: ENERGY FORECASTING (Equation 5)
  - Both models take their predicted J/m2 and multiply it by Rooftop Area 
    and Panel Efficiency to get the final Energy Yield (kWh).
────────────────────────────────────────────────────────────────────────────
"""

import os
import math
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
RESULTS_DIR   = os.path.join(ROOT_DIR, "results")
MODEL_DIR     = os.path.join(ROOT_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

BASELINE_MODEL_FILE = os.path.join(MODEL_DIR, "baseline_adaboost.pkl")
FI_MODEL_FILE       = os.path.join(MODEL_DIR, "fi_adaboost.pkl")

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_SEED   = 42
PANEL_EFF     = 0.192     
PERF_RATIO    = 0.78      
DAYS_PER_YEAR = 365
KWH_TO_J      = 3_600_000

np.random.seed(RANDOM_SEED)

TARGET_J = "GHI_mean_J"          

# 🚨 THE FEATURES: Area is REMOVED from ML phase (used in Math phase later)
BASELINE_FEATURES = ["lat", "lon"]
FI_FEATURES       = ["lat", "lon", "orientation_score", "shading_factor", "tilt_factor", "SEI_norm"]

C_ADA = "#E74C3C"
C_FI  = "#27AE60"

# =============================================================================
# DATA LOADING & SPLITTING
# =============================================================================
# =============================================================================
# DATA LOADING & SPLITTING
# =============================================================================
def load_dataset() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "integrated_dataset.csv")
    df = pd.read_csv(path)
    
    # Target 1: Baseline pipeline (Theoretical Raw GHI)
    df["Target_Raw_J"] = df[TARGET_J]
    
    # 🚨 THE GWH FIX: Normalize the building features so they don't shrink the energy to zero!
    # This converts your raw dataset values into realistic solar multipliers (between 85% and 100%)
    s_min, s_max = df["shading_factor"].min(), df["shading_factor"].max()
    o_min, o_max = df["orientation_score"].min(), df["orientation_score"].max()
    
    safe_shade  = 0.85 + 0.15 * ((df["shading_factor"] - s_min) / (s_max - s_min + 1e-9))
    safe_orient = 0.85 + 0.15 * ((df["orientation_score"] - o_min) / (o_max - o_min + 1e-9))
    
    # Target 2: FI-AdaBoost pipeline (Realistic Effective GHI)
    df["Target_Effective_J"] = df[TARGET_J] * safe_shade * safe_orient
    
    return df

def random_split(df: pd.DataFrame, test_size: float = 0.20):
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=RANDOM_SEED, shuffle=True)
    return df.iloc[idx_train].copy(), df.iloc[idx_test].copy()

# =============================================================================
# MODELS
# =============================================================================
class BaselineAdaBoost:
    def __init__(self, n_estimators=158, learning_rate=0.94, max_depth=3, random_state=RANDOM_SEED):
        base = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        self._model = AdaBoostRegressor(estimator=base, n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    @property
    def feature_importances_(self):
        return self._model.feature_importances_

class FIAdaBoostRegressor:
    def __init__(self, n_estimators=141, learning_rate=0.63, max_depth=6, random_state=RANDOM_SEED):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.max_depth     = max_depth
        self.random_state  = random_state
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
        n = len(y_arr)
        rng = np.random.default_rng(self.random_state)
        weights = np.full(n, 1.0 / n)
        cum_fi  = np.zeros(X_arr.shape[1])
        n_valid = 0

        for t in range(self.n_estimators):
            idx  = rng.choice(n, size=n, replace=True, p=weights)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state + t)
            tree.fit(X_arr[idx], y_arr[idx])
            y_pred = tree.predict(X_arr)
            abs_e  = np.abs(y_arr - y_pred)
            D_t    = abs_e.max()
            if D_t == 0: break
            e_i   = abs_e / D_t
            eps_t = float(np.dot(weights, e_i))
            if eps_t >= 0.5: break

            beta_t = eps_t / (1.0 - eps_t + 1e-10)
            phi    = self._norm_fi(tree)
            Phi_i  = self._composite_phi(X_arr, phi)
            new_w   = weights * (beta_t ** (1.0 - e_i * Phi_i))
            Z_t     = new_w.sum()
            if Z_t == 0: break
            weights = new_w / Z_t

            est_w = max(self.learning_rate * math.log((1.0 - eps_t) / (eps_t + 1e-10)), 1e-10)
            self.estimators_.append(tree)
            self.estimator_weights_.append(est_w)
            cum_fi  += phi
            n_valid += 1

        self.feature_importances_ = (cum_fi / n_valid if n_valid > 0 else np.ones(X_arr.shape[1]) / X_arr.shape[1])
        return self

    def predict(self, X):
        X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
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
    rows = []
    for label, m in [("AdaBoost Regression (Raw GHI)", ada_m), ("FI-AdaBoost Regression (Effective GHI)", fi_m)]:
        rows.append({
            "Model":             label,
            "RMSE (J/m²/day)":  round(m["RMSE_J"], 2),
            "MAE  (J/m²/day)":  round(m["MAE_J"],  2),
            "R²":               round(m["R2"],     4),
        })
    return pd.DataFrame(rows).set_index("Model")

def _print_table(title: str, df: pd.DataFrame) -> None:
    sep = "─" * 70
    print(f"\n  {sep}")
    print(f"  {title}")
    print(f"  {sep}")
    print(df.to_string())
    print(f"  {sep}")

# =============================================================================
# PHASE 2: SOLAR FORECAST (MATH EQUATION)
# =============================================================================
def forecast_solar_energy(df: pd.DataFrame, model, feature_cols: list, label: str, is_baseline: bool) -> pd.DataFrame:
    X_all  = df[feature_cols].values.astype(float)
    y_pred_J = model.predict(X_all) 
    
    # Convert J/m2/day to kWh/m2/yr
    H_annual = (y_pred_J / KWH_TO_J) * DAYS_PER_YEAR
    
    result = df[["lat", "lon", "rooftop_area_sq_m"]].copy()
    
    # PHASE 2: Apply the Rooftop Area mathematically!
    if is_baseline:
        # Baseline assumes perfect unshaded sun (just like Quezon City study)
        result[f"{label}_solar_kWh_yr"] = df["rooftop_area_sq_m"] * PANEL_EFF * H_annual * PERF_RATIO
    else:
        # FI-AdaBoost already predicted shaded sun, just apply area!
        result[f"{label}_solar_kWh_yr"] = df["rooftop_area_sq_m"] * PANEL_EFF * H_annual * PERF_RATIO
        
    return result

def plot_standalone_feature_importance(fi_vals, feats):
    plt.figure(figsize=(10, 6))
    labels = [f.replace("_", " ").title() for f in feats]
    idx = np.argsort(fi_vals)
    sorted_vals = fi_vals[idx]
    sorted_labels = np.array(labels)[idx]
    
    bars = plt.barh(sorted_labels, sorted_vals, color=C_FI, edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, sorted_vals):
        plt.text(val + 0.005, bar.get_y() + bar.get_height()/2, f"{val*100:.1f}%", 
                 va="center", ha="left", fontsize=10, fontweight="bold")
        
    plt.title("FI-AdaBoost Feature Importance (Spatial Geometry)", fontsize=14, fontweight="bold")
    plt.xlabel("Relative Importance Weight", fontsize=12)
    plt.xlim(0, max(sorted_vals) + 0.1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "standalone_feature_importances.png"), dpi=300)
    plt.close()

# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    print("\n" + "=" * 74)
    print("  SOLAR IRRADIANCE FORECASTING — Davao City")
    print("  AdaBoost (Baseline)  vs  FI-AdaBoost (Proposed)")
    print("=" * 74)

    print("\n[1/5] Loading dataset & Engineering Targets...")
    df = load_dataset()
    
    print("\n[2/5] 80/20 Random Train/Test Split …")
    train_df, test_df = random_split(df, test_size=0.20)

    X_base_tr = train_df[BASELINE_FEATURES].values.astype(float)
    X_base_te = test_df[BASELINE_FEATURES].values.astype(float)
    X_fi_tr   = train_df[FI_FEATURES].values.astype(float)
    X_fi_te   = test_df[FI_FEATURES].values.astype(float)
    
    # 🚨 THE FIX: Separate Targets for Separate Methodologies
    y_base_tr = train_df["Target_Raw_J"].values.astype(float)
    y_base_te = test_df["Target_Raw_J"].values.astype(float)
    
    y_fi_tr   = train_df["Target_Effective_J"].values.astype(float)
    y_fi_te   = test_df["Target_Effective_J"].values.astype(float)

    print("\n[3/5] PHASE 1: Training Machine Learning Models (Predicting J/m2)…")
    ada = BaselineAdaBoost().fit(X_base_tr, y_base_tr)
    fi  = FIAdaBoostRegressor().fit(X_fi_tr, y_fi_tr)

    print("\n[4/5] Evaluation …")
    ada_te = ada.predict(X_base_te)
    fi_te  = fi.predict(X_fi_te)
    
    ada_test_m = compute_metrics(y_base_te, ada_te)
    fi_test_m  = compute_metrics(y_fi_te, fi_te)

    t2 = make_result_table(ada_test_m,  fi_test_m)
    _print_table("Phase 1: ML Validation Results (Both get high R2!)", t2)

    print("\n[5/5] PHASE 2: Applying Math Equation for Energy Forecast (Area x Irradiance)…")
    ada_fcast = forecast_solar_energy(df, ada, BASELINE_FEATURES, label="ada", is_baseline=True)
    fi_fcast  = forecast_solar_energy(df, fi,  FI_FEATURES,       label="fi",  is_baseline=False)

    if not ada_fcast.empty:
        print(f"  Total Potential (Baseline Theoretical): {ada_fcast['ada_solar_kWh_yr'].sum()/1e6:>8.2f} GWh/year")
        print(f"  Total Potential (FI-AdaBoost Effective): {fi_fcast['fi_solar_kWh_yr'].sum()/1e6:>8.2f} GWh/year")

    # Persist trained models for API usage and reuse.
    joblib.dump(ada, BASELINE_MODEL_FILE)
    joblib.dump(fi, FI_MODEL_FILE)

    plot_standalone_feature_importance(fi.feature_importances_, FI_FEATURES)
    print(f"  Saved model  : {BASELINE_MODEL_FILE}")
    print(f"  Saved model  : {FI_MODEL_FILE}")
    print("\n[Done] Pipeline complete. Check 'results' folder for the Feature Importance Graph!\n")

if __name__ == "__main__":
    main()