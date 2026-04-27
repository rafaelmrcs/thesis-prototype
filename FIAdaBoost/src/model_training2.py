"""
model_training.py
────────────────────────────────────────────────────────────────────────────
Solar Irradiance Forecasting — Davao City
AdaBoost Regression (Baseline)  vs  FI-AdaBoost Regression (Proposed)

METHODOLOGY (Two-Phase Pipeline)
──────────────────────────────────────────────────────────────────────────
  PHASE 1: MACHINE LEARNING (fair comparison — same target for both models)
  - Both models predict RAW GHI (J/m²/day).
  - Baseline uses [lat, lon] only.
  - FI-AdaBoost uses [lat, lon, orientation_score, shading_factor, SEI_norm],
    with a novel feature-importance-weighted boosting update (the Φᵢ term).

  PHASE 2: ENERGY FORECASTING (Equation 5)
  - Baseline: theoretical maximum (area × GHI × efficiency × PR).
  - FI-AdaBoost: realistic estimate — applies building-level shading and
    orientation corrections derived from train-set statistics (no leakage).
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
import pvlib

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

TARGET_J = "GHI_mean_J"   # J/m²/day — ML metrics for Baseline reported in J/m²

# Area is excluded from ML phase — applied mathematically in Phase 2.
# tilt_factor is a constant (0.992) across all rows — zero variance, removed.
# FI model adds azimuth + log-transformed sparse features + predicts effective GHI.
BASELINE_FEATURES = ["lat", "lon"]
FI_FEATURES       = ["lat", "lon", "azimuth", "orientation_score", "shading_factor_log", "SEI_norm_log"]

# ── pvlib POA ratio helper ────────────────────────────────────────────────────
_poa_cache: dict[tuple, float] = {}

def _poa_ratio(lat: float, azimuth_deg: float, tilt_deg: float = 10.0) -> float:
    """Annual-average POA/GHI ratio for a surface at given azimuth and tilt."""
    key = (round(lat, 2), round(azimuth_deg, 1))
    if key in _poa_cache:
        return _poa_cache[key]
    location  = pvlib.location.Location(lat, 125.6128, altitude=30, tz="Asia/Manila")
    times     = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h", tz="Asia/Manila")
    solar_pos = location.get_solarposition(times)
    clearsky  = location.get_clearsky(times, model="ineichen")
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

C_ADA = "#E74C3C"
C_FI  = "#27AE60"

# =============================================================================
# DATA LOADING & SPLITTING
# =============================================================================
def load_dataset() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "integrated_dataset.csv")
    df = pd.read_csv(path)

    # Baseline target: raw GHI (lat/lon only model, replicating Quezon City study)
    df["Target_J"] = df[TARGET_J]

    # Log-transform zero-inflated features so tree splits can extract signal
    # (shading_factor=0 for 75% of rows; SEI_norm<0.01 for 92% of rows)
    df["shading_factor_log"] = np.log1p(df["shading_factor"] * 10)
    df["SEI_norm_log"]       = np.log1p(df["SEI_norm"] * 100)

    # FI target: geometry-adjusted effective GHI using pvlib POA ratio per building.
    # This gives real per-building variance (not just 2 discrete NASA grid values)
    # and is consistent with what the live API computes at inference time.
    print("  Computing pvlib POA ratios for effective GHI target …", flush=True)
    df["Target_eff_J"] = df.apply(
        lambda r: r[TARGET_J] * _poa_ratio(r["lat"], r["azimuth"]), axis=1
    )
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
    def __init__(self, n_estimators=141, learning_rate=0.63, max_depth=4, random_state=RANDOM_SEED):
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
    for label, m in [
        ("AdaBoost Regression (Baseline — lat/lon only)", ada_m),
        ("FI-AdaBoost Regression (Proposed — spatial features)", fi_m),
    ]:
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
def forecast_solar_energy(
    df: pd.DataFrame,
    model,
    feature_cols: list,
    label: str,
    is_baseline: bool,
    train_df: pd.DataFrame = None,
) -> pd.DataFrame:
    X_all    = df[feature_cols].values.astype(float)
    y_pred_J = model.predict(X_all)

    H_annual = (y_pred_J / KWH_TO_J) * DAYS_PER_YEAR
    result   = df[["lat", "lon", "rooftop_area_sq_m"]].copy()

    if is_baseline:
        # Baseline: theoretical maximum using raw sky GHI (no geometry correction).
        result[f"{label}_solar_kWh_yr"] = df["rooftop_area_sq_m"] * PANEL_EFF * H_annual * PERF_RATIO
    else:
        # FI-AdaBoost: model predicts effective GHI already corrected by pvlib POA
        # (orientation baked into the training target — no double-correction needed).
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
# RESULT EXPORTS
# =============================================================================
def save_metrics_csv(ada_m: dict, fi_m: dict) -> str:
    rows = [
        {"model": "AdaBoost (Baseline)", "target": "Raw GHI (J/m²/day)", **ada_m},
        {"model": "FI-AdaBoost (Proposed)", "target": "Effective GHI — pvlib POA-adjusted (J/m²/day)", **fi_m},
    ]
    path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_forecast_csv(ada_fcast: pd.DataFrame, fi_fcast: pd.DataFrame) -> str:
    merged = ada_fcast.merge(
        fi_fcast[["lat", "lon", "fi_solar_kWh_yr"]],
        on=["lat", "lon"],
        how="left",
    )
    merged["difference_kWh_yr"] = merged["ada_solar_kWh_yr"] - merged["fi_solar_kWh_yr"]
    path = os.path.join(RESULTS_DIR, "forecast_per_building.csv")
    merged.to_csv(path, index=False)
    return path


def plot_actual_vs_predicted(y_true_ada, y_pred_ada, y_true_fi, y_pred_fi):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, y_true, y_pred, color, title in [
        (axes[0], y_true_ada, y_pred_ada, C_ADA, "AdaBoost Baseline\n(Raw GHI)"),
        (axes[1], y_true_fi,  y_pred_fi,  C_FI,  "FI-AdaBoost Proposed\n(Effective GHI)"),
    ]:
        ax.scatter(y_true, y_pred, alpha=0.4, s=18, color=color, edgecolors="none")
        mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.2, label="Perfect fit")
        r2 = r2_score(y_true, y_pred)
        ax.set_title(f"{title}\nR² = {r2:.4f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Actual (J/m²/day)", fontsize=10)
        ax.set_ylabel("Predicted (J/m²/day)", fontsize=10)
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.suptitle("Actual vs Predicted — Phase 1 ML Validation", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "actual_vs_predicted.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_residuals(y_true_ada, y_pred_ada, y_true_fi, y_pred_fi):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, y_true, y_pred, color, title in [
        (axes[0], y_true_ada, y_pred_ada, C_ADA, "AdaBoost Baseline"),
        (axes[1], y_true_fi,  y_pred_fi,  C_FI,  "FI-AdaBoost Proposed"),
    ]:
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.4, s=18, color=color, edgecolors="none")
        ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
        ax.set_title(f"{title}\nResiduals", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted (J/m²/day)", fontsize=10)
        ax.set_ylabel("Residual (Actual − Predicted)", fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
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
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.suptitle("Model Performance Comparison — Phase 1 ML", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "metrics_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_energy_distribution(ada_fcast: pd.DataFrame, fi_fcast: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, fcast, col, color, title in [
        (axes[0], ada_fcast, "ada_solar_kWh_yr", C_ADA, "AdaBoost Baseline\n(Theoretical)"),
        (axes[1], fi_fcast,  "fi_solar_kWh_yr",  C_FI,  "FI-AdaBoost Proposed\n(Effective)"),
    ]:
        data = fcast[col].dropna()
        ax.hist(data, bins=40, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(), color="black", linestyle="--", linewidth=1.2, label=f"Mean: {data.mean():,.0f}")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Solar Energy Yield (kWh/year)", fontsize=10)
        ax.set_ylabel("Number of Buildings", fontsize=10)
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.suptitle("Per-Building Solar Energy Yield Distribution — Phase 2 Forecast", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "energy_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_total_energy_comparison(ada_fcast: pd.DataFrame, fi_fcast: pd.DataFrame):
    ada_kwh = ada_fcast["ada_solar_kWh_yr"].sum()
    fi_kwh  = fi_fcast["fi_solar_kWh_yr"].sum()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["AdaBoost\n(Theoretical)", "FI-AdaBoost\n(Effective)"],
                  [ada_kwh, fi_kwh], color=[C_ADA, C_FI], edgecolor="black", alpha=0.85, width=0.45)
    for bar, val in zip(bars, [ada_kwh, fi_kwh]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val:,.0f} kWh", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Solar Energy Yield (kWh/year)", fontsize=11)
    ax.set_title("Total Rooftop Solar Potential — Davao City\nPhase 2 Energy Forecast Comparison",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(ada_kwh, fi_kwh) * 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "total_energy_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    print("\n" + "=" * 74)
    print("  SOLAR IRRADIANCE FORECASTING — Davao City")
    print("  AdaBoost (Baseline)  vs  FI-AdaBoost (Proposed)")
    print("=" * 74)

    print("\n[1/5] Loading dataset …")
    df = load_dataset()

    print("\n[2/5] 80/20 Random Train/Test Split …")
    train_df, test_df = random_split(df, test_size=0.20)

    X_base_tr = train_df[BASELINE_FEATURES].values.astype(float)
    X_base_te = test_df[BASELINE_FEATURES].values.astype(float)
    X_fi_tr   = train_df[FI_FEATURES].values.astype(float)
    X_fi_te   = test_df[FI_FEATURES].values.astype(float)

    # Baseline: raw GHI (lat/lon only — replicates Quezon City study)
    y_base_tr = train_df["Target_J"].values.astype(float)
    y_base_te = test_df["Target_J"].values.astype(float)

    # FI: geometry-adjusted effective GHI (azimuth + building features)
    y_fi_tr = train_df["Target_eff_J"].values.astype(float)
    y_fi_te = test_df["Target_eff_J"].values.astype(float)

    print("\n[3/5] PHASE 1: Training ML Models …")
    ada = BaselineAdaBoost().fit(X_base_tr, y_base_tr)
    fi  = FIAdaBoostRegressor().fit(X_fi_tr, y_fi_tr)

    print("\n[4/5] Evaluation …")
    ada_te = ada.predict(X_base_te)
    fi_te  = fi.predict(X_fi_te)

    ada_test_m = compute_metrics(y_base_te, ada_te)
    fi_test_m  = compute_metrics(y_fi_te,   fi_te)

    t2 = make_result_table(ada_test_m, fi_test_m)
    _print_table("Phase 1: ML Validation Results", t2)

    print("\n[5/5] PHASE 2: Energy Forecast (Area × Irradiance) …")
    ada_fcast = forecast_solar_energy(df, ada, BASELINE_FEATURES, label="ada", is_baseline=True,  train_df=train_df)
    fi_fcast  = forecast_solar_energy(df, fi,  FI_FEATURES,       label="fi",  is_baseline=False, train_df=train_df)

    if not ada_fcast.empty:
        print(f"  Total Potential (Baseline — theoretical sky GHI): {ada_fcast['ada_solar_kWh_yr'].sum():>18,.2f} kWh/year")
        print(f"  Total Potential (FI-AdaBoost — pvlib POA-adjusted): {fi_fcast['fi_solar_kWh_yr'].sum():>18,.2f} kWh/year")

    # Persist trained models for API usage and reuse.
    joblib.dump(ada, BASELINE_MODEL_FILE)
    joblib.dump(fi, FI_MODEL_FILE)

    # ── Export results to disk ────────────────────────────────────────────────
    print("\n[6/6] Saving results to disk…")

    # CSVs
    p_metrics  = save_metrics_csv(ada_test_m, fi_test_m)
    p_forecast = save_forecast_csv(ada_fcast, fi_fcast)

    # Plots
    plot_standalone_feature_importance(fi.feature_importances_, FI_FEATURES)
    plot_actual_vs_predicted(y_base_te, ada_te, y_fi_te, fi_te)
    plot_residuals(y_base_te, ada_te, y_fi_te, fi_te)
    plot_metrics_comparison(ada_test_m, fi_test_m)
    plot_energy_distribution(ada_fcast, fi_fcast)
    plot_total_energy_comparison(ada_fcast, fi_fcast)

    saved = [
        ("CSV",  p_metrics),
        ("CSV",  p_forecast),
        ("Plot", os.path.join(RESULTS_DIR, "standalone_feature_importances.png")),
        ("Plot", os.path.join(RESULTS_DIR, "actual_vs_predicted.png")),
        ("Plot", os.path.join(RESULTS_DIR, "residuals.png")),
        ("Plot", os.path.join(RESULTS_DIR, "metrics_comparison.png")),
        ("Plot", os.path.join(RESULTS_DIR, "energy_distribution.png")),
        ("Plot", os.path.join(RESULTS_DIR, "total_energy_comparison.png")),
        ("Model", BASELINE_MODEL_FILE),
        ("Model", FI_MODEL_FILE),
    ]
    print(f"\n  {'Type':<8}  File")
    print(f"  {'─'*8}  {'─'*55}")
    for kind, p in saved:
        print(f"  {kind:<8}  {p}")
    print("\n[Done] All results saved to 'results/' folder.\n")

if __name__ == "__main__":
    main()