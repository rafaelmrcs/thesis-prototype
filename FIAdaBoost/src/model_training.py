"""
model_training.py
────────────────────────────────────────────────────────────────────────────
Solar Irradiance Forecasting — Davao City
AdaBoost Regression (Baseline)  vs  FI-AdaBoost Regression (Proposed)

METHODOLOGY
──────────────────────────────────────────────────────────────────────────
  Dataset : 3,000 rows (one per random spatial coordinate)
            Same size and design as Quezon City baseline study.

  Target  : GHI_mean_J = annual average GHI × 3,600,000 (J/m²/day)
            Equivalent to Quezon City's "solarrad" in J/m².

  Split   : 80/20 RANDOM  →  2,400 train / 600 test
            Same split used for BOTH models (fixed seed — reproducible).
            Matches Quezon City methodology (random, not chronological,
            because the dataset is spatial, not temporal).

  BASELINE AdaBoost
  ─────────────────
    Features X : [lat, lon]
    Model      : sklearn AdaBoostRegressor (DecisionTree base)
    This exactly replicates the Quezon City study:
      "The primary machine learning algorithm utilized ... is Ensemble
       Regression with Adaptive Boosting ... The process involves
       assigning initial weights to data points, training decision trees,
       calculating error rates, updating weights, and making final
       predictions by combining the weighted outputs of all trees."

  FI-AdaBoost (Proposed)
  ──────────────────────
    Features X : [lat, lon,
                  rooftop_area_sq_m, orientation_score,
                  shading_factor,    tilt_factor, SEI_norm]
    Model      : Custom FI-AdaBoost — same boosting framework but weight
                 update is guided by feature importances (§2.4.2)
    SAME y, SAME split as baseline → directly comparable metrics.

  WHY THE COMPARISON IS FAIR
  ──────────────────────────
    • Same 3,000 rows — identical data
    • Same target (GHI_mean_J) — metrics on same scale
    • Same 80/20 random split — identical train/test rows
    • Baseline uses only lat/lon (Quezon City standard)
    • FI-AdaBoost uses lat/lon PLUS building features — the hypothesis
      is that building-level features, combined with feature-importance
      weighting, improve prediction beyond the spatial baseline.

  After prediction:  E = A × r × H × PR  per building (§D.2 / Quezon City Eq.5)
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
import matplotlib.gridspec as gridspec
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
PANEL_EFF     = 0.192     # LG Neon 2 315W (same as Quezon City study)
PERF_RATIO    = 0.78      # US DoE 75-system average (same as Quezon City)
DAYS_PER_YEAR = 365
KWH_TO_J      = 3_600_000

np.random.seed(RANDOM_SEED)

# ── Feature & target definitions ──────────────────────────────────────────────
TARGET_J   = "GHI_mean_J"          # J/m²/day — used for RMSE/MAE (matches study)
TARGET_KWH = "GHI_mean_2024"       # kWh/m²/day — kept for energy forecast calc

# BASELINE: lat, lon ONLY — exactly as Quezon City study
BASELINE_FEATURES = ["lat", "lon"]

# FI-AdaBoost: lat, lon + per-building topographical features
FI_FEATURES = [
    "lat", "lon",
    "rooftop_area_sq_m",
    "orientation_score",
    "shading_factor",
    "tilt_factor",
    "SEI_norm",
]

C_ADA = "#E74C3C"
C_FI  = "#27AE60"


# =============================================================================
# SECTION 1 — DATA LOADING & VALIDATION
# =============================================================================

def load_dataset() -> pd.DataFrame:
    """Load the 3,000-row integrated spatial dataset."""
    path = os.path.join(PROCESSED_DIR, "integrated_dataset.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing: {path}\n"
            "Run the pipeline in order:\n"
            "  1. data_acquisition.py   (fetch_nasa_baseline_spatial + fetch_osm_data)\n"
            "  2. data_processing.py    (process_baseline_spatial + process_osm)\n"
            "  3. feature_engineering.py\n"
            "  4. data_integration.py\n"
            "  5. model_training.py     ← you are here"
        )

    df = pd.read_csv(path)
    print(f"[Load] Rows    : {len(df):,}")
    print(f"[Load] Columns : {list(df.columns)}")

    # Validate all required columns
    needed = set(FI_FEATURES + [TARGET_J, TARGET_KWH])
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"integrated_dataset.csv is missing columns: {sorted(missing)}\n"
            "Re-run feature_engineering.py and data_integration.py."
        )

    # Feature variance check
    print("\n[Load] Feature variance check:")
    for f in FI_FEATURES:
        var = df[f].var()
        if var > 1e-10:
            status = "OK"
        elif f == "tilt_factor":
            status = "CONSTANT — expected (flat roof assumption §2.2.3.4)"
        else:
            status = "WARNING — zero variance"
        print(f"  {f:<25} var={var:.6f}  {status}")

    return df


# =============================================================================
# SECTION 2 — 80/20 RANDOM SPLIT  (matches Quezon City methodology)
#
# Quezon City: "the dataset is strategically divided into two subsets:
#               2,500 instances for training and 500 instances for testing."
#               (random cvpartition in MATLAB)
#
# Here: 80/20 random split → 2,400 train / 600 test for 3,000 rows.
# Fixed random_state=42 ensures reproducibility.
# SAME split indices used for BOTH models.
# =============================================================================

def random_split(df: pd.DataFrame, test_size: float = 0.20):
    """80/20 random split — same indices for baseline and FI-AdaBoost."""
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, random_state=RANDOM_SEED, shuffle=True
    )
    return df.iloc[idx_train].copy(), df.iloc[idx_test].copy()


# =============================================================================
# SECTION 3 — BASELINE AdaBoost REGRESSION  (§2.3 / Quezon City replication)
#
# Exactly the same as the Quezon City study:
#   sklearn AdaBoostRegressor
#   DecisionTreeRegressor base estimator (max_depth=3)
#   n_estimators=100, learning_rate=0.1
# =============================================================================

class BaselineAdaBoost:
    """
    Standard AdaBoost Regression.
    Replicates Quezon City study (Ensemble Regression with Adaptive Boosting).
    Features: lat, lon only.
    """

    def __init__(self, n_estimators: int   = 100,
                 learning_rate:    float = 0.1,
                 max_depth:        int   = 3,
                 random_state:     int   = RANDOM_SEED):
        base = DecisionTreeRegressor(max_depth=max_depth,
                                     random_state=random_state)
        self._model = AdaBoostRegressor(
            estimator=base,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_


# =============================================================================
# SECTION 4 — PROPOSED FI-AdaBoost REGRESSION  (§2.4)
#
# Same base algorithm as baseline but:
#   (a) uses 7 features instead of 2
#   (b) weight update guided by feature importances (§2.4.2)
# =============================================================================

class FIAdaBoostRegressor:
    """
    Feature-Importance-Aware AdaBoost Regression (§2.4).

    Standard weight update (baseline):
        w_i^{t+1} = w_i^t · β_t^{1 − e_i^t}          / Z_t

    FI-AdaBoost weight update (proposed):
        w_i^{t+1} = w_i^t · β_t^{1 − e_i^t · Φ(x_i)} / Z_t

    where:
        φ(f_k)  = I(f_k) / Σ_j I(f_j)        — normalised feature importance
        Φ(x_i)  = Σ_k φ(f_k) · |x_{i,k}|_norm — composite sample importance

    With 7 features (including 5 building-specific ones), Φ(x_i) differs
    meaningfully across buildings — the FI mechanism directs attention
    toward samples where high-importance features (e.g. SEI_norm, orientation)
    drive large prediction errors.
    """

    def __init__(self, n_estimators: int   = 100,
                 learning_rate:    float = 0.1,
                 max_depth:        int   = 3,
                 random_state:     int   = RANDOM_SEED):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.max_depth     = max_depth
        self.random_state  = random_state

        self.estimators_          : list = []
        self.estimator_weights_   : list = []
        self.feature_importances_ : np.ndarray | None = None

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _norm_fi(tree: DecisionTreeRegressor) -> np.ndarray:
        """φ(f_k) = I(f_k) / Σ I(f_j)  (§2.4.1)."""
        raw = tree.feature_importances_
        s   = raw.sum()
        return raw / s if s > 0 else np.ones_like(raw) / len(raw)

    @staticmethod
    def _composite_phi(X: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Φ(x_i) = Σ_k φ(f_k) · |x_{i,k}|_norm  (§2.4.1)."""
        X_abs  = np.abs(X)
        col_mx = X_abs.max(axis=0)
        col_mx[col_mx == 0] = 1
        Phi    = (X_abs / col_mx * phi).sum(axis=1)
        p_max  = Phi.max()
        return Phi / p_max if p_max > 0 else Phi

    # ── training ─────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray):
        n       = len(y)
        rng     = np.random.default_rng(self.random_state)
        weights = np.full(n, 1.0 / n)
        cum_fi  = np.zeros(X.shape[1])
        n_valid = 0

        for t in range(self.n_estimators):
            idx  = rng.choice(n, size=n, replace=True, p=weights)
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         random_state=self.random_state + t)
            tree.fit(X[idx], y[idx])

            y_pred = tree.predict(X)
            abs_e  = np.abs(y - y_pred)
            D_t    = abs_e.max()
            if D_t == 0:
                break
            e_i   = abs_e / D_t
            eps_t = float(np.dot(weights, e_i))
            if eps_t >= 0.5:
                break

            beta_t = eps_t / (1.0 - eps_t + 1e-10)
            phi    = self._norm_fi(tree)
            Phi_i  = self._composite_phi(X, phi)

            new_w   = weights * (beta_t ** (1.0 - e_i * Phi_i))
            Z_t     = new_w.sum()
            if Z_t == 0:
                break
            weights = new_w / Z_t

            est_w = max(
                self.learning_rate * math.log(
                    (1.0 - eps_t) / (eps_t + 1e-10)
                ),
                1e-10,
            )
            self.estimators_.append(tree)
            self.estimator_weights_.append(est_w)
            cum_fi  += phi
            n_valid += 1

        self.feature_importances_ = (
            cum_fi / n_valid if n_valid > 0
            else np.ones(X.shape[1]) / X.shape[1]
        )
        return self

    # ── prediction (weighted median — §2.3.2) ────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("Call fit() first.")
        preds   = np.array([e.predict(X) for e in self.estimators_])
        weights = np.array(self.estimator_weights_)
        weights = weights / weights.sum()
        result  = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            p_i   = preds[:, i]
            order = np.argsort(p_i)
            cumw  = np.cumsum(weights[order])
            mid   = np.searchsorted(cumw, 0.5)
            result[i] = p_i[order[min(mid, len(p_i) - 1)]]
        return result


# =============================================================================
# SECTION 5 — EVALUATION METRICS  (§2.6 / Quezon City §C.5)
# Primary unit: J/m²/day — matches Quezon City study Table 2.
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    RMSE (§2.6.1), MAE (§2.6.2), R² (§2.6.3).
    Returned in both kWh and J to match Quezon City reporting convention.
    """
    rmse_j = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae_j  = float(mean_absolute_error(y_true, y_pred))
    r2     = float(r2_score(y_true, y_pred))
    return {
        "RMSE_J":   rmse_j,
        "MAE_J":    mae_j,
        "RMSE_kWh": rmse_j / KWH_TO_J,
        "MAE_kWh":  mae_j  / KWH_TO_J,
        "R2":       r2,
    }


def diebold_mariano(e1: np.ndarray, e2: np.ndarray) -> tuple:
    """DM test on squared-error series, Newey-West variance (§2.5.4)."""
    d  = e1**2 - e2**2
    n  = len(d)
    db = d.mean()
    g0 = float(np.var(d, ddof=1))
    g1 = float(np.mean((d[1:] - db) * (d[:-1] - db))) if n > 1 else 0.0
    nw = (g0 + 2.0 * g1) / n
    if nw <= 0:
        return np.nan, np.nan
    dm = db / math.sqrt(nw)
    pv = 2.0 * (1.0 - stats.norm.cdf(abs(dm)))
    return float(dm), float(pv)


# =============================================================================
# SECTION 6 — SOLAR ENERGY POTENTIAL FORECAST  (E = A × r × H × PR)
# Quezon City Eq. 5 — applied after GHI prediction
# =============================================================================

def forecast_solar_energy(df: pd.DataFrame,
                           model,
                           feature_cols: list,
                           label: str) -> pd.DataFrame:
    """
    Predict GHI for all 3,000 points, then compute per-building energy.
    E = A × r × H × PR
      A = rooftop_area_sq_m (m²)
      r = PANEL_EFF = 0.192
      H = predicted_GHI_kWh/day × 365  (annual irradiation, kWh/m²/yr)
      PR = PERF_RATIO = 0.78
    """
    X_all  = df[feature_cols].values.astype(float)
    y_pred_J   = model.predict(X_all)
    y_pred_kWh = y_pred_J / KWH_TO_J   # convert back to kWh/m²/day

    H = y_pred_kWh * DAYS_PER_YEAR     # annual irradiation kWh/m²/yr

    result = df[["lat", "lon", "rooftop_area_sq_m",
                 "SEI_norm", TARGET_KWH]].copy()
    result["predicted_GHI_kWh"] = y_pred_kWh
    result["annual_H_kWh_m2"]   = H
    result[f"{label}_solar_kWh_yr"] = (
        result["rooftop_area_sq_m"] * PANEL_EFF * H * PERF_RATIO
    )
    result[f"{label}_solar_kWh_mo"] = result[f"{label}_solar_kWh_yr"] / 12
    return result


# =============================================================================
# SECTION 7 — VISUALISATION
# =============================================================================

def plot_eda(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "EDA — Davao City  |  3,000 Spatial Coordinates\n"
        "Target: Annual Average GHI  (J/m²/day)",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    # Spatial scatter coloured by GHI
    ax1 = fig.add_subplot(gs[0, 0])
    sc  = ax1.scatter(df["lon"], df["lat"], c=df[TARGET_J],
                      cmap="YlOrRd", s=6, alpha=0.65)
    plt.colorbar(sc, ax=ax1, label="J/m²/day")
    ax1.set_title("Spatial Distribution\n& Annual GHI")
    ax1.set_xlabel("Longitude"); ax1.set_ylabel("Latitude")

    # GHI histogram (J/m²/day)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df[TARGET_J], bins=40, color="#E07B39",
             edgecolor="white", alpha=0.85)
    desc = df[TARGET_J].describe()
    ax2.text(0.97, 0.97,
             f"Mean:  {desc['mean']:,.0f}\nStd:   {desc['std']:,.0f}\n"
             f"Min:   {desc['min']:,.0f}\nMax:   {desc['max']:,.0f}",
             transform=ax2.transAxes, fontsize=8, va="top", ha="right",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax2.set_title("GHI Distribution (J/m²/day)")
    ax2.set_xlabel("J/m²/day")

    # Spatial heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    res  = 20
    lat_b = np.linspace(df["lat"].min(), df["lat"].max(), res)
    lon_b = np.linspace(df["lon"].min(), df["lon"].max(), res)
    hmap  = np.full((res-1, res-1), np.nan)
    for i in range(res-1):
        for j in range(res-1):
            m = ((df["lat"] >= lat_b[i]) & (df["lat"] < lat_b[i+1]) &
                 (df["lon"] >= lon_b[j]) & (df["lon"] < lon_b[j+1]))
            if m.sum() > 0:
                hmap[i, j] = df.loc[m, TARGET_J].mean()
    im = ax3.imshow(hmap, origin="lower", aspect="auto",
                    extent=[df["lon"].min(), df["lon"].max(),
                            df["lat"].min(), df["lat"].max()],
                    cmap="YlOrRd")
    plt.colorbar(im, ax=ax3, label="J/m²/day")
    ax3.set_title("Spatial Heatmap — Annual GHI")
    ax3.set_xlabel("Longitude"); ax3.set_ylabel("Latitude")

    # Lat vs GHI
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(df["lat"], df[TARGET_J], alpha=0.3, s=5, color="#3498DB")
    m, b = np.polyfit(df["lat"], df[TARGET_J], 1)
    xl   = np.array([df["lat"].min(), df["lat"].max()])
    ax4.plot(xl, m*xl+b, "r--", lw=1.5, label=f"slope={m:.0f}")
    ax4.set_title("Latitude vs GHI"); ax4.legend(fontsize=8)
    ax4.set_xlabel("Latitude"); ax4.set_ylabel("J/m²/day")

    # Lon vs GHI
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(df["lon"], df[TARGET_J], alpha=0.3, s=5, color="#9B59B6")
    m2, b2 = np.polyfit(df["lon"], df[TARGET_J], 1)
    xl2    = np.array([df["lon"].min(), df["lon"].max()])
    ax5.plot(xl2, m2*xl2+b2, "r--", lw=1.5, label=f"slope={m2:.0f}")
    ax5.set_title("Longitude vs GHI"); ax5.legend(fontsize=8)
    ax5.set_xlabel("Longitude"); ax5.set_ylabel("J/m²/day")

    # SEI_norm distribution
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(df["SEI_norm"], bins=40, color="#27AE60",
             edgecolor="white", alpha=0.85)
    ax6.set_title("SEI_norm Distribution\n(per building, attached to each point)")
    ax6.set_xlabel("SEI_norm (0–1)")

    plt.savefig(os.path.join(RESULTS_DIR, "eda.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  [Plot] eda.png")


def plot_results(res: dict) -> None:
    y_tr     = res["y_train"]; y_te     = res["y_test"]
    ada_tr   = res["ada_tr"];  ada_te   = res["ada_te"]
    fi_tr    = res["fi_tr"];   fi_te    = res["fi_te"]
    ada_tr_m = res["ada_train_m"]; fi_tr_m = res["fi_train_m"]
    ada_te_m = res["ada_test_m"];  fi_te_m = res["fi_test_m"]

    # ── Figure 1: Table 1 & Table 2 bar charts ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Table 1 (Training) & Table 2 (Test) — Performance Comparison\n"
        "RMSE / MAE in J/m²/day  |  R² dimensionless",
        fontsize=12, fontweight="bold",
    )
    for ax, title, ada_m, fi_m in [
        (axes[0], "Training Data (Table 1)", ada_tr_m, fi_tr_m),
        (axes[1], "Test Data (Table 2)",     ada_te_m, fi_te_m),
    ]:
        keys  = ["RMSE_J", "MAE_J", "R2"]
        xlbls = ["RMSE\n(J/m²/day)", "MAE\n(J/m²/day)", "R²"]
        x = np.arange(3); w = 0.35
        av = [ada_m[k] for k in keys]; fv = [fi_m[k] for k in keys]
        b1 = ax.bar(x - w/2, av, w, label="AdaBoost (Baseline)",
                    color=C_ADA, alpha=0.85, edgecolor="white")
        b2 = ax.bar(x + w/2, fv, w, label="FI-AdaBoost (Proposed)",
                    color=C_FI,  alpha=0.85, edgecolor="white")
        for bar, val in zip(list(b1) + list(b2), av + fv):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(av + fv) * 0.012,
                    f"{val:,.2f}", ha="center", fontsize=7, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(xlbls, fontsize=9)
        ax.set_title(title); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "tables_1_and_2.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Plot] tables_1_and_2.png")

    # ── Figure 2: Actual vs Predicted scatter (test) ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Predicted vs Actual GHI — Test Set (J/m²/day)",
                 fontsize=12, fontweight="bold")
    for ax, name, pred, color in [
        (axes[0], "AdaBoost (Baseline)\nFeatures: lat, lon",    ada_te, C_ADA),
        (axes[1], "FI-AdaBoost (Proposed)\nFeatures: lat, lon + building", fi_te,  C_FI),
    ]:
        ax.scatter(y_te, pred, alpha=0.5, s=10, color=color)
        lim = [min(y_te.min(), pred.min()), max(y_te.max(), pred.max())]
        ax.plot(lim, lim, "k--", lw=1)
        m = compute_metrics(y_te, pred)
        ax.set_title(
            f"{name}\nRMSE={m['RMSE_J']:,.2f} J  R²={m['R2']:.4f}",
            fontsize=8,
        )
        ax.set_xlabel("Actual (J/m²/day)"); ax.set_ylabel("Predicted (J/m²/day)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "predicted_vs_actual.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Plot] predicted_vs_actual.png")

    # ── Figure 3: Feature importances ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Feature Importances — Learned by Each Model",
                 fontsize=12, fontweight="bold")
    for ax, name, fi_vals, feats, color in [
        (axes[0], "AdaBoost (Baseline)\n[lat, lon]",
         res["ada_fi"], BASELINE_FEATURES, C_ADA),
        (axes[1], "FI-AdaBoost (Proposed)\n[lat, lon + building features]",
         res["fi_fi"],  FI_FEATURES, C_FI),
    ]:
        labels = [f.replace("_", " ") for f in feats]
        idx    = np.argsort(fi_vals)
        ax.barh(np.array(labels)[idx], fi_vals[idx],
                color=color, alpha=0.85, edgecolor="white")
        for i, v in enumerate(fi_vals[idx]):
            ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
        ax.set_title(name, fontsize=9); ax.set_xlabel("Importance")
        ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "feature_importances.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Plot] feature_importances.png")

    # ── Figure 4: Residuals ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Residual Analysis — Test Set (J/m²/day)",
                 fontsize=12, fontweight="bold")
    e_ada = y_te - ada_te; e_fi = y_te - fi_te
    axes[0].hist(e_ada, bins=30, alpha=0.7, color=C_ADA,
                 edgecolor="white", label="AdaBoost")
    axes[0].hist(e_fi,  bins=30, alpha=0.7, color=C_FI,
                 edgecolor="white", label="FI-AdaBoost")
    axes[0].axvline(0, color="black", lw=0.8, ls="--")
    axes[0].set_title("Residual Distribution"); axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Residual (J/m²/day)")
    axes[1].plot(np.arange(len(y_te)), np.cumsum(np.abs(e_ada)),
                 color=C_ADA, lw=1.5, label="AdaBoost")
    axes[1].plot(np.arange(len(y_te)), np.cumsum(np.abs(e_fi)),
                 color=C_FI,  lw=1.5, label="FI-AdaBoost")
    axes[1].set_title("Cumulative Absolute Error")
    axes[1].set_xlabel("Test Sample Index")
    axes[1].set_ylabel("Cumulative |Error| (J/m²/day)")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "residuals.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Plot] residuals.png")


def plot_forecast(ada_fcast: pd.DataFrame, fi_fcast: pd.DataFrame) -> None:
    if ada_fcast.empty or fi_fcast.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Per-Building Solar Energy Potential  (E = A × r × H × PR)\n"
        f"r={PANEL_EFF}  PR={PERF_RATIO}",
        fontsize=12, fontweight="bold",
    )
    for ax, df, col, name, color in [
        (axes[0], ada_fcast, "ada_solar_kWh_yr", "AdaBoost (Baseline)",    C_ADA),
        (axes[1], fi_fcast,  "fi_solar_kWh_yr",  "FI-AdaBoost (Proposed)", C_FI),
    ]:
        data = df[col].clip(upper=np.percentile(df[col], 95))
        ax.hist(data, bins=40, color=color, edgecolor="white", alpha=0.85)
        mn = df[col].mean()
        ax.axvline(mn, color="black", lw=1.5, ls="--",
                   label=f"Mean: {mn:,.0f} kWh/yr")
        ax.set_title(f"{name}"); ax.set_xlabel("kWh/year")
        ax.set_ylabel("Buildings"); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "solar_forecast.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Plot] solar_forecast.png")


# =============================================================================
# RESULT TABLE HELPERS
# =============================================================================

def make_result_table(ada_m: dict, fi_m: dict) -> pd.DataFrame:
    rows = []
    for label, m in [("AdaBoost Regression (Baseline)", ada_m),
                     ("FI-AdaBoost Regression (Proposed)", fi_m)]:
        rows.append({
            "Model":             label,
            "RMSE (J/m²/day)":  round(m["RMSE_J"],   2),
            "MAE  (J/m²/day)":  round(m["MAE_J"],    2),
            "RMSE (kWh/m²/day)": round(m["RMSE_kWh"], 6),
            "MAE  (kWh/m²/day)": round(m["MAE_kWh"],  6),
            "R²":               round(m["R2"],        4),
        })
    return pd.DataFrame(rows).set_index("Model")


def _print_table(title: str, df: pd.DataFrame) -> None:
    sep = "─" * 90
    print(f"\n  {sep}")
    print(f"  {title}")
    print(f"  {sep}")
    print(df.to_string())
    print(f"  {sep}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("\n" + "=" * 74)
    print("  SOLAR IRRADIANCE FORECASTING — Davao City")
    print("  AdaBoost (Baseline)  vs  FI-AdaBoost (Proposed)")
    print("  Dataset: 3,000 spatial coordinates  |  Split: 80/20 random")
    print("=" * 74)

    # ── 1. Load ───────────────────────────────────────────────────────────
    print("\n[1/7] Loading dataset …")
    df = load_dataset()
    print(f"\n  Target statistics (J/m²/day):")
    print(df[TARGET_J].describe().round(2).to_string())

    # ── 2. EDA ────────────────────────────────────────────────────────────
    print("\n[2/7] EDA …")
    plot_eda(df)

    # ── 3. 80/20 random split — same as Quezon City ───────────────────────
    print("\n[3/7] 80/20 Random Train/Test Split …")
    train_df, test_df = random_split(df, test_size=0.20)
    print(f"  Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows")

    # Feature matrices — SAME y for both models
    X_base_tr = train_df[BASELINE_FEATURES].values.astype(float)
    X_base_te = test_df[BASELINE_FEATURES].values.astype(float)
    X_fi_tr   = train_df[FI_FEATURES].values.astype(float)
    X_fi_te   = test_df[FI_FEATURES].values.astype(float)
    y_tr      = train_df[TARGET_J].values.astype(float)   # J/m²/day
    y_te      = test_df[TARGET_J].values.astype(float)

    print(f"\n  Baseline features ({len(BASELINE_FEATURES)})   : {BASELINE_FEATURES}")
    print(f"  FI-AdaBoost features ({len(FI_FEATURES)}) :")
    for f in FI_FEATURES:
        tag = "  [+building]" if f not in BASELINE_FEATURES else ""
        print(f"    • {f}{tag}")
    print(f"\n  Target : {TARGET_J}  (J/m²/day)  ← same for both models")

    # ── 4. Train both models ──────────────────────────────────────────────
    print("\n[4/7] Training models …")
    ada = BaselineAdaBoost(n_estimators=100, learning_rate=0.1,
                           max_depth=3, random_state=RANDOM_SEED)
    fi  = FIAdaBoostRegressor(n_estimators=100, learning_rate=0.1,
                              max_depth=3, random_state=RANDOM_SEED)

    print(f"  Training AdaBoost (Baseline) on {len(X_base_tr):,} samples …")
    print(f"    Features: {BASELINE_FEATURES}")
    ada.fit(X_base_tr, y_tr)

    print(f"  Training FI-AdaBoost (Proposed) on {len(X_fi_tr):,} samples …")
    print(f"    Features: {FI_FEATURES}")
    fi.fit(X_fi_tr, y_tr)
    print(f"    Fitted {len(fi.estimators_)} estimators.")

    # ── 5. Evaluate ───────────────────────────────────────────────────────
    print("\n[5/7] Evaluation …")
    ada_tr = ada.predict(X_base_tr); fi_tr = fi.predict(X_fi_tr)
    ada_te = ada.predict(X_base_te); fi_te = fi.predict(X_fi_te)

    ada_train_m = compute_metrics(y_tr, ada_tr)
    fi_train_m  = compute_metrics(y_tr, fi_tr)
    ada_test_m  = compute_metrics(y_te, ada_te)
    fi_test_m   = compute_metrics(y_te, fi_te)

    t1 = make_result_table(ada_train_m, fi_train_m)
    t2 = make_result_table(ada_test_m,  fi_test_m)
    _print_table("Table 1 — TRAINING Data", t1)
    _print_table("Table 2 — TEST Data",     t2)

    rmse_red         = ((ada_test_m["RMSE_J"] - fi_test_m["RMSE_J"])
                        / ada_test_m["RMSE_J"] * 100)
    e_ada            = y_te - ada_te
    e_fi             = y_te - fi_te
    dm_stat, dm_pval = diebold_mariano(e_ada, e_fi)
    sig = ("✓ Statistically significant (p < 0.05)"
           if (not np.isnan(dm_pval)) and dm_pval < 0.05
           else "✗ Not significant (p ≥ 0.05)")
    print(f"\n  RMSE Reduction (test)  : {rmse_red:+.2f}%  "
          f"({'FI-AdaBoost better' if rmse_red > 0 else 'AdaBoost better'})")
    print(f"  DM Statistic           : {dm_stat:.4f}")
    print(f"  p-value                : {dm_pval:.4f}  — {sig}")

    # ── 6. Solar energy forecast ──────────────────────────────────────────
    print("\n[6/7] Solar energy forecast (E = A × r × H × PR) …")
    ada_fcast = forecast_solar_energy(df, ada, BASELINE_FEATURES, label="ada")
    fi_fcast  = forecast_solar_energy(df, fi,  FI_FEATURES,      label="fi")

    if not ada_fcast.empty:
        print(f"\n  {'Metric':<40} {'AdaBoost':>10}  {'FI-AdaBoost':>12}")
        print(f"  {'─'*66}")
        print(f"  {'Total buildings':<40} "
              f"{len(ada_fcast):>10,}  {'(same)':>12}")
        print(f"  {'Total rooftop area (m²)':<40} "
              f"{ada_fcast['rooftop_area_sq_m'].sum():>10,.0f}  {'(same)':>12}")
        print(f"  {'Total potential (GWh/year)':<40} "
              f"{ada_fcast['ada_solar_kWh_yr'].sum()/1e6:>10.2f}  "
              f"{fi_fcast['fi_solar_kWh_yr'].sum()/1e6:>12.2f}")
        print(f"  {'Monthly equivalent (GWh/month)':<40} "
              f"{ada_fcast['ada_solar_kWh_yr'].sum()/12/1e6:>10.2f}  "
              f"{fi_fcast['fi_solar_kWh_yr'].sum()/12/1e6:>12.2f}")
        print(f"  {'Avg per building (kWh/yr)':<40} "
              f"{ada_fcast['ada_solar_kWh_yr'].mean():>10,.0f}  "
              f"{fi_fcast['fi_solar_kWh_yr'].mean():>12,.0f}")
        print(f"  (Quezon City baseline result: ~30.7 GWh/month)")

    # ── 7. Plots & save ───────────────────────────────────────────────────
    print("\n[7/7] Plots & output files …")
    res_dict = dict(
        y_train=y_tr, y_test=y_te,
        ada_tr=ada_tr, ada_te=ada_te,
        fi_tr=fi_tr,   fi_te=fi_te,
        ada_train_m=ada_train_m, ada_test_m=ada_test_m,
        fi_train_m=fi_train_m,   fi_test_m=fi_test_m,
        ada_fi=ada.feature_importances_,
        fi_fi=fi.feature_importances_,
    )
    plot_results(res_dict)
    plot_forecast(ada_fcast, fi_fcast)

    t1.to_csv(os.path.join(RESULTS_DIR, "table1_training.csv"))
    t2.to_csv(os.path.join(RESULTS_DIR, "table2_test.csv"))
    ada_fcast.to_csv(os.path.join(RESULTS_DIR, "ada_solar_forecast.csv"), index=False)
    fi_fcast.to_csv(os.path.join(RESULTS_DIR,  "fi_solar_forecast.csv"),  index=False)

    # Persist trained models for API usage and reuse.
    joblib.dump(ada, BASELINE_MODEL_FILE)
    joblib.dump(fi, FI_MODEL_FILE)

    print(f"\n  All outputs → {RESULTS_DIR}")
    print(f"  Saved model  → {BASELINE_MODEL_FILE}")
    print(f"  Saved model  → {FI_MODEL_FILE}")
    print("\n" + "=" * 74 + "\n")
    return res_dict, ada_fcast, fi_fcast


if __name__ == "__main__":
    main()