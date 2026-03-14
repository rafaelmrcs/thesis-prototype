# src/model_training.py
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =========================
# Helpers
# =========================
def evaluate(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def weighted_median(preds_2d, weights):
    """
    preds_2d: shape (n_estimators, n_samples)
    weights:  shape (n_estimators,)
    returns:  shape (n_samples,)
    """
    preds_2d = np.asarray(preds_2d, dtype=float)
    weights = np.asarray(weights, dtype=float)

    n_estimators, n_samples = preds_2d.shape
    out = np.empty(n_samples, dtype=float)

    wsum = np.sum(weights)
    if wsum <= 0:
        return np.median(preds_2d, axis=0)

    weights = weights / wsum

    for i in range(n_samples):
        p = preds_2d[:, i]
        order = np.argsort(p)
        p_sorted = p[order]
        w_sorted = weights[order]
        cw = np.cumsum(w_sorted)
        out[i] = p_sorted[np.searchsorted(cw, 0.5)]
    return out


# =========================
# FI-AdaBoost
# =========================
class FIAdaBoostRegressor:
    """
    Feature-Importance-Aware AdaBoost Regression (FI-AdaBoost)

    Paper-style:
    - e_i^t = |y_i - h_t(x_i)| / max_j |y_j - h_t(x_j)|
    - phi(f_k) = I(f_k) / sum_j I(f_j)
    - Phi(x_i) = sum_k phi(f_k) * x_norm_{i,k}
    - w_{t+1,i} = w_{t,i} * beta_t^(1 - e_i^t * Phi(x_i)) / Z_t
    - final prediction: weighted median
    """
    def __init__(self, n_estimators=83, max_depth=4, random_state=42, use_weighted_median=True):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.random_state = int(random_state)
        self.use_weighted_median = bool(use_weighted_median)
        self.estimators_ = []
        self.estimator_weights_ = []

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            y = pd.Series(y)

        X_vals = np.asarray(X.values, dtype=float)
        y_vals = np.asarray(pd.Series(y).values, dtype=float)

        n_samples, n_features = X_vals.shape
        weights = np.ones(n_samples, dtype=float) / n_samples

        # Min-max normalize features for Phi(x)
        x_min = X_vals.min(axis=0)
        x_max = X_vals.max(axis=0)
        x_range = x_max - x_min
        x_range[x_range == 0] = 1.0
        X_norm = (X_vals - x_min) / x_range

        self.estimators_.clear()
        self.estimator_weights_.clear()

        for t in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=self.random_state + t
            )
            tree.fit(X_vals, y_vals, sample_weight=weights)

            y_pred = tree.predict(X_vals)
            errors = np.abs(y_vals - y_pred)

            max_error = np.max(errors)
            if max_error <= 0:
                break

            e_norm = errors / max_error  # [0,1]

            raw_fi = tree.feature_importances_
            fi_sum = np.sum(raw_fi)
            if fi_sum <= 0:
                phi_f = np.ones(n_features, dtype=float) / n_features
            else:
                phi_f = raw_fi / fi_sum

            phi_x = X_norm @ phi_f  # composite importance per sample

            avg_error = np.sum(weights * e_norm)
            if avg_error >= 0.5:
                break

            beta_t = avg_error / (1.0 - avg_error)

            # FI-guided weight update
            exponent = 1.0 - (e_norm * phi_x)
            weights = weights * np.power(beta_t, exponent)

            zt = np.sum(weights)
            if zt <= 0:
                break
            weights /= zt

            self.estimators_.append(tree)
            self.estimator_weights_.append(np.log(1.0 / (beta_t + 1e-12)))

        return self

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_vals = np.asarray(X.values, dtype=float)

        if len(self.estimators_) == 0:
            return np.zeros(X_vals.shape[0], dtype=float)

        preds = np.array([est.predict(X_vals) for est in self.estimators_], dtype=float)
        w = np.asarray(self.estimator_weights_, dtype=float)

        if self.use_weighted_median:
            return weighted_median(preds, w)
        return np.average(preds, axis=0, weights=w)


# =========================
# Main
# =========================
if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(ROOT_DIR, "data", "processed")
    model_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    data_path = os.path.join(processed_dir, "integrated_dataset.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}. Run data_integration.py first.")

    df = pd.read_csv(data_path)

    # -------------------------
    # 🚨 OPTUNA'S WINNING NUMBERS PLUGGED IN HERE! 🚨
    # -------------------------
    BEST_N_ESTIMATORS = 83
    BEST_MAX_DEPTH = 4

    # -------------------------
    # Option B Target
    # -------------------------
    TARGET = "solar_energy_potential"
    if TARGET not in df.columns:
        raise ValueError(f"Missing target '{TARGET}'. Ensure data_integration.py created it.")

    # Ensure chronological order
    if "date" not in df.columns:
        raise ValueError("Missing 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # -------------------------
    # Build ONE shared feature set for FAIR comparison
    # (baseline vs FI differs ONLY by weighting mechanism)
    # -------------------------
    leakage_cols = ["sunshine_hours", "clear_sky_ratio", "sunshine_flag", "year_month"]

    drop_cols = [TARGET, "date", "element", "id", "month"] + leakage_cols

    # If lat/lon are constant, drop them automatically
    for c in ["lat", "lon"]:
        if c in df.columns and df[c].nunique(dropna=False) == 1:
            drop_cols.append(c)

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=["number"]).copy()
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    y = pd.to_numeric(df[TARGET], errors="coerce").fillna(0.0).astype(float)

    # Chronological split
    split_idx = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print("\n[Training] Shared feature set for BOTH models")
    print(f"Rows: train={len(X_train)}, test={len(X_test)}")
    print(f"Features used: {X.shape[1]}")
    print(f"Using Optuna-Optimized Parameters: n_estimators={BEST_N_ESTIMATORS}, max_depth={BEST_MAX_DEPTH}")

    # -------------------------
    # Baseline AdaBoost (standard)
    # -------------------------
    print("\n[1/2] Training BASELINE AdaBoostRegressor...")
    baseline = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=BEST_MAX_DEPTH, random_state=42),
        n_estimators=BEST_N_ESTIMATORS,
        learning_rate=1.0,
        random_state=42
    )
    baseline.fit(X_train, y_train)
    pred_base = baseline.predict(X_test)
    res_base = evaluate(y_test, pred_base)

    # -------------------------
    # FI-AdaBoost (proposed)
    # -------------------------
    print("[2/2] Training FI-AdaBoostRegressor...")
    fi = FIAdaBoostRegressor(
        n_estimators=BEST_N_ESTIMATORS,
        max_depth=BEST_MAX_DEPTH,
        random_state=42,
        use_weighted_median=True
    )
    fi.fit(X_train, y_train)
    pred_fi = fi.predict(X_test)
    res_fi = evaluate(y_test, pred_fi)

    print("\n--- RESULTS (Solar Energy Potential Forecasting) ---")
    print(f"Baseline AdaBoost: RMSE={res_base['RMSE']:.4f}, MAE={res_base['MAE']:.4f}, R2={res_base['R2']:.4f}")
    print(f"FI-AdaBoost:       RMSE={res_fi['RMSE']:.4f}, MAE={res_fi['MAE']:.4f}, R2={res_fi['R2']:.4f}")

    joblib.dump(baseline, os.path.join(model_dir, "baseline_adaboost.pkl"))
    joblib.dump(fi, os.path.join(model_dir, "fi_adaboost.pkl"))
    print("\nSaved: models/baseline_adaboost.pkl, models/fi_adaboost.pkl")

    