import os
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# -------------------------
# Helpers
# -------------------------
def evaluate(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def weighted_median(preds_2d, weights):
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


class FIAdaBoostRegressor:
    def __init__(self, n_estimators=150, max_depth=4, random_state=42, use_weighted_median=True):
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

        x_min = X_vals.min(axis=0)
        x_max = X_vals.max(axis=0)
        x_range = x_max - x_min
        x_range[x_range == 0] = 1.0
        X_norm = (X_vals - x_min) / x_range

        self.estimators_.clear()
        self.estimator_weights_.clear()

        for t_idx in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=self.random_state + t_idx
            )
            tree.fit(X_vals, y_vals, sample_weight=weights)

            y_pred = tree.predict(X_vals)
            errors = np.abs(y_vals - y_pred)
            max_error = np.max(errors)
            if max_error <= 0:
                break

            e_norm = errors / max_error

            raw_fi = tree.feature_importances_
            fi_sum = np.sum(raw_fi)
            phi_f = (raw_fi / fi_sum) if fi_sum > 0 else (np.ones(n_features) / n_features)

            phi_x = X_norm @ phi_f

            avg_error = np.sum(weights * e_norm)
            if avg_error >= 0.5:
                break

            beta_t = avg_error / (1.0 - avg_error)

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


# -------------------------
# Main tuning routine
# -------------------------
TARGET = "solar_energy_potential"
LEAKAGE_COLS = ["sunshine_hours", "clear_sky_ratio", "sunshine_flag", "year_month"]

# Keep this small for laptop speed
N_SPLITS = 3
SAMPLE_FRAC = 0.20  # 20% rows to make tuning fast; set None to use all rows


def build_X_y(df: pd.DataFrame):
    # Sort by time
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Shared feature set
    drop_cols = [TARGET, "date", "element", "id", "month"] + LEAKAGE_COLS

    for c in ["lat", "lon"]:
        if c in df.columns and df[c].nunique(dropna=False) == 1:
            drop_cols.append(c)

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    y = pd.to_numeric(df[TARGET], errors="coerce").fillna(0.0).astype(float)
    return X, y


def cv_score_model(model, X, y, tscv):
    rmses, maes, r2s = [], [], []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        m = evaluate(y_test, pred)

        rmses.append(m["RMSE"])
        maes.append(m["MAE"])
        r2s.append(m["R2"])

    return float(np.mean(rmses)), float(np.mean(maes)), float(np.mean(r2s))


def main():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(ROOT_DIR, "data", "processed")
    out_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)

    data_path = os.path.join(processed_dir, "integrated_dataset.csv")
    df = pd.read_csv(data_path)

    if TARGET not in df.columns:
        raise ValueError(f"Missing target '{TARGET}'")

    if SAMPLE_FRAC is not None:
        df = df.sample(frac=float(SAMPLE_FRAC), random_state=42).copy()
        print(f"[Downsample] Using {SAMPLE_FRAC:.2f} of rows => {len(df)} rows")

    X, y = build_X_y(df)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    results = []

    # ---- Baseline grid (8 runs)
    baseline_grid = []
    for n_est in [50, 100]:
        for depth in [3, 4]:
            for lr in [0.5, 1.0]:
                baseline_grid.append((n_est, depth, lr))

    print(f"\n[Tuning] Baseline grid runs: {len(baseline_grid)}")
    for n_est, depth, lr in baseline_grid:
        model = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=depth, random_state=42),
            n_estimators=n_est,
            learning_rate=lr,
            random_state=42
        )
        rmse, mae, r2 = cv_score_model(model, X, y, tscv)
        results.append({
            "model": "Baseline",
            "n_estimators": n_est,
            "max_depth": depth,
            "learning_rate": lr,
            "cv_rmse": rmse,
            "cv_mae": mae,
            "cv_r2": r2
        })
        print(f"Baseline n={n_est} depth={depth} lr={lr} => RMSE={rmse:.2f} MAE={mae:.2f} R2={r2:.4f}")

    # ---- FI grid (4 runs)
    fi_grid = []
    for n_est in [100, 150]:
        for depth in [3, 4]:
            fi_grid.append((n_est, depth))

    print(f"\n[Tuning] FI grid runs: {len(fi_grid)}")
    for n_est, depth in fi_grid:
        model = FIAdaBoostRegressor(
            n_estimators=n_est,
            max_depth=depth,
            random_state=42,
            use_weighted_median=True
        )
        rmse, mae, r2 = cv_score_model(model, X, y, tscv)
        results.append({
            "model": "FI-AdaBoost",
            "n_estimators": n_est,
            "max_depth": depth,
            "learning_rate": np.nan,
            "cv_rmse": rmse,
            "cv_mae": mae,
            "cv_r2": r2
        })
        print(f"FI n={n_est} depth={depth} => RMSE={rmse:.2f} MAE={mae:.2f} R2={r2:.4f}")

    res_df = pd.DataFrame(results).sort_values(["model", "cv_rmse"])
    out_path = os.path.join(out_dir, "tuning_quick_results.csv")
    res_df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print("\nTop configs (lowest CV RMSE):")
    print(res_df.groupby("model").head(3)[["model","n_estimators","max_depth","learning_rate","cv_rmse","cv_mae","cv_r2"]])


if __name__ == "__main__":
    main()