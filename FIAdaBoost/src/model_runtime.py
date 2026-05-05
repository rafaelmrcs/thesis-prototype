from __future__ import annotations

import __main__
import math

import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

RANDOM_SEED = 42
KWH_TO_J = 3_600_000
TARGET_COL = "Target_eff_J"

SHARED_FEATURES = [
    "lat",
    "lon",
    "azimuth",
    "orientation_score",
    "shading_factor",
    "SEI_norm",
    "clear_sky_ratio",
    "sunshine_hours",
]

# Learned SEI normalization reference from the training rooftop feature set.
DEFAULT_SEI_NORMALIZER = 5368.320870273783

# Coverage of the current 10,000-point training sample.
DEFAULT_LAT_RANGE = (6.965095132565466, 7.602425835788297)
DEFAULT_LON_RANGE = (125.21874705155416, 125.69388469799664)


class BaselineAdaBoost:
    def __init__(
        self,
        n_estimators: int = 158,
        learning_rate: float = 0.94,
        max_depth: int = 3,
        random_state: int = RANDOM_SEED,
    ) -> None:
        base = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        self._model = AdaBoostRegressor(
            estimator=base,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
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
    def __init__(
        self,
        n_estimators: int = 141,
        learning_rate: float = 0.63,
        max_depth: int = 4,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators_ = []
        self.estimator_weights_ = []
        self.feature_importances_ = None

    @staticmethod
    def _norm_fi(tree) -> np.ndarray:
        raw = tree.feature_importances_
        total = raw.sum()
        return raw / total if total > 0 else np.ones_like(raw) / len(raw)

    @staticmethod
    def _composite_phi(X: np.ndarray, phi: np.ndarray) -> np.ndarray:
        X_abs = np.abs(X)
        col_max = X_abs.max(axis=0)
        col_max[col_max == 0] = 1
        phi_i = (X_abs / col_max * phi).sum(axis=1)
        max_phi = phi_i.max()
        return phi_i / max_phi if max_phi > 0 else phi_i

    def fit(self, X, y):
        X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

        n_rows = len(y_arr)
        rng = np.random.default_rng(self.random_state)
        weights = np.full(n_rows, 1.0 / n_rows)
        cumulative_fi = np.zeros(X_arr.shape[1])
        valid_estimators = 0

        for step in range(self.n_estimators):
            sample_idx = rng.choice(n_rows, size=n_rows, replace=True, p=weights)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=self.random_state + step,
            )
            tree.fit(X_arr[sample_idx], y_arr[sample_idx])

            prediction = tree.predict(X_arr)
            abs_error = np.abs(y_arr - prediction)
            max_error = abs_error.max()
            if max_error == 0:
                break

            norm_error = abs_error / max_error
            eps_t = float(np.dot(weights, norm_error))
            if eps_t >= 0.5:
                break

            beta_t = eps_t / (1.0 - eps_t + 1e-10)
            phi = self._norm_fi(tree)
            phi_i = self._composite_phi(X_arr, phi)

            new_weights = weights * (beta_t ** (1.0 - norm_error * phi_i))
            z_t = new_weights.sum()
            if z_t == 0:
                break
            weights = new_weights / z_t

            estimator_weight = max(
                self.learning_rate * math.log((1.0 - eps_t) / (eps_t + 1e-10)),
                1e-10,
            )
            self.estimators_.append(tree)
            self.estimator_weights_.append(estimator_weight)
            cumulative_fi += phi
            valid_estimators += 1

        self.feature_importances_ = (
            cumulative_fi / valid_estimators
            if valid_estimators > 0
            else np.ones(X_arr.shape[1]) / X_arr.shape[1]
        )
        return self

    def predict(self, X):
        X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        predictions = np.array([est.predict(X_arr) for est in self.estimators_])
        weights = np.asarray(self.estimator_weights_, dtype=float)
        weights = weights / weights.sum()

        result = np.zeros(X_arr.shape[0])
        for row_idx in range(X_arr.shape[0]):
            row_preds = predictions[:, row_idx]
            order = np.argsort(row_preds)
            cumulative = np.cumsum(weights[order])
            midpoint = np.searchsorted(cumulative, 0.5)
            result[row_idx] = row_preds[order[min(midpoint, len(row_preds) - 1)]]
        return result


def register_pickle_compat() -> None:
    if not hasattr(__main__, "BaselineAdaBoost"):
        setattr(__main__, "BaselineAdaBoost", BaselineAdaBoost)
    if not hasattr(__main__, "FIAdaBoostRegressor"):
        setattr(__main__, "FIAdaBoostRegressor", FIAdaBoostRegressor)
