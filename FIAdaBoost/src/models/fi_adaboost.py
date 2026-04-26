import math
import numpy as np
from sklearn.tree import DecisionTreeRegressor

RANDOM_SEED = 42

class FIAdaBoostRegressor:
    def __init__(self, n_estimators=181, learning_rate=0.7000, max_depth=10, random_state=RANDOM_SEED):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators_ = []
        self.estimator_weights_ = []
        self.feature_importances_ = None

    @staticmethod
    def _norm_fi(tree):
        raw = tree.feature_importances_
        s = raw.sum()
        return raw / s if s > 0 else np.ones_like(raw) / len(raw)

    @staticmethod
    def _composite_phi(X, phi):
        X_abs = np.abs(X)
        col_mx = X_abs.max(axis=0)
        col_mx[col_mx == 0] = 1
        Phi = (X_abs / col_mx * phi).sum(axis=1)
        p_max = Phi.max()
        return Phi / p_max if p_max > 0 else Phi

    def fit(self, X, y):
        n = len(y)
        rng = np.random.default_rng(self.random_state)
        weights = np.full(n, 1.0 / n)
        cum_fi = np.zeros(X.shape[1])
        n_valid = 0

        for t in range(self.n_estimators):
            idx = rng.choice(n, size=n, replace=True, p=weights)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state + t)
            tree.fit(X[idx], y[idx])
            y_pred = tree.predict(X)
            abs_e = np.abs(y - y_pred)
            D_t = abs_e.max()
            if D_t == 0:
                break

            e_i = abs_e / D_t
            eps_t = float(np.dot(weights, e_i))
            if eps_t >= 0.5:
                break

            beta_t = eps_t / (1.0 - eps_t + 1e-10)
            phi = self._norm_fi(tree)
            Phi_i = self._composite_phi(X, phi)
            new_w = weights * (beta_t ** (1.0 - e_i * Phi_i))
            Z_t = new_w.sum()
            if Z_t == 0:
                break

            weights = new_w / Z_t

            est_w = max(
                self.learning_rate * math.log((1.0 - eps_t) / (eps_t + 1e-10)),
                1e-10
            )
            self.estimators_.append(tree)
            self.estimator_weights_.append(est_w)
            cum_fi += phi
            n_valid += 1

        self.feature_importances_ = (
            cum_fi / n_valid if n_valid > 0 else np.ones(X.shape[1]) / X.shape[1]
        )
        return self

    def predict(self, X):
        preds = np.array([e.predict(X) for e in self.estimators_])
        weights = np.array(self.estimator_weights_)
        weights = weights / weights.sum()

        result = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            p_i = preds[:, i]
            order = np.argsort(p_i)
            cumw = np.cumsum(weights[order])
            mid = np.searchsorted(cumw, 0.5)
            result[i] = p_i[order[min(mid, len(p_i) - 1)]]
        return result