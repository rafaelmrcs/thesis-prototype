import os
import numpy as np
import pandas as pd
import joblib

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def evaluate(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }

if __name__ == "__main__":
    year = "2024"
    path = os.path.join(PROCESSED_DIR, f"baseline_spatial_clean_{year}.csv")
    df = pd.read_csv(path)

    target = f"GHI_mean_{year}"
    X = df[["lat", "lon"]]
    y = df[target].astype(float)

    # Match paper-ish split 2500 train / 500 test (~16.7% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.167, random_state=42
    )

    model = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=3, random_state=42),
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    res = evaluate(y_test, pred)
    print("--- BASELINE REPRODUCTION (NASA POWER) ---")
    print(f"RMSE={res['RMSE']:.4f}, MAE={res['MAE']:.4f}, R2={res['R2']:.4f}")

    joblib.dump(model, os.path.join(MODEL_DIR, "baseline_reproduction_adaboost.pkl"))
    print("Saved: models/baseline_reproduction_adaboost.pkl")
