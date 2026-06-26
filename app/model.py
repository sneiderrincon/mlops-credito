import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path("models/model.joblib")

def train_and_save():
    from sklearn.datasets import make_classification
    from xgboost import XGBClassifier

    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        random_state=42
    )

    model = XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")

def load_model():
    if not MODEL_PATH.exists():
        train_and_save()
    return joblib.load(MODEL_PATH)

def predict(features: list) -> dict:
    model = load_model()
    X = np.array(features).reshape(1, -1)
    prob = model.predict_proba(X)[0][1]
    return {
        "abandono_probable": bool(prob > 0.5),
        "probabilidad": round(float(prob), 4)
    }