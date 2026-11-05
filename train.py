"""Train a heart disease classifier and save it with joblib.

Usage:
    1. Download heart.csv from
       https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
    2. Put it at data/heart.csv
    3. python train.py
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/heart.csv")
MODEL_PATH = Path("model/heart_model.joblib")

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"Dataset not found at {DATA_PATH}. Download heart.csv from Kaggle first.")

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
