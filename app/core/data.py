"""
Data handling and machine learning utilities.

This module provides functions to load the Pima Indians Diabetes dataset,
train a simple machine learning model, and generate risk predictions.
The default model is a logistic regression classifier trained on all features.

Note: A CSV file named ``diabetes.csv`` must be placed in the ``data`` directory
within the project root. This file should have a column named ``Outcome`` with
binary labels (1 for diabetes, 0 for no diabetes) and the following feature
columns:

    - Pregnancies
    - Glucose
    - BloodPressure
    - SkinThickness
    - Insulin
    - BMI
    - DiabetesPedigreeFunction
    - Age
"""
from __future__ import annotations

"""
Human-friendly data utilities for the Diabetes Risk & Care System.

This module provides:
- A safe loader that tries to read `data/diabetes.csv` and, if missing,
  builds a small synthetic dataset so the app *always* runs.
- A tiny training routine (logistic regression).
- A simple risk predictor returning a probability + human label.
"""

from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Columns we expect in the dataset
FEATURES: List[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET: str = "Outcome"


def _add_headers_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the CSV lacks headers, assign typical Pima feature names.
    """
    if df.columns.tolist() == list(range(len(df.columns))):
        # Assign only up to the number of columns present
        default_cols = FEATURES + [TARGET]
        df.columns = default_cols[: len(df.columns)]
    return df


def load_diabetes_csv_or_synthesize() -> Tuple[pd.DataFrame, str]:
    """
    Try to load `data/diabetes.csv`. If not found or unreadable, return a
    realistic synthetic dataset that looks like Pima data.

    Returns:
        (df, source)
        - df: pandas DataFrame with FEATURES + TARGET columns
        - source: 'file:<path>' or 'synthetic'
    """
    candidate_paths = [
        Path("data/diabetes.csv"),
        Path(__file__).resolve().parents[2] / "data" / "diabetes.csv",
        Path("diabetes.csv"),
    ]

    for p in candidate_paths:
        if p.exists():
            try:
                df = pd.read_csv(p)
                df = _add_headers_if_missing(df)
                # Ensure we at least have the feature columns
                missing = [c for c in FEATURES if c not in df.columns]
                if missing:
                    raise ValueError(f"CSV is missing columns: {missing}")
                # If Outcome is missing, create a placeholder (not ideal, but keeps app running)
                if TARGET not in df.columns:
                    df[TARGET] = 0
                return df, f"file:{p}"
            except Exception:
                # Fall through to synthetic if parsing fails
                pass

    # --- Fallback: synthesize realistic-ish data so app always runs ---
    rng = np.random.default_rng(42)
    n = 600
    pregnancies = rng.integers(0, 15, n)
    glucose = rng.normal(120, 30, n).clip(60, 300)
    bp = rng.normal(72, 12, n).clip(40, 140)
    skin = rng.normal(23, 8, n).clip(0, 99)
    insulin = rng.normal(80, 60, n).clip(0, 300)
    bmi = rng.normal(30, 6, n).clip(15, 60)
    dpf = rng.normal(0.5, 0.25, n).clip(0.05, 2.5)
    age = rng.integers(18, 80, n)

    # Logit-like risk; higher glucose/BMI/age/DPF increase risk
    z = (
        0.015 * glucose
        + 0.02 * bmi
        + 0.01 * age
        + 0.008 * bp
        + 0.5 * dpf
        + 0.02 * pregnancies
        - 10.0
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    outcome = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame(
        {
            "Pregnancies": pregnancies,
            "Glucose": glucose.round(0),
            "BloodPressure": bp.round(0),
            "SkinThickness": skin.round(0),
            "Insulin": insulin.round(0),
            "BMI": bmi.round(1),
            "DiabetesPedigreeFunction": dpf.round(3),
            "Age": age,
            "Outcome": outcome,
        }
    )
    return df, "synthetic"


def train_model(df: pd.DataFrame) -> LogisticRegression:
    """
    Train a small logistic regression model.

    Notes:
      - This is intentionally simple and fast.
      - We validate the required columns before training.
    """
    df = df.copy()

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    X = df[FEATURES].astype(float).values
    y = df[TARGET].astype(int).values

    # Simple holdout split (no persistence for this demo)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=300, solver="lbfgs")
    model.fit(X_train, y_train)
    return model


def _to_label(probability: float) -> str:
    """
    Convert a 0..1 probability into a human label.
    """
    if probability < 0.33:
        return "Low"
    if probability < 0.66:
        return "Moderate"
    return "High"


def predict_risk(
    model: LogisticRegression,
    pregnancies: float,
    glucose: float,
    blood_pressure: float,
    skin_thickness: float,
    insulin: float,
    bmi: float,
    dpf: float,
    age: float,
) -> tuple[float, str]:
    """
    Predict the probability of diabetes with a trained model.
    Returns (probability, label).
    """
    X = np.array(
        [
            [
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                dpf,
                age,
            ]
        ],
        dtype=float,
    )

    if hasattr(model, "predict_proba"):
        p = float(model.predict_proba(X)[0, 1])
    else:
        # Fallback if the model lacks predict_proba
        raw = float(model.decision_function(X))
        p = 1.0 / (1.0 + np.exp(-raw))

    return p, _to_label(p)
