#!/usr/bin/env python3
"""Train hierarchical failure classifier (Sentry + Specialist).

Sentry (Stage 1): binary classifier detecting Machine failure.
 - Trained on SMOTE-balanced training set (`X_train_res.csv`, `y_train_res.csv`).
 - Uses XGBoost if available, otherwise RandomForest.
 - Optimizes for recall using randomized search.

Specialist (Stage 2): multi-class classifier predicting Failure_Type.
 - Trained on original rows where `Machine failure` == 1.
 - Maps TWF,HDF,PWF,OSF,RNF -> Failure_Type {0..4}.
 - Uses LightGBM (if available) with class_weight='balanced'.

Saves models to `models/` and prints evaluation metrics.
"""
import os
import sys
import joblib
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

RANDOM_STATE = 42
ROOT = os.path.dirname(__file__)
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_smote_data():
    # Prefer files produced by preprocessing_oversampling.py
    X_path = os.path.join(ROOT, "X_train_res.csv")
    y_path = os.path.join(ROOT, "y_train_res.csv")
    X_test_path = os.path.join(ROOT, "X_test.csv")
    y_test_path = os.path.join(ROOT, "y_test.csv")

    if not (os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(X_test_path) and os.path.exists(y_test_path)):
        print("Balanced train/test CSVs not found. Run preprocessing_oversampling.py first.")
        sys.exit(1)

    X_train_res = pd.read_csv(X_path)
    y_train_res = pd.read_csv(y_path).squeeze()
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()

    return X_train_res, y_train_res, X_test, y_test


def train_sentry(X_train, y_train, X_test, y_test):
    # Use XGBoost if available
    try:
        from xgboost import XGBClassifier

        estimator = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE)
        param_dist = {
            "estimator__n_estimators": [50, 100, 200],
            "estimator__max_depth": [3, 5, 7],
            "estimator__learning_rate": [0.01, 0.1, 0.2],
        }
        print("Using XGBoost for Sentry.")
    except Exception:
        estimator = RandomForestClassifier(random_state=RANDOM_STATE)
        param_dist = {
            "estimator__n_estimators": [100, 200],
            "estimator__max_depth": [None, 10, 20],
        }
        print("XGBoost not available; using RandomForest for Sentry.")

    pipe = Pipeline([("scaler", StandardScaler()), ("estimator", estimator)])

    # Use a smaller search and avoid parallel jobs to keep runtime predictable in this environment
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=4,
        scoring="recall",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    search.fit(X_train, y_train)

    best = search.best_estimator_
    preds = best.predict(X_test)

    print("Sentry - Best params:", search.best_params_)
    print("Sentry - Test classification report (recall-focused):")
    print(classification_report(y_test, preds, digits=4))
    print("Sentry - Confusion matrix:\n", confusion_matrix(y_test, preds))

    joblib.dump(best, os.path.join(MODELS_DIR, "sentry_pipeline.pkl"))
    print("Saved Sentry pipeline to models/sentry_pipeline.pkl")

    return best


def build_failure_type(df):
    # Deprecated: specialist data preparation handled in train_specialist
    raise RuntimeError("build_failure_type is deprecated; specialist data is prepared inside train_specialist")


def train_specialist(df):
    # Prepare Specialist dataset from original features CSV
    FILE_ORIG = os.path.join(ROOT, "ai4i2020_features.csv")
    if not os.path.exists(FILE_ORIG):
        print("Original features file not found for Specialist training:", FILE_ORIG)
        return None

    print("Preparing Specialist Data (Original Failures Only)...")
    df_full = pd.read_csv(FILE_ORIG)

    # 1. Encode Type if present
    mapping = {"L": 0, "M": 1, "H": 2}
    if "Type" in df_full.columns and "Type_encoded" not in df_full.columns:
        df_full["Type_encoded"] = df_full["Type"].map(mapping)

    failure_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    for c in failure_cols:
        if c not in df_full.columns:
            df_full[c] = 0

    # Remove ambiguous failures where Machine failure==1 but no specific type flagged
    df_full["sum_failures"] = df_full[failure_cols].sum(axis=1)
    mask_valid = (df_full.get("Machine failure", 0) == 1) & (df_full["sum_failures"] > 0)
    df_failures = df_full[mask_valid].copy()
    print(f"Filtered down to {len(df_failures)} valid, labeled failure rows.")

    if df_failures.empty:
        print("No valid labeled failures found after filtering ambiguous rows.")
        return None

    # Create multi-class target using idxmax on the failure columns
    y_type = df_failures[failure_cols].idxmax(axis=1)
    type_map = {"TWF": 0, "HDF": 1, "PWF": 2, "OSF": 3, "RNF": 4}
    y_multiclass = y_type.map(type_map)

    # Drop identifier and target columns to form features
    cols_to_drop = ["UDI", "Product ID", "Type", "Machine failure", "sum_failures"] + failure_cols
    cols_to_drop = [c for c in cols_to_drop if c in df_failures.columns]
    X_specialist = df_failures.drop(columns=cols_to_drop)

    # Ensure numeric
    for c in X_specialist.select_dtypes(exclude=["number"]).columns:
        X_specialist[c] = pd.to_numeric(X_specialist[c], errors="coerce")
    X_specialist = X_specialist.fillna(0)

    # Split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_specialist, y_multiclass, test_size=0.2, stratify=y_multiclass, random_state=RANDOM_STATE
    )

    print("Specialist dataset class distribution:", Counter(y_tr))

    try:
        import lightgbm as lgb

        clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=5,
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
        print("Using LightGBM for Specialist.")
    except Exception:
        clf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
        print("LightGBM not available; using RandomForest for Specialist.")

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_val)

    print("Specialist - Validation classification report:")
    print(classification_report(y_val, preds, digits=4))
    print("Specialist - Confusion matrix:\n", confusion_matrix(y_val, preds))

    joblib.dump(pipe, os.path.join(MODELS_DIR, "specialist_pipeline.pkl"))
    print("Saved Specialist pipeline to models/specialist_pipeline.pkl")

    return pipe


def main():
    # Load smote-balanced train/test for Sentry
    X_train_res, y_train_res, X_test, y_test = load_smote_data()

    print("Sentry training dataset distribution:", Counter(y_train_res))

    sentry = train_sentry(X_train_res, y_train_res, X_test, y_test)

    # Load original features for Specialist
    orig_path = os.path.join(ROOT, "ai4i2020_features.csv")
    if not os.path.exists(orig_path):
        print("Original features CSV not found: ai4i2020_features.csv")
        sys.exit(1)

    df = pd.read_csv(orig_path)

    # Ensure Type encoded exists; if not, compute
    if "Type_encoded" not in df.columns and "Type" in df.columns:
        mapping = {"L": 0, "M": 1, "H": 2}
        df["Type_encoded"] = df["Type"].map(mapping)

    specialist = train_specialist(df)


if __name__ == "__main__":
    main()
