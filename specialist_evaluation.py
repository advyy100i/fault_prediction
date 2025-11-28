#!/usr/bin/env python3
"""Evaluate the Specialist (multi-class) model and save reports/plots.

Produces:
- `specialist_evaluation.txt` : text summary with classification report and metrics
- `specialist_confusion.png` : confusion matrix heatmap
- `fi_specialist.png` : top feature importances (model or permutation)

Run: `python specialist_evaluation.py`
"""
import os
import sys
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive backend to avoid GUI hangs in headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.inspection import permutation_importance

ROOT = os.path.dirname(__file__)
MODELS_DIR = os.path.join(ROOT, "models")
SPECIALIST_PATH = os.path.join(MODELS_DIR, "specialist_pipeline.pkl")
FEATURES_PATH = os.path.join(ROOT, "ai4i2020_features.csv")

OUT_TXT = os.path.join(ROOT, "specialist_evaluation.txt")
OUT_CONF = os.path.join(ROOT, "specialist_confusion.png")
OUT_FI = os.path.join(ROOT, "fi_specialist.png")


def prepare_specialist_data(df):
    failure_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    for c in failure_cols:
        if c not in df.columns:
            df[c] = 0

    df["sum_failures"] = df[failure_cols].sum(axis=1)
    mask_valid = (df.get("Machine failure", 0) == 1) & (df["sum_failures"] > 0)
    df_failures = df[mask_valid].copy()
    if df_failures.empty:
        raise RuntimeError("No valid labeled failures found for Specialist evaluation.")

    y_type = df_failures[failure_cols].idxmax(axis=1)
    type_map = {"TWF": 0, "HDF": 1, "PWF": 2, "OSF": 3, "RNF": 4}
    y = y_type.map(type_map)

    cols_to_drop = ["UDI", "Product ID", "Type", "Machine failure", "sum_failures"] + failure_cols
    cols_to_drop = [c for c in cols_to_drop if c in df_failures.columns]
    X = df_failures.drop(columns=cols_to_drop)

    # Ensure numeric
    for c in X.select_dtypes(exclude=["number"]).columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0)

    return X, y, df_failures


def main():
    if not os.path.exists(SPECIALIST_PATH):
        print("Specialist model not found:", SPECIALIST_PATH)
        sys.exit(1)
    if not os.path.exists(FEATURES_PATH):
        print("Features file not found:", FEATURES_PATH)
        sys.exit(1)

    pipe = joblib.load(SPECIALIST_PATH)
    df = pd.read_csv(FEATURES_PATH)
    X, y, df_failures = prepare_specialist_data(df)

    print("Specialist evaluation rows:", len(X))

    # Align columns to what the pipeline saw during training to avoid feature-name mismatch
    expected = None
    try:
        # check scaler or pipeline for feature names
        if hasattr(pipe, "named_steps"):
            for step in pipe.named_steps.values():
                if hasattr(step, "feature_names_in_"):
                    expected = list(step.feature_names_in_)
                    break

        if expected is None and hasattr(pipe, "feature_names_in_"):
            expected = list(pipe.feature_names_in_)
    except Exception:
        expected = None

    if expected is not None:
        missing = [c for c in expected if c not in X.columns]
        if missing:
            print(f"Adding missing columns for evaluation: {missing}")
            for c in missing:
                X[c] = 0
        # Reorder to expected
        X = X[expected]
    else:
        # Best-effort fallback: ensure Type_encoded exists (common cause of mismatch)
        if "Type_encoded" not in X.columns:
            print("Warning: `Type_encoded` missing; adding with zeros as fallback.")
            X["Type_encoded"] = 0

    preds = pipe.predict(X)

    report = classification_report(y, preds, digits=4, output_dict=False)
    report_dict = classification_report(y, preds, digits=4, output_dict=True)
    cm = confusion_matrix(y, preds)
    acc = accuracy_score(y, preds)
    f1_macro = f1_score(y, preds, average="macro")
    f1_micro = f1_score(y, preds, average="micro")

    # Save textual report
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("Specialist Model Evaluation\n")
        f.write("===========================\n\n")
        f.write(f"Rows evaluated: {len(X)}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 macro: {f1_macro:.4f}\n")
        f.write(f"F1 micro: {f1_micro:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    print("Saved evaluation text to", OUT_TXT)

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Specialist Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUT_CONF)
    plt.close()
    print("Saved confusion matrix to", OUT_CONF)

    # Feature importances: try to use model feature_importances_, otherwise use permutation
    try:
        # Extract estimator from pipeline if present
        if hasattr(pipe, "named_steps") and "clf" in pipe.named_steps:
            est = pipe.named_steps["clf"]
        else:
            # If pipeline has different naming, try last step
            est = list(pipe.steps)[-1][1]

        if hasattr(est, "feature_importances_"):
            importances = est.feature_importances_
            fi = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        else:
            raise AttributeError("No feature_importances_ attribute")
    except Exception:
        print("Falling back to permutation importance (may be slower)...")
        perm = permutation_importance(pipe, X, y, n_repeats=10, random_state=42, n_jobs=1)
        fi = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

    topk = fi.head(20)
    plt.figure(figsize=(8, max(4, 0.2 * len(topk))))
    sns.barplot(x=topk.values, y=topk.index, palette="viridis")
    plt.xlabel("Importance")
    plt.title("Specialist - Top feature importances")
    plt.tight_layout()
    plt.savefig(OUT_FI)
    plt.close()
    print("Saved feature importances to", OUT_FI)

    # Also print summary to stdout
    print("Accuracy:", acc)
    print("F1 macro:", f1_macro)
    print(report)


if __name__ == "__main__":
    main()
