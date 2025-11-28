import os
import sys
import joblib
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.metrics import confusion_matrix, f1_score, classification_report

ROOT = os.path.dirname(__file__)
MODELS_DIR = os.path.join(ROOT, "models")


def load_test():
    X_test_path = os.path.join(ROOT, "X_test.csv")
    y_test_path = os.path.join(ROOT, "y_test.csv")
    if not (os.path.exists(X_test_path) and os.path.exists(y_test_path)):
        print("Missing test files. Run preprocessing_oversampling.py first.")
        sys.exit(1)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()
    return X_test, y_test


def load_sentry():
    sentry_path = os.path.join(MODELS_DIR, "sentry_pipeline.pkl")
    if not os.path.exists(sentry_path):
        print("Sentry model not found. Run train_hierarchical.py first.")
        sys.exit(1)
    return joblib.load(sentry_path)


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    # F1 for failure class (label 1)
    f1_failure = f1_score(y_true, y_pred, pos_label=1)
    report = classification_report(y_true, y_pred, digits=4)
    return cm, f1_failure, report


def feature_importance(pipe, feature_names, top_n=10):
    # Extract the final estimator
    try:
        estimator = pipe.steps[-1][1]
    except Exception:
        estimator = pipe

    importances = None
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        if coef.ndim == 1:
            importances = np.abs(coef)
        else:
            importances = np.sum(np.abs(coef), axis=0)

    if importances is None:
        print("Could not extract feature importances from the sentry estimator.")
        return None

    # align lengths
    L = min(len(importances), len(feature_names))
    importances = importances[:L]
    feature_names = feature_names[:L]

    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    return fi


def check_physics_in_top(fi, physics_aliases, top_k=5):
    top_features = fi.head(top_k)["feature"].tolist()
    found = []
    for phys, aliases in physics_aliases.items():
        present = any(a in top_features for a in aliases)
        found.append((phys, present, [f for f in top_features if any(a == f for a in aliases)]))
    return top_features, found


def main():
    X_test, y_test = load_test()
    sentry = load_sentry()

    feature_names = list(X_test.columns)

    y_pred = sentry.predict(X_test)

    cm, f1_failure, report = compute_metrics(y_test, y_pred)

    print("\n--- PHASE 5 EVALUATION ---\n")
    print("Confusion Matrix (rows=true, cols=pred):\n", cm)
    print(f"\nF1 score (failure class=1): {f1_failure:.4f}\n")
    print("Full classification report:\n")
    print(report)

    fi = feature_importance(sentry, feature_names, top_n=20)
    if fi is not None:
        print("\nTop 20 features by importance (Sentry):")
        print(fi.head(20).to_string(index=False))

        physics_aliases = {
            "Power": ["Power_W", "Power"],
            "Temp_Diff": ["Temp_Diff_K", "Temp_Diff", "Temp_DiffK"],
            "Strain_Load": ["Strain_Load", "StrainLoad"],
        }

        top_feats, found = check_physics_in_top(fi, physics_aliases, top_k=5)

        print(f"\nTop-5 features: {top_feats}")
        for phys, present, matches in found:
            print(f"- {phys}: in_top5={present}; matching_names_in_top5={matches}")

        success = all(present for _, present, _ in found)
        if success:
            print("\nSUCCESS: All three physics-based features appear in the Top 5 features.")
        else:
            print("\nRESULT: Not all physics features are in Top 5. Consider feature tuning or model reweighting.")


if __name__ == "__main__":
    main()
