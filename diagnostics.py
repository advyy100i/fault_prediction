import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(__file__)
MODELS_DIR = os.path.join(ROOT, "models")


def load_test_data():
    X_test_path = os.path.join(ROOT, "X_test.csv")
    y_test_path = os.path.join(ROOT, "y_test.csv")
    if not (os.path.exists(X_test_path) and os.path.exists(y_test_path)):
        print("Test CSVs not found. Run preprocessing_oversampling.py first.")
        sys.exit(1)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()
    return X_test, y_test


def load_models():
    sentry_path = os.path.join(MODELS_DIR, "sentry_pipeline.pkl")
    specialist_path = os.path.join(MODELS_DIR, "specialist_pipeline.pkl")

    if not os.path.exists(sentry_path):
        print("Sentry model not found at models/sentry_pipeline.pkl")
        sys.exit(1)

    sentry = joblib.load(sentry_path)
    specialist = None
    if os.path.exists(specialist_path):
        specialist = joblib.load(specialist_path)

    return sentry, specialist


def get_feature_names(X_test):
    return list(X_test.columns)


def safe_column(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def analyze_false_negatives(X_test, y_test, y_pred):
    analysis_df = X_test.copy()
    analysis_df["True_Label"] = y_test.values
    analysis_df["Predicted_Label"] = y_pred

    fn = analysis_df[(analysis_df["True_Label"] == 1) & (analysis_df["Predicted_Label"] == 0)]
    tp = analysis_df[(analysis_df["True_Label"] == 1) & (analysis_df["Predicted_Label"] == 1)]

    print(f"Total False Negatives: {len(fn)}")

    # columns may be named 'Temp_Diff_K' in our pipeline
    power_col = safe_column(analysis_df, "Power_W", "Power")
    temp_col = safe_column(analysis_df, "Temp_Diff_K", "Temp_Diff", "Temp_DiffK")
    strain_col = safe_column(analysis_df, "Strain_Load", "StrainLoad")

    cols = [c for c in (power_col, temp_col, strain_col) if c]
    if not cols:
        print("No physics-derived columns found in test set to summarize.")
        return

    print("Average Physics Values for the Missed Failures:")
    print(fn[cols].mean())

    print("\nVs. Average Physics Values for CAUGHT Failures:")
    print(tp[cols].mean())

    print("\nINTERPRETATION:")
    print("If the 'Missed' values are lower/normal compared to 'Caught',")
    print("it suggests those misses could be Random Failures or lacked signal in these physics features.")


def plot_importance_from_pipeline(pipeline, model_name, feature_names, top_n=10, out_file=None):
    # pipeline is expected to be a sklearn Pipeline
    # extract last estimator
    try:
        estimator = pipeline.steps[-1][1]
    except Exception:
        estimator = pipeline

    # retrieve importances
    importances = None
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        if coef.ndim == 1:
            importances = np.abs(coef)
        else:
            # multiclass - sum abs across classes
            importances = np.sum(np.abs(coef), axis=0)

    if importances is None:
        print(f"Cannot extract importance for {model_name}")
        return

    # Some estimators trained without feature names (LightGBM warnings); align by length
    if len(importances) != len(feature_names):
        # attempt to trim or pad
        L = min(len(importances), len(feature_names))
        importances = importances[:L]
        feature_names = feature_names[:L]

    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=fi, palette="viridis")
    plt.title(f"Top {top_n} Features - {model_name}")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
        print(f"Saved feature importance plot to {out_file}")
    plt.show()


def main():
    X_test, y_test = load_test_data()
    sentry, specialist = load_models()

    feature_names = get_feature_names(X_test)

    # Ensure columns order matches what pipeline expects; pipeline will handle scaling
    y_pred_sentry = sentry.predict(X_test)

    analyze_false_negatives(X_test, y_test, y_pred_sentry)

    print("\n--- DIAGNOSTIC 2: PROVING PHYSICS DRIVES THE MODEL ---\n")
    # Plot for Sentry
    plot_importance_from_pipeline(sentry, "Sentry (Detection)", feature_names, out_file=os.path.join(ROOT, "fi_sentry.png"))

    # Plot for Specialist if available
    if specialist is not None:
        # Specialist was trained on failed-only rows; feature names were the original feature set minus failure cols.
        plot_importance_from_pipeline(specialist, "Specialist (Diagnosis)", feature_names, out_file=os.path.join(ROOT, "fi_specialist.png"))


if __name__ == "__main__":
    main()
