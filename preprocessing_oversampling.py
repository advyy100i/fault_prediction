#!/usr/bin/env python3
"""Preprocessing and oversampling (SMOTE) for ai4i2020 features.

Steps:
- Read `ai4i2020_features.csv` (produced by feature_engineering.py)
- Ordinal-encode `Type`: L->0, M->1, H->2
- Split into train/test (80/20) with `stratify=y` on `Machine failure`
- Apply SMOTE only to training set
- Save balanced training set and test set to CSV

Outputs:
- `X_train_res.csv`, `y_train_res.csv`, `X_test.csv`, `y_test.csv`

If `imbalanced-learn` is missing, the script prints an install hint.
"""
import os
import sys
from collections import Counter

import pandas as pd


INPUT = "ai4i2020_features.csv"
RANDOM_STATE = 42


def check_file():
    if not os.path.exists(INPUT):
        print(f"Input file '{INPUT}' not found. Run feature_engineering.py first.")
        sys.exit(1)


def ordinal_encode_type(df):
    mapping = {"L": 0, "M": 1, "H": 2}
    if "Type" in df.columns:
        df["Type_encoded"] = df["Type"].map(mapping)
    else:
        raise KeyError("Column 'Type' not found in input dataframe")
    return df


def prepare_features_and_target(df):
    # Identify target
    if "Machine failure" not in df.columns:
        raise KeyError("Target column 'Machine failure' not found")

    y = df["Machine failure"].astype(int)

    # Columns to drop from features (identifiers and explicit failure columns)
    drop_cols = [
        "UDI",
        "Product ID",
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
        "Type",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)

    # If there are non-numeric columns left, attempt to numeric-convert or drop
    non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        # Attempt to convert any leftover to numeric; otherwise drop with a warning
        for c in non_numeric:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        still_non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
        if still_non_numeric:
            print("Dropping non-numeric columns from features:", still_non_numeric)
            X = X.drop(columns=still_non_numeric)

    return X, y


def split_data(X, y):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    try:
        from imblearn.over_sampling import SMOTE
    except Exception:
        print("Package 'imbalanced-learn' not installed. Install it with:")
        print("  pip install -U imbalanced-learn")
        sys.exit(1)

    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res


def save_outputs(X_train_res, y_train_res, X_test, y_test):
    X_train_res.to_csv("X_train_res.csv", index=False)
    pd.Series(y_train_res, name="Machine failure").to_csv("y_train_res.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    pd.Series(y_test, name="Machine failure").to_csv("y_test.csv", index=False)


def print_distribution(msg, y):
    c = Counter(y)
    total = sum(c.values())
    print(f"{msg}: {dict(c)} (total={total})")


def main():
    check_file()

    df = pd.read_csv(INPUT)

    df = ordinal_encode_type(df)

    X, y = prepare_features_and_target(df)

    print_distribution("Original distribution", y)

    X_train, X_test, y_train, y_test = split_data(X, y)

    print_distribution("Train distribution before SMOTE", y_train)
    print_distribution("Test distribution", y_test)

    X_res, y_res = apply_smote(X_train, y_train)

    print_distribution("Train distribution after SMOTE", y_res)

    save_outputs(X_res, y_res, X_test, y_test)

    print("Saved: X_train_res.csv, y_train_res.csv, X_test.csv, y_test.csv")


if __name__ == "__main__":
    main()
