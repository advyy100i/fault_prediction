#!/usr/bin/env python3
"""Feature engineering for ai4i2020 dataset.

Creates three physics-based synthetic features:
- Power_W: 2*pi*Rotational_speed[rpm]*Torque[Nm] / 60
- Temp_Diff_K: Process temperature [K] - Air temperature [K]
- Strain_Load: Tool wear [min] * Torque [Nm]

Reads `ai4i2020.csv` in the current directory and writes
`ai4i2020_features.csv` with the new columns.
"""
import os
import sys
import math
import pandas as pd


INPUT = "ai4i2020.csv"
OUTPUT = "ai4i2020_features.csv"


def main():
    if not os.path.exists(INPUT):
        print(f"Input file '{INPUT}' not found in the current directory.")
        sys.exit(1)

    df = pd.read_csv(INPUT)

    # Required column names (expected in this dataset)
    req = [
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Air temperature [K]",
        "Process temperature [K]",
        "Tool wear [min]",
    ]

    missing = [c for c in req if c not in df.columns]
    if missing:
        print("The input file is missing required columns:", missing)
        print("Available columns:", list(df.columns))
        sys.exit(1)

    # Ensure numeric types (coerce errors to NaN)
    df["Rotational speed [rpm]"] = pd.to_numeric(df["Rotational speed [rpm]"], errors="coerce")
    df["Torque [Nm]"] = pd.to_numeric(df["Torque [Nm]"], errors="coerce")
    df["Air temperature [K]"] = pd.to_numeric(df["Air temperature [K]"], errors="coerce")
    df["Process temperature [K]"] = pd.to_numeric(df["Process temperature [K]"], errors="coerce")
    df["Tool wear [min]"] = pd.to_numeric(df["Tool wear [min]"], errors="coerce")

    # Compute Power in Watts
    df["Power_W"] = 2 * math.pi * df["Rotational speed [rpm]"] * df["Torque [Nm]"] / 60.0

    # Temperature difference (Process - Air)
    df["Temp_Diff_K"] = df["Process temperature [K]"] - df["Air temperature [K]"]

    # Strain load: Tool wear multiplied by torque
    df["Strain_Load"] = df["Tool wear [min]"] * df["Torque [Nm]"]

    df.to_csv(OUTPUT, index=False)

    print(f"Wrote {OUTPUT} with {len(df)} rows and {len(df.columns)} columns.")
    print("Sample of new columns:")
    print(df[["Power_W", "Temp_Diff_K", "Strain_Load"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
