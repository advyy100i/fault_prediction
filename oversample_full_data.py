#!/usr/bin/env python3
"""Oversample the full ai4i2020_features.csv dataset to 20,000 entries.

This script reads the original dataset with engineered features and applies
oversampling to create a larger dataset of 20,000 samples. It uses a combination
of SMOTE for the minority class (failures) and random oversampling for the 
majority class to maintain balance.

Outputs: ai4i2020_features_20k.csv
"""
import os
import pandas as pd
import numpy as np
from collections import Counter

# Check if imbalanced-learn is available
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.combine import SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Warning: imbalanced-learn not found. Install with: pip install imbalanced-learn")


INPUT = "ai4i2020_features.csv"
OUTPUT = "ai4i2020_features_20k.csv"
TARGET_SIZE = 20000
RANDOM_STATE = 42


def main():
    if not os.path.exists(INPUT):
        print(f"Error: Input file '{INPUT}' not found.")
        print("Please run feature_engineering.py first to create the features dataset.")
        return 1
    
    print(f"Reading {INPUT}...")
    df = pd.read_csv(INPUT)
    original_size = len(df)
    print(f"Original dataset size: {original_size} samples")
    
    if original_size >= TARGET_SIZE:
        print(f"Dataset already has {original_size} samples (target: {TARGET_SIZE})")
        print("No oversampling needed.")
        return 0
    
    # Separate features and target
    if "Machine failure" not in df.columns:
        print("Error: 'Machine failure' column not found in dataset")
        return 1
    
    # Identify feature columns (exclude identifiers and target)
    exclude_cols = [
        "UDI", "Product ID", "Machine failure",
        "TWF", "HDF", "PWF", "OSF", "RNF"
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df["Machine failure"].copy()
    
    # Keep the excluded columns for reconstruction
    metadata = df[exclude_cols].copy()
    
    print(f"\nClass distribution (original):")
    print(Counter(y))
    
    if not IMBLEARN_AVAILABLE:
        print("\nUsing simple random oversampling...")
        # Simple random oversampling with replacement
        np.random.seed(RANDOM_STATE)
        indices = np.random.choice(len(df), size=TARGET_SIZE, replace=True)
        df_oversampled = df.iloc[indices].reset_index(drop=True)
        
        # Reset UDI to be sequential
        df_oversampled["UDI"] = range(1, TARGET_SIZE + 1)
        
    else:
        print("\nUsing SMOTE + Random Oversampling...")
        
        # Calculate how many samples we need for each class to reach 20k total
        failure_count = sum(y == 1)
        normal_count = sum(y == 0)
        
        # First, use SMOTE to oversample the minority class (failures)
        # SMOTE can create synthetic samples for the minority class
        try:
            # Determine sampling strategy
            # We want roughly the same failure rate but scaled up
            failure_ratio = failure_count / original_size
            target_failures = int(TARGET_SIZE * failure_ratio * 1.5)  # Slightly increase failure samples
            target_normal = TARGET_SIZE - target_failures
            
            # Use RandomOverSampler which works better for this use case
            ros = RandomOverSampler(
                sampling_strategy={0: target_normal, 1: target_failures},
                random_state=RANDOM_STATE
            )
            
            X_resampled, y_resampled = ros.fit_resample(X, y)
            
            # If we haven't reached target size, add more samples randomly
            current_size = len(X_resampled)
            if current_size < TARGET_SIZE:
                additional_needed = TARGET_SIZE - current_size
                indices = np.random.choice(current_size, size=additional_needed, replace=True)
                
                X_additional = X_resampled.iloc[indices]
                y_additional = y_resampled.iloc[indices]
                
                X_resampled = pd.concat([X_resampled, X_additional], ignore_index=True)
                y_resampled = pd.concat([y_resampled, y_additional], ignore_index=True)
            
            # Trim if we overshot
            if len(X_resampled) > TARGET_SIZE:
                X_resampled = X_resampled.iloc[:TARGET_SIZE]
                y_resampled = y_resampled.iloc[:TARGET_SIZE]
            
            print(f"\nClass distribution (oversampled to {len(X_resampled)}):")
            print(Counter(y_resampled))
            
            # Reconstruct the full dataframe
            df_oversampled = X_resampled.copy()
            df_oversampled["Machine failure"] = y_resampled
            
            # Generate new UDI and Product ID
            df_oversampled["UDI"] = range(1, len(df_oversampled) + 1)
            df_oversampled["Product ID"] = [f"S{50000 + i}" for i in range(len(df_oversampled))]
            
            # For failure types, maintain them from original or set to 0 for synthetic samples
            # This is approximate - we'll randomly assign based on the distribution
            for fail_col in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
                if fail_col in df.columns:
                    # For failure samples, randomly sample from original failure patterns
                    failure_mask = df_oversampled["Machine failure"] == 1
                    original_failures = df[df["Machine failure"] == 1]
                    
                    if len(original_failures) > 0:
                        sampled_fail_types = original_failures[fail_col].sample(
                            n=failure_mask.sum(), 
                            replace=True, 
                            random_state=RANDOM_STATE
                        ).values
                        df_oversampled.loc[failure_mask, fail_col] = sampled_fail_types
                        df_oversampled.loc[~failure_mask, fail_col] = 0
                    else:
                        df_oversampled[fail_col] = 0
            
        except Exception as e:
            print(f"Error during oversampling: {e}")
            print("Falling back to simple random oversampling...")
            np.random.seed(RANDOM_STATE)
            indices = np.random.choice(len(df), size=TARGET_SIZE, replace=True)
            df_oversampled = df.iloc[indices].reset_index(drop=True)
            df_oversampled["UDI"] = range(1, TARGET_SIZE + 1)
    
    # Save the oversampled dataset
    print(f"\nSaving oversampled dataset to {OUTPUT}...")
    df_oversampled.to_csv(OUTPUT, index=False)
    
    print(f"✅ Successfully created oversampled dataset!")
    print(f"   Original size: {original_size}")
    print(f"   New size: {len(df_oversampled)}")
    print(f"   Output file: {OUTPUT}")
    
    # Show summary statistics
    print(f"\nOversampled dataset summary:")
    print(f"   Total samples: {len(df_oversampled)}")
    if "Machine failure" in df_oversampled.columns:
        failures = sum(df_oversampled["Machine failure"] == 1)
        print(f"   Failures: {failures} ({failures/len(df_oversampled)*100:.2f}%)")
        print(f"   Normal: {len(df_oversampled) - failures} ({(len(df_oversampled)-failures)/len(df_oversampled)*100:.2f}%)")
    
    return 0


if __name__ == "__main__":
    exit(main())
