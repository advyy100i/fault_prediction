"""
04_safety_net.py - Unsupervised Anomaly Detection for Predictive Maintenance
==============================================================================

PURPOSE: The "Safety Net" - A Second Line of Defense
-----------------------------------------------------
Our supervised XGBoost model is excellent at detecting KNOWN failure modes
(TWF, HDF, PWF, OSF, RNF). But what about failures it has never seen?

This is the fundamental limitation of supervised learning:
- It can only recognize patterns it was trained on.
- Novel failure modes, sensor drift, or rare edge cases slip through.

THE SOLUTION: Isolation Forest (Unsupervised Anomaly Detection)
---------------------------------------------------------------
Instead of learning "what does a failure look like?", we flip the question:
"What does NORMAL operation look like?"

Any data point that deviates significantly from learned normality is flagged
as an anomaly - even if we've never seen that specific failure before.

WHY ISOLATION FOREST?
---------------------
1. It's based on the principle that anomalies are "few and different".
2. Anomalies are easier to ISOLATE - they require fewer random splits.
3. Normal points are clustered and require many splits to isolate.
4. It's fast, scales well, and doesn't require distance calculations.

TRAINING PHILOSOPHY:
--------------------
We train ONLY on healthy machines (Target == 0).
This teaches the model the "shape" of normality in feature space.
Any new observation that falls outside this learned shape is suspicious.

Author: Advay
Date: January 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Features selected for anomaly detection
# WHY THESE FEATURES?
# - Rotational speed [rpm]: Core operational parameter
# - Torque [Nm]: Load indicator
# - Tool wear [min]: Degradation over time
# - Power_W: Computed power (Speed × Torque × 2π/60) - physics-based
# - Temp_Diff_K: Temperature differential - thermal stress indicator
# - Strain_Load: Torque × Tool wear - compound stress metric
FEATURES = [
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
    'Power_W',
    'Temp_Diff_K',
    'Strain_Load'
]

TARGET = 'Machine failure'
DATA_PATH = 'ai4i2020_features.csv'
MODEL_PATH = 'models/safety_net.pkl'
SCALER_PATH = 'models/safety_net_scaler.pkl'

# Isolation Forest hyperparameters
# contamination=0.02: We expect ~2% of healthy data to be edge cases/noise
# This is conservative - better to flag a few false positives than miss anomalies
CONTAMINATION = 0.02
RANDOM_STATE = 42


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    """
    Load the preprocessed feature dataset.
    
    Returns:
        DataFrame with all features and target variable
    """
    print(f"[INFO] Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Failure distribution:\n{df[TARGET].value_counts()}")
    return df


# =============================================================================
# TRAINING LOGIC
# =============================================================================

def train_isolation_forest(df: pd.DataFrame) -> tuple:
    """
    Train Isolation Forest on HEALTHY data only.
    
    WHY TRAIN ON HEALTHY DATA ONLY?
    --------------------------------
    This is the core principle of one-class classification:
    - We define "normal" by learning from healthy examples only.
    - The model creates a decision boundary around normal operating conditions.
    - Anything outside this boundary is flagged as anomalous.
    
    If we trained on both healthy and failed machines, the model would
    learn a mixed representation, diluting its ability to detect anomalies.
    
    Returns:
        tuple: (trained_model, scaler, healthy_data_scaled, failure_data_scaled)
    """
    print("\n" + "="*60)
    print("TRAINING ISOLATION FOREST - THE SAFETY NET")
    print("="*60)
    
    # Step 1: Separate healthy and failed machines
    healthy_mask = df[TARGET] == 0
    failure_mask = df[TARGET] == 1
    
    healthy_data = df.loc[healthy_mask, FEATURES].copy()
    failure_data = df.loc[failure_mask, FEATURES].copy()
    
    print(f"\n[INFO] Healthy samples (training): {len(healthy_data)}")
    print(f"[INFO] Failure samples (held out): {len(failure_data)}")
    
    # Step 2: Scale the features
    # WHY SCALE?
    # Isolation Forest is tree-based and technically doesn't require scaling.
    # However, scaling helps with:
    # - PCA visualization (required)
    # - Consistent feature importance interpretation
    # - Better decision boundary visualization
    scaler = StandardScaler()
    healthy_scaled = scaler.fit_transform(healthy_data)
    failure_scaled = scaler.transform(failure_data)
    
    print(f"\n[INFO] Features scaled using StandardScaler")
    
    # Step 3: Train Isolation Forest on HEALTHY data only
    print(f"\n[INFO] Training Isolation Forest...")
    print(f"       - contamination: {CONTAMINATION}")
    print(f"       - random_state: {RANDOM_STATE}")
    
    iso_forest = IsolationForest(
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_estimators=100,      # Number of trees in the forest
        max_samples='auto',    # Subsample size for each tree
        n_jobs=-1,             # Use all CPU cores
        verbose=0
    )
    
    iso_forest.fit(healthy_scaled)
    
    print(f"[INFO] Isolation Forest trained successfully!")
    
    return iso_forest, scaler, healthy_scaled, failure_scaled


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_model(model, scaler) -> None:
    """
    Save the trained model and scaler for production use.
    
    WHY SAVE BOTH?
    - The model expects scaled input, so we need the exact same scaler
      that was fitted on training data.
    - Saving them together ensures consistency in production.
    """
    print(f"\n[INFO] Saving model to: {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)
    
    print(f"[INFO] Saving scaler to: {SCALER_PATH}")
    joblib.dump(scaler, SCALER_PATH)
    
    print("[INFO] Model and scaler saved successfully!")


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_decision_boundary(model, scaler, healthy_scaled, failure_scaled) -> None:
    """
    Create a 2D visualization of the Isolation Forest decision boundary.
    
    WHY PCA?
    - Our feature space is 6-dimensional - impossible to visualize directly.
    - PCA projects data to 2D while preserving maximum variance.
    - This gives us a meaningful view of the data structure.
    
    WHY SHOW DECISION BOUNDARY?
    - Visual confirmation that failures fall OUTSIDE the normal region.
    - Helps understand how the model separates normal from anomalous.
    - Useful for explaining the model to stakeholders.
    """
    print("\n[INFO] Creating visualization with PCA...")
    
    # Step 1: Reduce dimensions with PCA
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    healthy_pca = pca.fit_transform(healthy_scaled)
    failure_pca = pca.transform(failure_scaled)
    
    explained_var = sum(pca.explained_variance_ratio_) * 100
    print(f"[INFO] PCA explained variance: {explained_var:.1f}%")
    
    # Step 2: Create mesh grid for decision boundary
    # Combine all points to determine plot boundaries
    all_points = np.vstack([healthy_pca, failure_pca])
    x_min, x_max = all_points[:, 0].min() - 1, all_points[:, 0].max() + 1
    y_min, y_max = all_points[:, 1].min() - 1, all_points[:, 1].max() + 1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    
    # Step 3: Get predictions for mesh grid
    # We need to transform mesh points back to original feature space
    # This is an approximation since PCA is not perfectly invertible
    mesh_points_pca = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_original = pca.inverse_transform(mesh_points_pca)
    
    # Get anomaly scores for visualization
    Z = model.decision_function(mesh_points_original)
    Z = Z.reshape(xx.shape)
    
    # Step 4: Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot decision boundary (contour)
    # Negative scores = anomalies, Positive scores = normal
    contour = ax.contourf(xx, yy, Z, levels=50, cmap='RdYlBu', alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2, linestyles='--')
    
    # Plot healthy points (training data)
    ax.scatter(
        healthy_pca[:, 0], 
        healthy_pca[:, 1], 
        c='blue', 
        s=20, 
        alpha=0.5, 
        label=f'Healthy (n={len(healthy_pca)})',
        edgecolors='none'
    )
    
    # Plot failure points (anomalies)
    ax.scatter(
        failure_pca[:, 0], 
        failure_pca[:, 1], 
        c='red', 
        s=60, 
        alpha=0.9, 
        label=f'Failures (n={len(failure_pca)})',
        edgecolors='black',
        marker='X'
    )
    
    # Formatting
    ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title(
        'Isolation Forest: Safety Net for Anomaly Detection\n'
        'Blue Region = Normal | Red Region = Anomalous | Black Line = Decision Boundary',
        fontsize=14,
        fontweight='bold'
    )
    ax.legend(loc='upper right', fontsize=11)
    
    # Add colorbar for anomaly scores
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Anomaly Score\n(Negative = Anomaly, Positive = Normal)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('safety_net_visualization.png', dpi=150, bbox_inches='tight')
    print("[INFO] Visualization saved to: safety_net_visualization.png")
    plt.show()
    
    # Step 5: Evaluate detection on actual failures
    print("\n" + "="*60)
    print("ANOMALY DETECTION PERFORMANCE")
    print("="*60)
    
    healthy_predictions = model.predict(healthy_scaled)
    failure_predictions = model.predict(failure_scaled)
    
    # In Isolation Forest: -1 = anomaly, 1 = normal
    healthy_flagged = (healthy_predictions == -1).sum()
    failures_detected = (failure_predictions == -1).sum()
    
    print(f"\n[RESULTS]")
    print(f"  Healthy machines flagged as anomaly: {healthy_flagged}/{len(healthy_predictions)} ({100*healthy_flagged/len(healthy_predictions):.1f}%)")
    print(f"  Actual failures detected as anomaly: {failures_detected}/{len(failure_predictions)} ({100*failures_detected/len(failure_predictions):.1f}%)")
    
    if failures_detected / len(failure_predictions) > 0.5:
        print("\n  ✓ SUCCESS: Majority of actual failures detected as anomalies!")
    else:
        print("\n  ⚠ WARNING: Detection rate is low - consider tuning contamination or features.")


# =============================================================================
# GHOST MACHINE SIMULATION
# =============================================================================

def simulate_ghost_machine(model, scaler) -> None:
    """
    Simulate a "Ghost Machine" with physically impossible values.
    
    THE CONCEPT:
    ------------
    A "Ghost Machine" has values that look normal individually but are
    physically impossible when considered together.
    
    Example: Low Speed + Low Torque but HIGH Power
    - Power = Torque × Angular Velocity
    - If speed and torque are low, power MUST be low (physics!)
    - High power with low speed/torque violates conservation of energy
    
    This tests if our model can detect physics-violating anomalies that
    simple threshold-based systems would miss.
    """
    print("\n" + "="*60)
    print("GHOST MACHINE SIMULATION")
    print("="*60)
    
    # Create a physically impossible machine state
    # Normal ranges (from data):
    # - Rotational speed: ~1200-2000 rpm
    # - Torque: ~20-60 Nm
    # - Power_W: typically Speed × Torque × 2π/60
    
    ghost_machine = {
        'Rotational speed [rpm]': 1200,      # Low-ish speed (normal individually)
        'Torque [Nm]': 25,                   # Low torque (normal individually)
        'Tool wear [min]': 100,              # Moderate wear (normal individually)
        'Power_W': 15000,                    # IMPOSSIBLE! Should be ~3140W with this speed/torque
        'Temp_Diff_K': 8,                    # Normal temperature difference
        'Strain_Load': 50                    # Low strain (normal individually)
    }
    
    # Calculate what Power_W SHOULD be based on physics
    expected_power = ghost_machine['Rotational speed [rpm]'] * ghost_machine['Torque [Nm]'] * 2 * np.pi / 60
    
    print("\n[GHOST MACHINE PROFILE]")
    print("-" * 40)
    for feature, value in ghost_machine.items():
        print(f"  {feature}: {value}")
    
    print(f"\n[PHYSICS CHECK]")
    print(f"  Expected Power (P = τω): {expected_power:.1f} W")
    print(f"  Reported Power: {ghost_machine['Power_W']:.1f} W")
    print(f"  Power Discrepancy: {ghost_machine['Power_W'] - expected_power:.1f} W (IMPOSSIBLE!)")
    
    # Prepare for prediction
    ghost_df = pd.DataFrame([ghost_machine])
    ghost_scaled = scaler.transform(ghost_df[FEATURES])
    
    # Get prediction
    prediction = model.predict(ghost_scaled)[0]
    anomaly_score = model.decision_function(ghost_scaled)[0]
    
    print(f"\n[ISOLATION FOREST VERDICT]")
    print(f"  Prediction: {prediction} (-1 = ANOMALY, 1 = Normal)")
    print(f"  Anomaly Score: {anomaly_score:.4f}")
    
    if prediction == -1:
        print("\n  ✓ SUCCESS: Ghost Machine detected as ANOMALY!")
        print("    The Safety Net caught a physics-violating state.")
    else:
        print("\n  ✗ MISSED: Ghost Machine was classified as normal.")
        print("    Consider adjusting contamination or adding physics constraints.")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution pipeline for the Safety Net system.
    """
    print("\n" + "="*60)
    print("SAFETY NET: UNSUPERVISED ANOMALY DETECTION SYSTEM")
    print("="*60)
    print("\nPhilosophy: Learn normality, detect deviations.")
    print("           The best defense against unknown unknowns.\n")
    
    # Step 1: Load data
    df = load_data(DATA_PATH)
    
    # Step 2: Train Isolation Forest on healthy data only
    model, scaler, healthy_scaled, failure_scaled = train_isolation_forest(df)
    
    # Step 3: Save model for production
    save_model(model, scaler)
    
    # Step 4: Visualize decision boundary
    visualize_decision_boundary(model, scaler, healthy_scaled, failure_scaled)
    
    # Step 5: Test with Ghost Machine
    simulate_ghost_machine(model, scaler)
    
    print("\n" + "="*60)
    print("SAFETY NET TRAINING COMPLETE")
    print("="*60)
    print(f"\nArtifacts saved:")
    print(f"  - Model: {MODEL_PATH}")
    print(f"  - Scaler: {SCALER_PATH}")
    print(f"  - Visualization: safety_net_visualization.png")
    print("\nThe Safety Net is now ready for deployment alongside XGBoost.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
