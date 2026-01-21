"""
04_safety_net.py - BULLETPROOF Unsupervised Anomaly Detection
==============================================================

VERSION 2.0: Multi-Layer Defense System
----------------------------------------
The original Isolation Forest alone missed 65% of failures. Unacceptable.

This version implements a THREE-LAYER defense:

LAYER 1: Statistical Bounds (Z-Score)
    - Any feature > 3 standard deviations from healthy mean = ANOMALY
    - Catches egregious univariate outliers instantly
    - No model needed - pure statistics

LAYER 2: Physics Constraints
    - Power = Torque × Angular_Velocity (P = τω)
    - If reported power deviates >20% from physics = ANOMALY
    - Catches sensor failures and data corruption

LAYER 3: Isolation Forest (Multivariate)
    - Catches subtle multivariate anomalies
    - Tuned with higher contamination for sensitivity

FINAL VERDICT: ANY layer flags = ANOMALY
    - Defense in depth: multiple independent detectors
    - One layer's miss is another layer's catch

Author: Advay
Date: January 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

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
BOUNDS_PATH = 'models/safety_net_bounds.pkl'

# Tuned for HIGH SENSITIVITY - we'd rather have false alarms than miss failures
CONTAMINATION = 0.05  # Increased from 0.02 - more aggressive
Z_SCORE_THRESHOLD = 3.0  # Flag anything beyond 3 sigma
POWER_TOLERANCE = 0.25  # 25% deviation from physics = suspicious
RANDOM_STATE = 42


# =============================================================================
# LAYER 1: STATISTICAL BOUNDS DETECTOR
# =============================================================================

class StatisticalBoundsDetector:
    """
    Layer 1: Flag any value that is statistically extreme.
    
    WHY THIS WORKS:
    - Healthy machines operate within predictable ranges
    - Any feature > 3σ from healthy mean is highly suspicious
    - This catches egregious outliers that Isolation Forest might miss
      because IF looks at isolation depth, not absolute values
    """
    
    def __init__(self, z_threshold: float = 3.0):
        self.z_threshold = z_threshold
        self.means = None
        self.stds = None
        self.mins = None
        self.maxs = None
        self.iqr_bounds = None
        
    def fit(self, X: pd.DataFrame):
        """Learn bounds from healthy data."""
        self.means = X.mean()
        self.stds = X.std().replace(0, 1e-10)  # Avoid division by zero
        self.mins = X.min()
        self.maxs = X.max()
        
        # IQR-based bounds (more robust to outliers in training data)
        Q1 = X.quantile(0.01)  # 1st percentile
        Q3 = X.quantile(0.99)  # 99th percentile
        IQR = Q3 - Q1
        self.iqr_bounds = {
            'lower': Q1 - 1.5 * IQR,
            'upper': Q3 + 1.5 * IQR
        }
        
        print(f"[LAYER 1] Statistical bounds learned from {len(X)} healthy samples")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns: -1 for anomaly, 1 for normal
        """
        predictions = np.ones(len(X))
        
        for idx in range(len(X)):
            row = X.iloc[idx]
            
            for col in X.columns:
                value = row[col]
                z_score = abs((value - self.means[col]) / self.stds[col])
                
                # Check Z-score
                if z_score > self.z_threshold:
                    predictions[idx] = -1
                    break
                
                # Check hard bounds (beyond anything seen in training)
                if value < self.iqr_bounds['lower'][col] or value > self.iqr_bounds['upper'][col]:
                    predictions[idx] = -1
                    break
        
        return predictions
    
    def get_z_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return Z-scores for each feature."""
        return (X - self.means) / self.stds


# =============================================================================
# LAYER 2: PHYSICS CONSTRAINTS DETECTOR
# =============================================================================

class PhysicsConstraintDetector:
    """
    Layer 2: Enforce physical laws that cannot be violated.
    
    KEY PHYSICS RELATIONSHIPS:
    1. Power = Torque × Angular_Velocity
       P = τ × ω = τ × (2π × RPM / 60)
       
    2. High tool wear + high torque = high strain (multiplicative)
    
    3. Temperature differential has physical limits based on power
    
    If data violates these relationships, it's either:
    - Sensor malfunction
    - Data corruption
    - Genuine anomalous machine state
    
    ALL of these should be flagged.
    """
    
    def __init__(self, power_tolerance: float = 0.25):
        self.power_tolerance = power_tolerance
        self.power_ratio_mean = None
        self.power_ratio_std = None
        self.strain_ratio_mean = None
        self.strain_ratio_std = None
        
    def fit(self, X: pd.DataFrame):
        """Learn physics relationships from healthy data."""
        # Power relationship: P = τω
        expected_power = X['Rotational speed [rpm]'] * X['Torque [Nm]'] * 2 * np.pi / 60
        actual_power = X['Power_W']
        
        # Ratio should be ~1 for physically consistent data
        power_ratio = actual_power / expected_power.replace(0, 1e-10)
        self.power_ratio_mean = power_ratio.mean()
        self.power_ratio_std = power_ratio.std()
        
        # Strain relationship: Strain ≈ Torque × Tool_Wear
        expected_strain = X['Torque [Nm]'] * X['Tool wear [min]']
        actual_strain = X['Strain_Load']
        strain_ratio = actual_strain / expected_strain.replace(0, 1e-10)
        self.strain_ratio_mean = strain_ratio.mean()
        self.strain_ratio_std = strain_ratio.std()
        
        print(f"[LAYER 2] Physics constraints learned:")
        print(f"         Power ratio: {self.power_ratio_mean:.3f} ± {self.power_ratio_std:.3f}")
        print(f"         Strain ratio: {self.strain_ratio_mean:.3f} ± {self.strain_ratio_std:.3f}")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns: -1 for physics violation, 1 for consistent
        """
        predictions = np.ones(len(X))
        
        # Check power consistency
        expected_power = X['Rotational speed [rpm]'] * X['Torque [Nm]'] * 2 * np.pi / 60
        actual_power = X['Power_W']
        power_ratio = actual_power / expected_power.replace(0, 1e-10)
        
        # Flag if ratio deviates significantly from learned mean
        power_z = abs(power_ratio - self.power_ratio_mean) / max(self.power_ratio_std, 0.01)
        predictions[power_z > 3] = -1
        
        # Also flag if absolute deviation is large
        power_deviation = abs(actual_power - expected_power) / expected_power.replace(0, 1e-10)
        predictions[power_deviation > self.power_tolerance] = -1
        
        # Check strain consistency
        expected_strain = X['Torque [Nm]'] * X['Tool wear [min]']
        actual_strain = X['Strain_Load']
        strain_ratio = actual_strain / expected_strain.replace(0, 1e-10)
        strain_z = abs(strain_ratio - self.strain_ratio_mean) / max(self.strain_ratio_std, 0.01)
        predictions[strain_z > 4] = -1  # Slightly looser for strain
        
        # Temperature sanity check
        if 'Temp_Diff_K' in X.columns:
            predictions[X['Temp_Diff_K'] < 0] = -1  # Negative temp diff is impossible
            predictions[X['Temp_Diff_K'] > 25] = -1  # Extreme temp diff
        
        return predictions
    
    def get_physics_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return physics consistency scores."""
        expected_power = X['Rotational speed [rpm]'] * X['Torque [Nm]'] * 2 * np.pi / 60
        actual_power = X['Power_W']
        
        return pd.DataFrame({
            'power_deviation_%': 100 * abs(actual_power - expected_power) / expected_power.replace(0, 1e-10),
            'expected_power': expected_power,
            'actual_power': actual_power
        })


# =============================================================================
# LAYER 3: ENSEMBLE ISOLATION DETECTOR
# =============================================================================

class EnsembleIsolationDetector:
    """
    Layer 3: Multiple Isolation Forest models voting together.
    
    WHY ENSEMBLE?
    - Different hyperparameters catch different types of anomalies
    - Voting reduces individual model noise
    """
    
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = RobustScaler()  # Robust to outliers
        
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=200,  # More trees for stability
            max_samples=0.8,   # Subsample for diversity
            n_jobs=-1
        )
        
        self.isolation_forest_aggressive = IsolationForest(
            contamination=contamination * 2,  # More aggressive
            random_state=random_state + 1,
            n_estimators=150,
            max_samples=0.5,
            n_jobs=-1
        )
        
    def fit(self, X: pd.DataFrame):
        """Fit all ensemble members on healthy data."""
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"[LAYER 3] Training Isolation Forest ensemble...")
        self.isolation_forest.fit(X_scaled)
        self.isolation_forest_aggressive.fit(X_scaled)
        
        print(f"[LAYER 3] Ensemble trained on {len(X)} healthy samples")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns: -1 for anomaly, 1 for normal
        Defensive: ANY model flagging = anomaly
        """
        X_scaled = self.scaler.transform(X)
        
        pred_if = self.isolation_forest.predict(X_scaled)
        pred_if_agg = self.isolation_forest_aggressive.predict(X_scaled)
        
        # Defensive ensemble: anomaly if ANY model says so
        votes = np.column_stack([pred_if, pred_if_agg])
        anomaly_votes = (votes == -1).sum(axis=1)
        
        predictions = np.where(anomaly_votes >= 1, -1, 1)
        return predictions
    
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores (lower = more anomalous)."""
        X_scaled = self.scaler.transform(X)
        return self.isolation_forest.decision_function(X_scaled)


# =============================================================================
# MASTER SAFETY NET: COMBINES ALL LAYERS
# =============================================================================

class BulletproofSafetyNet:
    """
    The Master Safety Net - Three Layers of Defense
    
    PHILOSOPHY: Defense in Depth
    - Layer 1 catches egregious statistical outliers
    - Layer 2 catches physics violations
    - Layer 3 catches subtle multivariate patterns
    
    ANY layer flagging = ANOMALY
    
    This ensures we NEVER miss an obviously broken machine.
    """
    
    def __init__(self, 
                 z_threshold: float = Z_SCORE_THRESHOLD,
                 power_tolerance: float = POWER_TOLERANCE,
                 contamination: float = CONTAMINATION):
        
        self.layer1 = StatisticalBoundsDetector(z_threshold=z_threshold)
        self.layer2 = PhysicsConstraintDetector(power_tolerance=power_tolerance)
        self.layer3 = EnsembleIsolationDetector(contamination=contamination)
        
        self.features = FEATURES
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame):
        """Train all layers on healthy data."""
        print("\n" + "="*60)
        print("TRAINING BULLETPROOF SAFETY NET")
        print("="*60)
        
        X_features = X[self.features].copy()
        
        self.layer1.fit(X_features)
        self.layer2.fit(X_features)
        self.layer3.fit(X_features)
        
        self.is_fitted = True
        print("\n[MASTER] All three layers trained successfully!")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies using all three layers.
        Returns: -1 for anomaly, 1 for normal
        """
        X_features = X[self.features].copy()
        
        pred_l1 = self.layer1.predict(X_features)
        pred_l2 = self.layer2.predict(X_features)
        pred_l3 = self.layer3.predict(X_features)
        
        # DEFENSIVE: Any layer flags = anomaly
        is_anomaly = (pred_l1 == -1) | (pred_l2 == -1) | (pred_l3 == -1)
        return np.where(is_anomaly, -1, 1)
    
    def predict_detailed(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict with detailed breakdown by layer.
        Useful for debugging and explanation.
        """
        X_features = X[self.features].copy()
        
        pred_l1 = self.layer1.predict(X_features)
        pred_l2 = self.layer2.predict(X_features)
        pred_l3 = self.layer3.predict(X_features)
        
        final_pred = self.predict(X)
        
        return pd.DataFrame({
            'Layer1_Stats': pred_l1,
            'Layer2_Physics': pred_l2,
            'Layer3_Isolation': pred_l3,
            'Final_Prediction': final_pred,
            'Anomaly_Score': self.layer3.decision_function(X_features)
        })
    
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores from Layer 3."""
        X_features = X[self.features].copy()
        return self.layer3.decision_function(X_features)


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

def train_safety_net(df: pd.DataFrame) -> BulletproofSafetyNet:
    """Train the bulletproof safety net on healthy data only."""
    
    # Filter to healthy machines only
    healthy_mask = df[TARGET] == 0
    healthy_data = df.loc[healthy_mask].copy()
    
    print(f"\n[INFO] Training on {len(healthy_data)} healthy samples")
    print(f"[INFO] Held out {(~healthy_mask).sum()} failure samples for testing")
    
    # Create and train the safety net
    safety_net = BulletproofSafetyNet()
    safety_net.fit(healthy_data)
    
    return safety_net


def evaluate_safety_net(safety_net: BulletproofSafetyNet, df: pd.DataFrame):
    """Evaluate the safety net's detection performance."""
    
    print("\n" + "="*60)
    print("EVALUATION: BULLETPROOF SAFETY NET")
    print("="*60)
    
    healthy_mask = df[TARGET] == 0
    failure_mask = df[TARGET] == 1
    
    healthy_data = df.loc[healthy_mask]
    failure_data = df.loc[failure_mask]
    
    # Predictions
    healthy_preds = safety_net.predict(healthy_data)
    failure_preds = safety_net.predict(failure_data)
    
    # Detailed breakdown
    failure_detailed = safety_net.predict_detailed(failure_data)
    
    # Statistics
    healthy_flagged = (healthy_preds == -1).sum()
    failures_detected = (failure_preds == -1).sum()
    
    print(f"\n[RESULTS]")
    print(f"  Healthy machines flagged: {healthy_flagged}/{len(healthy_preds)} ({100*healthy_flagged/len(healthy_preds):.1f}%)")
    print(f"  Failures detected: {failures_detected}/{len(failure_preds)} ({100*failures_detected/len(failure_preds):.1f}%)")
    
    # Layer-by-layer breakdown for failures
    print(f"\n[LAYER BREAKDOWN - Failures Detected]")
    print(f"  Layer 1 (Statistics): {(failure_detailed['Layer1_Stats'] == -1).sum()}/{len(failure_preds)}")
    print(f"  Layer 2 (Physics):    {(failure_detailed['Layer2_Physics'] == -1).sum()}/{len(failure_preds)}")
    print(f"  Layer 3 (Isolation):  {(failure_detailed['Layer3_Isolation'] == -1).sum()}/{len(failure_preds)}")
    print(f"  Combined (ANY):       {failures_detected}/{len(failure_preds)}")
    
    if failures_detected / len(failure_preds) > 0.7:
        print("\n  ✓ SUCCESS: Detecting >70% of actual failures!")
    elif failures_detected / len(failure_preds) > 0.5:
        print("\n  ~ ACCEPTABLE: Detecting >50% of failures, but could improve.")
    else:
        print("\n  ✗ WARNING: Detection rate below 50% - needs tuning.")
    
    return failure_detailed


def visualize_safety_net(safety_net: BulletproofSafetyNet, df: pd.DataFrame):
    """Create visualization of the safety net's decision boundary."""
    
    print("\n[INFO] Creating visualization...")
    
    healthy_mask = df[TARGET] == 0
    failure_mask = df[TARGET] == 1
    
    healthy_data = df.loc[healthy_mask, FEATURES]
    failure_data = df.loc[failure_mask, FEATURES]
    
    # Scale data for PCA
    all_data = pd.concat([healthy_data, failure_data])
    scaler = StandardScaler()
    all_scaled = scaler.fit_transform(all_data)
    
    healthy_scaled = all_scaled[:len(healthy_data)]
    failure_scaled = all_scaled[len(healthy_data):]
    
    # PCA
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    healthy_pca = pca.fit_transform(healthy_scaled)
    failure_pca = pca.transform(failure_scaled)
    
    # Get predictions
    healthy_preds = safety_net.predict(healthy_data)
    failure_preds = safety_net.predict(failure_data)
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: All data with predictions
    ax1 = axes[0]
    
    # Healthy - correctly identified
    healthy_normal = healthy_preds == 1
    healthy_flagged = healthy_preds == -1
    
    ax1.scatter(healthy_pca[healthy_normal, 0], healthy_pca[healthy_normal, 1],
                c='blue', s=15, alpha=0.3, label=f'Healthy (Normal): {healthy_normal.sum()}')
    ax1.scatter(healthy_pca[healthy_flagged, 0], healthy_pca[healthy_flagged, 1],
                c='orange', s=30, alpha=0.6, label=f'Healthy (Flagged): {healthy_flagged.sum()}')
    
    # Failures
    failure_detected = failure_preds == -1
    failure_missed = failure_preds == 1
    
    ax1.scatter(failure_pca[failure_detected, 0], failure_pca[failure_detected, 1],
                c='red', s=60, alpha=0.8, marker='X', label=f'Failure (Detected): {failure_detected.sum()}')
    ax1.scatter(failure_pca[failure_missed, 0], failure_pca[failure_missed, 1],
                c='green', s=80, alpha=0.9, marker='s', label=f'Failure (MISSED): {failure_missed.sum()}')
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax1.set_title('Safety Net: Detection Results', fontweight='bold')
    ax1.legend(loc='best')
    
    # Plot 2: Decision boundary
    ax2 = axes[1]
    
    all_pca = np.vstack([healthy_pca, failure_pca])
    x_min, x_max = all_pca[:, 0].min() - 1, all_pca[:, 0].max() + 1
    y_min, y_max = all_pca[:, 1].min() - 1, all_pca[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    mesh_pca = np.c_[xx.ravel(), yy.ravel()]
    mesh_original = pca.inverse_transform(mesh_pca)
    mesh_unscaled = scaler.inverse_transform(mesh_original)
    mesh_df = pd.DataFrame(mesh_unscaled, columns=FEATURES)
    
    # Use only Layer 3 for smooth decision boundary
    Z = safety_net.layer3.decision_function(mesh_df)
    Z = Z.reshape(xx.shape)
    
    contour = ax2.contourf(xx, yy, Z, levels=30, cmap='RdYlBu', alpha=0.6)
    ax2.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2, linestyles='--')
    
    ax2.scatter(healthy_pca[:, 0], healthy_pca[:, 1], c='blue', s=10, alpha=0.2)
    ax2.scatter(failure_pca[:, 0], failure_pca[:, 1], c='red', s=40, alpha=0.8, marker='X')
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax2.set_title('Isolation Forest Decision Boundary', fontweight='bold')
    plt.colorbar(contour, ax=ax2, label='Anomaly Score')
    
    plt.tight_layout()
    plt.savefig('safety_net_visualization.png', dpi=150, bbox_inches='tight')
    print("[INFO] Visualization saved to: safety_net_visualization.png")
    plt.show()


def simulate_ghost_machine(safety_net: BulletproofSafetyNet):
    """Test with physics-violating ghost machines."""
    
    print("\n" + "="*60)
    print("GHOST MACHINE SIMULATION")
    print("="*60)
    
    # Create multiple ghost machines with different anomalies
    ghost_machines = [
        {
            'name': 'Physics Violation (Power)',
            'Rotational speed [rpm]': 1200,
            'Torque [Nm]': 25,
            'Tool wear [min]': 100,
            'Power_W': 15000,  # Should be ~3140W
            'Temp_Diff_K': 8,
            'Strain_Load': 50
        },
        {
            'name': 'Extreme Speed',
            'Rotational speed [rpm]': 5000,  # Way too high
            'Torque [Nm]': 40,
            'Tool wear [min]': 50,
            'Power_W': 20944,  # Correct for this speed
            'Temp_Diff_K': 12,
            'Strain_Load': 2000
        },
        {
            'name': 'Negative Temperature',
            'Rotational speed [rpm]': 1500,
            'Torque [Nm]': 40,
            'Tool wear [min]': 100,
            'Power_W': 6283,
            'Temp_Diff_K': -5,  # Impossible
            'Strain_Load': 4000
        },
        {
            'name': 'Extreme Strain',
            'Rotational speed [rpm]': 1500,
            'Torque [Nm]': 40,
            'Tool wear [min]': 100,
            'Power_W': 6283,
            'Temp_Diff_K': 10,
            'Strain_Load': 50000  # Should be ~4000
        }
    ]
    
    for ghost in ghost_machines:
        name = ghost.pop('name')
        ghost_df = pd.DataFrame([ghost])
        
        # Get detailed prediction
        detailed = safety_net.predict_detailed(ghost_df)
        final_pred = detailed['Final_Prediction'].iloc[0]
        
        print(f"\n[{name}]")
        print(f"  Layer 1 (Stats):    {'ANOMALY' if detailed['Layer1_Stats'].iloc[0] == -1 else 'normal'}")
        print(f"  Layer 2 (Physics):  {'ANOMALY' if detailed['Layer2_Physics'].iloc[0] == -1 else 'normal'}")
        print(f"  Layer 3 (Isolation):{'ANOMALY' if detailed['Layer3_Isolation'].iloc[0] == -1 else 'normal'}")
        print(f"  FINAL VERDICT:      {'🚨 ANOMALY' if final_pred == -1 else '✓ Normal'}")
        
        ghost['name'] = name  # Restore for potential reuse


def save_safety_net(safety_net: BulletproofSafetyNet):
    """Save the trained safety net."""
    
    print(f"\n[INFO] Saving safety net to: {MODEL_PATH}")
    joblib.dump(safety_net, MODEL_PATH)
    
    # Also save layer 1 bounds separately for dashboard use
    bounds = {
        'means': safety_net.layer1.means,
        'stds': safety_net.layer1.stds,
        'mins': safety_net.layer1.mins,
        'maxs': safety_net.layer1.maxs,
        'iqr_bounds': safety_net.layer1.iqr_bounds,
        'power_ratio_mean': safety_net.layer2.power_ratio_mean,
        'power_ratio_std': safety_net.layer2.power_ratio_std
    }
    joblib.dump(bounds, BOUNDS_PATH)
    
    print(f"[INFO] Bounds saved to: {BOUNDS_PATH}")
    print("[INFO] Safety Net saved successfully!")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main training pipeline."""
    
    print("\n" + "="*60)
    print("BULLETPROOF SAFETY NET v2.0")
    print("Three Layers of Defense Against Unknown Failures")
    print("="*60)
    
    # Load data
    print(f"\n[INFO] Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Failure distribution:\n{df[TARGET].value_counts()}")
    
    # Train
    safety_net = train_safety_net(df)
    
    # Evaluate
    evaluate_safety_net(safety_net, df)
    
    # Save
    save_safety_net(safety_net)
    
    # Visualize
    visualize_safety_net(safety_net, df)
    
    # Test ghost machines
    simulate_ghost_machine(safety_net)
    
    print("\n" + "="*60)
    print("BULLETPROOF SAFETY NET TRAINING COMPLETE")
    print("="*60)
    print(f"\nArtifacts:")
    print(f"  - Model: {MODEL_PATH}")
    print(f"  - Bounds: {BOUNDS_PATH}")
    print(f"  - Visualization: safety_net_visualization.png")
    print("\nThe Safety Net is now ready for deployment.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
