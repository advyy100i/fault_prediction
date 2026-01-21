# ⚙️ Physics-Aware Hierarchical Failure Diagnosis

### **Executive Summary**

This project implements a **Multi-Stage Machine Learning System** for industrial machinery failure prediction with three distinct architectures:
1. **Two-Stage Hierarchical System** (Sentry + Specialist)
2. **Three-Layer Unsupervised Safety Net** (Statistical + Physics + ML)
3. **Interactive Streamlit Dashboard** for Real-Time Monitoring

Unlike standard "black box" models, this system integrates **Domain Knowledge (Physics)** to predict not only *if* a machine will break, but exactly *how* it will break (Power Failure, Heat Failure, Overstrain, Tool Wear). It addresses severe class imbalance (3% failure rate) using multiple complementary approaches.

---

## 📁 Project Structure

```
fault_prediction/
├── Data Files
│   ├── ai4i2020.csv                      # Original AI4I 2020 dataset
│   ├── ai4i2020_features.csv             # Dataset with engineered features
│   ├── X_train_res.csv                   # SMOTE-balanced training features
│   ├── y_train_res.csv                   # SMOTE-balanced training labels
│   ├── X_test.csv                        # Test features
│   └── y_test.csv                        # Test labels
│
├── Core Pipeline Scripts
│   ├── feature_engineering.py            # Physics-based feature generation
│   ├── preprocessing_oversampling.py     # Data preprocessing & SMOTE
│   ├── train_hierarchical.py             # Train Sentry + Specialist models
│   ├── 04_safety_net.py                  # Unsupervised anomaly detection system
│   ├── specialist_evaluation.py          # Specialist model evaluation
│   ├── evaluate_phase5.py                # Sentry model evaluation
│   └── diagnostics.py                    # Model diagnostics utilities
│
├── Interactive Applications
│   ├── dash.py                           # Streamlit dashboard (main UI)
│   └── test_osf.py                       # Safety Net testing script
│
├── Analysis & Reporting
│   ├── ai4i_fault_report.ipynb           # Comprehensive analysis notebook
│   ├── results_visualization.ipynb       # Results visualization notebook
│   ├── results_summary.txt               # Performance metrics summary
│   ├── specialist_evaluation_summary.md  # Specialist evaluation report
│   └── specialist_evaluation.txt         # Detailed specialist metrics
│
├── Saved Models
│   ├── sentry_pipeline.pkl               # Stage 1: Binary failure detector
│   ├── specialist_pipeline.pkl           # Stage 2: Failure type classifier
│   ├── safety_net.pkl                    # Unsupervised anomaly detector
│   ├── safety_net_bounds.pkl             # Statistical bounds for safety net
│   └── safety_net_scaler.pkl             # Scaler for safety net
│
└── Documentation
    ├── readme.md                         # This file
    └── safety_net_visualization.png      # Safety Net architecture diagram
```

---

## 🧠 The Novel Approach: Physics-Based Feature Engineering

Standard datasets provide raw telemetry (Torque, Speed, Temperature). However, machines don't fail because "Torque is high"; they fail because physical limits are crossed.

I engineered synthetic features to represent these physical limits, which became the strongest predictors in the model:

| Derived Feature | Formula | Physics Logic |
|:---|:---|:---|
| **Power Output (W)** | $P = 2\pi \times \text{Speed} \times \text{Torque} / 60$ | High speed + High torque = Massive energy stress |
| **Temp Difference (K)** | $\Delta T = T_{process} - T_{air}$ | Low difference indicates failed heat dissipation |
| **Strain Load** | $S = \text{Tool Wear} \times \text{Torque}$ | Worn tools under pressure are liable to snap (Overstrain) |

These features were implemented in `feature_engineering.py` and proved critical to model performance.

---

## 🏗️ System Architectures

### **Architecture 1: Hierarchical Supervised System**

#### **Stage 1: The "Sentry" (Detection)**
- **Goal:** Flag *potential* failures with high sensitivity
- **Model:** XGBoost Classifier (Binary)
- **Strategy:** Optimized for **Recall** (catching failures is critical)
- **Handling Imbalance:** SMOTE oversampling (97:3 → 50:50 class ratio)
- **Performance:** 
  - Recall: **82.4%** (caught 56/68 failures)
  - Precision: 63.6%
  - F1-Score: 0.63

#### **Stage 2: The "Specialist" (Diagnosis)**
- **Goal:** Identify specific root cause (TWF, HDF, PWF, OSF)
- **Model:** LightGBM Multi-Class Classifier
- **Input:** Only cases flagged by the Sentry
- **Strategy:** Leverage physics features for precise diagnosis
- **Performance:**
  - Accuracy: **99.7%**
  - F1-Score (macro): 0.996
  - Per-Class Precision/Recall: >98% for all failure types

### **Architecture 2: Three-Layer Safety Net (Unsupervised)**

An unsupervised anomaly detection system with defense-in-depth:

#### **Layer 1: Statistical Bounds Detector**
- Uses Z-score (3σ threshold) for univariate outlier detection
- Catches extreme deviations in any single feature
- No model training required

#### **Layer 2: Physics Constraints Detector**
- Validates mechanical power equation: $P = τω$
- Flags >20% deviation from theoretical power
- Detects sensor failures and data corruption

#### **Layer 3: Ensemble Isolation Forest**
- Multiple Isolation Forest models with varied hyperparameters
- Captures complex multivariate anomalies
- Tuned for higher sensitivity (contamination factor)

**Final Verdict:** ANY layer flags → ANOMALY (OR logic for maximum sensitivity)

**Implementation:** `04_safety_net.py` - 688 lines implementing all three layers

### **Architecture 3: Interactive Streamlit Dashboard**

Real-time monitoring interface with:
- Live prediction using all three systems
- Feature importance visualization
- Confusion matrix analysis
- Individual sample diagnosis
- Safety Net multi-layer breakdown
- Physics constraint validation

**Launch:** `streamlit run dash.py`

---

## 📊 Complete Performance Metrics

### **Sentry Model (Binary Classification)**
```
              precision    recall  f1-score   support
           0     0.9942    0.9715    0.9827      1932
           1     0.5089    0.8382    0.6333        68

    accuracy                         0.9670      2000
   macro avg     0.7516    0.9049    0.8080      2000
weighted avg     0.9777    0.9670    0.9708      2000
```

**Feature Importance (Top 5):**
1. Rotational speed [rpm]: 35.7%
2. **Power_W** (Engineered): 16.8%
3. Tool wear [min]: 16.1%
4. **Temp_Diff_K** (Engineered): 9.1%
5. Torque [Nm]: 8.7%

### **Specialist Model (Multi-Class Classification)**
```
Class 0 (TWF - Tool Wear Failure):    precision=1.00, recall=0.98
Class 1 (HDF - Heat Dissipation):     precision=1.00, recall=1.00
Class 2 (PWF - Power Failure):        precision=0.99, recall=1.00
Class 3 (OSF - Overstrain):           precision=1.00, recall=1.00
```

### **Safety Net (Unsupervised)**
- Multi-layer defense system
- No labeled data required
- Catches anomalies missed by supervised models
- Useful for deployment scenarios with evolving failure modes

---

## 🚀 Complete Pipeline Execution

### **Prerequisites**
```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn joblib matplotlib seaborn streamlit
```

### **Full Training Pipeline**

**Step 1: Feature Engineering**
```bash
python feature_engineering.py
```
- Reads: `ai4i2020.csv`
- Creates: Physics features (Power_W, Temp_Diff_K, Strain_Load)
- Outputs: `ai4i2020_features.csv`

**Step 2: Data Preprocessing & SMOTE**
```bash
python preprocessing_oversampling.py
```
- Reads: `ai4i2020_features.csv`
- Ordinal encodes Type: L→0, M→1, H→2
- Splits 80/20 train/test (stratified)
- Applies SMOTE to training set
- Outputs: `X_train_res.csv`, `y_train_res.csv`, `X_test.csv`, `y_test.csv`

**Step 3: Train Hierarchical Models**
```bash
python train_hierarchical.py
```
- Trains Sentry (XGBoost) on SMOTE-balanced data
- Trains Specialist (LightGBM) on labeled failures
- Saves models to `models/` directory
- Prints evaluation metrics and confusion matrices

**Step 4: Train Safety Net (Unsupervised)**
```bash
python 04_safety_net.py
```
- Trains three-layer anomaly detection system
- Fits statistical bounds on healthy samples
- Configures physics constraints
- Trains ensemble Isolation Forest
- Saves: `safety_net.pkl`, `safety_net_bounds.pkl`, `safety_net_scaler.pkl`
- Generates: `safety_net_visualization.png`

**Step 5: Evaluate Models**
```bash
# Evaluate Sentry
python evaluate_phase5.py

# Evaluate Specialist
python specialist_evaluation.py
```

**Step 6: Launch Dashboard**
```bash
streamlit run dash.py
```
- Interactive UI at `http://localhost:8501`
- Real-time predictions
- Visual analytics
- Model interpretability

---

## 📊 Analysis Notebooks

### **ai4i_fault_report.ipynb**
Comprehensive exploratory data analysis:
- Dataset statistics and distributions
- Class imbalance analysis
- Feature correlations
- Physics feature validation
- Failure mode patterns

### **results_visualization.ipynb**
Performance visualization:
- Confusion matrices (heatmaps)
- ROC curves
- Precision-Recall curves
- Feature importance plots
- Model comparison charts

---

## 🔬 Key Scripts Explained

### **feature_engineering.py**
Creates physics-based features from raw telemetry:
- Mechanical power calculation
- Thermal differential
- Tool strain under load
- Handles missing values and type coercion

### **preprocessing_oversampling.py**
Data preparation pipeline:
- Ordinal encoding for categorical features
- Stratified train-test split
- SMOTE oversampling (training only)
- Feature scaling
- CSV output for reproducibility

### **train_hierarchical.py**
Two-stage model training:
- Sentry: RandomizedSearchCV with XGBoost
- Specialist: Class-weighted LightGBM
- Hyperparameter optimization
- Cross-validation
- Model persistence

### **04_safety_net.py**
Unsupervised anomaly detection (688 lines):
- `StatisticalBoundsDetector`: Z-score based
- `PhysicsConstraintDetector`: Domain knowledge validation
- `EnsembleIsolationDetector`: Multiple Isolation Forests
- `BulletproofSafetyNet`: Orchestrator class
- Defense-in-depth architecture

### **dash.py**
Streamlit interactive dashboard (1001 lines):
- Model loading and caching
- Real-time prediction interface
- Feature importance visualization
- Confusion matrix display
- Safety Net layer breakdown
- Individual sample analysis

### **specialist_evaluation.py**
Detailed specialist model assessment:
- Per-class metrics (precision, recall, F1)
- Confusion matrix generation
- Feature importance analysis
- SHAP value calculation (optional)
- Classification report export

### **evaluate_phase5.py**
Sentry model evaluation:
- Confusion matrix
- Classification report
- Feature importance ranking
- Physics feature validation
- Performance metrics export

### **diagnostics.py**
Utility functions for model analysis:
- Error analysis
- Misclassification investigation
- Feature distribution comparisons
- Model debugging tools

### **test_osf.py**
Safety Net testing and validation:
- Load trained Safety Net models
- Test on validation samples
- Layer-by-layer analysis
- Anomaly detection metrics
- False positive/negative analysis

---

## 🎯 Use Cases & Applications

### **Manufacturing Quality Control**
- Real-time failure prediction
- Preventive maintenance scheduling
- Reduce unplanned downtime
- Minimize production losses

### **Predictive Maintenance**
- Monitor equipment health
- Identify failure patterns early
- Optimize maintenance intervals
- Extend machine lifespan

### **Root Cause Analysis**
- Diagnose specific failure types
- Guide repair strategies
- Improve maintenance efficiency
- Reduce diagnostic time

### **Anomaly Detection**
- Detect unusual operating conditions
- Flag sensor malfunctions
- Catch data quality issues
- Identify novel failure modes

---

## 📈 Model Performance Summary

| Model | Type | Accuracy | Recall | Precision | F1-Score |
|:---|:---|:---:|:---:|:---:|:---:|
| **Sentry** | Binary | 96.7% | 83.8% | 50.9% | 0.633 |
| **Specialist** | Multi-Class | 99.7% | 98-100% | 99-100% | 0.996 |
| **Safety Net** | Unsupervised | N/A | Adjustable | Adjustable | N/A |

---

## 🔧 Configuration & Customization

### **Hyperparameters**
Edit in respective scripts:
- `RANDOM_STATE = 42` (reproducibility)
- SMOTE sampling strategy (in `preprocessing_oversampling.py`)
- XGBoost parameters (in `train_hierarchical.py`)
- Isolation Forest contamination (in `04_safety_net.py`)

### **Feature Selection**
Modify feature lists in:
- `feature_engineering.py`: Add new physics features
- `dash.py`: Update `SAFETY_NET_FEATURES`

### **Thresholds**
Adjust detection sensitivity:
- Statistical bounds: Z-score threshold (default: 3σ)
- Physics constraints: Power deviation tolerance (default: 20%)
- Isolation Forest: Contamination factor (default: 0.05)

---

## 📝 Results Files

### **results_summary.txt**
Consolidated performance metrics:
- Confusion matrices
- Classification reports
- Feature importance rankings
- Top-5 feature validation

### **specialist_evaluation_summary.md**
Detailed specialist analysis:
- Per-class performance breakdown
- Training/testing methodology
- Evaluation caveats
- Recommended next steps

### **specialist_evaluation.txt**
Raw numerical output:
- Precision, recall, F1 per class
- Support counts
- Macro/weighted averages

---

## 🚀 Quick Start Examples

### **Make a Single Prediction**
```python
import joblib
import pandas as pd

# Load models
sentry = joblib.load('models/sentry_pipeline.pkl')
specialist = joblib.load('models/specialist_pipeline.pkl')

# Prepare sample
sample = pd.DataFrame([{
    'Rotational speed [rpm]': 1500,
    'Torque [Nm]': 40,
    'Tool wear [min]': 100,
    'Power_W': 6283,
    'Temp_Diff_K': 8.5,
    'Strain_Load': 4000,
    'Type_encoded': 1,
    'Air temperature [K]': 300,
    'Process temperature [K]': 308.5
}])

# Predict
if sentry.predict(sample)[0] == 1:
    failure_type = specialist.predict(sample)[0]
    print(f"FAILURE DETECTED: Type {failure_type}")
else:
    print("Machine operating normally")
```

### **Use Safety Net**
```python
import joblib
from importlib import import_module

# Load Safety Net
safety_net = joblib.load('models/safety_net.pkl')

# Detect anomalies
predictions = safety_net.predict(X_test[['Rotational speed [rpm]', 'Torque [Nm]', 
                                         'Tool wear [min]', 'Power_W', 
                                         'Temp_Diff_K', 'Strain_Load']])
# -1 = anomaly, 1 = normal
```

---

## 🤝 Contributing

This project demonstrates advanced ML engineering principles:
- Physics-informed feature engineering
- Hierarchical model architecture
- Handling severe class imbalance
- Unsupervised anomaly detection
- Production-ready deployment with Streamlit

Feel free to:
- Extend to other datasets
- Add new physics features
- Improve model architectures
- Enhance the dashboard UI

---

## 📚 References & Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset  
**Source:** UCI Machine Learning Repository  
**Features:** 10,000 samples with sensor readings and failure labels  
**Classes:** Machine failure (binary) + 5 failure types (TWF, HDF, PWF, OSF, RNF)

---

## 📧 Contact & Repository

**GitHub:** [https://github.com/advyy100i/fault_prediction](https://github.com/advyy100i/fault_prediction)  
**Author:** Advay  
**Last Updated:** January 2026

---

## 🎓 Key Learnings

1. **Physics > Raw Data**: Engineered features outperformed raw telemetry
2. **Defense in Depth**: Multiple detection layers catch different failure modes
3. **Class Imbalance**: SMOTE + Recall optimization critical for rare events
4. **Interpretability**: Physics-based features provide explainable predictions
5. **Unsupervised Backup**: Safety Net catches novel failures without labels

---

## 🚀 Next Steps

- [ ] Deploy as REST API (FastAPI)
- [ ] Add SHAP explanations to dashboard
- [ ] Implement real-time data streaming
- [ ] Cross-validation with other machinery datasets
- [ ] A/B testing in production environment
- [ ] Edge device deployment (IoT sensors)

---

**⭐ If this project helped you, please star the repository!**
  python train_hierarchical.py
  ```

  Optional: Run the Phase 5 evaluation and diagnostics/visualization

  ```bash
  python evaluate_phase5.py          # prints confusion matrix, F1, and top features
  python diagnostics.py             # creates plots and saved images (fi_sentry.png, fi_specialist.png)
  # Open the notebook for interactive visualization:
  results_visualization.ipynb
  ```

## 🛠️ Technologies Used

  * **Python 3.8+**
  * **Pandas / NumPy:** Data Engineering.
  * **Scikit-Learn:** Pipeline management & Metrics.
  * **Imbalanced-Learn:** SMOTE implementation.
  * **XGBoost / LightGBM:** Gradient Boosting algorithms.
  * **Matplotlib / Seaborn:** Visualization.

-----

