

# ⚙️ Physics-Aware Hierarchical Failure Diagnosis

### **Executive Summary**

This project builds a **Two-Stage Hierarchical Machine Learning System** to predict industrial machinery failures.

Unlike standard "black box" models that simply predict *if* a machine will break, this system integrates **Domain Knowledge (Physics)** to predict exactly *how* it will break (Power Failure, Heat Failure, Overstrain, etc.). It addresses severe class imbalance (3% failure rate) using a tiered architecture, achieving **100% diagnostic accuracy** on known failure modes.

-----

## 🧠 The Novel Approach: Physics-Based Feature Engineering

Standard datasets provide raw telemetry (Torque, Speed, Temperature). However, machines don't fail because "Torque is high"; they fail because physical limits are crossed.

I engineered synthetic features to represent these physical limits, which became the strongest predictors in the model:

| Derived Feature | Formula | Physics Logic |
| :--- | :--- | :--- |
| **Power Output (W)** | $P = 2 \pi \times \text{Speed} \times \text{Torque} / 60$ | High speed + High torque = Massive energy stress. |
| **Temp Difference (K)** | $\Delta T = T_{process} - T_{air}$ | Low difference indicates failed heat dissipation. |
| **Strain Load** | $S = \text{Tool Wear} \times \text{Torque}$ | Old tools under pressure are liable to snap (Overstrain). |

-----

## 🏗️ System Architecture

The solution uses a **Hierarchical (Triage)** architecture rather than a single classifier.

### **Stage 1: The "Sentry" (Detection)**

  * **Goal:** Flag *potential* failures with high sensitivity.
  * **Model:** XGBoost Classifier (Binary).
  * **Strategy:** Optimized for **Recall**. In predictive maintenance, a False Negative (missed failure) is catastrophic, while a False Positive (maintenance check) is a manageable cost.
  * **Handling Imbalance:** Applied SMOTE (Synthetic Minority Over-sampling) on training data to balance the 97:3 class ratio.

### **Stage 2: The "Specialist" (Diagnosis)**

  * **Goal:** Identify the specific Root Cause (Power, Heat, Tool Wear, Overstrain).
  * **Model:** LightGBM Classifier (Multi-Class).
  * **Input:** Only receives cases flagged by the Sentry.
  * **Strategy:** Leverages the Physics features to map telemetry to specific failure modes.

-----

## 📊 Results & Performance

### **1. Sentry Model (Detection)**

The model successfully prioritizes safety.

  * **Recall (Sensitivity):** **82.4%** (Caught 56/68 failures).
  * **Precision:** 63.6% (Acceptable trade-off for high safety).
  * **Feature Importance:** `Power_W` (Engineered) ranked \#2, beating raw `Torque` (Rank \#4), proving the value of physics features.

### **2. Specialist Model (Diagnosis)**

  * **Accuracy:** **100%** on validation set.
  * **Insight:** The model achieved perfect scores not through overfitting, but because the **Physics Features** effectively acted as an "answer key."
      * *Heat Failures* were perfectly predicted by `Temp_Diff_K`.
      * *Power Failures* were perfectly predicted by `Power_W`.

-----


## 🚀 How to Run

1.  **Clone the repository**

    ```bash
    git clone https://github.com/yourusername/physics-aware-failure.git
    cd physics-aware-failure
    ```

2.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Pipeline**

  The repository scripts in this workspace are in the project root. Run them from the project directory:

  *Step 1: Generate Physics Features*

  ```bash
  python feature_engineering.py
  ```

  *Step 2: Preprocess & Balance Data* (includes ordinal encoding and SMOTE on the training split)

  ```bash
  python preprocessing_oversampling.py
  ```

  *Step 3: Train & Evaluate (Hierarchical pipeline - Sentry + Specialist)*

  ```bash
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

