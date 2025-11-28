**Specialist Model Evaluation — Summary**

- **Files produced:** `specialist_evaluation.txt`, `specialist_confusion.png`, `fi_specialist.png`
- **Rows evaluated:** 330 (filtered labeled failures from `ai4i2020_features.csv`)
- **Accuracy:** 0.99697
- **F1 (macro):** 0.99589

- **Per-class performance (from `specialist_evaluation.txt`):**
  - Class 0 (TWF): precision=1.0000, recall=0.9783, support=46
  - Class 1 (HDF): precision=1.0000, recall=1.0000, support=115
  - Class 2 (PWF): precision=0.9891, recall=1.0000, support=91
  - Class 3 (OSF): precision=1.0000, recall=1.0000, support=78

- **Notes / Caveats:**
  - The evaluation used the saved `models/specialist_pipeline.pkl` and ran predictions on all labeled failure rows extracted from `ai4i2020_features.csv` (rows where `Machine failure == 1` and a subtype flag exists).
  - Because the saved Specialist was trained earlier from a train split, the evaluation above includes both training and non-training rows (it is *not* a held-out test that was guaranteed unseen during training). The very high scores are likely optimistic. Use a hold-out set or cross-validation for a more realistic estimate.
  - Feature importances were saved to `fi_specialist.png`. If the model lacked `feature_importances_`, permutation importance was used as a fallback.

- **Recommended next steps:**
  - Evaluate Specialist only on a held-out test set (rows excluded during Specialist training) or re-train with a saved validation/test split and then evaluate on that hold-out split.
  - Compute SHAP values or permutation importance with more repeats for robust interpretability.
  - Evaluate the two-stage pipeline end-to-end: run the Sentry on the whole test set, pass only Sentry-positive rows to Specialist, and compute the combined system confusion/detection accuracy.
  - If small class counts persist, consider cross-validation with stratification or data augmentation for rare failure types.

- **Next actions I can take for you:**
  - Run an end-to-end evaluation (Sentry → Specialist) on your held-out data.
  - Re-run Specialist training while saving a dedicated hold-out split, then evaluate it.
  - Compute SHAP explanations for the Specialist model.
