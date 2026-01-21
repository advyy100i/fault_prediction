import pandas as pd
import joblib
import numpy as np

X_test = pd.read_csv('X_test.csv')
sentry = joblib.load('models/sentry_pipeline.pkl')

# Test various OSF-like conditions
base_sample = X_test.mean()

print('Testing OSF Detection at Various Levels:')
print('=' * 60)

test_cases = [
    {'Tool wear [min]': 172, 'Torque [Nm]': 46, 'Strain_Load': 11000},  # Min OSF threshold
    {'Tool wear [min]': 190, 'Torque [Nm]': 52, 'Strain_Load': 11500},  # Low OSF
    {'Tool wear [min]': 200, 'Torque [Nm]': 55, 'Strain_Load': 11800},  # Medium OSF
    {'Tool wear [min]': 210, 'Torque [Nm]': 60, 'Strain_Load': 12500},  # High OSF
    {'Tool wear [min]': 230, 'Torque [Nm]': 70, 'Strain_Load': 15000},  # Very High OSF
]

for case in test_cases:
    sample = base_sample.copy()
    for k, v in case.items():
        sample[k] = v
    sample_df = pd.DataFrame([sample])
    pred = sentry.predict(sample_df)[0]
    prob = sentry.predict_proba(sample_df)[0][1]
    status = 'DETECTED' if prob >= 0.25 else 'MISSED'
    print(f"TW={case['Tool wear [min]']:3d}, Torque={case['Torque [Nm]']:4.1f}, Strain={case['Strain_Load']:5.0f} -> Prob: {prob:.4f} [{status}]")

# Check actual OSF failures in test set
print('\n' + '=' * 60)
print('Checking actual failures in test set with OSF-like features:')
y_test = pd.read_csv('y_test.csv').squeeze()

# Get all predictions
probs = sentry.predict_proba(X_test)[:, 1]
preds = (probs >= 0.25).astype(int)

# OSF-like conditions
osf_mask = (X_test['Tool wear [min]'] > 170) & (X_test['Torque [Nm]'] > 46) & (X_test['Strain_Load'] > 11000)
osf_samples = X_test[osf_mask]
osf_probs = probs[osf_mask]
osf_actual = y_test[osf_mask]

print(f'OSF-like samples: {len(osf_samples)}')
print(f'Actual failures in OSF-like: {osf_actual.sum()}')
print(f'Detected by Sentry (prob >= 0.25): {(osf_probs >= 0.25).sum()}')
print(f'Detection rate: {100 * (osf_probs >= 0.25).sum() / len(osf_samples):.1f}%')

print('\nOSF-like sample details:')
osf_df = osf_samples.copy()
osf_df['Actual_Failure'] = osf_actual.values
osf_df['Sentry_Prob'] = osf_probs
osf_df['Detected'] = osf_probs >= 0.25
print(osf_df[['Tool wear [min]', 'Torque [Nm]', 'Strain_Load', 'Power_W', 'Actual_Failure', 'Sentry_Prob', 'Detected']])
