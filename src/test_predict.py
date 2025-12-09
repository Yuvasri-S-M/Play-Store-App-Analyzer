import numpy as np

def safe_log1p(X):
    X = np.array(X, dtype=float)
    mask = ~np.isfinite(X)
    if mask.any():
        X[mask] = 0.0
    return np.log1p(X)

import joblib, pandas as pd
from pathlib import Path

p = Path('models') / 'app_success_clf.pkl'
print("Loading model from:", p)

m = joblib.load(p)

row = pd.DataFrame([{
    'Category': 'Social',
    'Rating': 4.3,
    'Reviews': 150000,
    'Price': 0.0,
    'DeveloperTop': 1
}])

prob = m.predict_proba(row)[:,1][0]
print(f"Pred prob of ≥1M installs: {prob:.4f}")


