from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import argparse

BASE = Path(__file__).resolve().parents[1]
CLEANED = BASE / "data" / "cleaned" / "Playstore_cleaned.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)
OUT_MODEL_PATH = MODEL_DIR / "app_success_clf.pkl"

def safe_log1p(X):
    """
    A safe log1p transformer for numeric arrays.
    Must be top-level (not nested) so it can be pickled.
    """
    import numpy as _np
    X = _np.array(X, dtype=float)
    mask = ~_np.isfinite(X)
    if mask.any():
        X[mask] = 0.0
    return _np.log1p(X)

def load_data(path: Path = CLEANED, sample: bool = False, sample_size: int = 50000, random_state: int = 1) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cleaned CSV not found at {path}")
    df = pd.read_csv(path, low_memory=False)
    if sample:
        df = df.sample(sample_size, random_state=random_state)
    return df

def prepare_features(df: pd.DataFrame) -> (pd.DataFrame, list):
    df = df.copy()

    if 'Installs' not in df.columns:
        raise KeyError("Installs column missing in cleaned CSV")

    df['target'] = (pd.to_numeric(df['Installs'], errors='coerce').fillna(0) >= 1_000_000).astype(int)

    for c in ['Rating', 'Reviews', 'Price']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan

    if 'Developer' in df.columns:
        top_devs = df.groupby('Developer')['Installs'].sum().sort_values(ascending=False).head(50).index
        df['DeveloperTop'] = df['Developer'].isin(top_devs).astype(int)
    else:
        df['DeveloperTop'] = 0

    feature_cols = ['Category', 'Rating', 'Reviews', 'Price', 'DeveloperTop']

    if 'Category' in df.columns:
        df['Category'] = df['Category'].fillna('Unknown')
    else:
        df['Category'] = 'Unknown'

    return df, feature_cols

def make_pipeline(cat_cols: list, num_cols: list):
    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
        ('log1p', FunctionTransformer(func=safe_log1p, validate=False)),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, cat_cols),
            ('num', num_transformer, num_cols),
        ],
        remainder='drop',
        sparse_threshold=0
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', clf)
    ])

    return pipeline

def train_and_save(sample: bool = False):
    print("Loading data...")
    df = load_data(sample=sample)
    df, feature_cols = prepare_features(df)

    X = df[feature_cols]
    y = df['target']

    cat_cols = ['Category']
    num_cols = [c for c in ['Rating', 'Reviews', 'Price', 'DeveloperTop'] if c in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = make_pipeline(cat_cols, num_cols)

    print(f"Training on {len(X_train)} rows...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on test set...")
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    print("Classification report:")
    print(classification_report(y_test, preds, digits=4))

    if probs is not None:
        try:
            auc = roc_auc_score(y_test, probs)
            print(f"ROC AUC: {auc:.4f}")
        except Exception as e:
            print("ROC AUC could not be computed:", e)

    joblib.dump(pipeline, OUT_MODEL_PATH)
    print("Saved model pipeline to:", OUT_MODEL_PATH)
    return OUT_MODEL_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Train on sample (faster)")
    args = parser.parse_args()

    model_path = train_and_save(sample=args.sample)
    print("Done. Model saved to :", model_path)

