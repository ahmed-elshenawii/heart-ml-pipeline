import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, Dict

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COL_CANDIDATES = ["target", "Target", "num", "disease"]

def infer_target_column(df: pd.DataFrame) -> str:
    for c in TARGET_COL_CANDIDATES:
        if c in df.columns:
            return c
    # Fallback: assume last column is target
    return df.columns[-1]

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # Drop exact duplicates
    df = df.drop_duplicates().copy()

    # Coerce numeric where possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")

    # Handle missing: fill numeric with median, categorical with mode
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())
        else:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].mode().iloc[0])

    # Simple outlier clipping for numeric columns (5th–95th percentile)
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            lo, hi = df[c].quantile([0.05, 0.95])
            df[c] = df[c].clip(lo, hi)

    return df

def prepare_splits(df: pd.DataFrame) -> Dict[str, object]:
    target_col = infer_target_column(df)
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # one-hot if needed
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return {
        "target_col": target_col,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

def pca_fit_transform(X: pd.DataFrame, n_components: int = 2):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    return pca, scaler, X_pca
