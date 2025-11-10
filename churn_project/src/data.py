from __future__ import annotations
import pandas as pd
from typing import Tuple
TARGET = "churn"
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df
def split_X_y(df: pd.DataFrame, target: str = TARGET) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data")
    X = df.drop(columns=[target])
    if 'customer_id' in X.columns:
        X = X.drop(columns=['customer_id'])
    y = df[target].astype(int)
    return X, y
