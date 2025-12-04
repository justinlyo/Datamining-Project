from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_data(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Drops any duplicates and rows missing audio features.
    """
    # Remove duplicate entries
    df_clean = df.drop_duplicates().copy()

    # Get rid of data without all rows of data
    df_clean = df_clean.dropna(subset=feature_cols)

    return df_clean


def preprocess_for_clustering(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler, List[str]]:

    df_clean = clean_data(df, feature_cols)

    # Extract a feature matrix
    X = df_clean[feature_cols].values

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df_clean, X_scaled, scaler, feature_cols
