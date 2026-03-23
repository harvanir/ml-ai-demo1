import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Any, Optional, Union
from app.models.response import Anomaly
from sklearn.ensemble import IsolationForest
from app.config import settings

def detect_anomalies_zscore(df: pd.DataFrame, threshold: float = 2.5) -> List[Anomaly]:
    """
    Detect anomalies using z-score method.

    High-level algorithm:
    1. For each numeric column:
       - Calculate mean (μ) and standard deviation (σ)
       - For each value x: z = (x - μ) / σ
       - If |z| > threshold (default 2.5) = ANOMALY

    Advantages:
    - Measures deviation in standard units
    - Good for normally distributed data
    - Provides continuous scores (not just outlier/not)
    - Easy for dynamic thresholding
    """
    anomalies = []
    numeric_cols = df.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        if df[col].isnull().all():
            continue
        # Get clean series and their original indices
        clean_series = df[col].dropna()
        print(f"DEBUG: Column {col} values: {clean_series.values}")
        if len(clean_series) < 3:  # Need at least 3 values
            continue
        z_scores = np.abs(stats.zscore(clean_series))
        print(f"DEBUG: Z-scores for {col}: {z_scores}")
        outlier_indices = np.where(z_scores > threshold)[0]
        print(f"DEBUG: Outlier indices for {col}: {outlier_indices}")

        for idx in outlier_indices:
            original_idx = clean_series.index[idx]
            anomalies.append(Anomaly(
                column=col,
                value=float(clean_series.iloc[idx]),
                index=int(original_idx),
                z_score=float(z_scores[idx]),
                is_outlier=True
            ))

    return anomalies

def detect_anomalies_iqr(df: pd.DataFrame) -> List[Anomaly]:
    """
    Detect anomalies using IQR method.

    High-level algorithm:
    1. For each numeric column:
       - Calculate Q1 (25th percentile) and Q3 (75th percentile)
       - Calculate IQR = Q3 - Q1
       - Determine lower bound = Q1 - 1.5 × IQR
       - Determine upper bound = Q3 + 1.5 × IQR
       - Data outside bounds = ANOMALY

    Advantages:
    - Robust against extreme outliers
    - No assumption of normal distribution
    - Good for small datasets
    - Easy to understand and interpret
    """
    anomalies = []
    numeric_cols = df.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        if df[col].isnull().all():
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        for idx, row in outliers.iterrows():
            anomalies.append(Anomaly(
                column=col,
                value=row[col],
                index=int(idx),
                is_outlier=True
            ))

    return anomalies

def detect_anomalies_isolation_forest(
    df: pd.DataFrame,
    contamination: Optional[float] = None,
    n_estimators: Optional[int] = None,
    max_samples: Optional[Union[str, float]] = None,
    random_state: Optional[int] = None,
    max_features: Optional[float] = None
) -> List[Anomaly]:
    """
    Detect anomalies using Isolation Forest method with configurable parameters.

    High-level algorithm:
    1. Build ensemble of isolation trees (random forests)
    2. Each tree isolates samples with random splits
    3. Anomaly score = average path length across all trees
    4. Anomalies = samples with shortest path lengths (easiest to isolate)

    Advantages:
    - Highly scalable O(n log n) for large datasets (100k+)
    - Memory efficient, doesn't store entire dataset
    - Good for high-dimensional data
    - Unsupervised, no need for labeled anomalies
    - Fast training and prediction

    Parameters:
    - contamination: Expected proportion of anomalies (uses config if None)
    - n_estimators: Number of trees (uses config if None)
    - max_samples: Samples per tree (uses config if None)
    - random_state: Random seed (uses config if None)
    - max_features: Max features per tree (uses config if None)
    """
    # Use configuration values if parameters not provided
    config = settings.isolation_forest
    contamination = contamination if contamination is not None else config.contamination
    n_estimators = n_estimators if n_estimators is not None else config.n_estimators
    max_samples = max_samples if max_samples is not None else (config.max_samples if config.max_samples != "auto" else "auto")
    random_state = random_state if random_state is not None else config.random_state
    max_features = max_features if max_features is not None else config.max_features

    anomalies = []
    numeric_cols = df.select_dtypes(include=['number']).columns

    if len(numeric_cols) == 0:
        return anomalies

    # Prepare data (drop NaN for training)
    X = df[numeric_cols].dropna()

    if len(X) < 10:  # Need minimum samples
        return anomalies

    # Initialize Isolation Forest with configured parameters
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        max_samples=max_samples,
        max_features=max_features,
        n_jobs=-1  # Use all CPU cores
    )

    # Fit and predict
    iso_forest.fit(X)
    predictions = iso_forest.predict(X)  # -1 = anomaly, 1 = normal
    scores = iso_forest.decision_function(X)  # Anomaly scores (lower = more anomalous)

    # Convert predictions to our format
    anomaly_indices = np.where(predictions == -1)[0]

    for idx in anomaly_indices:
        original_idx = X.index[idx]
        # Find which column has the most anomalous value
        row_values = X.iloc[idx]
        # Use the column with highest absolute z-score as primary anomaly column
        col_z_scores = np.abs(stats.zscore(row_values))
        primary_col = numeric_cols[np.argmax(col_z_scores)]

        anomalies.append(Anomaly(
            column=primary_col,
            value=float(row_values[primary_col]),
            index=int(original_idx),
            z_score=float(scores[idx]),  # Isolation Forest score
            is_outlier=True
        ))

    return anomalies

# Default to IQR for better small dataset handling
# Recommendation: Use IQR for small datasets or non-normal data
# Use Z-Score for large datasets with normal distribution
# Use Isolation Forest for medium/large datasets (100k+) or high-dimensional data
detect_anomalies = detect_anomalies_iqr