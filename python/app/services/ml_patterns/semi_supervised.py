"""
Semi-Supervised Learning Pattern
Demonstrates learning with partial labeled data
"""

import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List

def analyze_semi_supervised(df: pd.DataFrame, labeled_ratio: float = 0.3, **kwargs) -> Dict[str, Any]:
    """
    Semi-supervised learning analysis: learning with partial labels

    For demo purposes, we'll:
    1. Assume the last column is the target
    2. Randomly mask some labels as unlabeled (-1)
    3. Use label propagation to predict missing labels
    """

    if df.empty:
        return {"error": "Empty dataset"}

    # Assume last column is target
    target_column = df.columns[-1]
    feature_cols = df.columns[:-1].tolist()

    if not feature_cols:
        return {"error": "No feature columns available"}

    # Clean data
    df_clean = df.dropna()
    if len(df_clean) < 5:
        return {"error": "Need at least 5 samples for semi-supervised learning"}

    X = df_clean[feature_cols]
    y = df_clean[target_column]

    # Convert categorical features
    X_processed = X.copy()
    for col in X_processed.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col].astype(str))

    # Convert target to numeric if needed
    if y.dtype == 'object':
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y.astype(str))
        target_classes = target_encoder.classes_
    else:
        y_encoded = y.values
        target_classes = None

    # Create semi-supervised scenario: mask some labels
    n_labeled = max(2, int(len(y_encoded) * labeled_ratio))
    labeled_indices = np.random.choice(len(y_encoded), n_labeled, replace=False)

    y_semi = y_encoded.copy()
    y_semi[~np.isin(np.arange(len(y_semi)), labeled_indices)] = -1  # Unlabeled

    # Apply label propagation
    label_prop = LabelPropagation(kernel='knn', n_neighbors=3)
    label_prop.fit(X_processed, y_semi)

    # Get predictions for unlabeled data
    y_pred = label_prop.predict(X_processed)

    # Calculate confidence scores
    confidence_scores = label_prop.predict_proba(X_processed).max(axis=1)

    # Analyze results
    labeled_correct = np.sum(y_pred[labeled_indices] == y_encoded[labeled_indices])
    labeled_accuracy = labeled_correct / len(labeled_indices) if len(labeled_indices) > 0 else 0

    unlabeled_indices = np.setdiff1d(np.arange(len(y_encoded)), labeled_indices)
    unlabeled_predictions = y_pred[unlabeled_indices]

    # Group predictions
    prediction_summary = {}
    for pred_class in np.unique(y_pred):
        count = np.sum(y_pred == pred_class)
        confidence = np.mean(confidence_scores[y_pred == pred_class])
        prediction_summary[str(pred_class)] = {
            "count": int(count),
            "confidence": round(confidence, 4)
        }

    result = {
        "pattern": "semi_supervised_learning",
        "setup": {
            "total_samples": len(df_clean),
            "labeled_samples": n_labeled,
            "unlabeled_samples": len(df_clean) - n_labeled,
            "labeled_ratio": labeled_ratio
        },
        "performance": {
            "labeled_accuracy": round(labeled_accuracy, 4),
            "predictions": prediction_summary
        },
        "insights": [
            f"🎯 Semi-supervised learning with {n_labeled}/{len(df_clean)} labeled samples",
            f"📊 Label propagation accuracy on labeled data: {labeled_accuracy:.1%}",
            f"🔍 Predicted labels for {len(unlabeled_indices)} unlabeled samples",
            f"📈 Average prediction confidence: {np.mean(confidence_scores):.3f}",
            "💡 Semi-supervised learning leverages both labeled and unlabeled data"
        ]
    }

    return result