"""
Supervised Learning Pattern
Demonstrates classification and regression with labeled data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, Any, List

def analyze_supervised(df: pd.DataFrame, target_column: str = None, **kwargs) -> Dict[str, Any]:
    """
    Supervised learning analysis: classification or regression

    For demo purposes, we'll:
    1. Try to identify a target column (last numeric column)
    2. Use other columns as features
    3. Perform classification if target is categorical, regression if numeric
    """

    if df.empty:
        return {"error": "Empty dataset"}

    # Auto-detect target column (last column if not specified)
    if target_column is None:
        target_column = df.columns[-1]

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found"}

    # Prepare features and target
    feature_cols = [col for col in df.columns if col != target_column]
    if not feature_cols:
        return {"error": "No feature columns available"}

    # Clean data
    df_clean = df.dropna()
    if len(df_clean) < 10:
        return {"error": "Need at least 10 samples for supervised learning"}

    X = df_clean[feature_cols]
    y = df_clean[target_column]

    # Convert categorical features to numeric
    X_processed = X.copy()
    label_encoders = {}

    for col in X_processed.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        label_encoders[col] = le

    # Determine if classification or regression
    is_classification = False
    if y.dtype == 'object' or len(y.unique()) < len(y) * 0.1:  # Less than 10% unique values
        is_classification = True
        # Convert target to numeric for classification
        if y.dtype == 'object':
            target_encoder = LabelEncoder()
            y_encoded = target_encoder.fit_transform(y.astype(str))
        else:
            y_encoded = y.values
    else:
        y_encoded = y.values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=0.3, random_state=42
    )

    # Train model
    if is_classification:
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        result = {
            "pattern": "supervised_learning",
            "type": "classification",
            "target_column": target_column,
            "feature_columns": feature_cols,
            "samples": len(df_clean),
            "accuracy": round(accuracy, 4),
            "feature_importance": dict(zip(feature_cols,
                                         [round(x, 4) for x in model.feature_importances_])),
            "insights": [
                f"🎯 Classification model trained on {len(feature_cols)} features",
                f"📊 Model accuracy: {accuracy:.1%}",
                f"🔍 Most important feature: {feature_cols[np.argmax(model.feature_importances_)]}",
                "💡 Supervised learning uses labeled data to learn patterns for prediction"
            ]
        }
    else:
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        result = {
            "pattern": "supervised_learning",
            "type": "regression",
            "target_column": target_column,
            "feature_columns": feature_cols,
            "samples": len(df_clean),
            "rmse": round(rmse, 4),
            "feature_importance": dict(zip(feature_cols,
                                         [round(x, 4) for x in model.feature_importances_])),
            "insights": [
                f"🎯 Regression model trained on {len(feature_cols)} features",
                f"📊 Root Mean Square Error: {rmse:.4f}",
                f"🔍 Most important feature: {feature_cols[np.argmax(model.feature_importances_)]}",
                "💡 Supervised learning predicts continuous values from labeled examples"
            ]
        }

    return result