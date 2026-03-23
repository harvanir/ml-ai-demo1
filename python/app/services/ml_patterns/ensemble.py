"""
Ensemble Learning Pattern
Demonstrates combining multiple models for improved performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from typing import Dict, Any, List

def analyze_ensemble_learning(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Ensemble learning analysis: combining multiple models

    Demonstrates:
    1. Individual model performance
    2. Ensemble methods (voting, bagging, boosting)
    3. Performance comparison
    """

    if df.empty:
        return {"error": "Empty dataset"}

    # Prepare data
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) < 2:
        return {"error": "Need at least 2 numeric columns for ensemble learning demo"}

    # Use last numeric column as target
    target_col = numeric_cols[-1]
    feature_cols = numeric_cols[:-1]

    df_clean = df[feature_cols + [target_col]].dropna()
    if len(df_clean) < 20:
        return {"error": "Need at least 20 samples for ensemble learning"}

    X = df_clean[feature_cols].values
    y = df_clean[target_col].values

    # Determine task type
    unique_values = len(np.unique(y))
    is_classification = unique_values < len(y) * 0.1  # Less than 10% unique = classification

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    if is_classification:
        # Classification ensemble
        n_classes = len(np.unique(y))

        # Individual models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=10, random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }

        # Train individual models and collect predictions
        individual_scores = {}
        predictions = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            score = accuracy_score(y_test, pred)
            individual_scores[name] = round(score, 4)
            predictions[name] = pred

        # Ensemble methods
        ensemble_models = {
            'Voting (Hard)': VotingClassifier(
                estimators=[(name, model) for name, model in models.items()],
                voting='hard'
            ),
            'Voting (Soft)': VotingClassifier(
                estimators=[(name, model) for name, model in models.items()],
                voting='soft'
            ),
            'Random Forest (Bagging)': RandomForestClassifier(n_estimators=50, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42)
        }

        ensemble_scores = {}
        for name, model in ensemble_models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            score = accuracy_score(y_test, pred)
            ensemble_scores[name] = round(score, 4)

        # Best individual and ensemble performance
        best_individual = max(individual_scores.values())
        best_ensemble = max(ensemble_scores.values())
        improvement = best_ensemble - best_individual

        result = {
            "pattern": "ensemble_learning",
            "type": "classification",
            "individual_models": individual_scores,
            "ensemble_models": ensemble_scores,
            "performance_comparison": {
                "best_individual_accuracy": best_individual,
                "best_ensemble_accuracy": best_ensemble,
                "improvement": round(improvement, 4)
            }
        }

    else:
        # Regression ensemble
        # Individual models
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=10, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=10, random_state=42),
            'SVR': SVR(kernel='rbf')
        }

        # Train individual models and collect predictions
        individual_scores = {}
        predictions = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            individual_scores[name] = round(rmse, 4)
            predictions[name] = pred

        # Ensemble methods
        ensemble_models = {
            'Voting': VotingRegressor(
                estimators=[(name, model) for name, model in models.items()]
            ),
            'Random Forest (Bagging)': RandomForestRegressor(n_estimators=50, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42)
        }

        ensemble_scores = {}
        for name, model in ensemble_models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            ensemble_scores[name] = round(rmse, 4)

        # Best individual and ensemble performance (lower RMSE is better)
        best_individual = min(individual_scores.values())
        best_ensemble = min(ensemble_scores.values())
        improvement = best_individual - best_ensemble  # Positive = improvement

        result = {
            "pattern": "ensemble_learning",
            "type": "regression",
            "individual_models": individual_scores,
            "ensemble_models": ensemble_scores,
            "performance_comparison": {
                "best_individual_rmse": best_individual,
                "best_ensemble_rmse": best_ensemble,
                "improvement": round(improvement, 4)
            }
        }

    # Common insights
    metric_name = "accuracy" if is_classification else "rmse"
    best_improvement = result["performance_comparison"]["improvement"]

    result["insights"] = [
        f"🎯 Ensemble learning combines {len(models)} individual models",
        f"📊 Task: {result['type']} with {len(feature_cols)} features",
        f"🏆 Best ensemble {metric_name}: {result['performance_comparison'][f'best_ensemble_{metric_name}']:.4f}",
        f"📈 Improvement over best individual: {best_improvement:.4f}",
        "💡 Ensemble methods often outperform individual models through diversity"
    ]

    return result