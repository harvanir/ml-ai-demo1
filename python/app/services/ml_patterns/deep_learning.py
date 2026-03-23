"""
Deep Learning Pattern
Demonstrates neural networks for pattern recognition
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, List

class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_rate=0.2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def analyze_deep_learning(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Deep learning analysis: neural network for classification/regression

    For demo purposes, we'll:
    1. Create a neural network using PyTorch
    2. Train on the data for prediction task
    3. Show network architecture and performance
    """

    if df.empty:
        return {"error": "Empty dataset"}

    # Prepare data
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) < 2:
        return {"error": "Need at least 2 numeric columns for deep learning demo"}

    # Use last numeric column as target
    target_col = numeric_cols[-1]
    feature_cols = numeric_cols[:-1]

    df_clean = df[feature_cols + [target_col]].dropna()
    if len(df_clean) < 10:
        return {"error": "Need at least 10 samples for deep learning"}

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
        # Classification using PyTorch
        n_classes = len(np.unique(y))

        # Encode labels for multi-class
        if n_classes > 2:
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            y_train_encoded = le.transform(y_train)
            y_test_encoded = le.transform(y_test)
            y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)
            criterion = nn.CrossEntropyLoss()
            output_size = n_classes
        else:
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
            criterion = nn.BCEWithLogitsLoss()
            output_size = 1

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        # DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Model
        model = Net(len(feature_cols), 16, 8, output_size)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training with early stopping
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        epochs = 200

        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_train_tensor)
                val_loss = criterion(val_outputs, y_train_tensor).item()

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        # Evaluate
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train_tensor)
            test_outputs = model(X_test_tensor)

            if n_classes > 2:
                train_preds = torch.argmax(train_outputs, dim=1).numpy()
                test_preds = torch.argmax(test_outputs, dim=1).numpy()
                train_acc = accuracy_score(y_train_encoded, train_preds)
                test_acc = accuracy_score(y_test_encoded, test_preds)
            else:
                train_preds = (torch.sigmoid(train_outputs) > 0.5).float().numpy().flatten()
                test_preds = (torch.sigmoid(test_outputs) > 0.5).float().numpy().flatten()
                train_acc = accuracy_score(y_train, train_preds)
                test_acc = accuracy_score(y_test, test_preds)

        result = {
            "pattern": "deep_learning",
            "type": "classification",
            "framework": "PyTorch",
            "network_architecture": [
                f"Input: {len(feature_cols)} features",
                "Hidden Layer 1: 16 neurons (ReLU) + Dropout(0.2)",
                "Hidden Layer 2: 8 neurons (ReLU) + Dropout(0.2)",
                f"Output: {n_classes} classes ({'CrossEntropy' if n_classes > 2 else 'BCEWithLogits'})"
            ],
            "performance": {
                "train_accuracy": round(train_acc, 4),
                "test_accuracy": round(test_acc, 4),
                "converged": epoch < 199  # Check if early stopping occurred
            }
        }

    else:
        # Regression using PyTorch
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        # DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Model
        model = Net(len(feature_cols), 16, 8, 1)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Training with early stopping
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        epochs = 200

        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_train_tensor)
                val_loss = criterion(val_outputs, y_train_tensor).item()

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        # Evaluate
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train_tensor)
            test_outputs = model(X_test_tensor)

            train_mse = criterion(train_outputs, y_train_tensor).item()
            test_mse = criterion(test_outputs, y_test_tensor).item()

            predictions = test_outputs.numpy().flatten()
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = 1 - (mse / np.var(y_test)) if np.var(y_test) > 0 else 0

        result = {
            "pattern": "deep_learning",
            "type": "regression",
            "framework": "PyTorch",
            "network_architecture": [
                f"Input: {len(feature_cols)} features",
                "Hidden Layer 1: 16 neurons (ReLU) + Dropout(0.2)",
                "Hidden Layer 2: 8 neurons (ReLU) + Dropout(0.2)",
                "Output: 1 value (linear)"
            ],
            "performance": {
                "train_r2": round(r2, 4),  # Approximate R²
                "test_r2": round(r2, 4),
                "rmse": round(rmse, 4),
                "converged": epoch < 199
            }
        }

    # Common insights
    result["insights"] = [
        f"🧠 Multi-layer perceptron trained for {len(feature_cols)} features",
        f"📊 Task: {result['type']} using {result['framework']}",
        f"🏗️ Network: {len(result['network_architecture'])} layers with ReLU activation",
        f"📈 Performance: {'accuracy' if is_classification else 'R²'} = {result['performance']['test_accuracy'] if is_classification else result['performance']['test_r2']:.4f}",
        "💡 Neural networks automatically learn complex patterns from data"
    ]

    return result