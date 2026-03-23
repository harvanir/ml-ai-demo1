#!/usr/bin/env python3
"""
Test Isolation Forest configurations directly without API server.
This demonstrates how different parameter configurations affect anomaly detection.
"""

import sys
import os
from pathlib import Path

# Add the python directory to the path so we can import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

import pandas as pd
import time
from app.services.anomaly_detector import detect_anomalies_isolation_forest
from app.config import settings

def test_configs():
    """Test different Isolation Forest configurations"""

    # Load sample data
    file_path = Path("sample_data/large_sample.csv")
    if not file_path.exists():
        print("❌ Sample data file not found. Please run generate_large_data.py first.")
        return

    print("📊 Loading sample data...")
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Different configurations to test
    configs = [
        {
            "name": "Default Config",
            "params": {}
        },
        {
            "name": "High Sensitivity (contamination=0.05)",
            "params": {
                "contamination": 0.05,
                "n_estimators": 50
            }
        },
        {
            "name": "Conservative (contamination=0.2)",
            "params": {
                "contamination": 0.2,
                "n_estimators": 200,
                "max_features": 0.8
            }
        },
        {
            "name": "Fast Processing (fewer trees)",
            "params": {
                "contamination": 0.1,
                "n_estimators": 25,
                "max_samples": 0.5
            }
        },
        {
            "name": "High Precision (more trees)",
            "params": {
                "contamination": 0.08,
                "n_estimators": 300,
                "max_features": 0.9,
                "random_state": 123
            }
        }
    ]

    print("\n🧪 Testing Isolation Forest Configurations")
    print("=" * 70)
    print(f"{'Configuration':<25} {'Anomalies':<10} {'Time(s)':<8} {'Parameters'}")
    print("=" * 70)

    for config in configs:
        start_time = time.time()

        # Run anomaly detection with specific config
        anomalies = detect_anomalies_isolation_forest(df, **config['params'])

        end_time = time.time()
        processing_time = end_time - start_time

        # Format parameters for display
        params_str = ", ".join([f"{k}={v}" for k, v in config['params'].items()])

        print("<25")

    print("=" * 70)
    print("\n📋 Configuration Details:")
    print("-" * 40)

    # Show current config settings
    config = settings.isolation_forest
    print("🔧 Current Default Configuration:")
    print(f"   contamination: {config.contamination}")
    print(f"   n_estimators: {config.n_estimators}")
    print(f"   max_samples: {config.max_samples}")
    print(f"   random_state: {config.random_state}")
    print(f"   max_features: {config.max_features}")

    print("\n💡 Tips for Configuration:")
    print("   • contamination: Lower = more sensitive (detects more anomalies)")
    print("   • n_estimators: Higher = more accurate but slower")
    print("   • max_samples: Lower = faster but potentially less accurate")
    print("   • max_features: Lower = reduces overfitting on high-dimensional data")

    print("\n✅ Configuration testing completed!")

if __name__ == "__main__":
    test_configs()