#!/usr/bin/env python3
"""
Test Isolation Forest configurations directly.
Run this from the python directory: python test_configs.py
"""

import pandas as pd
import time
from pathlib import Path
from app.services.anomaly_detector import detect_anomalies_isolation_forest
from app.config import settings

def test_configs():
    """Test different Isolation Forest configurations"""

    # Load sample data
    file_path = Path("../sample_data/large_sample.csv")
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
            "name": "High Sensitivity",
            "params": {
                "contamination": 0.05,
                "n_estimators": 50
            }
        },
        {
            "name": "Conservative",
            "params": {
                "contamination": 0.2,
                "n_estimators": 200,
                "max_features": 0.8
            }
        },
        {
            "name": "Fast Processing",
            "params": {
                "contamination": 0.1,
                "n_estimators": 25,
                "max_samples": 0.5
            }
        }
    ]

    print("\n🧪 Testing Isolation Forest Configurations")
    print("=" * 60)
    print(f"{'Configuration':<20} {'Anomalies':<10} {'Time(s)':<8}")
    print("=" * 60)

    for config in configs:
        start_time = time.time()

        # Run anomaly detection with specific config
        anomalies = detect_anomalies_isolation_forest(df, **config['params'])

        end_time = time.time()
        processing_time = end_time - start_time

        print("<20")

    print("=" * 60)
    print("\n📋 Current Default Configuration:")
    config = settings.isolation_forest
    print(f"   contamination: {config.contamination}")
    print(f"   n_estimators: {config.n_estimators}")
    print(f"   max_samples: {config.max_samples}")
    print(f"   random_state: {config.random_state}")
    print(f"   max_features: {config.max_features}")

    print("\n✅ Configuration testing completed!")

if __name__ == "__main__":
    test_configs()