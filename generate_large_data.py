import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def generate_large_sample_data(n_samples: int = 100000, n_features: int = 6, anomaly_ratio: float = 0.05):
    """
    Generate large sample dataset with anomalies for testing.

    Parameters:
    - n_samples: Number of total samples (default 100k)
    - n_features: Number of features/columns
    - anomaly_ratio: Proportion of anomalies (default 5%)
    """
    np.random.seed(42)  # For reproducibility

    # Calculate number of normal and anomalous samples
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies

    print(f"Generating {n_samples:,} samples ({n_normal:,} normal + {n_anomalies:,} anomalies)")

    # Generate normal data (multivariate normal distribution)
    mean = [50, 100, 4.5, 25, 150, 4.2]  # price, quantity, rating, sales, etc.
    cov = np.array([
        [100, 20, 0.5, 10, 50, 0.2],   # price variance and covariances
        [20, 400, 1.0, 40, 100, 0.8],
        [0.5, 1.0, 0.25, 2, 5, 0.1],
        [10, 40, 2, 100, 200, 1.5],
        [50, 100, 5, 200, 1000, 3],
        [0.2, 0.8, 0.1, 1.5, 3, 0.5]
    ])

    # Generate normal samples
    normal_data = np.random.multivariate_normal(mean, cov, n_normal)

    # Generate anomalous data (extreme values)
    anomaly_data = []
    for _ in range(n_anomalies):
        # Create anomalies by multiplying normal values by extreme factors
        base_sample = np.random.multivariate_normal(mean, cov, 1)[0]

        # Randomly choose which features to make anomalous
        anomaly_features = np.random.choice(n_features, size=np.random.randint(1, 3), replace=False)

        for feat_idx in anomaly_features:
            # Make anomalies by extreme multiplication or addition
            if np.random.random() > 0.5:
                base_sample[feat_idx] *= np.random.uniform(3, 10)  # Multiply by 3-10x
            else:
                base_sample[feat_idx] += np.random.uniform(500, 2000)  # Add large value

        anomaly_data.append(base_sample)

    anomaly_data = np.array(anomaly_data)

    # Combine normal and anomalous data
    all_data = np.vstack([normal_data, anomaly_data])

    # Create DataFrame
    columns = ['price', 'quantity', 'rating', 'sales', 'value', 'score']
    df = pd.DataFrame(all_data, columns=columns)

    # Add categorical columns
    products = ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Phone', 'Tablet', 'Headphones', 'Speaker']
    categories = ['Electronics', 'Accessories', 'Gaming', 'Office', 'Mobile']

    df['name'] = np.random.choice(products, n_samples)
    df['category'] = np.random.choice(categories, n_samples)

    # Ensure numeric columns are properly typed
    numeric_cols = ['price', 'quantity', 'rating', 'sales', 'value', 'score']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Add some missing values randomly (realistic scenario)
    for col in numeric_cols:
        mask = np.random.random(n_samples) < 0.02  # 2% missing
        df.loc[mask, col] = np.nan

    return df

def main():
    parser = argparse.ArgumentParser(description='Generate large sample dataset for anomaly detection testing')
    parser.add_argument('--samples', type=int, default=100000, help='Number of samples (default: 100,000)')
    parser.add_argument('--features', type=int, default=6, help='Number of numeric features (default: 6)')
    parser.add_argument('--anomalies', type=float, default=0.05, help='Anomaly ratio (default: 0.05 = 5 percent)')
    parser.add_argument('--output', type=str, default='sample_data/large_sample.csv', help='Output file path')

    args = parser.parse_args()

    print("🚀 Generating large sample dataset...")
    print(f"📊 Samples: {args.samples:,}")
    print(f"📈 Features: {args.features}")
    print(f"🎯 Anomaly ratio: {args.anomalies:.1%}")
    print(f"💾 Output: {args.output}")
    print()

    # Generate data
    df = generate_large_sample_data(args.samples, args.features, args.anomalies)

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)

    print("✅ Dataset generated successfully!")
    print(f"📁 Saved to: {output_path}")
    print(f"📊 Shape: {df.shape}")
    print(f"📈 Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

    # Show sample statistics
    print("\n📈 Data Statistics:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe())

    # Count anomalies (approximate)
    print(f"\n🎯 Expected anomalies: ~{int(args.samples * args.anomalies):,}")

if __name__ == "__main__":
    main()