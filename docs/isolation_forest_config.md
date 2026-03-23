# Isolation Forest Configuration Guide

## Overview

Isolation Forest is an advanced anomaly detection algorithm used in this project for identifying outliers in large datasets (100k+ rows). It works by building an ensemble of isolation trees where anomalies are isolated closer to the root of the trees.

## Environment Variables Setup

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` file with your desired configuration:
   ```bash
   # Isolation Forest Configuration
   ISOLATION_FOREST_CONTAMINATION=0.1      # Expected anomaly proportion (0.0-0.5)
   ISOLATION_FOREST_N_ESTIMATORS=100      # Number of trees (1-1000)
   ISOLATION_FOREST_MAX_SAMPLES=          # Samples per tree (leave empty for auto)
   ISOLATION_FOREST_RANDOM_STATE=42       # Random seed for reproducibility
   ISOLATION_FOREST_MAX_FEATURES=1.0      # Max features per tree (0.1-1.0)
   ```

## Configuration Parameters

- `ISOLATION_FOREST_CONTAMINATION`: Expected proportion of anomalies in the dataset (0.0-0.5)
  - Lower values (e.g., 0.05) = More sensitive, detects more anomalies but may have more false positives
  - Higher values (e.g., 0.2) = More conservative, fewer false positives but may miss anomalies

- `ISOLATION_FOREST_N_ESTIMATORS`: Number of trees in the forest (1-1000)
  - More trees = Better accuracy but slower performance
  - Fewer trees = Faster but potentially less accurate

- `ISOLATION_FOREST_MAX_SAMPLES`: Number of samples to draw for each tree
  - 'auto' or empty = Use all samples
  - Float (0.0-1.0) = Fraction of total samples
  - Int = Absolute number of samples

- `ISOLATION_FOREST_RANDOM_STATE`: Random seed for reproducible results
  - Set to any integer for consistent results across runs
  - Leave unset for truly random behavior

- `ISOLATION_FOREST_MAX_FEATURES`: Maximum features to consider per tree (0.1-1.0)
  - 1.0 = Use all features
  - Lower values = Random feature subset for each tree

## Recommended Configurations by Use Case

| **Use Case** | **contamination** | **n_estimators** | **max_samples** | **max_features** | **Best For** |
|-------------|------------------|------------------|-----------------|------------------|-------------|
| **⚡ Speed Optimized** | 0.1 | 25-50 | 0.5 | 1.0 | Fast processing, limited resources |
| **🎯 Balanced (Default)** | 0.1 | 100 | auto | 1.0 | General purpose, good accuracy-speed balance |
| **🔍 High Accuracy** | 0.1 | 200-300 | auto | 0.8-1.0 | Maximum accuracy, slower processing |
| **🔬 High Sensitivity** | 0.05 | 100 | auto | 1.0 | Catch more anomalies, more false positives |
| **🛡️ Conservative** | 0.2 | 100 | auto | 0.8 | Fewer false positives, may miss anomalies |
| **📊 Large Datasets (1M+)** | 0.1 | 50-100 | 0.3-0.5 | 0.8 | Memory efficient for huge datasets |

## API Usage with Custom Parameters

```bash
# Use environment config
curl -X POST "http://localhost:8000/analyze?method=isolation_forest" \
     -F "file=@data.csv"

# Override specific parameters
curl -X POST "http://localhost:8000/analyze?method=isolation_forest&contamination=0.05&n_estimators=50" \
     -F "file=@data.csv"

# Speed optimized config
curl -X POST "http://localhost:8000/analyze?method=isolation_forest&contamination=0.1&n_estimators=25&max_samples=0.5" \
     -F "file=@data.csv"
```

## Technical Details

- **Algorithm**: Ensemble of isolation trees
- **Complexity**: O(n log n) for training, O(log n) for scoring
- **Memory Usage**: Scales well with dataset size
- **Best For**: High-dimensional data, large datasets, no assumption of data distribution
- **Limitations**: May not perform well on very small datasets (<100 samples)</content>
<parameter name="filePath">c:\Users\ahmad\OneDrive\Documents\dev\github\harvanir\ml-ai-demo1\docs\isolation_forest_config.md