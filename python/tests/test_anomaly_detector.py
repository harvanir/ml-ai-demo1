import pytest
import pandas as pd
import numpy as np
from app.services.anomaly_detector import detect_anomalies_zscore

def test_detect_anomalies_zscore():
    # Create test data with known outliers
    data = {
        'normal': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'with_outlier': [1, 2, 3, 4, 5, 6, 7, 8, 9, 2000]  # 2000 is clear outlier
    }
    df = pd.DataFrame(data)

    anomalies = detect_anomalies_zscore(df)

    # Should detect the outlier in 'with_outlier' column
    assert len(anomalies) > 0
    assert any(a.column == 'with_outlier' and a.value == 2000 for a in anomalies)