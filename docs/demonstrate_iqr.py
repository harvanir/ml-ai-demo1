#!/usr/bin/env python3
"""
Demonstrate IQR Anomaly Detection
"""

import pandas as pd
import numpy as np

def demonstrate_iqr():
    # Load sample data
    df = pd.read_csv('sample_data/sample.csv')

    # Calculate IQR for price column
    prices = df['price'].values
    Q1 = np.percentile(prices, 25)
    Q3 = np.percentile(prices, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print('📊 IQR Analysis untuk Kolom Price:')
    print(f'Q1 (25%): ${Q1:.2f}')
    print(f'Q3 (75%): ${Q3:.2f}')
    print(f'IQR: ${IQR:.2f}')
    print(f'Batas Bawah: ${lower_bound:.2f}')
    print(f'Batas Atas: ${upper_bound:.2f}')
    print()

    print('📋 Status Harga Products:')
    anomalies = []
    for idx, row in df.iterrows():
        price = row['price']
        is_anomaly = price < lower_bound or price > upper_bound
        status = '🚨 ANOMALI' if is_anomaly else '✅ Normal'
        print(f'{row["name"]}: ${price:.2f} {status}')
        if is_anomaly:
            anomalies.append((row['name'], price))

    print(f'\n🎯 Total Anomali: {len(anomalies)} dari {len(df)} products')
    for name, price in anomalies:
        print(f'   - {name}: ${price:.2f}')

if __name__ == "__main__":
    demonstrate_iqr()