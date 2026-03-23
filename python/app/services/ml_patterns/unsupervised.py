"""
Unsupervised Learning Pattern
Demonstrates clustering and anomaly detection without labeled data
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from typing import Dict, Any, List
from app.services.anomaly_detector import detect_anomalies_iqr

def analyze_unsupervised(df: pd.DataFrame, n_clusters: int = 3, **kwargs) -> Dict[str, Any]:
    """
    Unsupervised learning analysis: clustering and anomaly detection

    Combines:
    1. K-means clustering to find natural groups
    2. Statistical anomaly detection (IQR method)
    """

    if df.empty:
        return {"error": "Empty dataset"}

    # Get numeric columns for analysis
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_cols:
        return {"error": "No numeric columns found for unsupervised learning"}

    # Prepare data for clustering
    df_clean = df[numeric_cols].dropna()
    if len(df_clean) < n_clusters:
        n_clusters = max(2, len(df_clean) // 2)  # Adjust clusters if needed

    if len(df_clean) < 3:
        return {"error": "Need at least 3 samples for clustering"}

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Calculate silhouette score (measure of cluster quality)
    try:
        silhouette_avg = silhouette_score(X_scaled, clusters)
    except:
        silhouette_avg = 0.0  # Can happen with small datasets

    # Get cluster centers (in original scale)
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

    # Analyze clusters
    cluster_sizes = pd.Series(clusters).value_counts().sort_index()
    cluster_stats = []

    for i in range(n_clusters):
        cluster_data = df_clean[clusters == i]
        stats = {
            "cluster": i,
            "size": int(cluster_sizes[i]),
            "percentage": round(float(cluster_sizes[i] / len(df_clean) * 100), 1),
            "centroid": {col: round(float(center), 4) for col, center in zip(numeric_cols, cluster_centers[i])},
            "feature_ranges": {}
        }

        # Calculate feature ranges for this cluster
        for col in numeric_cols:
            values = cluster_data[col]
            stats["feature_ranges"][col] = {
                "min": round(float(values.min()), 4),
                "max": round(float(values.max()), 4),
                "mean": round(float(values.mean()), 4),
                "std": round(float(values.std()), 4) if len(values) > 1 else 0.0
            }

        cluster_stats.append(stats)

    # Perform anomaly detection
    anomalies = detect_anomalies_iqr(df)

    # Find which clusters have anomalies
    anomaly_clusters = set()
    for anomaly in anomalies:
        if anomaly.column in numeric_cols:
            # Find which cluster this anomaly belongs to
            row_idx = anomaly.index
            if row_idx < len(clusters):
                anomaly_clusters.add(int(clusters[row_idx]))

    result = {
        "pattern": "unsupervised_learning",
        "clustering": {
            "n_clusters": n_clusters,
            "silhouette_score": round(silhouette_avg, 4),
            "cluster_sizes": [int(x) for x in cluster_sizes],
            "clusters": cluster_stats
        },
        "anomalies": {
            "count": len(anomalies),
            "details": [{"column": a.column, "value": a.value, "index": a.index}
                       for a in anomalies[:10]],  # Limit to first 10
            "anomaly_clusters": list(anomaly_clusters)
        },
        "insights": [
            f"🎯 Found {n_clusters} natural clusters in the data",
            f"📊 Clustering quality (silhouette): {silhouette_avg:.3f}",
            f"🔍 Detected {len(anomalies)} statistical anomalies",
            f"📈 Largest cluster: {int(cluster_sizes.idxmax())} ({int(cluster_sizes.max())} samples)",
            "💡 Unsupervised learning discovers hidden patterns without labeled data"
        ]
    }

    return result