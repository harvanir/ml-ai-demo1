#!/usr/bin/env python3
"""
Test script for ML-AI Demo with all learning patterns
"""

import requests
import json
import os

BASE_URL = "http://localhost:8000"

def test_ml_patterns():
    """Test all ML learning patterns"""

    print("🧠 Testing ML-AI Demo with All Learning Patterns")
    print("=" * 60)

    # Test health endpoint
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print("❌ Health check failed")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return

    # Test available patterns
    try:
        response = requests.get(f"{BASE_URL}/ml-patterns")
        if response.status_code == 200:
            patterns = response.json()["patterns"]
            print(f"✅ Available ML patterns: {', '.join(patterns)}")
        else:
            print("❌ Failed to get patterns")
            return
    except Exception as e:
        print(f"❌ Error getting patterns: {e}")
        return

    # Test each pattern with sample data
    sample_file = "sample_data/sample.csv"
    if not os.path.exists(sample_file):
        print(f"❌ Sample file not found: {sample_file}")
        return

    print("\n🔬 Testing each ML pattern:")
    print("-" * 40)

    for pattern in patterns:
        try:
            with open(sample_file, 'rb') as f:
                files = {'file': ('sample.csv', f, 'text/csv')}
                data = {'pattern': pattern}
                response = requests.post(
                    f"{BASE_URL}/analyze-pattern",
                    files=files,
                    data=data
                )

            if response.status_code == 200:
                result = response.json()
                ml_result = result.get("ml_pattern", {})

                if "error" in ml_result:
                    print(f"⚠️  {pattern}: {ml_result['error']}")
                else:
                    pattern_type = ml_result.get("type", "unknown")
                    insights = ml_result.get("insights", [])
                    print(f"✅ {pattern} ({pattern_type}): {insights[0] if insights else 'Completed'}")
            else:
                print(f"❌ {pattern}: HTTP {response.status_code}")

        except Exception as e:
            print(f"❌ {pattern}: Error - {str(e)}")

    print("\n🎉 ML Pattern Testing Complete!")
    print("\n💡 Try the interactive API docs at: http://localhost:8000/docs")

if __name__ == "__main__":
    test_ml_patterns()