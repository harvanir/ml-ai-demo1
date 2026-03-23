import requests
import time
from pathlib import Path

def test_isolation_forest_configs():
    """Test different Isolation Forest configurations"""

    url = "http://localhost:8000/analyze"
    file_path = "sample_data/large_sample.csv"

    # Different configurations to test
    configs = [
        {
            "name": "Default Config",
            "params": {"method": "isolation_forest"}
        },
        {
            "name": "High Sensitivity (contamination=0.05)",
            "params": {
                "method": "isolation_forest",
                "contamination": 0.05,
                "n_estimators": 50
            }
        },
        {
            "name": "Conservative (contamination=0.2)",
            "params": {
                "method": "isolation_forest",
                "contamination": 0.2,
                "n_estimators": 200,
                "max_features": 0.8
            }
        },
        {
            "name": "Fast Processing (fewer trees)",
            "params": {
                "method": "isolation_forest",
                "contamination": 0.1,
                "n_estimators": 25,
                "max_samples": 0.5
            }
        }
    ]

    print("🧪 Testing Isolation Forest with Different Configurations")
    print("=" * 60)

    for config in configs:
        print(f"\n🔬 {config['name']}")
        print("-" * 40)

        start_time = time.time()

        try:
            with open(file_path, "rb") as f:
                files = {"file": f}
                data = {"use_ai": "false", **config['params']}
                response = requests.post(url, files=files, data=data)

            end_time = time.time()
            processing_time = end_time - start_time

            if response.status_code == 200:
                result = response.json()
                anomalies = result.get('anomalies', [])
                summary = result.get('summary', {})

                print(f"⏱️  Processing Time: {processing_time:.2f}s")
                print(f"📊 Anomalies Found: {len(anomalies)}")
                print(f"📈 Total Rows: {summary.get('total_rows', 'N/A')}")

                # Show parameter summary
                params_summary = []
                for key, value in config['params'].items():
                    if key != 'method':
                        params_summary.append(f"{key}={value}")
                if params_summary:
                    print(f"⚙️  Parameters: {', '.join(params_summary)}")

            else:
                print(f"❌ Error: {response.status_code}")
                print(f"   {response.text}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")

    print("\n" + "=" * 60)
    print("✅ Configuration testing completed!")

if __name__ == "__main__":
    # Check if sample file exists
    if not Path("sample_data/large_sample.csv").exists():
        print("❌ Sample data file not found. Please run generate_large_data.py first.")
        exit(1)

    test_isolation_forest_configs()