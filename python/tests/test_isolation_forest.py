import requests
import time

# Test Isolation Forest with large dataset
url = "http://localhost:8000/analyze"
file_path = "sample_data/large_sample.csv"

print("🚀 Testing Isolation Forest with 100k dataset...")
print(f"📁 File: {file_path}")
print(f"🔗 URL: {url}")
print()

start_time = time.time()

try:
    with open(file_path, "rb") as f:
        files = {"file": f}
        data = {"method": "isolation_forest", "use_ai": "false"}
        response = requests.post(url, files=files, data=data)

    end_time = time.time()

    print(f"📊 Status Code: {response.status_code}")
    print(f"⏱️  Processing Time: {end_time - start_time:.2f} seconds")
    print()

    if response.status_code == 200:
        result = response.json()
        anomalies = result.get('anomalies', [])
        summary = result.get('summary', {})

        print("📈 Data Summary:")
        print(f"   Total Rows: {summary.get('total_rows', 'N/A')}")
        print(f"   Total Columns: {summary.get('total_columns', 'N/A')}")
        print(f"   Numeric Columns: {len(summary.get('numeric_columns', []))}")
        print()

        print("🎯 Anomaly Detection Results:")
        print(f"   Method: Isolation Forest")
        print(f"   Anomalies Found: {len(anomalies)}")
        expected_anomalies = int(100000 * 0.05)  # 5% of 100k
        print(f"   Expected Anomalies: ~{expected_anomalies}")
        print()

        if anomalies:
            print("🚨 Sample Anomalies:")
            for i, anomaly in enumerate(anomalies[:5]):  # Show first 5
                print(f"   {i+1}. Column: {anomaly['column']}, Value: {anomaly['value']:.2f}, Score: {anomaly.get('z_score', 'N/A')}")
            if len(anomalies) > 5:
                print(f"   ... and {len(anomalies) - 5} more")
        else:
            print("✅ No anomalies detected")

        print()
        print("🤖 AI Summary:")
        ai_summary = result.get('ai_summary', 'No AI summary available')
        print(f"   {ai_summary}")

    else:
        print(f"❌ Error: {response.text}")

except FileNotFoundError:
    print(f"❌ Error: File '{file_path}' not found")
except requests.exceptions.ConnectionError:
    print("❌ Error: Cannot connect to server. Make sure the API is running on http://localhost:8000")
except Exception as e:
    print(f"❌ Error: {str(e)}")

print()
print("🏁 Test completed!")