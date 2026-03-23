#!/usr/bin/env python3
"""
Test script untuk ML-AI-Demo1
Menguji endpoint /analyze dengan file sample
"""

import requests
import json
import os

def test_ml_analysis():
    """Test the ML analysis endpoint"""

    # URL endpoint
    url = "http://localhost:8000/analyze"

    # Path ke file sample
    sample_file = os.path.join(os.path.dirname(__file__), "..", "..", "sample_data", "sample.csv")

    if not os.path.exists(sample_file):
        print(f"❌ File sample tidak ditemukan: {sample_file}")
        return

    print("🚀 Testing ML Analysis...")
    print(f"📁 File: {sample_file}")
    print(f"🔗 URL: {url}")
    print("-" * 50)

    try:
        # Upload file
        with open(sample_file, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)

        print(f"📊 Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis berhasil!")
            print("\n📈 DATA SUMMARY:")
            print(f"   - Total Rows: {result['summary']['total_rows']}")
            print(f"   - Total Columns: {result['summary']['total_columns']}")
            print(f"   - Numeric Columns: {result['summary']['numeric_columns']}")

            print(f"\n🔍 ANOMALIES DETECTED: {len(result['anomalies'])}")
            if result['anomalies']:
                for anomaly in result['anomalies']:
                    z_score = anomaly.get('z_score')
                    z_str = f"{z_score:.2f}" if z_score is not None else "N/A"
                    print(f"   - Column: {anomaly['column']}, Value: {anomaly['value']}, Z-Score: {z_str}")
            else:
                print("   ✅ No anomalies detected")

            # Debug: print raw data types
            print(f"\n🐛 DEBUG: Raw response keys: {list(result.keys())}")
            if 'debug' in result:
                print(f"   Debug info: {result['debug']}")

            if result.get('ai_summary'):
                print(f"\n🤖 AI Summary: {result['ai_summary']}")
            else:
                print("\n🤖 AI Summary: Not requested (set use_ai=true for explanation)")

        else:
            print(f"❌ Error: {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Pastikan server berjalan di http://localhost:8000")
        print("   Jalankan: cd python && python -m app.main")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_with_anomaly():
    """Test dengan data yang mengandung anomaly"""

    print("\n" + "="*60)
    print("🧪 TESTING WITH ANOMALY DATA")
    print("="*60)

    # Buat data test dengan anomaly
    test_data = """name,price,quantity,rating
Product A,10.99,100,4.5
Product B,15.50,50,4.0
Product C,99999.99,1,5.0
Product D,20.00,75,3.8
Product E,12.49,200,4.2"""

    url = "http://localhost:8000/analyze"

    try:
        # Upload sebagai file in-memory
        from io import BytesIO
        files = {"file": ("test_anomaly.csv", BytesIO(test_data.encode()), "text/csv")}
        response = requests.post(url, files=files)

        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis dengan anomaly berhasil!")
            print(f"🔍 Anomalies: {len(result['anomalies'])}")
            for anomaly in result['anomalies']:
                z_score = anomaly.get('z_score')
                z_str = f"{z_score:.2f}" if z_score is not None else "N/A"
                print(f"   - {anomaly['column']}: {anomaly['value']} (z-score: {z_str})")
        else:
            print(f"❌ Error: {response.text}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("🤖 ML-AI-Demo1 Test Script")
    print("="*40)

    # Test normal
    test_ml_analysis()

    # Test dengan anomaly
    test_with_anomaly()

    print("\n" + "="*40)
    print("🎉 Test selesai!")
    print("📖 Lihat dokumentasi API: http://localhost:8000/docs")