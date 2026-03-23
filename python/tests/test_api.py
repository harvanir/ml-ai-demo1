import requests
import os

# Test the analyze API
url = "http://localhost:8000/analyze"
file_path = os.path.join(os.path.dirname(__file__), "..", "..", "sample_data", "sample.csv")

if __name__ == "__main__":
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")