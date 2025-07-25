# test_api.py
import requests
import json

# Test the API locally
def test_api():
    url = "http://localhost:8000/hackrx/run"
    headers = {
        "Authorization": "Bearer 78b25ddaad17f4e8d85cde3dca81ade8319272062cf10b73ba148b425151f2fd",
        "Content-Type": "application/json"
    }
    
    # Test with a simple question first
    data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=example",
        "questions": [
            "What is the waiting period for cataract surgery?"
        ]
    }
    
    try:
        print("Testing API at:", url)
        print("Sending request...")
        response = requests.post(url, headers=headers, json=data, timeout=120)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success! Answers received:")
            for i, answer in enumerate(result.get("answers", []), 1):
                print(f"{i}. {answer}")
        elif response.status_code == 401:
            print("❌ Authentication error - check your Bearer token")
        elif response.status_code == 422:
            print("❌ Request format error:")
            print(response.json())
        else:
            print("❌ Error:", response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Make sure the API is running with 'python api.py'")
    except requests.exceptions.Timeout:
        print("❌ Request timed out. The document processing is taking too long.")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_health():
    """Test health endpoint"""
    url = "http://localhost:8000/health"
    try:
        response = requests.get(url, timeout=10)
        print(f"✅ Health check: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   Status: {health_data.get('status')}")
            print(f"   Models loaded: {health_data.get('models_loaded')}")
        else:
            print("❌ Health check failed")
    except Exception as e:
        print(f"❌ Health check failed: {e}")

def test_root():
    """Test root endpoint"""
    url = "http://localhost:8000/"
    try:
        response = requests.get(url, timeout=10)
        print(f"✅ Root endpoint: {response.status_code}")
        if response.status_code == 200:
            print("   API is running successfully")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")

if __name__ == "__main__":
    print("=== Testing API Endpoints ===")
    test_root()
    print()
    test_health()
    print()
    print("=== Testing Main Endpoint ===")
    print("This may take 15-30 seconds as it downloads and processes the document...")
    test_api()