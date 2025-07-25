# test_improved.py - Better test with working PDF URLs
import requests
import json
import time

def test_api_with_working_pdf():
    """Test with a reliable PDF URL"""
    url = "http://localhost:8000/hackrx/run"
    headers = {
        "Authorization": "Bearer 78b25ddaad17f4e8d85cde3dca81ade8319272062cf10b73ba148b425151f2fd",
        "Content-Type": "application/json"
    }
    
    # Using a more reliable PDF URL for testing
    test_cases = [
        {
            "name": "Sample Research Paper",
            "url": "https://arxiv.org/pdf/1706.03762.pdf",  # Attention paper
            "questions": ["What is this paper about?", "Who are the authors?"]
        },
        {
            "name": "HackRx Sample Policy",
            "url": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=your_signature",
            "questions": [
                "What is the grace period for premium payment?",
                "What is the waiting period for pre-existing diseases?"
            ]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n=== Test Case {i}: {test_case['name']} ===")
        
        data = {
            "documents": test_case["url"],
            "questions": test_case["questions"]
        }
        
        try:
            print(f"Testing with: {test_case['url'][:80]}...")
            start_time = time.time()
            
            response = requests.post(url, headers=headers, json=data, timeout=120)
            
            end_time = time.time()
            print(f"Response time: {end_time - start_time:.2f} seconds")
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ SUCCESS!")
                print("\nQuestions and Answers:")
                for j, (question, answer) in enumerate(zip(test_case["questions"], result.get("answers", [])), 1):
                    print(f"\nQ{j}: {question}")
                    print(f"A{j}: {answer}")
                    
            else:
                print("❌ Error Response:")
                try:
                    error_detail = response.json()
                    print(json.dumps(error_detail, indent=2))
                except:
                    print(response.text)
                    
        except requests.exceptions.Timeout:
            print("❌ Request timed out (>120 seconds)")
        except Exception as e:
            print(f"❌ Error: {e}")

def test_health_endpoint():
    """Test the health check endpoint"""
    print("\n=== Testing Health Endpoint ===")
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        print(f"Health Status: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API is healthy")
            print(f"Models loaded: {health_data.get('models_loaded', 'Unknown')}")
        else:
            print("❌ API health check failed")
    except Exception as e:
        print(f"❌ Health check error: {e}")

def test_with_hackrx_sample():
    """Test with the actual HackRx sample if available"""
    print("\n=== Testing with HackRx Sample Questions ===")
    
    url = "http://localhost:8000/hackrx/run"
    headers = {
        "Authorization": "Bearer 78b25ddaad17f4e8d85cde3dca81ade8319272062cf10b73ba148b425151f2fd",
        "Content-Type": "application/json"
    }
    
    # These are the exact questions from the hackathon problem statement
    hackrx_questions = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
    
    # Try with a public insurance policy PDF (fallback)
    data = {
        "documents": "https://www.irdai.gov.in/ADMINCMS/cms/frmGeneral_Layout.aspx?page=PageNo234&flag=1",
        "questions": hackrx_questions[:3]  # Test with first 3 questions
    }
    
    try:
        print("Testing with insurance policy questions...")
        response = requests.post(url, headers=headers, json=data, timeout=180)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ HackRx-style test SUCCESS!")
            for i, (q, a) in enumerate(zip(data["questions"], result.get("answers", [])), 1):
                print(f"\n{i}. Q: {q}")
                print(f"   A: {a}")
        else:
            print(f"❌ Status: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ HackRx test error: {e}")

def main():
    print("🚀 Starting comprehensive API tests...")
    print("Make sure your API is running on http://localhost:8000")
    
    # Test health first
    test_health_endpoint()
    
    # Test with working PDFs
    test_api_with_working_pdf()
    
    # Test with HackRx-style questions
    test_with_hackrx_sample()
    
    print("\n" + "="*50)
    print("🎯 Testing Complete!")
    print("If you see ✅ SUCCESS messages, your API is working!")
    print("Deploy to Railway/Render/Heroku and submit the URL to HackRx")
    print("="*50)

if __name__ == "__main__":
    main()