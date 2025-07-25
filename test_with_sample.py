# test_with_sample.py
import requests
import json

def test_api_with_sample():
    """Test with a publicly available PDF"""
    url = "http://localhost:8000/hackrx/run"
    headers = {
        "Authorization": "Bearer 78b25ddaad17f4e8d85cde3dca81ade8319272062cf10b73ba148b425151f2fd",
        "Content-Type": "application/json"
    }
    
    # Using a simple public PDF for testing
    data = {
        "documents": "https://www.adobe.com/support/products/enterprise/knowledgecenter/media/c4611_sample_explain.pdf",
        "questions": [
            "What is this document about?"
        ]
    }
    
    try:
        print("Testing API with sample document...")
        print("Document URL:", data["documents"])
        response = requests.post(url, headers=headers, json=data, timeout=120)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Your API is working correctly!")
            print("\nAnswers received:")
            for i, answer in enumerate(result.get("answers", []), 1):
                print(f"{i}. {answer}")
                
            print("\n🎉 Your API is ready for the hackathon submission!")
            print("Deploy it to Railway/Render/Heroku and submit the URL")
            
        else:
            print("❌ Error:", response.json())
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_with_local_file():
    """Instructions for testing with your own PDF"""
    print("\n=== Testing with Your Own PDF ===")
    print("If you want to test with your own PDF file:")
    print("1. Upload your PDF to a public URL (like Google Drive, Dropbox)")
    print("2. Replace the 'documents' URL in the test")
    print("3. Run the test again")
    print("\nFor Google Drive:")
    print("- Share the file publicly")
    print("- Use format: https://drive.google.com/uc?id=FILE_ID")

if __name__ == "__main__":
    test_api_with_sample()
    test_with_local_file()