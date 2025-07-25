# simple_test.py
import requests
import json
import os

def test_permissions():
    """Test if we can create temporary files"""
    import tempfile
    import time
    
    try:
        temp_dir = tempfile.gettempdir()
        test_file = os.path.join(temp_dir, f"test_{int(time.time())}.txt")
        
        with open(test_file, 'w') as f:
            f.write("test")
        
        os.unlink(test_file)
        print("✅ File permissions OK")
        return True
    except Exception as e:
        print(f"❌ File permission issue: {e}")
        return False

def test_simple_request():
    """Test API with a very simple request"""
    url = "http://localhost:8000/hackrx/run"
    headers = {
        "Authorization": "Bearer 78b25ddaad17f4e8d85cde3dca81ade8319272062cf10b73ba148b425151f2fd",
        "Content-Type": "application/json"
    }
    
    # Try with a different PDF
    test_urls = [
        "https://www.clickdimensions.com/links/TestPDFfile.pdf",
        "https://file-examples.com/storage/fe86c5d4707a05da3c56813/2017/10/file-sample_150kB.pdf"
    ]
    
    for i, pdf_url in enumerate(test_urls, 1):
        print(f"\n=== Test {i} ===")
        data = {
            "documents": pdf_url,
            "questions": ["What is in this document?"]
        }
        
        try:
            print(f"Testing with: {pdf_url}")
            response = requests.post(url, headers=headers, json=data, timeout=60)
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ SUCCESS!")
                print(f"Answer: {result['answers'][0]}")
                return True
            else:
                print(f"❌ Error: {response.json()}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
    
    return False

def test_without_document():
    """Test what happens with an invalid document"""
    url = "http://localhost:8000/hackrx/run"
    headers = {
        "Authorization": "Bearer 78b25ddaad17f4e8d85cde3dca81ade8319272062cf10b73ba148b425151f2fd",
        "Content-Type": "application/json"
    }
    
    data = {
        "documents": "https://httpbin.org/status/404",  # This will fail
        "questions": ["Test question"]
    }
    
    try:
        print("\n=== Testing Error Handling ===")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        print(f"Status: {response.status_code}")
        if response.status_code != 200:
            print("✅ Error handling works correctly")
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing File Permissions ===")
    if test_permissions():
        print("\n=== Testing PDF Processing ===")
        if test_simple_request():
            print("\n🎉 Your API is working perfectly!")
        else:
            print("\n❌ PDF processing has issues")
    
    test_without_document()