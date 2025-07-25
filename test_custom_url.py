# test_custom_url.py - Test with your own PDF URL
import requests
import json
import time

def test_with_your_pdf():
    """Test with your own PDF URL"""
    url = "http://localhost:8000/hackrx/run"
    headers = {
        "Authorization": "Bearer 78b25ddaad17f4e8d85cde3dca81ade8319272062cf10b73ba148b425151f2fd",
        "Content-Type": "application/json"
    }
    
    YOUR_PDF_URL = "https://drive.google.com/uc?export=download&id=1nZpiq8W1_0gBD62TxjLML1Jr_WhocRV3"
    
    test_questions = [
        "What is this document about?",
        "What are the main points covered?",
        "Who published this document?"
    ]
    
    insurance_questions = [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?",
        "Does this policy cover maternity expenses?",
        "What is the definition of a hospital?",
        "Are there any sub-limits on room rent?"
    ]
    
    # Choose your questions
    questions_to_use = test_questions  # Change to insurance_questions if testing policy
    
    data = {
        "documents": YOUR_PDF_URL,  
        "questions": questions_to_use
    }
    
    print("🧪 Testing with your custom PDF URL")
    print(f"📄 Document: {YOUR_PDF_URL}")
    print(f"❓ Questions: {len(questions_to_use)} questions")
    print("-" * 60)
    
    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, json=data, timeout=120)
        end_time = time.time()
        
        print(f"⏱️  Response Time: {end_time - start_time:.2f} seconds")
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            answers = result.get("answers", [])
            
            # Check if we got real answers
            real_answers = [a for a in answers if not ("Unable to" in a)]
            
            if real_answers:
                print("✅ SUCCESS! Got real answers from your document!")
                print("\n📋 Results:")
                for i, (q, a) in enumerate(zip(questions_to_use, answers), 1):
                    print(f"\n{i}. Q: {q}")
                    print(f"   A: {a}")
                    
                print(f"\n🎉 Your API is working perfectly!")
                print(f"🚀 Ready for HackRx submission!")
                
            else:
                print("⚠️  Got fallback responses - check your PDF URL")
                print("💡 Make sure the URL is publicly accessible")
                
        else:
            print(f"❌ Error: {response.status_code}")
            error_data = response.json() if response.headers.get('content-type') == 'application/json' else response.text
            print(f"Details: {error_data}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def quick_test_multiple_urls():
    """Test multiple URLs quickly"""
    
    #🔥 PUT MULTIPLE TEST URLS HERE 🔥
    test_urls = [
         
        "https://drive.google.com/uc?export=download&id=1nZpiq8W1_0gBD62TxjLML1Jr_WhocRV3",  # Add your URLs here
       
    ]
    
    url = "http://localhost:8000/hackrx/run"
    headers = {
        "Authorization": "Bearer 78b25ddaad17f4e8d85cde3dca81ade8319272062cf10b73ba148b425151f2fd",
        "Content-Type": "application/json"
    }
    
    simple_question = ["What is this document about?"]
    
    print("🔄 Quick testing multiple URLs...")
    
    for i, pdf_url in enumerate(test_urls, 1):
        print(f"\n--- Test {i} ---")
        print(f"URL: {pdf_url[:50]}...")
        
        data = {"documents": pdf_url, "questions": simple_question}
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            if response.status_code == 200:
                result = response.json()
                answer = result.get("answers", ["No answer"])[0]
                
                if "Unable to" not in answer:
                    print(f"✅ SUCCESS: {answer[:100]}...")
                else:
                    print(f"⚠️  Fallback: {answer[:50]}...")
            else:
                print(f"❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Failed: {str(e)[:50]}...")

def how_to_get_pdf_urls():
    """Instructions for getting PDF URLs"""
    print("\n" + "="*60)
    print("📚 HOW TO GET A WORKING PDF URL")
    print("="*60)
    
    print("""
🎯 OPTION 1: Google Drive
1. Upload your PDF to Google Drive
2. Right-click → Share → Change to "Anyone with the link"  
3. Copy the share link (looks like: https://drive.google.com/file/d/FILE_ID/view)
4. Convert to direct link: https://drive.google.com/uc?id=FILE_ID

🎯 OPTION 2: Dropbox
1. Upload PDF to Dropbox
2. Right-click → Share → Create Link
3. Change the URL ending from ?dl=0 to ?dl=1

🎯 OPTION 3: GitHub (if you have an account)
1. Upload PDF to any GitHub repository
2. Go to the file → Click "Raw"
3. Copy the raw URL

🎯 OPTION 4: File hosting services
- WeTransfer
- SendSpace  
- MediaFire (get direct link)

🎯 OPTION 5: Create a simple test PDF
1. Create text file with sample insurance content:
   
   SAMPLE INSURANCE POLICY
   Grace Period: 30 days
   Waiting Period: 2 years for pre-existing conditions
   Coverage: Medical expenses up to $100,000
   
2. Convert to PDF (Word → Save as PDF)
3. Upload using any method above
""")

if __name__ == "__main__":
    print("🧪 Custom PDF URL Tester")
    print("="*60)
    
    # Show instructions first
    how_to_get_pdf_urls()
    
    print("\n🔧 USAGE:")
    print("1. Edit YOUR_PDF_URL variable in this file")
    print("2. Run: python test_custom_url.py")
    print("3. Check if you get real answers!")
    
    # Quick test with known URLs
    print("\n🚀 Running quick tests with sample URLs...")
    quick_test_multiple_urls()
    
    print("\n💡 To test YOUR PDF:")
    print("Edit the YOUR_PDF_URL variable at the top of this file!")
    
    # Uncomment the line below after adding your URL
    test_with_your_pdf()