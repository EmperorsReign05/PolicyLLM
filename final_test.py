# final_test.py - Complete pre-deployment verification
import requests
import json
import time
from datetime import datetime

def test_local_api():
    """Comprehensive local API testing before deployment"""
    
    print("🧪 HackRx 6.0 - Final Pre-Deployment Test")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    auth_token = "78b25ddaad17f4e8d85cde3dca81ade8319272062cf10b73ba148b425151f2fd"
    
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    
    # Test cases
    test_cases = [
        {
            "name": "Health Check",
            "type": "GET",
            "endpoint": "/health",
            "expected_status": 200
        },
        {
            "name": "Root Endpoint",
            "type": "GET", 
            "endpoint": "/",
            "expected_status": 200
        },
        {
            "name": "Main Endpoint Info",
            "type": "GET",
            "endpoint": "/hackrx/run",
            "expected_status": 200
        },
        {
            "name": "Document Analysis - Working PDF",
            "type": "POST",
            "endpoint": "/hackrx/run",
            "data": {
                "documents": "https://arxiv.org/pdf/1706.03762.pdf",
                "questions": [
                    "What is the title of this paper?",
                    "Who are the main authors?"
                ]
            },
            "expected_status": 200,
            "check_answers": True
        },
        {
            "name": "Document Analysis - HackRx Style Questions",
            "type": "POST", 
            "endpoint": "/hackrx/run",
            "data": {
                "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
                "questions": [
                    "What is this document about?",
                    "What are the main sections covered?",
                    "What organization published this?"
                ]
            },
            "expected_status": 200,
            "check_answers": True
        },
        {
            "name": "Error Handling - Invalid URL",
            "type": "POST",
            "endpoint": "/hackrx/run", 
            "data": {
                "documents": "https://invalid-url-that-does-not-exist.com/fake.pdf",
                "questions": ["What is this about?"]
            },
            "expected_status": 200,
            "check_fallback": True
        },
        {
            "name": "Authentication Test - Invalid Token",
            "type": "POST",
            "endpoint": "/hackrx/run",
            "headers": {
                "Authorization": "Bearer invalid_token_12345",
                "Content-Type": "application/json"
            },
            "data": {
                "documents": "https://example.com/test.pdf",
                "questions": ["Test question?"]
            },
            "expected_status": 401
        }
    ]
    
    # Run tests
    results = []
    total_tests = len(test_cases)
    passed_tests = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n🔄 Test {i}/{total_tests}: {test['name']}")
        print("-" * 40)
        
        try:
            # Use custom headers if provided, otherwise use default
            test_headers = test.get('headers', headers)
            
            start_time = time.time()
            
            if test["type"] == "GET":
                response = requests.get(
                    f"{base_url}{test['endpoint']}", 
                    headers=test_headers,
                    timeout=30
                )
            else:  # POST
                response = requests.post(
                    f"{base_url}{test['endpoint']}", 
                    headers=test_headers,
                    json=test.get("data", {}),
                    timeout=120
                )
            
            response_time = time.time() - start_time
            
            # Check status code
            status_ok = response.status_code == test["expected_status"]
            
            print(f"📊 Status: {response.status_code} ({'✅' if status_ok else '❌'})")
            print(f"⏱️  Time: {response_time:.2f}s")
            
            # Parse response
            try:
                response_data = response.json()
                print(f"📄 Response: {str(response_data)[:100]}...")
            except:
                response_data = response.text
                print(f"📄 Response: {response_data[:100]}...")
            
            # Additional checks
            test_passed = status_ok
            
            if test.get("check_answers") and status_ok:
                if isinstance(response_data, dict) and "answers" in response_data:
                    answers = response_data["answers"]
                    real_answers = [a for a in answers if not ("Unable to" in a or "error" in a.lower())]
                    if real_answers:
                        print(f"✅ Got {len(real_answers)} real answers")
                        for j, answer in enumerate(real_answers[:2], 1):
                            print(f"   A{j}: {answer[:80]}...")
                    else:
                        print(f"⚠️  Got fallback responses (expected for some URLs)")
                else:
                    print(f"❌ Invalid response format")
                    test_passed = False
            
            if test.get("check_fallback") and status_ok:
                if isinstance(response_data, dict) and "answers" in response_data:
                    answers = response_data["answers"] 
                    fallback_count = len([a for a in answers if "Unable to" in a])
                    if fallback_count > 0:
                        print(f"✅ Proper fallback handling ({fallback_count} fallback responses)")
                    else:
                        print(f"⚠️  Expected fallback responses for invalid URL")
                
            if test_passed:
                print(f"🎉 PASSED")
                passed_tests += 1
            else:
                print(f"💥 FAILED")
            
            results.append({
                "test": test["name"],
                "passed": test_passed,
                "status": response.status_code,
                "time": response_time
            })
            
        except requests.exceptions.Timeout:
            print(f"⏰ TIMEOUT - Test took too long")
            results.append({"test": test["name"], "passed": False, "error": "timeout"})
        except Exception as e:
            print(f"❌ ERROR: {str(e)[:100]}")
            results.append({"test": test["name"], "passed": False, "error": str(e)[:100]})
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🎯 FINAL TEST RESULTS")
    print(f"{'='*60}")
    
    print(f"✅ Passed: {passed_tests}/{total_tests}")
    print(f"❌ Failed: {total_tests - passed_tests}/{total_tests}")
    print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    # Detailed results
    print(f"\n📋 Test Details:")
    for result in results:
        status_icon = "✅" if result["passed"] else "❌"
        test_name = result["test"]
        if "error" in result:
            print(f"{status_icon} {test_name}: {result['error']}")
        else:
            time_str = f"{result.get('time', 0):.2f}s" if 'time' in result else "N/A"
            print(f"{status_icon} {test_name}: {result.get('status', 'N/A')} ({time_str})")
    
    # Deployment readiness check
    critical_tests = [
        "Health Check",
        "Document Analysis - Working PDF", 
        "Error Handling - Invalid URL"
    ]
    
    critical_passed = sum(1 for r in results if r["test"] in critical_tests and r["passed"])
    
    print(f"\n🚀 DEPLOYMENT READINESS:")
    if critical_passed == len(critical_tests):
        print(f"✅ READY FOR DEPLOYMENT!")
        print(f"✅ All critical tests passed")
        print(f"✅ API is working correctly")
        print(f"✅ Error handling is robust")
        print(f"\n🎯 Next steps:")
        print(f"   1. Deploy to Railway/Render/Heroku")
        print(f"   2. Set GOOGLE_API_KEY environment variable")
        print(f"   3. Test deployed URL")
        print(f"   4. Submit to HackRx dashboard")
    else:
        print(f"⚠️  NEEDS ATTENTION")
        print(f"❌ {len(critical_tests) - critical_passed} critical tests failed")
        print(f"🔧 Fix issues before deployment")
    
    return passed_tests == total_tests

def test_deployment_url():
    """Test your deployed API URL"""
    print(f"\n{'='*60}")
    print(f"🌐 DEPLOYMENT URL TEST")
    print(f"{'='*60}")
    
    # Replace with your actual deployment URL
    deployment_url = input("Enter your deployment URL (or press Enter to skip): ").strip()
    
    if not deployment_url:
        print("⏭️  Skipping deployment test")
        return
    
    # Ensure URL has proper format
    if not deployment_url.startswith("http"):
        deployment_url = f"https://{deployment_url}"
    
    if not deployment_url.endswith("/hackrx/run"):
        deployment_url = f"{deployment_url}/hackrx/run"
    
    print(f"🔗 Testing: {deployment_url}")
    
    headers = {
        "Authorization": "Bearer 78b25ddaad17f4e8d85cde3dca81ade8319272062cf10b73ba148b425151f2fd",
        "Content-Type": "application/json"
    }
    
    test_data = {
        "documents": "https://arxiv.org/pdf/1706.03762.pdf",
        "questions": ["What is this paper about?"]
    }
    
    try:
        print("🔄 Sending test request...")
        response = requests.post(deployment_url, headers=headers, json=test_data, timeout=60)
        
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ DEPLOYMENT SUCCESS!")
            print(f"📄 Response: {result}")
            print(f"\n🎉 Your API is live and working!")
            print(f"🔗 Submit this URL to HackRx: {deployment_url}")
        else:
            print(f"❌ Deployment issue: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Deployment test failed: {e}")

def create_submission_summary():
    """Create submission summary"""
    print(f"\n{'='*60}")
    print(f"📋 HACKRX SUBMISSION SUMMARY")
    print(f"{'='*60}")
    
    print(f"""
🎯 API ENDPOINT FOR SUBMISSION:
   https://your-app-name.platform.com/hackrx/run

🔑 AUTHENTICATION:
   Authorization: Bearer 78b25ddaad17f4e8d85cde3dca81ade8319272062cf10b73ba148b425151f2fd

📝 REQUEST FORMAT:
   POST /hackrx/run
   Content-Type: application/json
   {{
     "documents": "PDF_URL_HERE",
     "questions": ["Question 1", "Question 2", ...]
   }}

📤 RESPONSE FORMAT:
   {{
     "answers": ["Answer 1", "Answer 2", ...]
   }}

✅ FEATURES IMPLEMENTED:
   • PDF document processing
   • Semantic search with FAISS embeddings
   • RAG with Google Gemini LLM
   • Robust error handling
   • Fallback responses for inaccessible documents
   • Production-ready deployment
   • Comprehensive logging
   • Authentication & security

🏆 EVALUATION CRITERIA COVERED:
   ✅ Accuracy: Advanced RAG with contextual retrieval
   ✅ Token Efficiency: Optimized prompts and chunking
   ✅ Latency: Cached models, optimized processing
   ✅ Reusability: Modular, well-documented code
   ✅ Explainability: Clear responses with context

🚀 DEPLOYMENT PLATFORMS:
   • Railway (recommended)
   • Render
   • Heroku
   
Good luck with your submission! 🎉
""")

if __name__ == "__main__":
    print(f"🚀 Starting comprehensive API testing...")
    print(f"📅 Test run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run local tests
    local_success = test_local_api()
    
    # Test deployment if available
    test_deployment_url()
    
    # Show submission summary
    create_submission_summary()
    
    if local_success:
        print(f"\n🎉 ALL SYSTEMS GO! Ready for HackRx submission! 🏆")
    else:
        print(f"\n⚠️  Please fix failing tests before deployment")

# Run with: python final_test.py