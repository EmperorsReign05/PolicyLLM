# api.py - Ultra-Light Cloud Deployment Version
import os
import requests
import tempfile
import time
import logging
import re
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import urllib3
from urllib.parse import urlparse

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Light imports that work on all cloud platforms
import PyPDF2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
AUTH_TOKEN = "78b25ddaad17f4e8d85cde3dca81ade8319272062cf10b73ba148b425151f2fd"
MAX_RESPONSE_TIME = 30
MAX_CHUNK_SIZE = 800
TOP_K_RETRIEVAL = 3

# Authentication
auth_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != AUTH_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return credentials.credentials

# Pydantic models
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Global variables
vectorizer = None
llm = None
app_start_time = time.time()

def initialize_models():
    """Initialize AI models"""
    global vectorizer, llm
    
    try:
        if vectorizer is None:
            logger.info("Loading TF-IDF vectorizer...")
            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                max_df=0.95,
                min_df=2
            )
            logger.info("✅ Vectorizer loaded")
        
        if llm is None:
            logger.info("Loading LLM...")
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found")
            
            genai.configure(api_key=api_key)
            llm = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("✅ LLM loaded")
            
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        raise

def extract_text_from_pdf(pdf_content):
    """Extract text from PDF using PyPDF2"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(pdf_content)
            temp_file_path = temp_file.name
        
        text_chunks = []
        
        with open(temp_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if text.strip():
                        # Simple chunking by sentences
                        sentences = re.split(r'[.!?]+', text)
                        current_chunk = ""
                        
                        for sentence in sentences:
                            sentence = sentence.strip()
                            if not sentence:
                                continue
                                
                            if len(current_chunk) + len(sentence) < MAX_CHUNK_SIZE:
                                current_chunk += sentence + ". "
                            else:
                                if current_chunk.strip():
                                    text_chunks.append({
                                        'text': current_chunk.strip(),
                                        'page': page_num + 1
                                    })
                                current_chunk = sentence + ". "
                        
                        # Add remaining chunk
                        if current_chunk.strip():
                            text_chunks.append({
                                'text': current_chunk.strip(),
                                'page': page_num + 1
                            })
                            
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num + 1}: {e}")
                    continue
        
        # Cleanup temp file
        try:
            os.unlink(temp_file_path)
        except:
            pass
            
        return text_chunks
        
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return []

def search_documents(query, text_chunks):
    """Search for relevant documents using TF-IDF"""
    try:
        if not text_chunks:
            return []
            
        # Prepare texts
        texts = [chunk['text'] for chunk in text_chunks]
        all_texts = texts + [query]
        
        # Fit vectorizer and transform
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate similarities
        query_vector = tfidf_matrix[-1]  # Last item is the query
        document_vectors = tfidf_matrix[:-1]  # All except query
        
        similarities = cosine_similarity(query_vector, document_vectors).flatten()
        
        # Get top k results
        top_indices = similarities.argsort()[-TOP_K_RETRIEVAL:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                results.append({
                    'text': texts[idx],
                    'score': float(similarities[idx]),
                    'page': text_chunks[idx]['page']
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        # Fallback: return first few chunks
        return text_chunks[:TOP_K_RETRIEVAL]

def generate_answer(question, context_docs):
    """Generate answer using Gemini"""
    try:
        if not context_docs:
            return "No relevant information found in the document."
            
        # Prepare context
        context = "\n\n".join([f"[Page {doc['page']}]: {doc['text']}" for doc in context_docs])
        
        prompt = f"""You are an expert document analyst. Based on the provided context, answer the question accurately and concisely.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Provide a direct, accurate answer based ONLY on the context provided
- Include specific details when available
- If the information isn't in the context, state "The provided document does not contain specific information about this topic"
- Keep answers concise but complete (50-150 words)
- Use professional language

ANSWER:"""
        
        response = llm.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return "Unable to generate answer due to an error."

def download_document(url, max_retries=3):
    """Download document with retries"""
    logger.info(f"📥 Downloading: {url[:100]}...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30, verify=False, stream=True)
            response.raise_for_status()
            
            content = b''
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    content += chunk
                    
            logger.info(f"✅ Downloaded {len(content):,} bytes")
            return content
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    raise Exception(f"Failed to download after {max_retries} attempts")

# FastAPI App
app = FastAPI(
    title="HackRx 6.0 AI Policy Analyzer",
    description="Cloud-optimized API for document analysis",
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting HackRx API...")
    try:
        initialize_models()
        logger.info("✅ Startup completed")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.get("/")
async def root():
    uptime = time.time() - app_start_time
    return {
        "status": "🚀 HackRx 6.0 API Online",
        "version": "2.2.0",
        "uptime_seconds": round(uptime, 2),
        "models_loaded": vectorizer is not None and llm is not None,
        "endpoint": "/hackrx/run"
    }

@app.get("/health")
async def health_check():
    uptime = time.time() - app_start_time
    return {
        "status": "healthy",
        "uptime_seconds": round(uptime, 2),
        "models_loaded": vectorizer is not None and llm is not None,
        "google_api_configured": bool(os.getenv("GOOGLE_API_KEY"))
    }

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_analysis(request: QueryRequest, token: str = Depends(verify_token)):
    """Main analysis endpoint"""
    start_time = time.time()
    request_id = f"req_{int(start_time)}"
    
    logger.info(f"📋 [{request_id}] Processing {len(request.questions)} questions")
    
    try:
        # Ensure models are loaded
        if vectorizer is None or llm is None:
            initialize_models()
        
        # Download document
        try:
            pdf_content = download_document(request.documents)
        except Exception as e:
            logger.error(f"[{request_id}] Download failed: {e}")
            fallback_answers = [
                "Unable to access the specified document. Please ensure the document URL is publicly accessible and points to a valid PDF file."
                for _ in request.questions
            ]
            return {"answers": fallback_answers}
        
        # Extract text
        text_chunks = extract_text_from_pdf(pdf_content)
        if not text_chunks:
            logger.error(f"[{request_id}] No text extracted from PDF")
            fallback_answers = [
                "Unable to extract text from the document. The document may be corrupted, password-protected, or contain only images."
                for _ in request.questions
            ]
            return {"answers": fallback_answers}
        
        logger.info(f"[{request_id}] Extracted {len(text_chunks)} text chunks")
        
        # Process questions
        answers = []
        for i, question in enumerate(request.questions, 1):
            logger.info(f"[{request_id}] Processing Q{i}: {question[:50]}...")
            
            try:
                # Search for relevant context
                relevant_docs = search_documents(question, text_chunks)
                
                # Generate answer
                answer = generate_answer(question, relevant_docs)
                answers.append(answer)
                
                logger.info(f"[{request_id}] ✅ Q{i} completed")
                
            except Exception as e:
                logger.error(f"[{request_id}] Q{i} failed: {e}")
                answers.append("Unable to process this question due to an error.")
        
        total_time = time.time() - start_time
        logger.info(f"[{request_id}] 🎉 Completed in {total_time:.2f}s")
        
        return {"answers": answers}
        
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {e}")
        error_answers = [
            "An unexpected error occurred while processing your request. Please try again."
            for _ in request.questions
        ]
        return {"answers": error_answers}

@app.get("/hackrx/run")
async def hackrx_info():
    """Info about the main endpoint"""
    return {
        "message": "HackRx 6.0 Document Analysis Endpoint",
        "method": "POST",
        "endpoint": "/hackrx/run",
        "description": "Cloud-optimized document analysis with question answering",
        "features": [
            "PDF document processing with PyPDF2",
            "TF-IDF based document search", 
            "Google Gemini for answer generation",
            "Robust error handling",
            "Fast cloud deployment"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, log_level="info")