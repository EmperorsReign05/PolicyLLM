# api.py - Ultra-Fast Cloud Deployment Version
import os
import requests
import tempfile
import time
import logging
import re
import asyncio
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import urllib3
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import threading

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

# Configuration - OPTIMIZED FOR SPEED
AUTH_TOKEN = "78b25ddaad17f4e8d85cde3dca81ade8319272062cf10b73ba148b425151f2fd"
MAX_RESPONSE_TIME = 25  # Reduced from 30
MAX_CHUNK_SIZE = 600    # Reduced from 800 for faster processing
TOP_K_RETRIEVAL = 2     # Reduced from 3
MAX_PAGES = 50          # Limit pages processed
TIMEOUT_PER_QUESTION = 3  # Max time per question
DOWNLOAD_TIMEOUT = 10   # Reduced download timeout

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

# Global variables with thread safety
vectorizer = None
llm = None
app_start_time = time.time()
doc_cache = {}  # Simple document cache
cache_lock = threading.Lock()

def initialize_models():
    """Initialize AI models - OPTIMIZED"""
    global vectorizer, llm
    
    try:
        if vectorizer is None:
            logger.info("Loading optimized TF-IDF vectorizer...")
            vectorizer = TfidfVectorizer(
                max_features=2000,  # Reduced from 5000
                stop_words='english',
                ngram_range=(1, 2),
                max_df=0.9,  # More aggressive filtering
                min_df=1     # Less strict minimum
            )
            logger.info("✅ Vectorizer loaded")
        
        if llm is None:
            logger.info("Loading LLM...")
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found")
            
            genai.configure(api_key=api_key)
            # Use faster model configuration
            llm = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Lower temperature for faster, more deterministic responses
                    max_output_tokens=200,  # Limit output length
                    top_p=0.8,
                    top_k=20
                )
            )
            logger.info("✅ LLM loaded with speed optimizations")
            
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        raise

def extract_text_from_pdf_fast(pdf_content):
    """Extract text from PDF - SPEED OPTIMIZED"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(pdf_content)
            temp_file_path = temp_file.name
        
        text_chunks = []
        
        with open(temp_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = min(len(pdf_reader.pages), MAX_PAGES)  # Limit pages
            
            for page_num in range(total_pages):
                try:
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    if text.strip():
                        # Faster chunking - split by paragraphs instead of sentences
                        paragraphs = text.split('\n\n')
                        current_chunk = ""
                        
                        for para in paragraphs:
                            para = para.strip().replace('\n', ' ')
                            if not para:
                                continue
                                
                            if len(current_chunk) + len(para) < MAX_CHUNK_SIZE:
                                current_chunk += para + " "
                            else:
                                if current_chunk.strip():
                                    text_chunks.append({
                                        'text': current_chunk.strip(),
                                        'page': page_num + 1
                                    })
                                current_chunk = para + " "
                        
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

def search_documents_fast(query, text_chunks):
    """Search for relevant documents - SPEED OPTIMIZED"""
    try:
        if not text_chunks:
            return []
            
        # Use only top chunks if too many
        if len(text_chunks) > 20:
            text_chunks = text_chunks[:20]
            
        # Prepare texts
        texts = [chunk['text'] for chunk in text_chunks]
        all_texts = texts + [query]
        
        # Fit vectorizer and transform
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate similarities
        query_vector = tfidf_matrix[-1]
        document_vectors = tfidf_matrix[:-1]
        
        similarities = cosine_similarity(query_vector, document_vectors).flatten()
        
        # Get top k results with lower threshold for speed
        top_indices = similarities.argsort()[-TOP_K_RETRIEVAL:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.05:  # Lower threshold
                results.append({
                    'text': texts[idx],
                    'score': float(similarities[idx]),
                    'page': text_chunks[idx]['page']
                })
        
        return results if results else text_chunks[:TOP_K_RETRIEVAL]
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return text_chunks[:TOP_K_RETRIEVAL]

def generate_answer_fast(question, context_docs):
    """Generate answer using Gemini - SPEED OPTIMIZED"""
    try:
        if not context_docs:
            return "No relevant information found in the document."
            
        # Prepare shorter context
        context = "\n".join([doc['text'][:300] for doc in context_docs])  # Limit context length
        
        # Shorter, more direct prompt
        prompt = f"""Based on the policy document context, answer concisely:

Context: {context}

Question: {question}

Answer (max 100 words):"""
        
        response = llm.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return "Unable to generate answer due to an error."

def download_document_fast(url):
    """Download document - SPEED OPTIMIZED"""
    logger.info(f"📥 Fast downloading: {url[:50]}...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(
            url, 
            headers=headers, 
            timeout=DOWNLOAD_TIMEOUT,  # Reduced timeout
            verify=False, 
            stream=True
        )
        response.raise_for_status()
        
        # Read with size limit for speed
        content = b''
        max_size = 10 * 1024 * 1024  # 10MB limit
        
        for chunk in response.iter_content(chunk_size=16384):  # Larger chunks
            if chunk:
                content += chunk
                if len(content) > max_size:
                    logger.warning("Document too large, truncating")
                    break
                    
        logger.info(f"✅ Downloaded {len(content):,} bytes in {DOWNLOAD_TIMEOUT}s timeout")
        return content
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

# FastAPI App
app = FastAPI(
    title="HackRx 6.0 AI Policy Analyzer - SPEED OPTIMIZED",
    description="Ultra-fast cloud API for document analysis",
    version="2.3.0"
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
    logger.info("🚀 Starting OPTIMIZED HackRx API...")
    try:
        initialize_models()
        logger.info("✅ Fast startup completed")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.get("/")
async def root():
    uptime = time.time() - app_start_time
    return {
        "status": "⚡ HackRx 6.0 SPEED-OPTIMIZED API Online",
        "version": "2.3.0",
        "uptime_seconds": round(uptime, 2),
        "models_loaded": vectorizer is not None and llm is not None,
        "optimizations": [
            "Reduced chunk size for faster processing",
            "Limited pages and context length",
            "Parallel question processing",
            "Optimized TF-IDF parameters",
            "Faster Gemini configuration"
        ],
        "endpoint": "/hackrx/run"
    }

@app.get("/health")
async def health_check():
    uptime = time.time() - app_start_time
    return {
        "status": "healthy",
        "uptime_seconds": round(uptime, 2),
        "models_loaded": vectorizer is not None and llm is not None,
        "google_api_configured": bool(os.getenv("GOOGLE_API_KEY")),
        "performance_mode": "OPTIMIZED"
    }

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_analysis(request: QueryRequest, token: str = Depends(verify_token)):
    """Main analysis endpoint - SPEED OPTIMIZED"""
    start_time = time.time()
    request_id = f"req_{int(start_time)}"
    
    logger.info(f"⚡ [{request_id}] FAST processing {len(request.questions)} questions")
    
    try:
        # Ensure models are loaded
        if vectorizer is None or llm is None:
            initialize_models()
        
        # Check cache first
        doc_hash = str(hash(request.documents))
        text_chunks = None
        
        with cache_lock:
            if doc_hash in doc_cache:
                text_chunks = doc_cache[doc_hash]
                logger.info(f"[{request_id}] Using cached document")
        
        if text_chunks is None:
            # Download document
            try:
                pdf_content = download_document_fast(request.documents)
            except Exception as e:
                logger.error(f"[{request_id}] Download failed: {e}")
                fallback_answers = [
                    "Document access failed. Please check URL accessibility."
                    for _ in request.questions
                ]
                return {"answers": fallback_answers}
            
            # Extract text
            text_chunks = extract_text_from_pdf_fast(pdf_content)
            if not text_chunks:
                logger.error(f"[{request_id}] No text extracted")
                fallback_answers = [
                    "Unable to extract text from document."
                    for _ in request.questions
                ]
                return {"answers": fallback_answers}
            
            # Cache the result
            with cache_lock:
                doc_cache[doc_hash] = text_chunks
                # Keep cache small
                if len(doc_cache) > 5:
                    oldest_key = next(iter(doc_cache))
                    del doc_cache[oldest_key]
        
        logger.info(f"[{request_id}] Processing {len(text_chunks)} chunks")
        
        # Process questions in parallel using thread pool
        def process_question(question_data):
            i, question = question_data
            try:
                # Timeout per question
                start_q_time = time.time()
                
                relevant_docs = search_documents_fast(question, text_chunks)
                answer = generate_answer_fast(question, relevant_docs)
                
                q_time = time.time() - start_q_time
                if q_time > TIMEOUT_PER_QUESTION:
                    logger.warning(f"Q{i} took {q_time:.2f}s (over limit)")
                
                return answer
                
            except Exception as e:
                logger.error(f"Q{i} failed: {e}")
                return "Processing error occurred."
        
        # Execute questions in parallel
        question_data = list(enumerate(request.questions, 1))
        answers = list(executor.map(process_question, question_data))
        
        total_time = time.time() - start_time
        logger.info(f"[{request_id}] ⚡ COMPLETED in {total_time:.2f}s")
        
        return {"answers": answers}
        
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {e}")
        error_answers = [
            "Processing error occurred."
            for _ in request.questions
        ]
        return {"answers": error_answers}

@app.get("/hackrx/run")
async def hackrx_info():
    """Info about the main endpoint"""
    return {
        "message": "HackRx 6.0 SPEED-OPTIMIZED Document Analysis",
        "method": "POST",
        "endpoint": "/hackrx/run",
        "description": "Ultra-fast document analysis with aggressive optimizations",
        "speed_features": [
            "Document caching",
            "Parallel question processing", 
            "Reduced chunk sizes",
            "Limited context windows",
            "Fast PDF extraction",
            "Optimized TF-IDF search",
            "Speed-tuned Gemini config"
        ],
        "target_response_time": "< 25 seconds"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, log_level="info")