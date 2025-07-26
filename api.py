# api.py - Production-Ready HackRx 6.0 API
import os
import requests
import tempfile
import json
import time
import logging
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import urllib3
from urllib.parse import urlparse

# Disable SSL warnings for problematic URLs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Import LangChain components
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
AUTH_TOKEN = "78b25ddaad17f4e8d85cde3dca81ade8319272062cf10b73ba148b425151f2fd"
MAX_RESPONSE_TIME = 30  # seconds
MAX_CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K_RETRIEVAL = 4

# --- Authentication ---
auth_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != AUTH_TOKEN:
        logger.warning(f"Authentication failed: {credentials.credentials[:20]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
        )
    return credentials.credentials

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# --- Global Variables ---
embeddings = None
llm = None
app_start_time = time.time()

def initialize_models():
    """Initialize AI models once at startup"""
    global embeddings, llm
    
    try:
        if embeddings is None:
            logger.info("Loading embeddings model...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}  # Better similarity scores
            )
            logger.info("✅ Embeddings loaded successfully")
        
        if llm is None:
            logger.info("Loading LLM...")
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest", 
                temperature=0.1,  # Slightly higher for more natural responses
                max_output_tokens=400,  # Optimized for concise answers
                google_api_key=api_key
            )
            logger.info("✅ LLM loaded successfully")
            
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        raise

# Enhanced prompt for better answers
hackathon_prompt = PromptTemplate.from_template(
    """You are an expert document analyst specializing in insurance policies and legal documents.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Provide a direct, accurate answer based ONLY on the context provided
- Include specific details like amounts, periods, percentages, and conditions when available
- If the exact information isn't in the context, state "The provided document does not contain specific information about [topic]"
- Keep answers concise but complete (50-150 words)
- Use professional, clear language
- Quote specific terms or clauses when relevant

ANSWER:"""
)

def format_docs(docs):
    """Enhanced document formatting for better context"""
    formatted = []
    for i, doc in enumerate(docs):
        page_num = doc.metadata.get('page', 'Unknown')
        content = doc.page_content.strip()
        formatted.append(f"[Page {page_num}]: {content}")
    return "\n\n".join(formatted)

def download_document_robust(url, max_retries=3):
    """
    Production-ready document downloading with comprehensive error handling
    """
    logger.info(f"📥 Downloading document: {url[:100]}...")
    
    # Validate URL
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL format")
    except Exception as e:
        raise Exception(f"URL validation failed: {e}")
    
    # Multiple header configurations
    header_configs = [
        {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/pdf,application/octet-stream,*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        },
        {
            'User-Agent': 'HackRx-Bot/1.0 (+https://hackrx.in)',
            'Accept': 'application/pdf,*/*'
        },
        {
            'User-Agent': 'python-requests/2.31.0',
            'Accept': '*/*'
        }
    ]
    
    for attempt in range(max_retries):
        for config_idx, headers in enumerate(header_configs):
            try:
                logger.info(f"🔄 Attempt {attempt + 1}/{max_retries}, Config {config_idx + 1}")
                
                session = requests.Session()
                session.headers.update(headers)
                
                response = session.get(
                    url, 
                    timeout=(15, 45),  # Increased timeouts for production
                    verify=False,
                    allow_redirects=True,
                    stream=True
                )
                
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' not in content_type and 'octet-stream' not in content_type:
                    logger.warning(f"⚠️  Unexpected content type: {content_type}")
                
                # Stream download with size limit (50MB max)
                content = b''
                max_size = 50 * 1024 * 1024  # 50MB
                downloaded = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        downloaded += len(chunk)
                        if downloaded > max_size:
                            raise Exception("Document too large (>50MB)")
                        content += chunk
                
                logger.info(f"✅ Successfully downloaded {len(content):,} bytes")
                return content
                
            except requests.exceptions.Timeout:
                logger.warning(f"⏰ Timeout with config {config_idx + 1}")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"🔌 Connection error with config {config_idx + 1}: {str(e)[:100]}")
            except requests.exceptions.HTTPError as e:
                logger.warning(f"🌐 HTTP error with config {config_idx + 1}: {e}")
                if e.response.status_code == 403:
                    continue  # Try next config for 403 errors
                elif e.response.status_code >= 500:
                    continue  # Retry for server errors
                else:
                    break  # Don't retry for client errors like 404
            except Exception as e:
                logger.warning(f"❌ Unexpected error with config {config_idx + 1}: {str(e)[:100]}")
        
        if attempt < max_retries - 1:
            sleep_time = min((attempt + 1) * 2, 10)  # Max 10 second delay
            logger.info(f"💤 Waiting {sleep_time}s before retry...")
            time.sleep(sleep_time)
    
    raise Exception(f"Failed to download document after {max_retries} attempts")

# --- FastAPI Application ---
app = FastAPI(
    title="HackRx 6.0 AI Policy Analyzer",
    description="Production-ready API for intelligent document analysis and question answering",
    version="2.0.0",
    docs_url="/docs",  # Enable in production for testing
    redoc_url="/redoc"
)

# Enhanced CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.on_event("startup")
async def startup_event():
    """Initialize models and log startup info"""
    logger.info("🚀 Starting HackRx 6.0 API...")
    try:
        initialize_models()
        logger.info("✅ API startup completed successfully")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.get("/")
async def root():
    """Enhanced root endpoint with system info"""
    uptime = time.time() - app_start_time
    return {
        "status": "🚀 HackRx 6.0 API Online",
        "version": "2.0.0",
        "uptime_seconds": round(uptime, 2),
        "models_loaded": embeddings is not None and llm is not None,
        "endpoints": {
            "main": "POST /hackrx/run",
            "health": "GET /health",
            "docs": "GET /docs"
        },
        "team": "Your Team Name",  # Update this
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check for monitoring"""
    uptime = time.time() - app_start_time
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": round(uptime, 2),
        "models": {
            "embeddings_loaded": embeddings is not None,
            "llm_loaded": llm is not None,
            "google_api_key_configured": bool(os.getenv("GOOGLE_API_KEY"))
        },
        "system": {
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            "environment": os.getenv("ENVIRONMENT", "production")
        }
    }
    
    # Check if all critical components are ready
    if not all([embeddings, llm, os.getenv("GOOGLE_API_KEY")]):
        health_status["status"] = "unhealthy"
        return health_status, 503
    
    return health_status

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_analysis(request: QueryRequest, token: str = Depends(verify_token)):
    """
    🎯 Main HackRx endpoint - Production-ready document analysis
    """
    request_start_time = time.time()
    request_id = f"req_{int(request_start_time)}"
    
    document_url = request.documents
    questions = request.questions
    
    logger.info(f"📋 [{request_id}] Processing {len(questions)} questions")
    logger.info(f"📄 [{request_id}] Document: {document_url[:100]}...")

    try:
        # Validate inputs
        if not document_url or not questions:
            raise HTTPException(400, "Missing documents or questions")
        
        if len(questions) > 20:  # Reasonable limit
            raise HTTPException(400, "Too many questions (max 20)")
        
        # Ensure models are loaded
        if embeddings is None or llm is None:
            logger.warning(f"[{request_id}] Models not loaded, initializing...")
            initialize_models()

        # Download document with timeout protection
        try:
            pdf_content = download_document_robust(document_url)
        except Exception as download_error:
            logger.error(f"[{request_id}] Download failed: {download_error}")
            fallback_answers = [
                "Unable to access the specified document. Please ensure the document URL is publicly accessible and points to a valid PDF file."
                for _ in questions
            ]
            return {"answers": fallback_answers}

        # Process document with enhanced error handling
        temp_file_path = None
        try:
            # Create temporary file with proper cleanup
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_content)
                temp_file_path = temp_file.name
            
            logger.info(f"[{request_id}] 📖 Loading PDF content...")
            loader = PyMuPDFLoader(temp_file_path)
            docs = loader.load()
            
            if not docs:
                raise Exception("No content extracted from PDF")
            
            logger.info(f"[{request_id}] 📚 Loaded {len(docs)} pages")
            
            # Enhanced text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=MAX_CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
                length_function=len
            )
            splits = text_splitter.split_documents(docs)
            logger.info(f"[{request_id}] ✂️  Split into {len(splits)} chunks")
            
            # Create optimized vector store
            logger.info(f"[{request_id}] 🧠 Creating embeddings...")
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={'k': TOP_K_RETRIEVAL}
            )
            
        except Exception as processing_error:
            logger.error(f"[{request_id}] Processing failed: {processing_error}")
            fallback_answers = [
                "Unable to process the document content. The document may be corrupted, password-protected, or in an unsupported format."
                for _ in questions
            ]
            return {"answers": fallback_answers}
        
        finally:
            # Cleanup temp file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    logger.warning(f"[{request_id}] Failed to cleanup temp file")

        # Create enhanced RAG chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | hackathon_prompt
            | llm
            | StrOutputParser()
        )

        # Process questions with timeout protection
        logger.info(f"[{request_id}] 🤔 Processing questions...")
        answers = []
        
        for i, question in enumerate(questions, 1):
            question_start = time.time()
            logger.info(f"[{request_id}] Q{i}/{len(questions)}: {question[:60]}...")
            
            try:
                # Check timeout
                if time.time() - request_start_time > MAX_RESPONSE_TIME - 5:
                    logger.warning(f"[{request_id}] Approaching timeout, using fallback")
                    answers.append("Response time limit reached. Please try with fewer questions.")
                    continue
                
                answer = rag_chain.invoke(question)
                clean_answer = answer.strip()
                
                # Ensure answer quality
                if len(clean_answer) < 10:
                    clean_answer = "The document does not contain sufficient information to answer this question."
                
                answers.append(clean_answer)
                
                question_time = time.time() - question_start
                logger.info(f"[{request_id}] ✅ Q{i} answered in {question_time:.2f}s")
                
            except Exception as e:
                logger.error(f"[{request_id}] Q{i} failed: {str(e)[:100]}")
                answers.append("Unable to process this question due to an error.")
        
        total_time = time.time() - request_start_time
        logger.info(f"[{request_id}] 🎉 Completed in {total_time:.2f}s")
        
        return {"answers": answers}

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)[:200]}")
        error_answers = [
            "An unexpected error occurred while processing your request. Please try again."
            for _ in questions
        ]
        return {"answers": error_answers}

@app.get("/hackrx/run")
async def hackrx_info():
    """Info about the main endpoint"""
    return {
        "message": "HackRx 6.0 Document Analysis Endpoint",
        "method": "POST",
        "endpoint": "/hackrx/run",
        "description": "Intelligent document analysis with question answering",
        "required_headers": {
            "Authorization": "Bearer <token>",
            "Content-Type": "application/json"
        },
        "request_format": {
            "documents": "URL to PDF document",
            "questions": ["Array of questions to answer"]
        },
        "response_format": {
            "answers": ["Array of answers corresponding to questions"]
        },
        "features": [
            "Multi-format document support",
            "Semantic search with FAISS",
            "Advanced RAG with Gemini",
            "Robust error handling",
            "Production-ready performance"
        ]
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Endpoint not found",
        "message": "Use POST /hackrx/run for document analysis",
        "available_endpoints": ["/", "/health", "/hackrx/run", "/docs"]
    }

if __name__ == "__main__":
    import uvicorn
    
    # Production configuration
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")  # Important for deployment
    
    logger.info("🚀 Starting HackRx 6.0 Production API...")
    logger.info(f"📡 Server: http://{host}:{port}")
    logger.info(f"📚 Documentation: http://{host}:{port}/docs")
    logger.info(f"🎯 Main endpoint: POST http://{host}:{port}/hackrx/run")
    
    uvicorn.run(
        "api:app", 
        host=host, 
        port=port, 
        reload=False,  # Disabled for production
        access_log=True,
        log_level="info"
    )