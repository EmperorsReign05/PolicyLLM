# api.py - Fixed for HackRx 6.0 with improved error handling
import os
import requests
import tempfile
import json
import time
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

# --- Load Environment Variables ---
load_dotenv()

# --- Authentication ---
AUTH_TOKEN = "78b25ddaad17f4e8d85cde3dca81ade8319272062cf10b73ba148b425151f2fd"
auth_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
        )

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# --- Global Variables for Performance ---
embeddings = None
llm = None

def initialize_models():
    """Initialize AI models once at startup"""
    global embeddings, llm
    
    if embeddings is None:
        print("Loading embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("Embeddings loaded successfully.")
    
    if llm is None:
        print("Loading LLM...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest", 
            temperature=0, 
            max_output_tokens=500,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        print("LLM loaded successfully.")

# Optimized prompt for hackathon requirements
hackathon_prompt = PromptTemplate.from_template(
    """
    You are an expert insurance policy analyst. Based on the policy document context, provide a direct, concise answer to the question.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    INSTRUCTIONS:
    - Answer directly and concisely
    - Include specific details like waiting periods, coverage amounts, conditions
    - If information is not in the context, state "Information not available in the provided policy document"
    - Keep answers under 100 words

    ANSWER:
    """
)

def format_docs(docs):
    """Format documents for prompt context"""
    return "\n\n".join(f"Page {doc.metadata.get('page', 'N/A')}: {doc.page_content}" for doc in docs)

def download_document_robust(url, max_retries=3):
    """
    Robust document downloading with multiple fallback strategies
    """
    print(f"Attempting to download: {url}")
    
    # Multiple header configurations to try
    header_configs = [
        {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/pdf,application/octet-stream,*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Referer': 'https://www.google.com/'
        },
        {
            'User-Agent': 'curl/7.68.0',
            'Accept': '*/*'
        },
        {
            'User-Agent': 'python-requests/2.28.0',
            'Accept': 'application/pdf'
        }
    ]
    
    for attempt in range(max_retries):
        for i, headers in enumerate(header_configs):
            try:
                print(f"Attempt {attempt + 1}/{max_retries}, header config {i + 1}")
                
                # Create session for better connection handling
                session = requests.Session()
                session.headers.update(headers)
                
                # Configure timeouts and retries
                response = session.get(
                    url, 
                    timeout=(10, 30),  # (connect_timeout, read_timeout)
                    verify=False,  # Disable SSL verification for problematic certificates
                    allow_redirects=True,
                    stream=True  # Stream download for large files
                )
                
                response.raise_for_status()
                
                # Read content in chunks
                content = b''
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        content += chunk
                
                print(f"Successfully downloaded {len(content)} bytes")
                return content
                
            except requests.exceptions.SSLError as e:
                print(f"SSL Error with config {i + 1}: {e}")
                continue
            except requests.exceptions.ConnectionError as e:
                print(f"Connection Error with config {i + 1}: {e}")
                continue
            except requests.exceptions.Timeout as e:
                print(f"Timeout Error with config {i + 1}: {e}")
                continue
            except requests.exceptions.RequestException as e:
                print(f"Request Error with config {i + 1}: {e}")
                continue
        
        if attempt < max_retries - 1:
            sleep_time = (attempt + 1) * 2
            print(f"All header configs failed. Waiting {sleep_time}s before retry...")
            time.sleep(sleep_time)
    
    raise Exception(f"Failed to download document after {max_retries} attempts with all header configurations")

# --- FastAPI Application ---
app = FastAPI(
    title="HackRx 6.0 AI Policy Analyzer",
    description="API for processing insurance policy documents and answering questions.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    initialize_models()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "API is running", 
        "message": "HackRx 6.0 Policy Analyzer",
        "endpoints": {
            "health_check": "/health",
            "main_endpoint": "/hackrx/run (POST only)",
            "test_endpoint": "/test"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": embeddings is not None and llm is not None,
        "timestamp": time.time()
    }

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_analysis(request: QueryRequest, token: str = Depends(verify_token)):
    """
    Main endpoint for processing documents and answering questions
    """
    start_time = time.time()
    document_url = request.documents
    questions = request.questions
    
    print(f"Processing {len(questions)} questions for document: {document_url}")

    try:
        # Ensure models are loaded
        if embeddings is None or llm is None:
            initialize_models()

        # 1. Download the document with robust error handling
        print("Downloading document...")
        try:
            pdf_content = download_document_robust(document_url)
        except Exception as download_error:
            print(f"Document download failed: {download_error}")
            # Return fallback answers when document can't be accessed
            fallback_answers = [
                "Unable to access the document to provide accurate information. Please ensure the document URL is accessible."
                for _ in questions
            ]
            return {"answers": fallback_answers}

        # 2. Process the document
        temp_file_path = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_content)
                temp_file_path = temp_file.name
            
            print("Loading and splitting document...")
            loader = PyMuPDFLoader(temp_file_path)
            docs = loader.load()
            
            if not docs:
                raise Exception("No content extracted from PDF")
            
            # Optimized text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            splits = text_splitter.split_documents(docs)
            print(f"Document split into {len(splits)} chunks")
            
            # Create vector store
            print("Creating vector embeddings...")
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 3}
            )
            
        except Exception as processing_error:
            print(f"Document processing error: {processing_error}")
            # Return fallback answers when processing fails
            fallback_answers = [
                "Unable to process the document content. The document may be corrupted or in an unsupported format."
                for _ in questions
            ]
            return {"answers": fallback_answers}
        
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

        # 3. Create the RAG chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | hackathon_prompt
            | llm
            | StrOutputParser()
        )

        # 4. Process all questions
        print("Processing questions...")
        answers = []
        for i, question in enumerate(questions, 1):
            print(f"Processing question {i}/{len(questions)}: {question[:50]}...")
            try:
                answer = rag_chain.invoke(question)
                answers.append(answer.strip())
            except Exception as e:
                print(f"Error processing question {i}: {e}")
                answers.append("Unable to process this question due to an error.")
        
        processing_time = time.time() - start_time
        print(f"All questions processed in {processing_time:.2f} seconds")
        
        return {"answers": answers}

    except Exception as e:
        print(f"Unexpected error: {e}")
        # Return generic error answers
        error_answers = [
            "An unexpected error occurred while processing your question."
            for _ in questions
        ]
        return {"answers": error_answers}

@app.get("/hackrx/run")
async def hackrx_info():
    """Info about the main endpoint"""
    return {
        "message": "This endpoint accepts POST requests only",
        "method": "POST",
        "endpoint": "/hackrx/run",
        "required_headers": {
            "Authorization": "Bearer <token>",
            "Content-Type": "application/json"
        },
        "required_body": {
            "documents": "URL to PDF document",
            "questions": ["List of questions"]
        }
    }

@app.post("/test")
async def test_endpoint(request: dict):
    """Test endpoint for debugging"""
    return {
        "message": "Test successful",
        "received": request,
        "models_loaded": embeddings is not None and llm is not None
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting HackRx 6.0 API server...")
    print("API will be available at: http://localhost:8000")
    print("Health check: http://localhost:8000/health")
    print("Main endpoint: http://localhost:8000/hackrx/run (POST only)")
    print("API Documentation: http://localhost:8000/docs")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)