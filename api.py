# api.py
import os
import fitz  # PyMuPDF
import requests
import tempfile
import asyncio
import re
from typing import List, Dict, Any
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Init ---
load_dotenv()
AUTH_TOKEN = "78b25ddaad17f4e8d85cde3dca81ade8319272062cf10b73ba148b425151f2fd"
auth_scheme = HTTPBearer()
app = FastAPI(title="HackRx 6.0 Policy Analyzer - Optimized")

# Configure Google AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Auth ---
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != AUTH_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

# --- Models ---
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# --- Lightweight Text Processing ---
class LightweightRAG:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2
        )
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)  # Keep basic punctuation
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def retrieve_relevant_chunks(self, query: str, chunks: List[str], top_k: int = 3) -> List[str]:
        """Retrieve most relevant chunks using TF-IDF"""
        if not chunks:
            return []
            
        try:
            # Fit vectorizer on chunks + query
            all_texts = chunks + [query]
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Calculate similarity between query and chunks
            query_vector = tfidf_matrix[-1]  # Last item is the query
            chunk_vectors = tfidf_matrix[:-1]  # All except query
            
            similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
            
            # Get top k chunks
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [chunks[i] for i in top_indices if similarities[i] > 0.1]
            
        except Exception as e:
            print(f"Retrieval error: {e}")
            # Fallback: return first few chunks
            return chunks[:top_k]
    
    async def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using Gemini"""
        prompt = f"""Based on the following policy context, provide a direct and concise answer to the question.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=200,
                    candidate_count=1
                )
            )
            return response.text.strip()
        except Exception as e:
            print(f"Generation error: {e}")
            return f"Unable to process question: {question}"

# Initialize RAG system
rag_system = LightweightRAG()

# --- PDF Processing ---
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF"""
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        
        for page in doc:
            text = page.get_text()
            if text.strip():
                text_parts.append(text)
        
        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")

# --- Main Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_analysis(request: QueryRequest, token: str = Depends(verify_token)):
    try:
        # Download PDF with timeout
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(request.documents, headers=headers, timeout=30)
        response.raise_for_status()

        # Process PDF
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_file.flush()
            
            # Extract and clean text
            raw_text = extract_text_from_pdf(temp_file.name)
            clean_text = rag_system.clean_text(raw_text)
            
            # Create chunks
            chunks = rag_system.chunk_text(clean_text)
            
            if not chunks:
                raise HTTPException(status_code=400, detail="No text could be extracted from PDF")

        # Process all questions concurrently
        async def process_question(question: str) -> str:
            relevant_chunks = rag_system.retrieve_relevant_chunks(question, chunks)
            context = "\n\n".join(relevant_chunks)
            return await rag_system.generate_answer(question, context)
        
        # Generate all answers concurrently
        tasks = [process_question(q) for q in request.questions]
        answers = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in individual tasks
        final_answers = []
        for i, answer in enumerate(answers):
            if isinstance(answer, Exception):
                final_answers.append(f"Error processing question {i+1}: {str(answer)}")
            else:
                final_answers.append(answer)
        
        return QueryResponse(answers=final_answers)

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")
    except Exception as e:
        print(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Policy Analyzer API is running"}

# Root endpoint
@app.get("/")
async def root():
    return {"message": "HackRx 6.0 Policy Analyzer - Optimized Version"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)