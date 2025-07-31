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

# --- Lightweight RAG System ---
class LightweightRAG:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # Reduced for large docs
            stop_words='english',
            ngram_range=(1, 1),  # Only unigrams for speed
            max_df=0.95,
            min_df=1  # Allow rare terms in large docs
        )
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks
    
    def retrieve_relevant_chunks(self, query: str, chunks: List[str], top_k: int = 5) -> List[str]:
        if not chunks:
            return []
        try:
            # Extract key terms from question for better matching
            key_terms = self.extract_key_terms(query)
            
            # For large documents, pre-filter chunks by keyword matching
            if len(chunks) > 200:
                scored_chunks = []
                for i, chunk in enumerate(chunks):
                    chunk_lower = chunk.lower()
                    
                    # Score based on key terms and question keywords
                    score = 0
                    for term in key_terms:
                        if term in chunk_lower:
                            score += chunk_lower.count(term) * 2
                    
                    # Boost score for insurance-specific terms
                    insurance_terms = ['premium', 'coverage', 'waiting period', 'deductible', 'claim', 'policy', 'benefit', 'exclusion', 'limit']
                    for term in insurance_terms:
                        if term in chunk_lower:
                            score += 1
                    
                    if score > 0:
                        scored_chunks.append((i, chunk, score))
                
                # Sort by score and take top 100 for TF-IDF
                scored_chunks.sort(key=lambda x: x[2], reverse=True)
                filtered_chunks = [chunk for _, chunk, _ in scored_chunks[:100]]
            else:
                filtered_chunks = chunks
            
            if not filtered_chunks:
                return chunks[:top_k]
            
            # Apply TF-IDF on filtered chunks
            all_texts = filtered_chunks + [query]
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            query_vector = tfidf_matrix[-1]
            chunk_vectors = tfidf_matrix[:-1]
            similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Return chunks with minimum similarity threshold
            relevant_chunks = []
            for i in top_indices:
                if similarities[i] > 0.03:  # Lower threshold
                    relevant_chunks.append(filtered_chunks[i])
            
            return relevant_chunks if relevant_chunks else filtered_chunks[:3]
            
        except Exception as e:
            print(f"Retrieval error: {e}")
            return chunks[:top_k]
    
    def extract_key_terms(self, question: str) -> List[str]:
        """Extract important terms from question"""
        # Remove common question words
        stop_words = {'what', 'is', 'the', 'are', 'does', 'do', 'how', 'when', 'where', 'why', 'which', 'who', 'this', 'that', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', question.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        return key_terms

    async def generate_answer(self, question: str, context: str) -> str:
        """Generate single answer for one question"""
        
        # Analyze question type for better prompting
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['grace period', 'waiting period', 'period']):
            prompt_suffix = "State the exact time period (days/months/years) mentioned."
        elif question_lower.startswith(('does', 'is', 'are')):
            prompt_suffix = "Start with 'Yes' or 'No' then explain with specific details."
        elif 'define' in question_lower or 'definition' in question_lower:
            prompt_suffix = "Provide the exact definition as stated in the policy."
        elif any(word in question_lower for word in ['how much', 'amount', 'limit', 'percentage']):
            prompt_suffix = "Include specific amounts, percentages, or limits."
        else:
            prompt_suffix = "Be specific with numbers, conditions, and requirements."
        
        prompt = f"""Based on the insurance policy text, answer the question accurately and concisely.

POLICY EXCERPT:
{context}

QUESTION: {question}

INSTRUCTION: {prompt_suffix}

ANSWER:"""
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=130,
                    candidate_count=1
                )
            )
            answer = response.text.strip()
            
            # Clean up common LLM artifacts
            answer = re.sub(r'^(Answer:|A:)\s*', '', answer)
            answer = re.sub(r'^Based on.*?policy,?\s*', '', answer, flags=re.IGNORECASE)
            
            return answer
        except Exception as e:
            print(f"Generation error: {e}")
            return "Unable to process this question."

# --- PDF Processing ---
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        text_parts = [page.get_text() for page in doc if page.get_text().strip()]
        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")

# Initialize RAG system
rag_system = LightweightRAG()

# --- Main Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_analysis(request: QueryRequest, token: str = Depends(verify_token)):
    try:
        # Download PDF
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(request.documents, headers=headers, timeout=25)
        response.raise_for_status()

        # Process PDF
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_file.flush()

            raw_text = extract_text_from_pdf(temp_file.name)
            clean_text = rag_system.clean_text(raw_text)
            chunks = rag_system.chunk_text(clean_text)
            
            if not chunks:
                raise HTTPException(status_code=400, detail="No text could be extracted from PDF")

        # Process questions with better context
        answers = []
        for i, question in enumerate(request.questions):
            try:
                relevant_chunks = rag_system.retrieve_relevant_chunks(question, chunks, top_k=5)
                
                # Use more context but keep it focused
                context = "\n\n".join(relevant_chunks[:3])
                
                # If context is too short, add more chunks
                if len(context) < 500 and len(relevant_chunks) > 3:
                    context = "\n\n".join(relevant_chunks[:4])
                
                answer = await rag_system.generate_answer(question, context)
                answers.append(answer)
                print(f"Processed question {i+1}/{len(request.questions)}: {len(context)} chars context")
            except Exception as e:
                print(f"Error processing question {i+1}: {e}")
                answers.append("Unable to process this question.")

        return QueryResponse(answers=answers)

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")
    except Exception as e:
        print(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Policy Analyzer API is running"}

@app.get("/")
async def root():
    return {"message": "HackRx 6.0 Policy Analyzer - Optimized Version"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)