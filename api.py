
import os
import fitz  # PyMuPDF
import requests
import tempfile
import asyncio
import re
from typing import List, Dict, Any, Tuple
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AsyncOpenAI # Use the OpenAI library to connect to OpenRouter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Init ---
load_dotenv()
AUTH_TOKEN = "78b25ddaad17f4e8d85cde3dca81ade8319272062cf10b73ba148b425151f2fd"
auth_scheme = HTTPBearer()
app = FastAPI(title="HackRx 6.0 Policy Analyzer - OpenRouter Edition")

# --- Auth ---
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != AUTH_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# --- Insurance-Specific Knowledge Base ---
# This class remains unchanged
class InsuranceKnowledgeBase:
    def __init__(self):
        self.question_patterns = {
            'grace_period': ['grace period', 'grace time', 'payment period', 'premium due'],
            'waiting_period': ['waiting period', 'wait time', 'exclusion period'],
            'pre_existing': ['pre-existing', 'pre existing', 'PED', 'prior condition'],
            'coverage': ['coverage', 'covered', 'cover', 'benefits', 'include'],
            'exclusions': ['exclusion', 'exclude', 'not covered', 'limitation'],
            'claim': ['claim', 'reimbursement', 'settlement'],
            'premium': ['premium', 'payment', 'cost', 'price'],
            'sum_insured': ['sum insured', 'limit', 'maximum amount'],
            'maternity': ['maternity', 'pregnancy', 'childbirth', 'delivery'],
            'room_rent': ['room rent', 'accommodation', 'ICU', 'hospital charges'],
            'ayush': ['ayush', 'ayurveda', 'homeopathy', 'unani', 'siddha'],
            'cataract': ['cataract', 'eye surgery', 'vision'],
            'cumulative_bonus': ['cumulative bonus', 'no claim bonus', 'NCD', 'bonus'],
            'hospital_definition': ['hospital', 'healthcare facility', 'medical center'],
            'co_payment': ['co-payment', 'co payment', 'copay', 'deductible']
        }
        self.section_keywords = {
            'definitions': ['definition', 'means', 'defined as', 'refers to'],
            'coverage': ['coverage', 'benefits', 'indemnify', 'shall cover'],
            'exclusions': ['exclusion', 'shall not', 'excluded', 'not covered'],
            'waiting_periods': ['waiting period', 'months of continuous coverage'],
            'claims': ['claim procedure', 'reimbursement', 'cashless'],
            'general_conditions': ['general terms', 'conditions', 'renewal']
        }

    def classify_question(self, question: str) -> Tuple[str, List[str]]:
        question_lower = question.lower()
        for category, patterns in self.question_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                return category, patterns
        return 'general', []

    def get_section_boost_terms(self, question_type: str) -> List[str]:
        boost_terms = {
            'grace_period': ['grace period', 'thirty days', '30 days', 'premium due date'],
            'waiting_period': ['waiting period', '36 months', '24 months', 'continuous coverage'],
            'pre_existing': ['pre-existing disease', 'PED', '36 months', 'physician within'],
            'maternity': ['maternity', 'childbirth', 'pregnancy', 'delivery', 'female insured'],
            'cataract': ['cataract', '25%', 'Rs. 40,000', 'per eye'],
            'room_rent': ['room rent', '2% of sum insured', 'Rs. 5,000', 'ICU', '5%', 'Rs. 10,000'],
            'ayush': ['AYUSH', 'Ayurveda', 'Homeopathy', 'Unani', 'Siddha', 'Naturopathy'],
            'cumulative_bonus': ['cumulative bonus', '5%', 'claim free', 'maximum of 50%'],
            'hospital_definition': ['hospital means', 'institution established', '10 inpatient beds']
        }
        return boost_terms.get(question_type, [])

# --- Enhanced RAG System ---
class InsuranceRAG:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1, 2), max_df=0.9, min_df=1)
        self.kb = InsuranceKnowledgeBase()
        
        # --- UPDATED: Initialize OpenRouter Client ---
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("gemini_key"),
        )
        # -------------------------------------------

    # ... (clean_insurance_text, smart_chunk_text, enhanced_retrieve methods remain the same) ...
    def clean_insurance_text(self, text: str) -> str:
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s{3,}', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\%\₹\/]', ' ', text)
        replacements = {
            r'Rs\.\s*': 'Rs. ', r'INR\s*': 'Rs. ', r'₹\s*': 'Rs. ',
            r'\b(\d+)\s*%': r'\1%', r'\b(\d+)\s*months?\b': r'\1 months',
            r'\b(\d+)\s*days?\b': r'\1 days'
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text.strip()

    def smart_chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
        chunks = []
        current_chunk = ""
        for line in text.split('\n'):
            if len(current_chunk) + len(line) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                words = current_chunk.split()
                overlap_text = ' '.join(words[-overlap//5:]) if len(words) > overlap//5 else ""
                current_chunk = overlap_text + '\n' + line
            else:
                current_chunk += '\n' + line
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]

    def enhanced_retrieve(self, question: str, chunks: List[str], top_k: int = 4) -> List[str]:
        if not chunks: return []
        try:
            question_type, patterns = self.kb.classify_question(question)
            boost_terms = self.kb.get_section_boost_terms(question_type)
            scored_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_lower = chunk.lower()
                score = 0
                for pattern in patterns:
                    score += len(re.findall(re.escape(pattern), chunk_lower)) * 3
                for term in boost_terms:
                    if term.lower() in chunk_lower: score += 2
                if question_type == 'grace_period' and any(term in chunk_lower for term in ['grace', '30', 'thirty', 'days']): score += 5
                elif question_type == 'waiting_period' and any(term in chunk_lower for term in ['waiting', '36', 'months', '24']): score += 5
                elif question_type == 'maternity' and 'maternity' in chunk_lower: score += 5
                elif question_type == 'cataract' and 'cataract' in chunk_lower: score += 5
                if question_type in ['coverage', 'benefits'] and any(word in chunk_lower for word in ['coverage', 'benefits', 'indemnify']): score += 3
                elif question_type in ['exclusions'] and any(word in chunk_lower for word in ['exclusion', 'excluded', 'shall not']): score += 3
                if re.search(r'\d+\s*%|\d+\s*days?|\d+\s*months?|Rs\.\s*\d+', chunk): score += 1
                scored_chunks.append((i, chunk, score))
            
            scored_chunks.sort(key=lambda x: x[2], reverse=True)
            high_score_chunks = [chunk for _, chunk, score in scored_chunks if score >= 5]
            if high_score_chunks: return high_score_chunks[:top_k]
            else:
                top_scored = [chunk for _, chunk, _ in scored_chunks[:min(50, len(scored_chunks))]]
                if len(top_scored) > 1:
                    all_texts = top_scored + [question]
                    tfidf_matrix = self.vectorizer.fit_transform(all_texts)
                    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
                    top_indices = np.argsort(similarities)[-top_k:][::-1]
                    return [top_scored[i] for i in top_indices]
                else: return top_scored[:top_k]
        except Exception as e:
            print(f"Enhanced retrieval error: {e}")
            return chunks[:top_k]

    # --- UPDATED: generate_insurance_answer to use OpenRouter client ---
    async def generate_insurance_answer(self, question: str, context: str, question_type: str) -> str:
        specific_instruction = self.kb.get_section_boost_terms(question_type) or "Be specific with numbers, percentages, time periods, and conditions."
        prompt = f"""You are an expert insurance policy analyst. Answer the question based ONLY on the provided policy text.
POLICY TEXT: {context}
QUESTION: {question}
INSTRUCTIONS:
- {specific_instruction}
- Be concise but include all relevant conditions.
- If the information is not in the provided text, say so.
ANSWER:"""
        
        try:
            # Use the OpenAI-compatible client to call OpenRouter
            response = await self.client.chat.completions.create(
                model="google/gemini-pro", # Standard model ID for Gemini Pro on OpenRouter
                messages=[
                    {"role": "system", "content": "You are an expert insurance policy analyst. Provide clear, accurate, and concise answers based only on the given policy context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=250
            )
            answer = response.choices[0].message.content.strip()
            return answer if answer else "No relevant information found in the policy."
        except Exception as e:
            print(f"Generation error: {e}")
            return "Unable to process this question."
    # -------------------------------------------------------------

# ... (The rest of your file: extract_text_from_pdf, rag_system initialization, and the FastAPI endpoint logic remains the same) ...
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        text_parts = [page.get_text() for page in doc if page.get_text().strip()]
        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")

rag_system = InsuranceRAG()

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_analysis(request: QueryRequest, token: str = Depends(verify_token)):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(request.documents, headers=headers, timeout=20)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            raw_text = extract_text_from_pdf(temp_file.name)
            clean_text = rag_system.clean_insurance_text(raw_text)
            chunks = rag_system.smart_chunk_text(clean_text)
            if not chunks:
                raise HTTPException(status_code=400, detail="No processable text found in PDF")

        async def process_question(question: str) -> str:
            question_type, _ = rag_system.kb.classify_question(question)
            relevant_chunks = rag_system.enhanced_retrieve(question, chunks, top_k=4)
            context = "\n\n---\n\n".join(relevant_chunks)
            return await rag_system.generate_insurance_answer(question, context, question_type)

        tasks = [process_question(q) for q in request.questions]
        answers = await asyncio.gather(*tasks)
        return QueryResponse(answers=answers)
    except Exception as e:
        print(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"System error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}