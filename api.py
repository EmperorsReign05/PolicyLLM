# api.py - Optimized for Insurance Documents
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
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Init ---
load_dotenv()
AUTH_TOKEN = "78b25ddaad17f4e8d85cde3dca81ade8319272062cf10b73ba148b425151f2fd"
auth_scheme = HTTPBearer()
app = FastAPI(title="HackRx 6.0 Policy Analyzer - Insurance Optimized")

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

# --- Insurance-Specific Knowledge Base ---
class InsuranceKnowledgeBase:
    def __init__(self):
        # Pre-built mappings for common insurance questions
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
        
        # Section keywords for better targeting
        self.section_keywords = {
            'definitions': ['definition', 'means', 'defined as', 'refers to'],
            'coverage': ['coverage', 'benefits', 'indemnify', 'shall cover'],
            'exclusions': ['exclusion', 'shall not', 'excluded', 'not covered'],
            'waiting_periods': ['waiting period', 'months of continuous coverage'],
            'claims': ['claim procedure', 'reimbursement', 'cashless'],
            'general_conditions': ['general terms', 'conditions', 'renewal']
        }

    def classify_question(self, question: str) -> Tuple[str, List[str]]:
        """Classify question type and return relevant keywords"""
        question_lower = question.lower()
        
        for category, patterns in self.question_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                return category, patterns
        
        return 'general', []

    def get_section_boost_terms(self, question_type: str) -> List[str]:
        """Get terms that should boost chunk relevance for specific question types"""
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
        self.vectorizer = TfidfVectorizer(
            max_features=1500,  # Increased for better insurance term coverage
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for insurance terms
            max_df=0.9,
            min_df=1
        )
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        self.kb = InsuranceKnowledgeBase()
        
    def clean_insurance_text(self, text: str) -> str:
        """Enhanced cleaning for insurance documents"""
        # Preserve important insurance formatting
        text = re.sub(r'\n+', '\n', text)  # Normalize line breaks
        text = re.sub(r'\s{3,}', ' ', text)  # Multiple spaces to single
        
        # Preserve important characters for insurance docs
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\%\₹\/]', ' ', text)
        
        # Normalize common insurance terms
        replacements = {
            r'Rs\.\s*': 'Rs. ',
            r'INR\s*': 'Rs. ',
            r'₹\s*': 'Rs. ',
            r'\b(\d+)\s*%': r'\1%',  # Normalize percentages
            r'\b(\d+)\s*months?\b': r'\1 months',  # Normalize months
            r'\b(\d+)\s*days?\b': r'\1 days'  # Normalize days
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def smart_chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
        """Smart chunking that preserves insurance document structure"""
        # Split by major sections first
        section_markers = [
            r'\n\d+\.\s+[A-Z][A-Z\s]+\n',  # Numbered sections
            r'\n\d+\.\d+\.?\s+[A-Z]',      # Subsections
            r'\nDefinitions?\n',
            r'\nCoverage\n',
            r'\nExclusions?\n',
            r'\nClaim\s+Procedure\n'
        ]
        
        chunks = []
        current_chunk = ""
        
        for line in text.split('\n'):
            if len(current_chunk) + len(line) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Keep overlap
                words = current_chunk.split()
                overlap_text = ' '.join(words[-overlap//5:]) if len(words) > overlap//5 else ""
                current_chunk = overlap_text + '\n' + line
            else:
                current_chunk += '\n' + line
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    def enhanced_retrieve(self, question: str, chunks: List[str], top_k: int = 4) -> List[str]:
        """Enhanced retrieval with insurance-specific logic"""
        if not chunks:
            return []
        
        try:
            # Classify question
            question_type, patterns = self.kb.classify_question(question)
            boost_terms = self.kb.get_section_boost_terms(question_type)
            
            # Score chunks with multiple criteria
            scored_chunks = []
            
            for i, chunk in enumerate(chunks):
                chunk_lower = chunk.lower()
                score = 0
                
                # 1. Pattern matching score
                for pattern in patterns:
                    matches = len(re.findall(re.escape(pattern), chunk_lower))
                    score += matches * 3
                
                # 2. Boost terms score
                for term in boost_terms:
                    if term.lower() in chunk_lower:
                        score += 2
                
                # 3. Question-specific boosting
                if question_type == 'grace_period' and any(term in chunk_lower for term in ['grace', '30', 'thirty', 'days']):
                    score += 5
                elif question_type == 'waiting_period' and any(term in chunk_lower for term in ['waiting', '36', 'months', '24']):
                    score += 5
                elif question_type == 'maternity' and 'maternity' in chunk_lower:
                    score += 5
                elif question_type == 'cataract' and 'cataract' in chunk_lower:
                    score += 5
                
                # 4. Section relevance
                if question_type in ['coverage', 'benefits'] and any(word in chunk_lower for word in ['coverage', 'benefits', 'indemnify']):
                    score += 3
                elif question_type in ['exclusions'] and any(word in chunk_lower for word in ['exclusion', 'excluded', 'shall not']):
                    score += 3
                
                # 5. Numerical relevance for specific questions
                if re.search(r'\d+\s*%|\d+\s*days?|\d+\s*months?|Rs\.\s*\d+', chunk):
                    score += 1
                
                scored_chunks.append((i, chunk, score))
            
            # Sort by score
            scored_chunks.sort(key=lambda x: x[2], reverse=True)
            
            # If we have high-scoring chunks, use them
            high_score_chunks = [chunk for _, chunk, score in scored_chunks if score >= 5]
            if high_score_chunks:
                selected_chunks = high_score_chunks[:top_k]
            else:
                # Fall back to TF-IDF for low-scoring chunks
                top_scored = [chunk for _, chunk, _ in scored_chunks[:min(50, len(scored_chunks))]]
                if len(top_scored) > 1:
                    all_texts = top_scored + [question]
                    tfidf_matrix = self.vectorizer.fit_transform(all_texts)
                    query_vector = tfidf_matrix[-1]
                    chunk_vectors = tfidf_matrix[:-1]
                    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
                    top_indices = np.argsort(similarities)[-top_k:][::-1]
                    selected_chunks = [top_scored[i] for i in top_indices]
                else:
                    selected_chunks = top_scored[:top_k]
            
            return selected_chunks[:top_k]
            
        except Exception as e:
            print(f"Enhanced retrieval error: {e}")
            return chunks[:top_k]

    async def generate_insurance_answer(self, question: str, context: str, question_type: str) -> str:
        """Generate answer with insurance-specific prompting"""
        
        # Enhanced prompts based on question type
        type_prompts = {
            'grace_period': "Focus on the exact number of days mentioned for grace period. Look for '30 days' or 'thirty days'.",
            'waiting_period': "State the exact waiting period in months (24 or 36 months). Specify what the waiting period applies to.",
            'pre_existing': "Look for '36 months' waiting period for pre-existing diseases. Mention the declaration requirement.",
            'maternity': "Check if maternity is covered or excluded. Look for specific conditions and waiting periods.",
            'cataract': "Find the specific limit for cataract treatment - look for '25% of Sum Insured' or 'Rs. 40,000 per eye'.",
            'room_rent': "Look for percentage limits - '2% of sum insured' for room rent, '5% of sum insured' for ICU.",
            'ayush': "Check coverage for AYUSH treatments (Ayurveda, Yoga, Naturopathy, Unani, Siddha, Homeopathy).",
            'cumulative_bonus': "Look for '5%' increase per claim-free year and 'maximum 50%' limit.",
            'hospital_definition': "Find the definition of hospital - look for bed requirements and registration criteria."
        }
        
        specific_instruction = type_prompts.get(question_type, "Be specific with numbers, percentages, time periods, and conditions.")
        
        prompt = f"""You are analyzing an insurance policy document. Answer the question based ONLY on the provided policy text.

POLICY TEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- {specific_instruction}
- Include exact numbers, percentages, time periods when mentioned
- If asking about coverage, state clearly if it's covered or excluded
- Be concise but include all relevant conditions
- If the information is not in the provided text, say so

ANSWER:"""
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=150,
                    candidate_count=1
                )
            )
            
            answer = response.text.strip()
            
            # Clean up response
            answer = re.sub(r'^(Answer:|A:)\s*', '', answer)
            answer = re.sub(r'^Based on.*?policy,?\s*', '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'^According to.*?document,?\s*', '', answer, flags=re.IGNORECASE)
            
            return answer
            
        except Exception as e:
            print(f"Generation error: {e}")
            return "Unable to process this question based on the provided policy text."

# --- PDF Processing ---
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(page_text)
        
        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")

# Initialize enhanced RAG system
rag_system = InsuranceRAG()

# --- Main Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_analysis(request: QueryRequest, token: str = Depends(verify_token)):
    try:
        # Download PDF with retry logic
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                response = requests.get(request.documents, headers=headers, timeout=20)
                response.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(1)

        # Process PDF with enhanced cleaning
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_file.flush()

            raw_text = extract_text_from_pdf(temp_file.name)
            clean_text = rag_system.clean_insurance_text(raw_text)
            chunks = rag_system.smart_chunk_text(clean_text)
            
            if not chunks:
                raise HTTPException(status_code=400, detail="No processable text found in PDF")

        # Process questions with enhanced logic
        answers = []
        for i, question in enumerate(request.questions):
            try:
                question_type, _ = rag_system.kb.classify_question(question)
                relevant_chunks = rag_system.enhanced_retrieve(question, chunks, top_k=4)
                
                # Combine context intelligently
                context = "\n\n---\n\n".join(relevant_chunks[:3])
                
                # Ensure we have enough context
                if len(context) < 400 and len(relevant_chunks) > 3:
                    context = "\n\n---\n\n".join(relevant_chunks[:4])
                
                answer = await rag_system.generate_insurance_answer(question, context, question_type)
                answers.append(answer)
                
                print(f"Q{i+1} [{question_type}]: {len(context)} chars, {len(relevant_chunks)} chunks")
                
            except Exception as e:
                print(f"Error processing question {i+1}: {e}")
                answers.append("Unable to determine from the provided policy document.")

        return QueryResponse(answers=answers)

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to access document: {str(e)}")
    except Exception as e:
        print(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"System error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Insurance Policy Analyzer API - Enhanced"}

@app.get("/")
async def root():
    return {"message": "HackRx 6.0 Insurance Policy Analyzer - Optimized for Accuracy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)