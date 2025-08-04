# enhanced_api.py - Significantly Improved Insurance Document Analysis
import os
import fitz  # PyMuPDF
import requests
import tempfile
import asyncio
import re
from typing import List, Dict, Any, Tuple, Optional
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# --- Init ---
load_dotenv()
AUTH_TOKEN = "78b25ddaad17f4e8d85cde3dca81ade8319272062cf10b73ba148b425151f2fd"
auth_scheme = HTTPBearer()
app = FastAPI(title="HackRx 6.0 Policy Analyzer - Enhanced RAG")

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

# --- Enhanced Insurance Knowledge Base ---
class AdvancedInsuranceKB:
    def __init__(self):
        # Comprehensive insurance term mappings
        self.question_patterns = {
            'grace_period': [
                'grace period', 'grace time', 'payment period', 'premium due', 
                'pay premium', 'late payment', 'policy lapse'
            ],
            'waiting_period': [
                'waiting period', 'wait time', 'exclusion period', 'initial waiting',
                'cooling period', 'probation period'
            ],
            'pre_existing': [
                'pre-existing', 'pre existing', 'PED', 'prior condition', 
                'existing disease', 'medical history', 'declare disease'
            ],
            'coverage': [
                'coverage', 'covered', 'cover', 'benefits', 'include', 
                'indemnify', 'reimburse', 'eligible', 'entitled'
            ],
            'exclusions': [
                'exclusion', 'exclude', 'not covered', 'limitation', 
                'not eligible', 'shall not', 'except'
            ],
            'claim': [
                'claim', 'reimbursement', 'settlement', 'cashless', 
                'claim procedure', 'submit claim', 'claim process'
            ],
            'premium': [
                'premium', 'payment', 'cost', 'price', 'installment', 
                'annual premium', 'monthly premium'
            ],
            'sum_insured': [
                'sum insured', 'limit', 'maximum amount', 'coverage limit', 
                'benefit limit', 'sum assured'
            ],
            'maternity': [
                'maternity', 'pregnancy', 'childbirth', 'delivery', 
                'maternal', 'newborn', 'confinement'
            ],
            'room_rent': [
                'room rent', 'accommodation', 'ICU', 'hospital charges', 
                'room category', 'bed charges', 'hospital room'
            ],
            'ayush': [
                'ayush', 'ayurveda', 'homeopathy', 'unani', 'siddha', 
                'naturopathy', 'yoga', 'alternative medicine'
            ],
            'cataract': [
                'cataract', 'eye surgery', 'vision', 'lens replacement', 
                'eye treatment', 'ophthalmology'
            ],
            'cumulative_bonus': [
                'cumulative bonus', 'no claim bonus', 'NCD', 'bonus', 
                'claim free', 'renewal bonus', 'loyalty bonus'
            ],
            'hospital_definition': [
                'hospital', 'healthcare facility', 'medical center', 
                'nursing home', 'clinic', 'medical institution'
            ],
            'co_payment': [
                'co-payment', 'co payment', 'copay', 'deductible', 
                'patient contribution', 'out of pocket'
            ],
            'ambulance': [
                'ambulance', 'emergency transport', 'patient transport'
            ],
            'domiciliary': [
                'domiciliary', 'home treatment', 'home care', 'treatment at home'
            ],
            'day_care': [
                'day care', 'daycare', 'same day discharge', 'outpatient surgery'
            ]
        }
        
        # Enhanced section identification
        self.section_markers = {
            'definitions': [
                'definitions', 'defined terms', 'interpretation', 'meaning',
                'shall mean', 'means and includes', 'terminology'
            ],
            'coverage': [
                'coverage', 'benefits', 'what is covered', 'scope of cover',
                'indemnity', 'reimbursement', 'eligible expenses'
            ],
            'exclusions': [
                'exclusions', 'what is not covered', 'limitations', 
                'exceptions', 'shall not cover', 'excluded'
            ],
            'waiting_periods': [
                'waiting period', 'initial waiting', 'specific waiting',
                'cooling period', 'probation'
            ],
            'claims': [
                'claim procedure', 'claims process', 'how to claim',
                'reimbursement procedure', 'cashless facility'
            ],
            'conditions': [
                'general conditions', 'terms and conditions', 'policy conditions',
                'general terms', 'provisions'
            ]
        }

        # Specific value extractors for common insurance queries
        self.value_patterns = {
            'grace_period': [
                r'grace period.*?(\d+)\s*days?',
                r'(\d+)\s*days?.*?grace',
                r'thirty\s*days?',
                r'30\s*days?'
            ],
            'waiting_period': [
                r'waiting period.*?(\d+)\s*months?',
                r'(\d+)\s*months?.*?waiting',
                r'initial waiting.*?(\d+)',
                r'(\d+)\s*months?.*?continuous coverage'
            ],
            'room_rent': [
                r'room rent.*?(\d+)%',
                r'(\d+)%.*?sum insured.*?room',
                r'Rs\.?\s*(\d+,?\d*)',
                r'ICU.*?(\d+)%'
            ],
            'cataract': [
                r'cataract.*?(\d+)%',
                r'Rs\.?\s*(\d+,?\d*).*?per eye',
                r'25%.*?sum insured'
            ]
        }

    def extract_numerical_values(self, text: str, question_type: str) -> List[str]:
        """Extract specific numerical values relevant to question type"""
        if question_type not in self.value_patterns:
            return []
        
        values = []
        for pattern in self.value_patterns[question_type]:
            matches = re.findall(pattern, text, re.IGNORECASE)
            values.extend(matches)
        
        return values

    def classify_question_advanced(self, question: str) -> Tuple[str, List[str], float]:
        """Advanced question classification with confidence scoring"""
        question_lower = question.lower()
        scores = {}
        
        for category, patterns in self.question_patterns.items():
            score = 0
            matched_patterns = []
            
            for pattern in patterns:
                if pattern in question_lower:
                    # Weight longer patterns more heavily
                    weight = len(pattern.split()) * 2
                    score += weight
                    matched_patterns.append(pattern)
            
            if score > 0:
                scores[category] = (score, matched_patterns)
        
        if scores:
            best_category = max(scores.keys(), key=lambda k: scores[k][0])
            confidence = scores[best_category][0] / sum(s[0] for s in scores.values())
            return best_category, scores[best_category][1], confidence
        
        return 'general', [], 0.0

# --- Advanced RAG System ---
class AdvancedInsuranceRAG:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams for better insurance phrase matching
            max_df=0.85,
            min_df=1,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'  # Include alphanumeric tokens
        )
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        self.kb = AdvancedInsuranceKB()
        self.stop_words = set(stopwords.words('english'))
        
    def advanced_text_cleaning(self, text: str) -> str:
        """Advanced text cleaning specifically for insurance documents"""
        # Preserve important insurance document structure
        text = re.sub(r'\f', '\n', text)  # Form feed to newline
        text = re.sub(r'\r\n', '\n', text)  # Windows line endings
        text = re.sub(r'\r', '\n', text)  # Mac line endings
        
        # Preserve important formatting patterns
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'\s{4,}', ' ', text)  # Multiple spaces but preserve some spacing
        
        # Clean up common PDF artifacts while preserving structure
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)  # Control characters
        
        # Normalize insurance-specific formatting
        replacements = {
            # Currency normalization
            r'(?:Rs\.?|INR|₹)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)': r'Rs. \1',
            # Percentage normalization
            r'(\d+(?:\.\d+)?)\s*%': r'\1%',
            # Time period normalization
            r'(\d+)\s*(?:months?|month)': r'\1 months',
            r'(\d+)\s*(?:days?|day)': r'\1 days',
            r'(\d+)\s*(?:years?|year)': r'\1 years',
            # Common insurance abbreviations
            r'\bPED\b': 'Pre-Existing Disease',
            r'\bICU\b': 'Intensive Care Unit',
            r'\bOPD\b': 'Out Patient Department',
            r'\bIPD\b': 'In Patient Department'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text.strip()

    def semantic_chunking(self, text: str, base_chunk_size: int = 800, overlap: int = 150) -> List[Dict[str, Any]]:
        """Advanced semantic chunking that preserves document structure and meaning"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_sentences = []
        
        # Identify section breaks
        section_breaks = []
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower().strip()
            
            # Check for section headers
            if (len(sentence.strip()) < 100 and 
                any(marker in sentence_lower for section_type, markers in self.kb.section_markers.items() 
                    for marker in markers)):
                section_breaks.append(i)
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > base_chunk_size and current_chunk:
                # Create chunk with metadata
                chunk_info = {
                    'text': current_chunk.strip(),
                    'sentences': current_sentences.copy(),
                    'start_idx': i - len(current_sentences),
                    'end_idx': i - 1,
                    'section_type': self._identify_section_type(current_chunk)
                }
                chunks.append(chunk_info)
                
                # Handle overlap
                if len(current_sentences) > 3:
                    overlap_sentences = current_sentences[-2:]  # Keep last 2 sentences for overlap
                    current_chunk = ' '.join(overlap_sentences)
                    current_sentences = overlap_sentences.copy()
                else:
                    current_chunk = ""
                    current_sentences = []
            
            # Add current sentence
            current_chunk += (' ' if current_chunk else '') + sentence
            current_sentences.append(sentence)
            i += 1
        
        # Add final chunk
        if current_chunk.strip():
            chunk_info = {
                'text': current_chunk.strip(),
                'sentences': current_sentences,
                'start_idx': len(sentences) - len(current_sentences),
                'end_idx': len(sentences) - 1,
                'section_type': self._identify_section_type(current_chunk)
            }
            chunks.append(chunk_info)
        
        return [chunk for chunk in chunks if len(chunk['text'].strip()) > 100]

    def _identify_section_type(self, text: str) -> str:
        """Identify the type of section based on content"""
        text_lower = text.lower()
        
        for section_type, markers in self.kb.section_markers.items():
            if any(marker in text_lower for marker in markers):
                return section_type
        
        return 'general'

    def hybrid_retrieval(self, question: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[str]:
        """Hybrid retrieval combining multiple scoring methods"""
        if not chunks:
            return []
        
        try:
            # Get question classification
            question_type, patterns, confidence = self.kb.classify_question_advanced(question)
            
            chunk_scores = []
            chunk_texts = [chunk['text'] for chunk in chunks]
            
            # 1. TF-IDF Similarity
            try:
                all_texts = chunk_texts + [question]
                tfidf_matrix = self.vectorizer.fit_transform(all_texts)
                query_vector = tfidf_matrix[-1]
                chunk_vectors = tfidf_matrix[:-1]
                tfidf_similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
            except:
                tfidf_similarities = np.zeros(len(chunks))
            
            # 2. Pattern Matching Score
            for i, chunk in enumerate(chunks):
                text_lower = chunk['text'].lower()
                
                # Base scores
                tfidf_score = tfidf_similarities[i] if i < len(tfidf_similarities) else 0
                pattern_score = 0
                section_score = 0
                value_score = 0
                
                # Pattern matching
                for pattern in patterns:
                    pattern_count = text_lower.count(pattern)
                    pattern_score += pattern_count * len(pattern.split()) * 2
                
                # Section type bonus
                if chunk['section_type'] in ['definitions', 'coverage', 'exclusions']:
                    if question_type in ['coverage', 'benefits'] and chunk['section_type'] == 'coverage':
                        section_score = 3
                    elif question_type == 'exclusions' and chunk['section_type'] == 'exclusions':
                        section_score = 3
                    elif question_type in ['grace_period', 'waiting_period'] and chunk['section_type'] == 'definitions':
                        section_score = 2
                
                # Value extraction bonus
                if question_type in self.kb.value_patterns:
                    values = self.kb.extract_numerical_values(chunk['text'], question_type)
                    value_score = len(values) * 2
                
                # Question-specific boosting
                specific_score = 0
                if question_type == 'grace_period':
                    if any(term in text_lower for term in ['grace', '30', 'thirty', 'days', 'premium due']):
                        specific_score = 5
                elif question_type == 'waiting_period':
                    if any(term in text_lower for term in ['waiting', '36', '24', 'months', 'continuous']):
                        specific_score = 5
                elif question_type == 'maternity':
                    if 'maternity' in text_lower or 'pregnancy' in text_lower:
                        specific_score = 5
                elif question_type == 'cataract':
                    if 'cataract' in text_lower:
                        specific_score = 5
                elif question_type == 'room_rent':
                    if any(term in text_lower for term in ['room rent', '2%', '5%', 'icu']):
                        specific_score = 5
                
                # Combine scores with weights
                final_score = (
                    tfidf_score * 3 +           # TF-IDF similarity
                    pattern_score * 2 +         # Pattern matching
                    section_score * 2 +         # Section relevance
                    value_score * 1.5 +         # Numerical values
                    specific_score * 3          # Question-specific boost
                )
                
                chunk_scores.append((i, final_score, chunk['text']))
            
            # Sort by score and return top chunks
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Ensure we have high-quality chunks
            selected_chunks = []
            for i, score, text in chunk_scores[:top_k * 2]:  # Get more candidates
                if score > 1.0 or len(selected_chunks) < 3:  # Ensure minimum chunks
                    selected_chunks.append(text)
                if len(selected_chunks) >= top_k:
                    break
            
            return selected_chunks[:top_k]
            
        except Exception as e:
            print(f"Hybrid retrieval error: {e}")
            return [chunk['text'] for chunk in chunks[:top_k]]

    async def generate_enhanced_answer(self, question: str, context: str, question_type: str) -> str:
        """Generate answer with enhanced prompting based on question type"""
        
        # Question-specific instructions
        specific_instructions = {
            'grace_period': "Look for the exact number of days for grace period. It's typically 30 days. Quote the exact text mentioning grace period.",
            'waiting_period': "Find the specific waiting period in months (usually 24 or 36 months). Specify what the waiting period applies to.",
            'pre_existing': "Look for '36 months' or similar waiting period for pre-existing diseases. Mention declaration requirements.",
            'maternity': "Check if maternity is covered or excluded. Look for specific conditions, waiting periods, and coverage limits.",
            'cataract': "Find the specific coverage limit for cataract - look for percentages of Sum Insured or fixed amounts like Rs. 40,000.",
            'room_rent': "Look for percentage limits - typically 2% of Sum Insured for room rent, 5% for ICU, or fixed amounts.",
            'ayush': "Check coverage for AYUSH treatments (Ayurveda, Yoga, Naturopathy, Unani, Siddha, Homeopathy).",
            'cumulative_bonus': "Look for percentage increase per claim-free year (usually 5%) and maximum limit (usually 50%).",
            'hospital_definition': "Find the complete definition including bed requirements, registration criteria, and facilities required.",
            'co_payment': "Look for any co-payment clauses, percentage or fixed amounts the insured must pay.",
            'exclusions': "List the specific exclusions mentioned. Be comprehensive but concise."
        }
        
        instruction = specific_instructions.get(question_type, "Provide specific details with exact numbers, percentages, amounts, and conditions mentioned in the policy.")
        
        prompt = f"""You are analyzing an insurance policy document. Answer the question based ONLY on the provided policy text.

POLICY CONTENT:
{context}

QUESTION: {question}

SPECIFIC INSTRUCTIONS:
- {instruction}
- Quote exact figures, percentages, time periods, and amounts when available
- If information is not clearly stated, say "Not specified in the provided policy text"
- Be precise and include all relevant conditions or limitations
- Format numbers clearly (e.g., "30 days", "Rs. 40,000", "25% of Sum Insured")

ANSWER (be concise but complete):"""
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Very low temperature for factual accuracy
                    max_output_tokens=200,
                    candidate_count=1
                )
            )
            
            answer = response.text.strip()
            
            # Clean up response
            answer = re.sub(r'^(Answer:|A:)\s*', '', answer)
            answer = re.sub(r'^Based on.*?policy.*?,?\s*', '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'^According to.*?document.*?,?\s*', '', answer, flags=re.IGNORECASE)
            
            return answer
            
        except Exception as e:
            print(f"Generation error: {e}")
            return "Unable to determine from the provided policy document."

# --- Enhanced PDF Processing ---
def extract_text_with_structure(pdf_path: str) -> str:
    """Extract text while preserving document structure"""
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        
        for page_num, page in enumerate(doc):
            # Get text with layout preservation
            blocks = page.get_text("dict")
            page_text = ""
            
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                line_text += text + " "
                        if line_text.strip():
                            page_text += line_text.strip() + "\n"
                    page_text += "\n"  # Block separator
            
            if page_text.strip():
                text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
        
        doc.close()
        return "\n".join(text_parts)
        
    except Exception as e:
        print(f"PDF extraction error: {e}")
        # Fallback to simple extraction
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            return "\n".join(text_parts)
        except:
            raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

# Initialize enhanced RAG system
rag_system = AdvancedInsuranceRAG()

# --- Main Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_analysis(request: QueryRequest, token: str = Depends(verify_token)):
    try:
        # Enhanced PDF download with better error handling
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    request.documents, 
                    headers=headers, 
                    timeout=30,
                    stream=True
                )
                response.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")
                await asyncio.sleep(2 ** attempt)

        # Enhanced PDF processing
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file.flush()

            # Extract text with structure preservation
            raw_text = extract_text_with_structure(temp_file.name)
            
            if not raw_text or len(raw_text.strip()) < 100:
                raise HTTPException(status_code=400, detail="No readable text found in PDF")

            # Advanced text cleaning and chunking
            clean_text = rag_system.advanced_text_cleaning(raw_text)
            chunks = rag_system.semantic_chunking(clean_text)
            
            if not chunks:
                raise HTTPException(status_code=400, detail="No processable content found")

        # Process questions with enhanced retrieval
        answers = []
        for i, question in enumerate(request.questions):
            try:
                question_type, patterns, confidence = rag_system.kb.classify_question_advanced(question)
                
                # Use more chunks for better context
                relevant_chunks = rag_system.hybrid_retrieval(question, chunks, top_k=6)
                
                # Combine context more intelligently
                context = "\n\n" + "="*50 + "\n\n".join(relevant_chunks[:4])
                
                # Ensure sufficient context
                if len(context) < 500 and len(relevant_chunks) > 4:
                    context = "\n\n" + "="*50 + "\n\n".join(relevant_chunks[:6])
                
                answer = await rag_system.generate_enhanced_answer(question, context, question_type)
                answers.append(answer)
                
                print(f"Q{i+1} [{question_type}] (conf: {confidence:.2f}): {len(context)} chars, {len(relevant_chunks)} chunks")
                
            except Exception as e:
                print(f"Error processing question {i+1}: {e}")
                answers.append("Unable to process this question with the provided policy document.")

        return QueryResponse(answers=answers)

    except HTTPException:
        raise
    except Exception as e:
        print(f"System error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Enhanced Insurance Policy Analyzer API"}

@app.get("/")
async def root():
    return {"message": "HackRx 6.0 Enhanced Insurance Policy Analyzer - Optimized for Maximum Accuracy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)