# optimized_api.py - High-Performance Insurance Document Analysis with Sentence Transformers + FAISS
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
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import pickle
import hashlib
from collections import defaultdict

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
app = FastAPI(title="HackRx 6.0 Policy Analyzer - Advanced Semantic Search")

# Configure Google AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Global model cache
_embedding_model = None
def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("Loading sentence transformer model...")
        # Using a lightweight but effective model for insurance documents
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully!")
    return _embedding_model

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

# --- Advanced Insurance Knowledge Base ---
class AdvancedInsuranceKB:
    def __init__(self):
        # Enhanced insurance term mappings with semantic variations
        self.question_patterns = {
            'grace_period': [
                'grace period', 'grace time', 'payment period', 'premium due', 
                'pay premium', 'late payment', 'policy lapse', 'payment deadline',
                'premium payment time', 'due date extension'
            ],
            'waiting_period': [
                'waiting period', 'wait time', 'exclusion period', 'initial waiting',
                'cooling period', 'probation period', 'coverage begins', 'when covered',
                'months before coverage', 'initial exclusion'
            ],
            'pre_existing': [
                'pre-existing', 'pre existing', 'PED', 'prior condition', 
                'existing disease', 'medical history', 'declare disease',
                'previous illness', 'ongoing condition', 'chronic disease'
            ],
            'coverage': [
                'coverage', 'covered', 'cover', 'benefits', 'include', 
                'indemnify', 'reimburse', 'eligible', 'entitled', 'what is covered',
                'scope of cover', 'medical expenses', 'treatment covered'
            ],
            'exclusions': [
                'exclusion', 'exclude', 'not covered', 'limitation', 
                'not eligible', 'shall not', 'except', 'excluded conditions',
                'what is not covered', 'restrictions', 'not reimbursed'
            ],
            'maternity': [
                'maternity', 'pregnancy', 'childbirth', 'delivery', 
                'maternal', 'newborn', 'confinement', 'obstetric care',
                'prenatal', 'postnatal', 'labor', 'caesarean'
            ],
            'room_rent': [
                'room rent', 'accommodation', 'ICU', 'hospital charges', 
                'room category', 'bed charges', 'hospital room', 'daily room rent',
                'room limit', 'accommodation charges', 'private room'
            ],
            'cataract': [
                'cataract', 'eye surgery', 'vision', 'lens replacement', 
                'eye treatment', 'ophthalmology', 'intraocular lens',
                'eye operation', 'vision correction'
            ],
            'cumulative_bonus': [
                'cumulative bonus', 'no claim bonus', 'NCD', 'bonus', 
                'claim free', 'renewal bonus', 'loyalty bonus', 'NCB',
                'discount for no claims', 'bonus accumulation'
            ]
        }
        
        # Question type boost keywords for semantic search
        self.semantic_boosts = {
            'grace_period': [
                'thirty days premium payment', 'grace period policy renewal',
                'payment due date extension', 'premium payment deadline'
            ],
            'waiting_period': [
                'thirty six months waiting', 'continuous coverage period',
                'initial waiting exclusion', 'months before eligible'
            ],
            'maternity': [
                'maternity benefit coverage', 'pregnancy related expenses',
                'childbirth hospital expenses', 'delivery medical costs'
            ],
            'cataract': [
                'cataract surgery coverage limit', 'eye surgery maximum amount',
                'per eye treatment cost', 'ophthalmology procedure limit'
            ],
            'room_rent': [
                'daily room rent limit', 'accommodation charge restriction',
                'ICU charge percentage', 'private room cost limit'
            ]
        }

    def classify_question_semantic(self, question: str) -> Tuple[str, List[str], float]:
        """Enhanced question classification using semantic similarity"""
        question_lower = question.lower()
        scores = {}
        
        for category, patterns in self.question_patterns.items():
            score = 0
            matched_patterns = []
            
            for pattern in patterns:
                # Exact matching
                if pattern in question_lower:
                    weight = len(pattern.split()) * 3  # Higher weight for exact matches
                    score += weight
                    matched_patterns.append(pattern)
                
                # Fuzzy matching for typos and variations
                pattern_words = set(pattern.split())
                question_words = set(question_lower.split())
                overlap = len(pattern_words.intersection(question_words))
                if overlap > 0:
                    score += overlap * 2
            
            if score > 0:
                scores[category] = (score, matched_patterns)
        
        if scores:
            best_category = max(scores.keys(), key=lambda k: scores[k][0])
            confidence = scores[best_category][0] / sum(s[0] for s in scores.values())
            return best_category, scores[best_category][1], confidence
        
        return 'general', [], 0.0

# --- Advanced Semantic RAG System ---
class SemanticInsuranceRAG:
    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        self.kb = AdvancedInsuranceKB()
        self.stop_words = set(stopwords.words('english'))
        self.chunk_cache = {}
        self.embedding_cache = {}
        
    def advanced_text_cleaning(self, text: str) -> str:
        """Advanced text cleaning optimized for insurance documents"""
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
            r'\bIPD\b': 'In Patient Department',
            r'\bNCD\b': 'No Claim Discount'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text.strip()

    def intelligent_chunking(self, text: str, base_chunk_size: int = 600, overlap: int = 100) -> List[Dict[str, Any]]:
        """Intelligent chunking optimized for semantic search"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_sentences = []
        
        # Identify section breaks and important markers
        section_markers = [
            r'\b(?:definitions?|defined terms?)\b',
            r'\b(?:coverage|benefits?|scope)\b',
            r'\b(?:exclusions?|limitations?)\b',
            r'\b(?:waiting period|initial waiting)\b',
            r'\b(?:claims?|reimbursement)\b',
            r'\b(?:conditions?|terms?)\b'
        ]
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i].strip()
            
            # Check for section headers (give them priority)
            is_section_header = (
                len(sentence) < 80 and
                any(re.search(marker, sentence, re.IGNORECASE) for marker in section_markers)
            )
            
            # Decide whether to start new chunk
            should_break = (
                len(current_chunk) + len(sentence) > base_chunk_size and 
                current_chunk and 
                not is_section_header
            )
            
            if should_break:
                # Create chunk with metadata
                chunk_info = {
                    'text': current_chunk.strip(),
                    'sentences': current_sentences.copy(),
                    'start_idx': i - len(current_sentences),
                    'end_idx': i - 1,
                    'section_type': self._identify_section_type(current_chunk),
                    'importance_score': self._calculate_importance(current_chunk)
                }
                chunks.append(chunk_info)
                
                # Handle overlap intelligently
                if len(current_sentences) > 2:
                    # Keep sentences that contain important keywords
                    overlap_sentences = []
                    for sent in current_sentences[-3:]:  # Look at last 3 sentences
                        if any(keyword in sent.lower() for keyword in 
                               ['shall', 'means', 'includes', 'excludes', 'coverage', 'waiting']):
                            overlap_sentences.append(sent)
                    
                    if not overlap_sentences and current_sentences:
                        overlap_sentences = [current_sentences[-1]]  # At least keep last sentence
                    
                    current_chunk = ' '.join(overlap_sentences)
                    current_sentences = overlap_sentences.copy()
                else:
                    current_chunk = ""
                    current_sentences = []
            
            # Add current sentence
            if current_chunk:
                current_chunk += ' ' + sentence
            else:
                current_chunk = sentence
            current_sentences.append(sentence)
            i += 1
        
        # Add final chunk
        if current_chunk.strip():
            chunk_info = {
                'text': current_chunk.strip(),
                'sentences': current_sentences,
                'start_idx': len(sentences) - len(current_sentences),
                'end_idx': len(sentences) - 1,
                'section_type': self._identify_section_type(current_chunk),
                'importance_score': self._calculate_importance(current_chunk)
            }
            chunks.append(chunk_info)
        
        return [chunk for chunk in chunks if len(chunk['text'].strip()) > 100]

    def _identify_section_type(self, text: str) -> str:
        """Identify section type with improved accuracy"""
        text_lower = text.lower()
        
        section_patterns = {
            'definitions': [r'\bmeans?\b', r'\bdefined? as\b', r'\brefers? to\b', r'\binterpretation\b'],
            'coverage': [r'\bcoverage\b', r'\bbenefits?\b', r'\bindemnif', r'\breimburse', r'\bshall cover\b'],
            'exclusions': [r'\bexclus', r'\bshall not\b', r'\bnot covered\b', r'\blimitation\b'],
            'waiting_periods': [r'\bwaiting period\b', r'\binitial waiting\b', r'\bcontinuous coverage\b'],
            'claims': [r'\bclaim', r'\breimbursement\b', r'\bcashless\b', r'\bsettlement\b'],
            'conditions': [r'\bconditions?\b', r'\bterms?\b', r'\bprovisions?\b', r'\brenewal\b']
        }
        
        for section_type, patterns in section_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return section_type
        
        return 'general'

    def _calculate_importance(self, text: str) -> float:
        """Calculate importance score for chunk prioritization"""
        text_lower = text.lower()
        score = 0
        
        # Important keywords boost
        important_keywords = [
            'grace period', 'waiting period', 'coverage', 'exclusion', 'maternity',
            'cataract', 'room rent', 'sum insured', 'premium', 'claim', 'hospital',
            'shall', 'means', 'includes', 'per cent', '%', 'days', 'months', 'years'
        ]
        
        for keyword in important_keywords:
            score += text_lower.count(keyword) * 2
        
        # Numerical content boost
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        score += len(numbers) * 1.5
        
        # Length penalty for very short or very long chunks
        length_ratio = len(text) / 800  # Optimal chunk size
        if 0.3 <= length_ratio <= 1.5:
            score += 2
        
        return score

    def create_faiss_index(self, chunks: List[Dict[str, Any]]) -> Tuple[faiss.Index, np.ndarray]:
        """Create FAISS index for fast semantic similarity search"""
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        print(f"FAISS index created with {index.ntotal} vectors")
        return index, embeddings

    def semantic_retrieval(self, question: str, chunks: List[Dict[str, Any]], top_k: int = 6) -> List[str]:
        """Advanced semantic retrieval using FAISS + question-specific boosting"""
        if not chunks:
            return []
        
        try:
            # Create FAISS index
            faiss_index, chunk_embeddings = self.create_faiss_index(chunks)
            
            # Get question classification for boosting
            question_type, patterns, confidence = self.kb.classify_question_semantic(question)
            
            # Create enhanced query with semantic boosting
            enhanced_query = question
            if question_type in self.kb.semantic_boosts:
                boost_phrases = self.kb.semantic_boosts[question_type]
                enhanced_query = f"{question} {' '.join(boost_phrases[:2])}"  # Add top 2 boost phrases
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([enhanced_query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search with FAISS
            similarities, indices = faiss_index.search(query_embedding, min(top_k * 2, len(chunks)))
            
            # Re-rank results with additional factors
            scored_chunks = []
            for i, (sim_score, chunk_idx) in enumerate(zip(similarities[0], indices[0])):
                if chunk_idx >= len(chunks):  # Safety check
                    continue
                    
                chunk = chunks[chunk_idx]
                text_lower = chunk['text'].lower()
                
                # Base semantic similarity score
                final_score = float(sim_score) * 10  # Scale up
                
                # Question-specific pattern boost
                for pattern in patterns:
                    if pattern in text_lower:
                        final_score += 3 * len(pattern.split())
                
                # Section type boost
                if question_type == 'coverage' and chunk['section_type'] == 'coverage':
                    final_score += 5
                elif question_type == 'exclusions' and chunk['section_type'] == 'exclusions':
                    final_score += 5
                elif question_type in ['grace_period', 'waiting_period'] and chunk['section_type'] == 'definitions':
                    final_score += 4
                elif chunk['section_type'] == question_type:
                    final_score += 3
                
                # Importance score boost
                final_score += chunk['importance_score'] * 0.5
                
                # Specific keyword boosting for insurance terms
                keyword_boosts = {
                    'grace_period': ['thirty days', '30 days', 'premium due', 'renew', 'continue'],
                    'waiting_period': ['36 months', '24 months', 'continuous coverage', 'inception'],
                    'maternity': ['female insured', 'lawful child', 'delivery', 'pregnancy'],
                    'cataract': ['per eye', '25%', 'Rs. 40,000', 'sum insured'],
                    'room_rent': ['2%', '5%', 'daily room', 'ICU charges']
                }
                
                if question_type in keyword_boosts:
                    for keyword in keyword_boosts[question_type]:
                        if keyword in text_lower:
                            final_score += 4
                
                scored_chunks.append((chunk_idx, final_score, chunk['text']))
            
            # Sort by final score and return top chunks
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            
            # Select diverse, high-quality chunks
            selected_chunks = []
            seen_content = set()
            
            for chunk_idx, score, text in scored_chunks:
                # Avoid very similar content
                text_signature = ' '.join(text.lower().split()[:15])  # First 15 words
                if text_signature not in seen_content or len(selected_chunks) < 3:
                    selected_chunks.append(text)
                    seen_content.add(text_signature)
                    
                    if len(selected_chunks) >= top_k:
                        break
            
            return selected_chunks[:top_k]
            
        except Exception as e:
            print(f"Semantic retrieval error: {e}")
            # Fallback to simple selection
            return [chunk['text'] for chunk in chunks[:top_k]]

    async def generate_optimized_answer(self, question: str, context: str, question_type: str) -> str:
        """Generate answer optimized for insurance policy accuracy"""
        
        # Enhanced prompt engineering based on question type
        prompt = f"""You are an expert insurance policy analyst with deep knowledge of Indian insurance policies. Answer the question based EXCLUSIVELY on the provided policy document text.

POLICY DOCUMENT CONTENT:
{context}

QUESTION: {question}

CRITICAL INSTRUCTIONS:
1. Answer ONLY based on the policy text provided above
2. Use EXACT phrases and numbers from the policy document
3. Include specific time periods (e.g., "thirty-six (36) months", "thirty days")
4. Include exact monetary amounts and percentages as stated
5. For waiting periods: State the exact duration and what it applies to
6. For coverage limits: Include both percentage and fixed amounts if mentioned
7. For exclusions: Be specific about conditions and circumstances
8. Use formal insurance terminology as found in the document
9. If multiple conditions apply, list them clearly
10. If the exact information is not in the provided text, state "Not specified in the provided policy document"

RESPONSE STYLE:
- Start directly with the key information (no preambles)
- Use the same language style as the policy document
- Include relevant conditions and exceptions
- Be precise with numbers, dates, and percentages
- Maintain professional insurance document tone

ANSWER:"""
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.05,  # Very low for factual accuracy
                    max_output_tokens=300,  # More space for detailed responses
                    candidate_count=1
                )
            )
            
            answer = response.text.strip()
            
            # Clean up response while preserving formatting
            answer = re.sub(r'^(Answer:|A:|Response:)\s*', '', answer)
            answer = re.sub(r'^Based on.*?policy.*?,?\s*', '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'^According to.*?document.*?,?\s*', '', answer, flags=re.IGNORECASE)
            
            return answer
            
        except Exception as e:
            print(f"Generation error: {e}")
            return "Unable to determine from the provided policy document."

# --- Enhanced PDF Processing ---
def extract_text_with_structure(pdf_path: str) -> str:
    """Extract text while preserving document structure and layout"""
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        
        for page_num, page in enumerate(doc):
            # Extract text blocks to preserve structure
            blocks = page.get_text("dict")
            page_text = ""
            
            for block in blocks["blocks"]:
                if "lines" in block:
                    block_text = ""
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                # Preserve formatting for important elements
                                font_size = span.get("size", 12)
                                if font_size > 14:  # Likely headers
                                    text = f"\n{text}\n"
                                line_text += text + " "
                        if line_text.strip():
                            block_text += line_text.strip() + "\n"
                    if block_text.strip():
                        page_text += block_text + "\n"
            
            if page_text.strip():
                text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
        
        doc.close()
        full_text = "\n".join(text_parts)
        
        # Additional structure preservation
        full_text = re.sub(r'\n([A-Z][A-Z\s]+)\n', r'\n\n\1\n\n', full_text)  # Emphasize headers
        return full_text
        
    except Exception as e:
        print(f"PDF extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

# Initialize optimized RAG system
rag_system = SemanticInsuranceRAG()

# --- Main Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_analysis(request: QueryRequest, token: str = Depends(verify_token)):
    try:
        # Enhanced PDF download
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
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

        # Process PDF with enhanced extraction
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file.flush()

            # Extract text with structure preservation
            raw_text = extract_text_with_structure(temp_file.name)
            
            if not raw_text or len(raw_text.strip()) < 100:
                raise HTTPException(status_code=400, detail="No readable text found in PDF")

            # Advanced processing
            clean_text = rag_system.advanced_text_cleaning(raw_text)
            chunks = rag_system.intelligent_chunking(clean_text, base_chunk_size=650, overlap=120)
            
            if not chunks:
                raise HTTPException(status_code=400, detail="No processable content found")

        print(f"Created {len(chunks)} chunks for processing")

        # Process questions with semantic search
        answers = []
        for i, question in enumerate(request.questions):
            try:
                question_type, patterns, confidence = rag_system.kb.classify_question_semantic(question)
                
                # Use semantic retrieval for better accuracy
                relevant_chunks = rag_system.semantic_retrieval(question, chunks, top_k=5)
                
                # Combine context optimally
                context_parts = []
                for idx, chunk in enumerate(relevant_chunks):
                    context_parts.append(f"SECTION {idx + 1}:\n{chunk}")
                
                context = "\n\n" + ("="*60 + "\n\n").join(context_parts)
                
                # Generate answer
                answer = await rag_system.generate_optimized_answer(question, context, question_type)
                answers.append(answer)
                
                print(f"Q{i+1} [{question_type}] (conf: {confidence:.2f}): {len(context)} chars, {len(relevant_chunks)} chunks")
                
            except Exception as e:
                print(f"Error processing question {i+1}: {e}")
                answers.append("Unable to determine from the provided policy document.")

        return QueryResponse(answers=answers)

    except HTTPException:
        raise
    except Exception as e:
        print(f"System error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Optimized Insurance Policy Analyzer with Semantic Search"}

@app.get("/")
async def root():
    return {"message": "HackRx 6.0 Semantic Insurance Policy Analyzer - Maximum Accuracy Mode"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)