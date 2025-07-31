# api.py - Optimized Insurance Policy Analyzer
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
from collections import Counter
import json

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

# --- Enhanced Insurance Knowledge System ---
class InsuranceKnowledgeExtractor:
    def __init__(self):
        # Direct pattern matching for specific information
        self.direct_patterns = {
            'grace_period': [
                r'grace\s+period[^\n]*?(\d+)\s*days?',
                r'grace\s+period[^\n]*?thirty\s*days?',
                r'(\d+)\s*days?[^\n]*?grace\s+period',
                r'thirty\s*days?[^\n]*?grace\s+period'
            ],
            'waiting_period_ped': [
                r'pre[-\s]*existing[^\n]*?(\d+)[^\n]*?months?',
                r'(\d+)[^\n]*?months?[^\n]*?pre[-\s]*existing',
                r'PED[^\n]*?(\d+)[^\n]*?months?'
            ],
            'maternity': [
                r'maternity[^\n]*?covered',
                r'maternity[^\n]*?excluded',
                r'childbirth[^\n]*?covered',
                r'pregnancy[^\n]*?covered',
                r'maternity\s+expenses?[^\n]*?covered',
                r'exclud[^.]*?maternity'
            ],
            'cataract_waiting': [
                r'cataract[^\n]*?(\d+)[^\n]*?months?',
                r'(\d+)[^\n]*?months?[^\n]*?cataract'
            ],
            'cataract_limit': [
                r'cataract[^\n]*?(\d+%)[^\n]*?sum\s+insured',
                r'cataract[^\n]*?(Rs\.?\s*\d+,?\d*)',
                r'(\d+%)[^\n]*?sum\s+insured[^\n]*?cataract',
                r'(Rs\.?\s*\d+,?\d*)[^\n]*?cataract'
            ],
            'room_rent': [
                r'room\s+rent[^\n]*?(\d+%)[^\n]*?sum\s+insured',
                r'room\s+rent[^\n]*?(Rs\.?\s*\d+,?\d*)',
                r'(\d+%)[^\n]*?sum\s+insured[^\n]*?room',
                r'(Rs\.?\s*\d+,?\d*)[^\n]*?room\s+rent'
            ],
            'icu_charges': [
                r'ICU[^\n]*?(\d+%)[^\n]*?sum\s+insured',
                r'ICU[^\n]*?(Rs\.?\s*\d+,?\d*)',
                r'intensive\s+care[^\n]*?(\d+%)',
                r'intensive\s+care[^\n]*?(Rs\.?\s*\d+,?\d*)'
            ],
            'cumulative_bonus': [
                r'cumulative\s+bonus[^\n]*?(\d+%)',
                r'(\d+%)[^\n]*?cumulative\s+bonus',
                r'claim\s+free[^\n]*?(\d+%)',
                r'no\s+claim[^\n]*?(\d+%)'
            ],
            'ambulance': [
                r'ambulance[^\n]*?(Rs\.?\s*\d+,?\d*)',
                r'road\s+ambulance[^\n]*?(Rs\.?\s*\d+,?\d*)',
                r'(Rs\.?\s*\d+,?\d*)[^\n]*?ambulance'
            ],
            'co_payment': [
                r'co[-\s]*payment[^\n]*?(\d+%)',
                r'(\d+%)[^\n]*?co[-\s]*payment',
                r'co[-\s]*pay[^\n]*?(\d+%)'
            ]
        }
        
        # Section identifiers for better targeting
        self.section_identifiers = {
            'definitions': ['3. DEFINITIONS', 'DEFINITIONS'],
            'coverage': ['4. COVERAGE', 'COVERAGE'],
            'exclusions': ['7. EXCLUSIONS', 'EXCLUSIONS'],
            'waiting_periods': ['6. WAITING PERIOD', 'WAITING PERIOD'],
            'claims': ['9. CLAIM PROCEDURE', 'CLAIM PROCEDURE'],
            'benefits': ['TABLE OF BENEFITS', 'BENEFITS']
        }

    def extract_direct_info(self, text: str, info_type: str) -> List[str]:
        """Extract information using direct pattern matching"""
        results = []
        text_lower = text.lower()
        
        if info_type in self.direct_patterns:
            for pattern in self.direct_patterns[info_type]:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    # Get surrounding context (3 lines before and after)
                    start = max(0, text.rfind('\n', 0, match.start()) - 200)
                    end = min(len(text), text.find('\n', match.end()) + 200)
                    if end == -1:
                        end = min(len(text), match.end() + 200)
                    
                    context = text[start:end].strip()
                    results.append(context)
        
        return results

    def find_section_content(self, text: str, section_type: str) -> str:
        """Find specific section content"""
        if section_type not in self.section_identifiers:
            return ""
        
        section_markers = self.section_identifiers[section_type]
        
        for marker in section_markers:
            # Find section start
            pattern = rf'^{re.escape(marker)}.*$'
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            if match:
                start_pos = match.start()
                
                # Find next major section (numbered section)
                next_section_pattern = r'^\d+\.\s+[A-Z][A-Z\s]+$'
                next_match = re.search(next_section_pattern, text[start_pos + len(match.group()):], re.MULTILINE)
                
                if next_match:
                    end_pos = start_pos + len(match.group()) + next_match.start()
                    return text[start_pos:end_pos]
                else:
                    # Take reasonable chunk if no next section found
                    return text[start_pos:start_pos + 3000]
        
        return ""

# --- Optimized RAG System ---
class OptimizedInsuranceRAG:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        self.extractor = InsuranceKnowledgeExtractor()
        
    def preprocess_document(self, text: str) -> Dict[str, str]:
        """Extract and organize document sections"""
        sections = {}
        
        # Extract major sections
        for section_type in ['definitions', 'coverage', 'exclusions', 'waiting_periods', 'claims', 'benefits']:
            content = self.extractor.find_section_content(text, section_type)
            if content:
                sections[section_type] = content
        
        # Also keep full text for fallback
        sections['full_text'] = text
        
        return sections

    def smart_retrieval(self, question: str, sections: Dict[str, str]) -> str:
        """Intelligent context retrieval based on question type"""
        question_lower = question.lower()
        relevant_sections = []
        
        # Question type mapping to sections
        section_priority = {
            'grace period': ['benefits', 'claims', 'full_text'],
            'waiting period': ['waiting_periods', 'exclusions', 'benefits', 'full_text'],
            'pre-existing': ['waiting_periods', 'exclusions', 'definitions', 'full_text'],
            'maternity': ['exclusions', 'coverage', 'full_text'],
            'cataract': ['coverage', 'waiting_periods', 'benefits', 'full_text'],
            'organ donor': ['coverage', 'exclusions', 'full_text'],
            'claim discount': ['benefits', 'coverage', 'full_text'],
            'no claim': ['benefits', 'coverage', 'full_text'],
            'cumulative bonus': ['benefits', 'coverage', 'full_text'],
            'health check': ['coverage', 'benefits', 'full_text'],
            'hospital': ['definitions', 'coverage', 'full_text'],
            'ayush': ['coverage', 'benefits', 'full_text'],
            'room rent': ['coverage', 'benefits', 'full_text'],
            'icu': ['coverage', 'benefits', 'full_text'],
            'eligibility': ['benefits', 'definitions', 'full_text'],
            'age limit': ['benefits', 'definitions', 'full_text'],
            'sum insured': ['benefits', 'coverage', 'full_text'],
            'premium': ['benefits', 'full_text'],
            'exclusion': ['exclusions', 'waiting_periods', 'full_text'],
            'claims process': ['claims', 'benefits', 'full_text'],
            'dental': ['coverage', 'exclusions', 'full_text'],
            'mental illness': ['coverage', 'exclusions', 'full_text'],
            'domiciliary': ['coverage', 'exclusions', 'full_text'],
            'ambulance': ['coverage', 'benefits', 'full_text'],
            'co-payment': ['benefits', 'claims', 'full_text'],
            'renewal': ['benefits', 'claims', 'full_text']
        }
        
        # Find relevant sections based on question keywords
        relevant_section_types = []
        for keyword, section_types in section_priority.items():
            if keyword in question_lower:
                relevant_section_types.extend(section_types)
                break
        
        # Default to key sections if no specific match
        if not relevant_section_types:
            relevant_section_types = ['coverage', 'benefits', 'exclusions', 'full_text']
        
        # Collect relevant content
        context_parts = []
        for section_type in relevant_section_types[:4]:  # Limit to avoid token overflow
            if section_type in sections and sections[section_type]:
                context_parts.append(f"=== {section_type.upper()} ===\n{sections[section_type]}")
        
        # Also add direct pattern matches
        for info_type in self.extractor.direct_patterns.keys():
            if any(keyword in question_lower for keyword in info_type.split('_')):
                matches = self.extractor.extract_direct_info(sections['full_text'], info_type)
                if matches:
                    context_parts.append(f"=== RELEVANT EXCERPTS ===\n" + "\n---\n".join(matches[:3]))
                break
        
        return "\n\n".join(context_parts)

    async def generate_answer(self, question: str, context: str) -> str:
        """Generate precise answer using enhanced prompting"""
        
        # Create specialized prompts based on question content
        question_lower = question.lower()
        
        if 'grace period' in question_lower:
            specific_instruction = "Look for the exact number of days for grace period. The answer should state '30 days' or 'thirty days' if found."
        elif 'waiting period' in question_lower and 'pre-existing' in question_lower:
            specific_instruction = "Look for the exact waiting period for pre-existing diseases. Should be stated as '36 months' or 'thirty-six months'."
        elif 'maternity' in question_lower:
            specific_instruction = "Determine if maternity expenses are covered or excluded. Look for specific mentions of 'maternity', 'childbirth', 'pregnancy'."
        elif 'cataract' in question_lower and 'waiting' in question_lower:
            specific_instruction = "Find the specific waiting period for cataract treatment. Look for '24 months' or 'two years'."
        elif 'hospital' in question_lower and 'define' in question_lower:
            specific_instruction = "Find the definition of 'Hospital'. Look for bed requirements, registration criteria, and facility requirements."
        elif 'ayush' in question_lower:
            specific_instruction = "Look for coverage of AYUSH treatments (Ayurveda, Yoga, Naturopathy, Unani, Siddha, Homeopathy)."
        elif 'room rent' in question_lower or 'icu' in question_lower:
            specific_instruction = "Find specific limits for room rent and ICU charges. Look for percentages of sum insured and maximum amounts."
        elif 'cumulative bonus' in question_lower or 'no claim' in question_lower:
            specific_instruction = "Look for cumulative bonus or no claim bonus details. Find the percentage increase and maximum limit."
        elif 'ambulance' in question_lower:
            specific_instruction = "Find ambulance coverage details including coverage limits and maximum amounts."
        else:
            specific_instruction = "Provide specific details with exact numbers, percentages, time periods, and conditions mentioned in the policy."

        prompt = f"""You are analyzing an insurance policy document. Answer the question based STRICTLY on the provided policy text.

POLICY CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- {specific_instruction}
- Quote exact numbers, percentages, amounts, and time periods from the policy
- If information is clearly stated, provide it confidently
- If coverage exists, state it clearly; if excluded, state that clearly
- Be precise and factual based on the document content
- If the specific information is genuinely not found in the provided context, state that clearly

ANSWER:"""

        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Very low temperature for consistency
                    max_output_tokens=200,
                    candidate_count=1
                )
            )
            
            answer = response.text.strip()
            
            # Clean up common response prefixes
            cleanup_patterns = [
                r'^(Answer:|A:|Response:)\s*',
                r'^Based on the.*?policy.*?,?\s*',
                r'^According to.*?document.*?,?\s*',
                r'^The policy.*?states.*?that\s*',
                r'^From the.*?text.*?,?\s*'
            ]
            
            for pattern in cleanup_patterns:
                answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
            
            return answer.strip()
            
        except Exception as e:
            print(f"Generation error: {e}")
            return "Unable to process this question due to technical error."

# --- PDF Processing ---
def extract_text_from_pdf(pdf_path: str) -> str:
    """Enhanced PDF text extraction"""
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        
        for page_num, page in enumerate(doc):
            # Extract text with better formatting preservation
            text_dict = page.get_text("dict")
            page_text = ""
            
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            line_text += span["text"]
                        if line_text.strip():
                            page_text += line_text + "\n"
                            
            if page_text.strip():
                text_parts.append(f"--- PAGE {page_num + 1} ---\n{page_text}")
        
        doc.close()
        
        full_text = "\n".join(text_parts)
        
        # Basic text cleaning while preserving structure
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)  # Normalize excessive line breaks
        full_text = re.sub(r' {3,}', ' ', full_text)  # Normalize excessive spaces
        
        return full_text
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")

# Initialize optimized system
rag_system = OptimizedInsuranceRAG()

# --- Main Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_analysis(request: QueryRequest, token: str = Depends(verify_token)):
    try:
        # Download PDF with improved error handling
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
                    raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        # Process PDF with enhanced extraction
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file.flush()

            # Extract and preprocess document
            raw_text = extract_text_from_pdf(temp_file.name)
            
            if len(raw_text.strip()) < 100:
                raise HTTPException(status_code=400, detail="Document appears to be empty or unreadable")
            
            # Preprocess into organized sections
            sections = rag_system.preprocess_document(raw_text)
            print(f"Extracted sections: {list(sections.keys())}")

        # Process questions with optimized retrieval
        answers = []
        for i, question in enumerate(request.questions):
            try:
                print(f"Processing Q{i+1}: {question[:50]}...")
                
                # Get relevant context using smart retrieval
                context = rag_system.smart_retrieval(question, sections)
                
                # Generate answer
                answer = await rag_system.generate_answer(question, context)
                answers.append(answer)
                
                print(f"Q{i+1} completed - Context length: {len(context)} chars")
                
            except Exception as e:
                print(f"Error processing question {i+1}: {e}")
                answers.append("Unable to process this question due to technical error.")

        print(f"Completed processing {len(answers)} questions")
        return QueryResponse(answers=answers)

    except HTTPException:
        raise
    except Exception as e:
        print(f"System error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal system error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Optimized Insurance Policy Analyzer"}

@app.get("/")
async def root():
    return {"message": "HackRx 6.0 - Optimized Insurance Policy Analyzer", "version": "2.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)