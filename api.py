
# api.py
import os
import fitz  # PyMuPDF
import requests
import tempfile
import asyncio
import re
import hashlib
import json
from typing import List, Dict, Any
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
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
        self.model = genai.GenerativeModel("gemini-1.5-flash-latest")
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s\.\,\!\?\;\:\-\(\)]", "", text)
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    def save_embedding_cache(self, key: str, data: List[Dict[str, Any]], dir="embedding_cache"):
        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, f"{key}.json")
        with open(path, "w") as f:
            json.dump(data, f)

    def load_embedding_cache(self, key: str, dir="embedding_cache") -> List[Dict[str, Any]]:
        path = os.path.join(dir, f"{key}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return []

    def retrieve_relevant_chunks(self, query: str, chunks: List[str], top_k: int = 3) -> List[str]:
        cache_key = self._hash_text("".join(chunks))
        cached = self.load_embedding_cache(cache_key)

        if cached:
            embedded_chunks = [np.array(item["embedding"]) for item in cached]
            texts = [item["text"] for item in cached]
        else:
            embedded_chunks = []
            texts = []
            for chunk in chunks:
                try:
                    emb = self.model.embed_content(content=chunk, task_type="retrieval_document")["embedding"]
                    embedded_chunks.append(np.array(emb))
                    texts.append(chunk)
                except Exception as e:
                    print(f"Embedding failed: {e}")
            self.save_embedding_cache(cache_key, [{"text": t, "embedding": e.tolist()} for t, e in zip(texts, embedded_chunks)])

        try:
            query_emb = np.array(self.model.embed_content(content=query, task_type="retrieval_query")["embedding"])
        except Exception as e:
            print(f"Query embedding failed: {e}")
            return chunks[:top_k]

        sims = cosine_similarity([query_emb], embedded_chunks).flatten()
        top_indices = np.argsort(sims)[-top_k:][::-1]
        return [texts[i] for i in top_indices]

    async def generate_answer(self, question: str, context: str) -> str:
        prompt = f"""You are a highly knowledgeable and concise insurance policy assistant.

Given the following policy excerpt and user question, answer with clear, factual, and context-specific information ONLY from the policy text.

If the policy explicitly answers the question, include:
- Direct yes/no at the start (if applicable)
- Specific figures, time periods, and conditions as stated
- Definitions or clauses from the text (in summarized form)

If the policy does NOT provide a clear answer, respond with:
"Information not available in the provided policy document."

---

Policy Excerpt:
\"\"\"
{context}
\"\"\"

Question:
{question}

Answer in one or two sentences, in formal, bulletproof language. Avoid hedging or speculation.

Answer:"""
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
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(request.documents, headers=headers, timeout=30)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_file.flush()

            raw_text = extract_text_from_pdf(temp_file.name)
            clean_text = rag_system.clean_text(raw_text)
            chunks = rag_system.chunk_text(clean_text)
            if not chunks:
                raise HTTPException(status_code=400, detail="No text could be extracted from PDF")

        async def process_question(question: str) -> str:
            relevant_chunks = rag_system.retrieve_relevant_chunks(question, chunks)
            context = "\n\n".join(relevant_chunks)
            return await rag_system.generate_answer(question, context)

        tasks = [process_question(q) for q in request.questions]
        answers = await asyncio.gather(*tasks, return_exceptions=True)

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

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Policy Analyzer API is running"}

@app.get("/")
async def root():
    return {"message": "HackRx 6.0 Policy Analyzer - Optimized Version"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
