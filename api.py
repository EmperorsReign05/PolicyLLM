# api.py
import os
import fitz  # PyMuPDF
import requests
import tempfile
import asyncio
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# LangChain RAG components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- Init ---
load_dotenv()
AUTH_TOKEN = "78b25ddaad17f4e8d85cde3dca81ade8319272062cf10b73ba148b425151f2fd"
auth_scheme = HTTPBearer()
app = FastAPI(title="HackRx 6.0 Policy Analyzer")

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

# --- AI Model Setup ---
print("Loading AI models...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
prompt = PromptTemplate.from_template(
    "Based on the following policy context, provide a direct and concise answer to the question.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
)
print("AI models loaded.")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- PDF Reader ---
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return [page.get_text() for page in doc]

# --- Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_analysis(request: QueryRequest, token: str = Depends(verify_token)):
    try:
        # Download PDF
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(request.documents, headers=headers)
        response.raise_for_status()

        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

            # Load & split
            texts = extract_text_from_pdf(temp_file_path)
            all_text = "\n".join(texts)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = splitter.create_documents([all_text])

            # Vector store
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever()

        # RAG Chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        tasks = [rag_chain.ainvoke(q) for q in request.questions]
        answers = await asyncio.gather(*tasks)
        return {"answers": [a.strip() for a in answers]}

    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))
