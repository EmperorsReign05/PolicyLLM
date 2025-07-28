# api.py
import os
import requests
import tempfile
import asyncio
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# LangChain components for a robust RAG pipeline
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- Load Environment Variables & Auth ---
load_dotenv()
AUTH_TOKEN = "78b25ddaad17f4e8d85cde3dca81ade8319272062cf10b73ba148b425151f2fd"
auth_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != AUTH_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# --- AI Model and Prompt Setup (Loaded once at startup) ---
print("Loading AI models...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"))
print("AI models loaded.")

prompt = PromptTemplate.from_template(
    "Based on the following policy context, provide a direct and concise answer to the question. CONTEXT: {context} QUESTION: {question} ANSWER:"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- FastAPI App ---
app = FastAPI(title="HackRx 6.0 High-Performance Policy Analyzer")

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_analysis(request: QueryRequest, token: str = Depends(verify_token)):
    try:
        # 1. Download and process the document to create a retriever
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(request.documents, headers=headers)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            loader = PyMuPDFLoader(temp_file.name)
            docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150))
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever()

        # 2. Define the RAG chain for a single question
        rag_chain_for_one_question = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # 3. Process all questions IN PARALLEL for speed
        tasks = [rag_chain_for_one_question.ainvoke(q) for q in request.questions]
        answers = await asyncio.gather(*tasks)
        
        # Clean up answers
        cleaned_answers = [ans.strip() for ans in answers]

        return {"answers": cleaned_answers}

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))