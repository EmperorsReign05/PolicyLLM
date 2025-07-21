import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define paths
DOCS_PATH = "documents/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def create_vector_db():
    """
    Processes all PDF documents in DOCS_PATH, chunks them,
    creates embeddings, and stores them in a FAISS vector database.
    """
    # 1. Load Documents
    print("Loading documents...")
    all_docs = []
    for filename in os.listdir(DOCS_PATH):
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(DOCS_PATH, filename))
            all_docs.extend(loader.load())
    
    if not all_docs:
        print("No PDF documents found in the 'documents' folder.")
        return

    # 2. Chunk Text
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(all_docs)

    # 3. Create Embeddings
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # 4. Create and Save Vector Store
    print("Creating and saving FAISS vector store...")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(DB_FAISS_PATH)
    print(f"Vector store created and saved at {DB_FAISS_PATH}")

if __name__ == "__main__":
    create_vector_db()