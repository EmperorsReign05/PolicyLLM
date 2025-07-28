# Policy LLM

An AI-powered insurance policy analyzer that uses Retrieval-Augmented Generation (RAG) to process complex policy documents and answer natural language queries with high accuracy.

## 🚀 Features

- **Document Processing**: Extracts and processes text from PDF policy documents up to 80+ pages
- **Intelligent Search**: Uses TF-IDF vectorization and semantic search for optimal document retrieval
- **Natural Language Queries**: Answer insurance-specific questions in plain English
- **Fast Response Times**: Optimized to respond in under 4 seconds
- **RESTful API**: Clean FastAPI implementation with authentication
- **Caching System**: Intelligent document caching for improved performance
- **Insurance Domain Expertise**: Specialized keyword matching for insurance terms

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python
- **AI/ML**: Google Gemini 1.5 Flash, LangChain, FAISS
- **Document Processing**: PyPDF2, TF-IDF Vectorization
- **Vector Storage**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: HuggingFace Sentence Transformers
- **Deployment**: Uvicorn ASGI server

## 📋 Prerequisites

- Python 3.8+
- Google AI API Key
- Required Python packages (see requirements)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/policy-llm.git
   cd policy-llm
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
   ```

4. **Run the application**
   ```bash
   # Using uvicorn directly
   uvicorn api:app --host 0.0.0.0 --port 8000

   # Or using the main script
   python api.py
   ```


## 🏗️ Architecture

The system follows a RAG (Retrieval-Augmented Generation) architecture:

1. **Document Ingestion**: PDF documents are downloaded and processed
2. **Text Extraction**: PyPDF2 extracts text with structure preservation
3. **Chunking**: Documents are split into manageable chunks with overlap
4. **Vectorization**: TF-IDF creates searchable vectors
5. **Query Processing**: User questions are matched against document chunks
6. **Answer Generation**: Google Gemini generates contextual responses

## 🔍 Key Components

- **`api.py`**: Main FastAPI application with optimized endpoints
- **`main.py`**: Core RAG implementation with FAISS vector store
- **Document Processing**: Advanced PDF text extraction with cleanup
- **Search System**: Hybrid TF-IDF and keyword-based retrieval
- **LLM Integration**: Optimized prompts for insurance domain


