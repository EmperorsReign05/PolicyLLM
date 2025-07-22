# src/main.py
import os
import json
import time
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# Define paths
DB_FAISS_PATH = "vectorstore/db_faiss"

# Optimized prompt for faster responses
custom_rag_prompt = PromptTemplate.from_template(
    """
    You are an expert insurance claims adjudicator. Analyze the user's situation based on the provided policy clauses and determine if the claim should be approved or rejected.

    CONTEXT:
    {context}

    QUERY:
    {query}

    INSTRUCTIONS:
    1. Evaluate the claim against waiting periods, exclusions, and procedure types.
    2. Output ONLY a valid JSON object with no additional text.
    3. Required format:
    {{
        "decision": "Approved|Rejected|Needs More Information",
        "amount": 0,
        "justification": {{
            "summary": "Brief one-sentence summary",
            "reasoning": [
                {{
                    "description": "Key decision factor",
                    "evidence": [
                        {{
                            "finding": "Relevant policy clause",
                            "citation": {{"Source": "document", "Page": "page_num"}}
                        }}
                    ]
                }}
            ]
        }}
    }}
    """
)

def format_docs(docs):
    """Helper function to format retrieved documents for the prompt."""
    return "\n\n".join(f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}\nContent: {doc.page_content}" for doc in docs)

def run_policy_analyzer(query, **kwargs):
    """
    Analyzes a user query against the policy knowledge base.
    
    Args:
        query (str): The user's query
        **kwargs: Performance settings including:
            - model_type: Model speed preference
            - max_chunks: Maximum number of document chunks to retrieve
            - max_output_tokens: Maximum tokens in response
    """
    start_time = time.time()
    print("Loading knowledge base and models...")
    
    # Extract performance settings with defaults
    model_type = kwargs.get('model_type', 'Fast (Gemini 1.5 Flash)')
    max_chunks = kwargs.get('max_chunks', 3)
    max_output_tokens = kwargs.get('max_output_tokens', 800)
    
    # Choose model based on preference
    if 'Pro' in model_type:
        model_name = "gemini-1.5-pro-latest"
        temperature = 0
    elif 'Balanced' in model_type:
        model_name = "gemini-1.5-flash-latest"
        temperature = 0.1
    else:  # Fast
        model_name = "gemini-1.5-flash-latest"
        temperature = 0
    
    print(f"Using model: {model_name} with {max_chunks} chunks")
    
    # Load embeddings and vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Configure retriever with performance settings
    retriever = db.as_retriever(search_kwargs={'k': max_chunks})

    # Configure LLM with performance settings
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    print("Setting up and invoking RAG chain...")
    rag_chain = (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    
    # Invoke the chain
    response_str = rag_chain.invoke(query)
    
    # Track timing
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f} seconds")

    try:
        # Clean up response if it has markdown formatting
        if response_str.startswith("```json"):
            response_str = response_str[7:-3].strip()
        elif response_str.startswith("```"):
            # Handle other code block formats
            lines = response_str.split('\n')
            response_str = '\n'.join(lines[1:-1]).strip()
        
        # Parse JSON
        response_json = json.loads(response_str)
        
        # Add performance metadata
        response_json['_metadata'] = {
            'processing_time': round(processing_time, 2),
            'model_used': model_name,
            'chunks_retrieved': max_chunks
        }
        
        return response_json
        
    except json.JSONDecodeError as e:
        print(f"Error: The model did not return valid JSON. Error: {e}")
        print("Raw response:")
        print(response_str)
        
        # Return error response in expected format
        return {
            "decision": "Needs More Information",
            "amount": 0,
            "justification": {
                "summary": "Unable to process query due to response parsing error.",
                "reasoning": [
                    {
                        "description": "JSON parsing failed",
                        "evidence": [
                            {
                                "finding": f"Raw response: {response_str[:200]}...",
                                "citation": {"Source": "System Error", "Page": "N/A"}
                            }
                        ]
                    }
                ]
            },
            "_metadata": {
                "processing_time": round(processing_time, 2),
                "error": "JSON parsing failed"
            }
        }

# Cache for repeated queries
_query_cache = {}

def run_policy_analyzer_cached(query, **kwargs):
    """
    Cached version of the policy analyzer for faster repeated queries.
    """
    import hashlib
    
    # Create cache key
    cache_key = hashlib.md5(f"{query}_{str(sorted(kwargs.items()))}".encode()).hexdigest()
    
    # Check cache
    if cache_key in _query_cache:
        print("Retrieved from cache!")
        cached_result = _query_cache[cache_key].copy()
        cached_result['_metadata']['from_cache'] = True
        return cached_result
    
    # Run analysis
    result = run_policy_analyzer(query, **kwargs)
    
    # Cache result (limit cache size)
    if len(_query_cache) > 50:  # Limit cache size
        # Remove oldest entry
        oldest_key = next(iter(_query_cache))
        del _query_cache[oldest_key]
    
    _query_cache[cache_key] = result
    return result

if __name__ == "__main__":
    sample_query = "I am a 62 years old and have the ICICI Golden Shield policy for 7 months. I need cataract surgery. Will the policy cover it?"
    
    print(f"Analyzing Query: '{sample_query}'\n")
    
    # Test with performance settings
    performance_settings = {
        'model_type': 'Fast (Gemini 1.5 Flash)',
        'max_chunks': 3,
        'max_output_tokens': 800
    }
    
    final_decision = run_policy_analyzer_cached(sample_query, **performance_settings)
    
    if final_decision:
        print("\n--- Final Structured Response ---")
        print(json.dumps(final_decision, indent=2))
        print("---------------------------------")