import os
import json
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# Define paths
DB_FAISS_PATH = "vectorstore/db_faiss"

# Define Prompts
custom_rag_prompt = PromptTemplate.from_template(
    """
    You are an expert insurance claims adjudicator. Your task is to analyze the user's situation based ONLY on the provided policy clauses and determine if the claim should be approved or rejected.

    CONTEXT:
    {context}

    QUERY:
    {query}

    INSTRUCTIONS:
    1.  Carefully read the user's query and the provided context clauses.
    2.  Evaluate the claim against rules like waiting periods, specific exclusions, and procedure types mentioned in the context.
    3.  Your final output MUST be a single, valid JSON object with no additional text or explanations before or after it.
    4.  The JSON object must have three keys: "decision" (string: "Approved", "Rejected", or "Needs More Information"), "amount" (integer), and "justification" (an object).
    5.  The "justification" object must contain a "summary" (string) and a "reasoning" (list of objects).
    6.  Each object in the "reasoning" list should detail a step in your decision-making process, including a "description" and a "citation" from the context.
    7.  Base your decision STRICTLY on the provided context. If the context does not contain enough information, state "Needs More Information".
    """
)

def format_docs(docs):
    """Helper function to format retrieved documents for the prompt."""
    return "\n\n".join(f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}\nContent: {doc.page_content}" for doc in docs)

def run_policy_analyzer(query):
    """
    Analyzes a user query against the policy knowledge base.
    """
    print("Loading knowledge base...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={'k': 5}) # Retrieve top 5 relevant chunks

    print("Initializing LLM...")
    # UPDATED LINE: Use the latest stable model name
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)

    print("Setting up RAG chain...")
    rag_chain = (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    print("Invoking chain to get the decision...")
    response_str = rag_chain.invoke(query)

    try:
        # Clean the response to ensure it's a valid JSON
        # The model might sometimes wrap the JSON in markdown backticks
        if response_str.startswith("```json"):
            response_str = response_str[7:]
            if response_str.endswith("```"):
                response_str = response_str[:-3]
        
        response_json = json.loads(response_str)
        return response_json
    except json.JSONDecodeError:
        print("Error: The model did not return a valid JSON. Raw response:")
        print(response_str)
        return None

if __name__ == "__main__":
    # This is our sample query
    sample_query = "I am a 46-year-old male who needs a knee surgery. I will be getting it done in Pune. My policy is 3 months old. Is my procedure covered?"
    
    print(f"Analyzing Query: '{sample_query}'\n")
    
    final_decision = run_policy_analyzer(sample_query)
    
    if final_decision:
        print("\n--- Final Structured Response ---")
        # Print the JSON in a nicely formatted way
        print(json.dumps(final_decision, indent=2))
        print("---------------------------------")