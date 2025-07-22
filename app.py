# Optimized app.py with performance improvements
import streamlit as st
import json
import os
import time
import asyncio
from src.main import run_policy_analyzer_cached as run_policy_analyzer

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI Policy Analyzer",
    page_icon="⚖️",
    layout="wide"
)

# Enhanced caching with TTL
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_result(query_hash):
    """Get cached analysis result"""
    return None  # Will return cached result if exists

def set_cached_result(query_hash, result):
    """Set cached analysis result"""
    # Store in session state for this session
    if 'analysis_cache' not in st.session_state:
        st.session_state.analysis_cache = {}
    st.session_state.analysis_cache[query_hash] = result

@st.cache_resource
def get_analyzer_func():
    return run_policy_analyzer

# Add performance monitoring
def track_performance(func):
    """Decorator to track function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        st.sidebar.metric("Last Query Time", f"{end_time - start_time:.2f}s")
        return result
    return wrapper

# --- App Header ---
st.title("⚖️ AI-Powered Insurance Policy Analyzer")
st.write(
    "This tool uses a Retrieval-Augmented Generation (RAG) system to answer questions "
    "about your insurance policies. Enter a query below to get a structured decision with citations."
)

# --- Performance Settings Sidebar ---
with st.sidebar:
    st.header("⚡ Performance Settings")
    
    # Model selection
    model_speed = st.selectbox(
        "Model Speed vs Accuracy",
        ["Fast (Gemini 1.5 Flash)", "Balanced", "Accurate (Gemini 1.5 Pro)"],
        index=0  # Default to fastest
    )
    
    # Chunk limit
    max_chunks = st.slider("Max Document Chunks", 1, 10, 3, 
                          help="Fewer chunks = faster response")
    
    # Output length
    max_output = st.slider("Max Output Length", 200, 2000, 800,
                          help="Shorter output = faster response")
    
    # Show cache status
    if 'query_cache_size' not in st.session_state:
        st.session_state.query_cache_size = 0
    st.metric("Cached Queries", st.session_state.query_cache_size)
    
    if st.button("Clear Cache"):
        if 'analysis_cache' in st.session_state:
            st.session_state.analysis_cache.clear()
        st.session_state.query_cache_size = 0
        st.success("Cache cleared!")

# --- User Input with Smart Suggestions ---
if 'last_query' not in st.session_state:
    st.session_state.last_query = "I am a 62 years old and have the ICICI Golden Shield policy for 7 months. I need cataract surgery. Will the policy cover it?"

# Quick query templates
st.write("**Quick Templates:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Cataract Surgery Coverage"):
        st.session_state.last_query = "Does my policy cover cataract surgery?"
with col2:
    if st.button("Waiting Period"):
        st.session_state.last_query = "What is the waiting period for my condition?"
with col3:
    if st.button("Claim Process"):
        st.session_state.last_query = "How do I file a claim?"

query = st.text_area(
    "Enter your query here:",
    st.session_state.last_query,
    height=120,  # Reduced height
    help="Be specific for faster, more accurate results"
)

# --- Real-time character count ---
char_count = len(query)
if char_count > 500:
    st.warning(f"Long queries ({char_count} chars) may take longer to process. Consider shortening.")

# --- Submit Button with Enhanced Logic ---
col1, col2 = st.columns([1, 4])
with col1:
    analyze_button = st.button("🚀 Analyze Query", type="primary")
with col2:
    if st.button("💾 Save Query"):
        if 'saved_queries' not in st.session_state:
            st.session_state.saved_queries = []
        st.session_state.saved_queries.append(query)
        st.success("Query saved!")

if analyze_button:
    if query:
        # Create query hash for caching
        import hashlib
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # Check cache first
        cached_result = None
        if 'analysis_cache' in st.session_state:
            cached_result = st.session_state.analysis_cache.get(query_hash)
        
        if cached_result:
            st.info("📋 Retrieved from cache (instant response)")
            result = cached_result
        else:
            st.session_state.last_query = query
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Performance settings
            perf_settings = {
                'model_type': model_speed,
                'max_chunks': max_chunks,
                'max_output_tokens': max_output
            }
            
            try:
                status_text.text("🔍 Retrieving relevant documents...")
                progress_bar.progress(20)
                
                status_text.text("🤖 Analyzing with AI...")
                progress_bar.progress(60)
                
                start_time = time.time()
                analyzer = get_analyzer_func()
                
                # Pass performance settings to analyzer (now properly supported)
                result = analyzer(query, 
                                model_type=model_speed,
                                max_chunks=max_chunks, 
                                max_output_tokens=max_output)
                
                analysis_time = time.time() - start_time
                
                progress_bar.progress(100)
                status_text.text(f"✅ Analysis complete in {analysis_time:.1f}s")
                
                # Cache the result
                set_cached_result(query_hash, result)
                st.session_state.query_cache_size = len(st.session_state.get('analysis_cache', {}))
                
                time.sleep(0.5)  # Brief pause to show completion
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"An error occurred: {e}")
                st.stop()
        
        # --- Display Results with Performance Metrics ---
        st.subheader("Analysis Result")
        
        # Show performance metrics if available
        if result and isinstance(result, dict) and '_metadata' in result:
            metadata = result['_metadata']
            performance_col1, performance_col2, performance_col3 = st.columns(3)
            with performance_col1:
                response_time = metadata.get('processing_time', 0)
                st.metric("Response Time", f"{response_time}s")
            with performance_col2:
                chunks = metadata.get('chunks_retrieved', max_chunks)
                st.metric("Chunks Processed", chunks)
            with performance_col3:
                if metadata.get('from_cache'):
                    st.metric("Source", "📋 Cache")
                else:
                    model_used = metadata.get('model_used', 'Unknown')
                    model_display = "Flash" if "flash" in model_used.lower() else "Pro"
                    st.metric("Model", f"Gemini {model_display}")
        
        if result and isinstance(result, dict):
            # Display decision with color
            decision = result.get("decision", "N/A")
            if decision == "Approved":
                st.success(f"**Decision: {decision}**")
            elif decision == "Rejected":
                st.error(f"**Decision: {decision}**")
            else:
                st.warning(f"**Decision: {decision}**")

            # Display summary
            justification = result.get("justification", {})
            summary = justification.get("summary", "No summary provided.")
            st.info(f"**Summary:** {summary}")

            # Display detailed reasoning and citations (same as fixed version)
            st.write("**Detailed Justification:**")
            reasoning_steps = justification.get("reasoning", [])
            if reasoning_steps:
                for i, step in enumerate(reasoning_steps):
                    with st.expander(f"**Step {i+1}: {step.get('description', 'No description')}**", expanded=i==0):  # Only first expanded
                        evidence_list = step.get('evidence', [])
                        
                        if not isinstance(evidence_list, list):
                            evidence_list = [evidence_list]

                        for evidence_item in evidence_list:
                            if isinstance(evidence_item, dict):
                                st.markdown(f"**Evidence:** `{evidence_item.get('finding', 'N/A')}`")
                                
                                citation = evidence_item.get('citation', {})
                                if isinstance(citation, dict):
                                    source = citation.get('Source', 'N/A')
                                    page = citation.get('Page', 'N/A')
                                    st.caption(f"Source: {source}, Page: {page}")
                                elif isinstance(citation, str):
                                    st.caption(f"Citation: {citation}")
                                else:
                                    st.caption(f"Citation: {str(citation)}")
                            else:
                                st.markdown(f"**Evidence:** `{str(evidence_item)}`")
            else:
                st.write("No detailed reasoning provided.")

        else:
            st.error("The analysis failed or did not return a valid response.")
            
    else:
        st.warning("Please enter a query to analyze.")

# --- Saved Queries Section ---
if 'saved_queries' in st.session_state and st.session_state.saved_queries:
    st.sidebar.write("**💾 Saved Queries:**")
    for i, saved_query in enumerate(st.session_state.saved_queries[-5:]):  # Show last 5
        if st.sidebar.button(f"Query {i+1}: {saved_query[:30]}...", key=f"saved_{i}"):
            st.session_state.last_query = saved_query
            st.experimental_rerun()