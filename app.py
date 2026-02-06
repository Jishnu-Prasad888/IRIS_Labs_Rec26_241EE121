# app_odyssey_hierarchical.py
import streamlit as st
import json
import os
from pathlib import Path
from src.odyssey_hierarchical_pipeline import OdysseyHierarchicalPipeline
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="Odyssey Hierarchical RAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "system_ready" not in st.session_state:
    st.session_state.system_ready = False
if "show_hierarchy" not in st.session_state:
    st.session_state.show_hierarchy = False

# Sidebar
with st.sidebar:
    st.title("Odyssey Hierarchical RAG")
    st.markdown("""
    **Features:**
    - Semantic chunking (not fixed-length)
    - Hierarchy-aware retrieval
    - Adaptive question routing
    - Context-aware responses
    """)
    
    st.divider()
    
    # API Configuration
    st.subheader("API Configuration")
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if api_key:
        st.success("Google API Key loaded")
    else:
        st.error("Google API Key not found")
        st.info("Add GOOGLE_API_KEY to your .env file")
    
    st.divider()
    
    # File paths
    st.subheader("Data Configuration")
    html_path = st.text_input("HTML file path", "data/raw/odyssey.html")
    
    # Check if file exists
    if Path(html_path).exists():
        st.success(f"Found: {html_path}")
        file_size = Path(html_path).stat().st_size / 1024
        st.caption(f"File size: {file_size:.1f} KB")
    else:
        st.error(f"File not found: {html_path}")
    
    st.divider()
    
    # Retrieval settings
    st.subheader("Retrieval Settings")
    k_chunks = st.slider("Max chunks to retrieve", 1, 10, 5, 
                         help="How many chunks to retrieve for context")
    similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.25, 0.05,
                                    help="Minimum similarity score for chunks")
    
    # Strategy selection
    retrieval_strategy = st.selectbox(
        "Retrieval strategy",
        ["adaptive", "overview", "detail", "character", "structural"],
        help="Adaptive: Auto-selects based on question type"
    )
    
    st.divider()
    
    # Initialize button
    st.subheader("System Control")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Initialize System", type="primary", disabled=not api_key):
            with st.spinner("Building hierarchical RAG system..."):
                try:
                    st.session_state.pipeline = OdysseyHierarchicalPipeline(
                        html_path=html_path
                    )
                    st.session_state.system_ready = True
                    st.success("System ready!")
                    
                    # Show stats
                    pipeline = st.session_state.pipeline
                    st.info(f"""
                    **System Stats:**
                    - Semantic chunks: {len(pipeline.chunks)}
                    - Hierarchy levels: 1-4
                    - Processing complete
                    """)
                    
                except Exception as e:
                    st.error(f"Initialization failed: {e}")
                    st.session_state.system_ready = False
    
    with col2:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Hierarchy visualization
    st.divider()
    st.subheader("Hierarchy Info")
    
    if st.button("Show Document Structure"):
        st.session_state.show_hierarchy = not st.session_state.show_hierarchy
    
    if st.session_state.show_hierarchy and st.session_state.pipeline:
        hierarchy_file = "data/processed/hierarchy.json"
        if Path(hierarchy_file).exists():
            with open(hierarchy_file, "r") as f:
                hierarchy = json.load(f)
            
            st.json(hierarchy, expanded=False)
            
            # Show stats
            nodes = hierarchy.get("nodes", {})
            if nodes:
                st.metric("Total nodes", len(nodes))
                levels = {}
                for node_id, node in nodes.items():
                    level = node.get("level", 0)
                    levels[level] = levels.get(level, 0) + 1
                
                for level, count in sorted(levels.items()):
                    st.caption(f"Level {level} nodes: {count}")

# Main interface
st.title("The Odyssey - Hierarchical RAG")
st.markdown("""
Ask questions about Homer's Odyssey with intelligent, structure-aware retrieval.
The system understands book structure, characters, and narrative elements.
""")

# Status indicators
col1, col2, col3, col4 = st.columns(4)
with col1:
    if api_key:
        st.success("API: Ready")
    else:
        st.error("API: Missing")
with col2:
    if st.session_state.system_ready:
        pipeline = st.session_state.pipeline
        st.success(f"System: Ready ({len(pipeline.chunks)} chunks)")
    else:
        st.warning("System: Not ready")
with col3:
    if st.session_state.system_ready:
        st.info("Source: Odyssey HTML")
with col4:
    st.caption(f" Messages: {len(st.session_state.messages)}")

st.divider()

# Chat container
chat_container = st.container()

# Display chat history
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show metadata for assistant messages
            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]
                with st.expander("Retrieval Details"):
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Strategy", metadata.get("strategy", "N/A"))
                    with cols[1]:
                        st.metric("Type", metadata.get("question_type", "N/A"))
                    with cols[2]:
                        st.metric("Chunks", metadata.get("chunks_retrieved", 0))
                    with cols[3]:
                        if metadata.get("sources"):
                            avg_sim = sum(s["similarity"] for s in metadata["sources"]) / len(metadata["sources"])
                            st.metric("Avg Similarity", f"{avg_sim:.3f}")
                    
                    # Sources table
                    if metadata.get("sources"):
                        st.subheader("Retrieved Sources")
                        for i, source in enumerate(metadata["sources"]):
                            with st.expander(f"Source {i+1}: {source.get('chunk_type', 'chunk')}"):
                                st.caption(f"Similarity: {source['similarity']:.3f}")
                                st.caption(f"Level: {source.get('level', 'N/A')}")
                                st.caption(f"Has parent: {source.get('has_parent', False)}")
                                st.caption(f"Children: {source.get('child_count', 0)}")
                                st.text(source['text_preview'])

# Chat input
if prompt := st.chat_input("Ask about The Odyssey...", 
                          disabled=not st.session_state.system_ready):
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)
    
   
    with st.chat_message("assistant"):
     if not api_key:
         response = "Please configure your Google API key in the .env file."
         metadata = {}
         st.markdown(response)
     elif not st.session_state.system_ready:
         response = "Please initialize the system first using the sidebar button."
         metadata = {}
         st.markdown(response)
     else:
         with st.spinner("Retrieving with hierarchy-aware search..."):
             try:
                 pipeline = st.session_state.pipeline

                 # Handle different strategies
                 if retrieval_strategy != "adaptive":
                     # For non-adaptive strategies, we'll analyze the question
                     # but also adjust parameters
                     question_type = retrieval_strategy  # Use the strategy as question type

                     # Get retrieval parameters for this strategy
                     retrieval_kwargs = pipeline._get_retrieval_kwargs(
                         question_type, k_chunks, similarity_threshold
                     )

                     # Get results
                     retrieved_results = pipeline.retriever.retrieve_with_context(
                         query=prompt,
                         **retrieval_kwargs
                     )

                     # Format context
                     context = pipeline._format_context_with_hierarchy(retrieved_results, question_type)

                     # Create prompt
                     final_prompt = pipeline._create_odyssey_prompt(context, prompt, question_type)

                     # Generate answer
                     answer = pipeline.generator.generate(final_prompt, max_new_tokens=512)

                     # Prepare sources
                     sources = []
                     for result in retrieved_results:
                         source_info = {
                             "chunk_id": result.chunk_id,
                             "similarity": round(result.similarity, 3),
                             "chunk_type": result.metadata.get("chunk_type", "unknown"),
                             "level": result.metadata.get("level", "unknown"),
                             "has_parent": result.parent_text is not None,
                             "child_count": result.child_count,
                             "text_preview": result.text[:150] + "..." if len(result.text) > 150 else result.text
                         }
                         sources.append(source_info)

                     result = {
                         "answer": answer.strip(),
                         "sources": sources,
                         "question_type": question_type,
                         "strategy": retrieval_strategy,
                         "chunks_retrieved": len(retrieved_results)
                     }
                 else:
                     # Use the adaptive strategy
                     result = pipeline.answer_question(
                         question=prompt,
                         k=k_chunks,
                         threshold=similarity_threshold,
                         strategy="adaptive"
                     )

                 # Display answer
                 st.markdown(result["answer"])

                 # Store metadata for display
                 metadata = {
                     "strategy": result.get("strategy", "adaptive"),
                     "question_type": result.get("question_type", "general"),
                     "chunks_retrieved": result.get("chunks_retrieved", 0),
                     "sources": result.get("sources", [])
                 }

             except Exception as e:
                    response = f"Error generating response: {str(e)}"
                    metadata = {}
                    st.error(response)
                    result = {"answer": response, "sources": []}


    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": result.get("answer", "No response generated"),
        "metadata": metadata
    })

# Footer
st.divider()
footer_cols = st.columns(4)
with footer_cols[0]:
    if st.button("Reset System"):
        st.session_state.pipeline = None
        st.session_state.system_ready = False
        st.session_state.messages = []
        st.rerun()
with footer_cols[1]:
    if st.session_state.pipeline:
        st.caption(f" {len(st.session_state.pipeline.chunks)} semantic chunks")
with footer_cols[2]:
    st.caption("Adaptive hierarchical retrieval")
with footer_cols[3]:
    st.caption(" Homer's Odyssey")

# Example questions
st.divider()
st.subheader("Example Questions")



st.caption("Try: 'Tell me about the Cyclops', 'What is the role of Athena?', 'Summary of Book 1'")