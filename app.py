import streamlit as st
import json
import os
from src.rag_pipeline import RAGPipeline
from src.data_processor import OdysseyDataProcessor
from src.embedding_retriever import EmbeddingRetriever
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="Odyssey RAG Chatbot",
    layout="wide"
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "system_ready" not in st.session_state:
    st.session_state.system_ready = False

with st.sidebar:
    st.title("Odyssey RAG Chatbot")
    st.markdown("""
    **About this chatbot:**
    - Answers questions about Homer's Odyssey
    - Uses Gemini 2.5 Flash model
    - Retrieves context from Project Gutenberg text
    - Online operation with Google AI API
    """)
    
    st.divider()
    st.subheader("API Configuration")
    
    if GOOGLE_API_KEY:
        st.success("Google API Key loaded")
    else:
        st.error("Google API Key not found")
        st.info("Add GOOGLE_API_KEY to your .env file")
    
    st.divider()
    
    # File paths
    st.subheader("Data Configuration")
    chunks_path = st.text_input("Chunks JSON path", "data/processed/chunks.json")
    embeddings_path = st.text_input("Embeddings directory", "data/embeddings/faiss_index")
    html_path = st.text_input("HTML source path", "data/raw/odyssey.html")
    
    st.divider()
    
    # Initialize system
    st.subheader("System Setup")
    
    if not GOOGLE_API_KEY:
        st.warning("API key required to initialize system")
    
    if st.button("Initialize RAG System", disabled=not GOOGLE_API_KEY):
        with st.spinner("Setting up the system..."):
            # Check if data processing is needed
            data_needs_processing = False
            
            if not os.path.exists(chunks_path):
                data_needs_processing = True
                st.info("Chunks file not found, will process HTML...")
            
            if not os.path.exists(embeddings_path):
                data_needs_processing = True
                st.info("Embeddings not found, will create them...")
            
            # Process data if needed
            if data_needs_processing:
                # Ensure directories exist
                os.makedirs(os.path.dirname(chunks_path), exist_ok=True)
                os.makedirs(embeddings_path, exist_ok=True)
                
                if os.path.exists(html_path):
                    st.info("Processing text data...")
                    try:
                        processor = OdysseyDataProcessor(html_path)
                        chunks = processor.process_and_save(chunks_path)
                        st.success(f"Extracted {len(chunks)} chunks")
                    except Exception as e:
                        st.error(f"Error processing HTML: {e}")
                        st.stop()
                else:
                    st.error(f"HTML file not found at: {html_path}")
                    st.stop()
                
                # Create embeddings
                st.info("Creating embeddings...")
                try:
                    with open(chunks_path, 'r', encoding='utf-8') as f:
                        chunks_data = json.load(f)
                    
                    retriever = EmbeddingRetriever()
                    embeddings = retriever.create_embeddings(chunks_data)
                    retriever.save_index(embeddings_path, chunks_path)
                    st.success(f"Created embeddings for {len(chunks_data)} chunks")
                except Exception as e:
                    st.error(f"Error creating embeddings: {e}")
                    st.stop()
            
            # Initialize pipeline
            st.info("Initializing RAG pipeline with Gemini...")
            try:
                st.session_state.pipeline = RAGPipeline(
                    chunks_path,
                    embeddings_path
                )
                st.session_state.system_ready = True
                st.success("System ready !!!")
            except Exception as e:
                st.error(f"Error initializing pipeline: {e}")
                st.session_state.system_ready = False
            else:
                st.session_state.system_ready = True

# Main interface
st.title("The Odyssey RAG Chatbot")
st.markdown("Ask questions about Homer's Odyssey using Gemini 2.5 Flash")

# System status
col1, col2 = st.columns(2)
with col1:
    if GOOGLE_API_KEY:
        st.success("API Key: Configured")
    else:
        st.error("API Key: Missing")

with col2:
    if st.session_state.system_ready:
        st.success("RAG System: Ready")
    else:
        st.warning("RAG System: Not initialized")

# Chat container
chat_container = st.container()

# Display chat history
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander("View Sources"):
                        for source in message["sources"]:
                            st.markdown(f"**{source['chapter']}**")
                            st.markdown(f"Paragraphs: {source['paragraph_range']}")
                            st.markdown(f"Similarity: {source['similarity']:.3f}")
                else:
                    st.caption("No sources available")

# Chat input
if prompt := st.chat_input("Ask a question about The Odyssey", 
                          disabled=not st.session_state.system_ready):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        if not GOOGLE_API_KEY:
            response = "Please configure your Google API key in the .env file."
            sources = []
            st.markdown(response)
        elif not st.session_state.system_ready:
            response = "Please initialize the system first using the sidebar button."
            sources = []
            st.markdown(response)
        else:
            with st.spinner("Thinking with Gemini 2.5 Flash..."):
                try:
                    pipeline = st.session_state.pipeline
                    assert pipeline is not None

                    result = pipeline.answer_question(prompt)

                    response = result["answer"]
                    sources = result["sources"]
                    
                    # Display answer
                    st.markdown(response)
                    
                    # Display sources
                    if sources:
                        with st.expander("View Sources"):
                            for source in sources:
                                st.markdown(f"**{source['chapter']}**")
                                st.markdown(f"Paragraphs: {source['paragraph_range']}")
                                st.markdown(f"Similarity: {source['similarity']:.3f}")
                    else:
                        st.caption("No relevant sources found")
                except Exception as e:
                    response = f"Error generating response: {str(e)}"
                    sources = []
                    st.error(response)
    
    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "sources": sources
    })

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

with col2:
    st.caption(f"Messages in chat: {len(st.session_state.messages)}")

with col3:
    st.caption(" Gemini 2.5 Flash")