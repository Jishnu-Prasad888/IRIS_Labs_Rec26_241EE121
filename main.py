# main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from src.odyssey_hierarchical_pipeline import OdysseyHierarchicalPipeline

load_dotenv()

app = FastAPI(title="Odyssey Hierarchical RAG API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
_pipeline = None

# Pydantic models
class InitializeRequest(BaseModel):
    html_path: str = "data/raw/odyssey.html"

class QuestionRequest(BaseModel):
    question: str
    k_chunks: int = 5
    similarity_threshold: float = 0.25
    retrieval_strategy: str = "adaptive"

class Source(BaseModel):
    chunk_id: str
    similarity: float
    chunk_type: str
    level: str
    has_parent: bool
    child_count: int
    text_preview: str

class AnswerResponse(BaseModel):
    answer: str
    strategy: str
    question_type: str
    chunks_retrieved: int
    sources: List[Source]
    system_stats: Optional[Dict[str, Any]] = None

class SystemStatus(BaseModel):
    api_ready: bool
    system_ready: bool
    chunk_count: Optional[int] = None
    file_exists: bool
    file_size_kb: Optional[float] = None

# Dependency to get pipeline
def get_pipeline():
    global _pipeline
    return _pipeline

@app.get("/")
async def root():
    return {"message": "Odyssey Hierarchical RAG API"}

@app.get("/status")
async def get_status():
    """Get system status"""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Check default file
    default_path = "data/raw/odyssey.html"
    file_exists = Path(default_path).exists()
    file_size = None
    
    if file_exists:
        file_size = Path(default_path).stat().st_size / 1024
    
    global _pipeline
    return SystemStatus(
        api_ready=bool(api_key),
        system_ready=_pipeline is not None,
        chunk_count=len(_pipeline.chunks) if _pipeline else None,
        file_exists=file_exists,
        file_size_kb=file_size
    )

@app.post("/initialize")
async def initialize_system(request: InitializeRequest):
    """Initialize the RAG pipeline"""
    global _pipeline
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="Google API Key not found in .env file")
    
    if not Path(request.html_path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.html_path}")
    
    try:
        _pipeline = OdysseyHierarchicalPipeline(
            html_path=request.html_path
        )
        
        return {
            "status": "success",
            "message": "System initialized successfully",
            "stats": {
                "semantic_chunks": len(_pipeline.chunks),
                "hierarchy_levels": "1-4",
                "html_path": request.html_path
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

@app.post("/ask")
async def ask_question(request: QuestionRequest, pipeline=Depends(get_pipeline)):
    """Ask a question about The Odyssey"""
    if not pipeline:
        raise HTTPException(status_code=400, detail="System not initialized. Call /initialize first")
    
    try:
        # Handle different strategies
        if request.retrieval_strategy != "adaptive":
            question_type = request.retrieval_strategy
            
            # Get retrieval parameters
            retrieval_kwargs = pipeline._get_retrieval_kwargs(
                question_type, request.k_chunks, request.similarity_threshold
            )
            
            # Get results
            retrieved_results = pipeline.retriever.retrieve_with_context(
                query=request.question,
                **retrieval_kwargs
            )
            
            # Format context
            context = pipeline._format_context_with_hierarchy(retrieved_results, question_type)
            
            # Create prompt
            final_prompt = pipeline._create_odyssey_prompt(context, request.question, question_type)
            
            # Generate answer
            answer = pipeline.generator.generate(final_prompt, max_new_tokens=512)
            
            # Prepare sources
            sources = []
            for result in retrieved_results:
                source_info = Source(
                    chunk_id=result.chunk_id,
                    similarity=round(result.similarity, 3),
                    chunk_type=result.metadata.get("chunk_type", "unknown"),
                    level=result.metadata.get("level", "unknown"),
                    has_parent=result.parent_text is not None,
                    child_count=result.child_count,
                    text_preview=result.text[:150] + "..." if len(result.text) > 150 else result.text
                )
                sources.append(source_info)
            
            result_data = {
                "answer": answer.strip(),
                "strategy": request.retrieval_strategy,
                "question_type": question_type,
                "chunks_retrieved": len(retrieved_results),
                "sources": sources
            }
        else:
            # Use adaptive strategy
            result_data = pipeline.answer_question(
                question=request.question,
                k=request.k_chunks,
                threshold=request.similarity_threshold,
                strategy="adaptive"
            )
        
        return AnswerResponse(
            **result_data,
            system_stats={
                "chunk_count": len(pipeline.chunks),
                "strategy_used": request.retrieval_strategy
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/hierarchy")
async def get_hierarchy():
    """Get document hierarchy structure"""
    hierarchy_file = "data/processed/hierarchy.json"
    if not Path(hierarchy_file).exists():
        raise HTTPException(status_code=404, detail="Hierarchy file not found")
    
    with open(hierarchy_file, "r") as f:
        hierarchy = json.load(f)
    
    # Calculate stats
    nodes = hierarchy.get("nodes", {})
    level_counts = {}
    for node_id, node in nodes.items():
        level = node.get("level", 0)
        level_counts[level] = level_counts.get(level, 0) + 1
    
    return {
        "hierarchy": hierarchy,
        "stats": {
            "total_nodes": len(nodes),
            "level_counts": level_counts
        }
    }

@app.post("/reset")
async def reset_system():
    """Reset the pipeline"""
    global _pipeline
    _pipeline = None
    return {"status": "success", "message": "System reset successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)