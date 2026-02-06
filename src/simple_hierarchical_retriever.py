# src/simple_hierarchical_retriever.py (UPDATED)
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
import faiss
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class HierarchicalResult:
    """Result with hierarchy information"""
    chunk_id: str
    text: str
    similarity: float
    metadata: Dict
    parent_text: Optional[str] = None
    child_count: int = 0


class SimpleHierarchicalRetriever:
    """
    Simplified hierarchical retriever for HTML content
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Dict] = []
        self.chunk_map: Dict[str, Dict] = {}
        self.parent_map: Dict[str, str] = {}  # child_id -> parent_id
        self.child_map: Dict[str, List[str]] = defaultdict(list)  # parent_id -> child_ids
        
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    @staticmethod
    def _normalize(x: np.ndarray) -> None:
        """Normalize embeddings for cosine similarity"""
        faiss.normalize_L2(x)
    
    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Create embeddings and build hierarchy maps"""
        self.chunks = chunks
        self.chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
        
        # Build hierarchy maps
        for chunk in chunks:
            parent_id = chunk["metadata"].get("parent_id")
            if parent_id:
                self.parent_map[chunk["chunk_id"]] = parent_id
                self.child_map[parent_id].append(chunk["chunk_id"])
        
        # Create embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings_np = np.asarray(embeddings, dtype=np.float32)
        
        self._normalize(embeddings_np)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings_np)
        
        print(f"Created embeddings for {len(chunks)} chunks")
        print(f"Hierarchy: {len(self.parent_map)} parent-child relationships")
        
        return embeddings_np
    
    def retrieve_with_context(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.25,
        include_parent: bool = True,
        include_children: bool = False
    ) -> List[HierarchicalResult]:
        """
        Retrieve with optional parent/child context
        """
        if self.index is None:
            raise RuntimeError("Index not initialized")
        
        # Encode query
        query_embedding = self.model.encode([query])
        query_np = np.asarray(query_embedding, dtype=np.float32)
        self._normalize(query_np)
        
        # Search
        scores, indices = self.index.search(query_np, min(k * 3, len(self.chunks)))
        
        results = []
        seen_chunks = set()
        
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1 or score < threshold or idx >= len(self.chunks):
                continue
            
            chunk = self.chunks[idx]
            chunk_id = chunk["chunk_id"]
            
            # Skip if already seen
            if chunk_id in seen_chunks:
                continue
            
            # Get parent context if requested
            parent_text = None
            if include_parent:
                parent_text = self._get_parent_text(chunk_id)
            
            # Get child count
            child_count = len(self.child_map.get(chunk_id, []))
            
            result = HierarchicalResult(
                chunk_id=chunk_id,
                text=chunk["text"],
                similarity=float(score),
                metadata=chunk["metadata"],
                parent_text=parent_text,
                child_count=child_count
            )
            
            results.append(result)
            seen_chunks.add(chunk_id)
            
            # Include children if requested
            if include_children and child_count > 0:
                child_results = self._get_relevant_children(
                    chunk_id, query_np, threshold * 0.8
                )
                for child_chunk, child_score in child_results[:2]:  # Top 2 children
                    if child_chunk["chunk_id"] not in seen_chunks:
                        results.append(HierarchicalResult(
                            chunk_id=child_chunk["chunk_id"],
                            text=child_chunk["text"],
                            similarity=float(child_score),
                            metadata=child_chunk["metadata"],
                            parent_text=chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                            child_count=0
                        ))
                        seen_chunks.add(child_chunk["chunk_id"])
            
            if len(results) >= k:
                break
        
        # Sort by similarity
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:k]
    
    def _get_parent_text(self, chunk_id: str) -> Optional[str]:
        """Get text of parent chunk"""
        parent_id = self.parent_map.get(chunk_id)
        if parent_id and parent_id in self.chunk_map:
            parent_chunk = self.chunk_map[parent_id]
            # Return truncated parent text
            text = parent_chunk["text"]
            return text[:150] + "..." if len(text) > 150 else text
        return None
    
    def _get_relevant_children(
        self, 
        parent_id: str, 
        query_embedding: np.ndarray, 
        threshold: float
    ) -> List[Tuple[Dict, float]]:
        """Get relevant children of a chunk"""
        child_ids = self.child_map.get(parent_id, [])
        if not child_ids:
            return []
        
        # Get child chunks
        child_chunks = []
        for child_id in child_ids:
            if child_id in self.chunk_map:
                child_chunks.append(self.chunk_map[child_id])
        
        if not child_chunks:
            return []
        
        # Score children against query
        child_texts = [chunk["text"] for chunk in child_chunks]
        child_embeddings = self.model.encode(child_texts)
        child_embeddings_np = np.asarray(child_embeddings, dtype=np.float32)
        self._normalize(child_embeddings_np)
        
        # Calculate similarities
        similarities = np.dot(child_embeddings_np, query_embedding.T).flatten()
        
        # Pair chunks with scores and filter
        results = []
        for chunk, similarity in zip(child_chunks, similarities):
            if similarity >= threshold:
                results.append((chunk, float(similarity)))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def save_index(self, index_path: str, chunks_path: str, hierarchy_path: str = None) -> None:
        """Save index, chunks, and hierarchy"""
        if self.index is None:
            raise RuntimeError("No index to save")
        
        # Create directories
        Path(index_path).mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{index_path}/index.faiss")
        
        # Save chunks
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        
        # Save hierarchy if requested
        if hierarchy_path:
            hierarchy_data = {
                "parent_map": self.parent_map,
                "child_map": dict(self.child_map)
            }
            with open(hierarchy_path, "w", encoding="utf-8") as f:
                json.dump(hierarchy_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved index to {index_path}")
        print(f"Saved chunks to {chunks_path}")
        if hierarchy_path:
            print(f"Saved hierarchy to {hierarchy_path}")
    
    def load_index(self, index_path: str, chunks_path: str, hierarchy_path: str = None) -> None:
        """Load index, chunks, and hierarchy"""
        # Load FAISS index
        self.index = faiss.read_index(f"{index_path}/index.faiss")
        
        # Load chunks
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.chunk_map = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        
        # Load hierarchy if available
        if hierarchy_path and Path(hierarchy_path).exists():
            with open(hierarchy_path, "r", encoding="utf-8") as f:
                hierarchy_data = json.load(f)
            self.parent_map = hierarchy_data.get("parent_map", {})
            self.child_map = defaultdict(list, hierarchy_data.get("child_map", {}))
        else:
            # Rebuild hierarchy from chunks
            self.parent_map = {}
            self.child_map = defaultdict(list)
            for chunk in self.chunks:
                parent_id = chunk["metadata"].get("parent_id")
                if parent_id:
                    self.parent_map[chunk["chunk_id"]] = parent_id
                    self.child_map[parent_id].append(chunk["chunk_id"])
        
        print(f"Loaded index with {len(self.chunks)} chunks")