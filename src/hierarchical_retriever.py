import json
import numpy as np
from typing import List, Dict, Tuple, Optional
import faiss
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import networkx as nx
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RetrievalResult:
    """Enhanced retrieval result with hierarchy information"""
    chunk_id: str
    text: str
    similarity: float
    metadata: Dict
    parent_chunks: List[Dict]
    child_chunks: List[Dict]
    depth: int


class HierarchicalRetriever:
    """TreeRAG-inspired hierarchical retriever"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.flat_index: Optional[faiss.IndexFlatIP] = None
        self.graph = nx.DiGraph()
        self.chunks: List[Dict] = []
        self.chunk_map: Dict[str, Dict] = {}
        self.level_indices = {}
        self.level_chunks = defaultdict(list)
        self.dimension = self.model.get_sentence_embedding_dimension()

    # -------------------
    # Core Utilities
    # -------------------
    @staticmethod
    def _normalize(x: np.ndarray) -> None:
        faiss.normalize_L2(x)

    @staticmethod
    def _search(index: faiss.IndexFlatIP, x: np.ndarray, k: int):
        return index.search(x, k)

    @staticmethod
    def _merge_results(results1: List[Tuple], results2: List[Tuple], k: int) -> List[Tuple]:
        merged = results1 + results2
        seen = set()
        unique_results = []
        for chunk, sim in sorted(merged, key=lambda x: x[1], reverse=True):
            chunk_id = chunk["chunk_id"]
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique_results.append((chunk, sim))
            if len(unique_results) >= k * 2:
                break
        return unique_results[:k]

    # -------------------
    # Hierarchy Construction
    # -------------------
    def build_hierarchy(self, chunks: List[Dict]):
        self.chunks = chunks
        self.chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
        for chunk in chunks:
            self.graph.add_node(chunk["chunk_id"], **chunk["metadata"])
            parent_id = chunk["metadata"].get("parent_id")
            if parent_id and parent_id in self.chunk_map:
                self.graph.add_edge(parent_id, chunk["chunk_id"])
            level = chunk["metadata"].get("level", 0)
            self.level_chunks[level].append(chunk)

    def create_multi_level_embeddings(self, chunks: List[Dict]):
        self.build_hierarchy(chunks)
        for level, lvl_chunks in self.level_chunks.items():
            texts = [c["text"] for c in lvl_chunks]
            embeddings = np.asarray(self.model.encode(texts, show_progress_bar=False), dtype=np.float32)
            self._normalize(embeddings)
            index = faiss.IndexFlatIP(self.dimension)
            index.add(embeddings)
            self.level_indices[level] = {"index": index, "chunks": lvl_chunks, "embeddings": embeddings}

        all_texts = [c["text"] for c in chunks]
        all_embeddings = np.asarray(self.model.encode(all_texts, show_progress_bar=True), dtype=np.float32)
        self._normalize(all_embeddings)
        self.flat_index = faiss.IndexFlatIP(self.dimension)
        self.flat_index.add(all_embeddings)
        return all_embeddings

    # -------------------
    # Hierarchical Retrieval
    # -------------------
    def hierarchical_retrieve(self, query: str, k: int = 5, threshold: float = 0.3, strategy: str = "hybrid"):
        query_embedding = np.asarray(self.model.encode([query]), dtype=np.float32)
        self._normalize(query_embedding)
        results = []

        if strategy == "top_down":
            results = self.top_down_retrieval(query_embedding, k, threshold)
        elif strategy == "bottom_up":
            results = self.bottom_up_retrieval(query_embedding, k, threshold)
        else:
            top_down = self.top_down_retrieval(query_embedding, k//2, threshold)
            bottom_up = self.bottom_up_retrieval(query_embedding, k//2, threshold)
            results = self._merge_results(top_down, bottom_up, k)

        enriched = [self.enrich_with_hierarchy(chunk, sim) for chunk, sim in results]
        return enriched

    def top_down_retrieval(self, query_embedding: np.ndarray, k: int, threshold: float):
        results = []
        for level in sorted(self.level_indices.keys())[:3]:
            index_info = self.level_indices[level]
            scores, indices = self._search(index_info["index"], query_embedding, min(k*2, len(index_info["chunks"])))
            for idx, score in zip(indices[0], scores[0]):
                if idx != -1 and score >= threshold:
                    chunk = index_info["chunks"][idx]
                    results.append((chunk, float(score)))
        return sorted(results, key=lambda x: x[1], reverse=True)[:k]

    def bottom_up_retrieval(self, query_embedding: np.ndarray, k: int, threshold: float):
        results = []
        for level in sorted(self.level_indices.keys(), reverse=True)[:2]:
            index_info = self.level_indices[level]
            scores, indices = self._search(index_info["index"], query_embedding, min(k*3, len(index_info["chunks"])))
            for idx, score in zip(indices[0], scores[0]):
                if idx != -1 and score >= threshold:
                    chunk = index_info["chunks"][idx]
                    results.append((chunk, float(score)))
        return sorted(results, key=lambda x: x[1], reverse=True)[:k]

    def enrich_with_hierarchy(self, chunk: Dict, similarity: float) -> RetrievalResult:
        chunk_id = chunk["chunk_id"]
        parent_chunks, depth = [], 0
        current = chunk["metadata"].get("parent_id")
        while current and current in self.chunk_map:
            parent_chunks.append(self.chunk_map[current])
            current = self.chunk_map[current]["metadata"].get("parent_id")
            depth += 1

        child_chunks = [self.chunk_map[cid] for cid in self.graph.successors(chunk_id) if cid in self.chunk_map]
        return RetrievalResult(
            chunk_id=chunk_id,
            text=chunk["text"],
            similarity=similarity,
            metadata=chunk["metadata"],
            parent_chunks=parent_chunks,
            child_chunks=child_chunks[:3],
            depth=depth
        )
