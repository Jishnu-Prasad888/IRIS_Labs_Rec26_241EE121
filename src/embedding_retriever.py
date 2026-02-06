# src/embedding_retriever.py

import json
from typing import List, Dict, Tuple, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Dict] = []

        # Explicit cast fixes Pylance
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    # -------------------------
    # FAISS helpers (Pylance-safe)
    # -------------------------

    @staticmethod
    def _normalize(x: np.ndarray) -> None:
        faiss.normalize_L2(x)

    @staticmethod
    def _search(index: faiss.IndexFlatIP, x: np.ndarray, k: int):
        return index.search(x, k)

    # -------------------------
    # Index creation
    # -------------------------

    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings_np = np.asarray(embeddings, dtype=np.float32)

        self._normalize(embeddings_np)

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings_np)

        return embeddings_np

    def save_index(self, index_path: str, chunks_path: str) -> None:
        if self.index is None:
            raise RuntimeError("No FAISS index to save.")

        faiss.write_index(self.index, f"{index_path}/index.faiss")

        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)

    def load_index(self, index_path: str, chunks_path: str) -> None:
        self.index = faiss.read_index(f"{index_path}/index.faiss")

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    # -------------------------
    # Retrieval
    # -------------------------

    def retrieve(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.3,
    ) -> Tuple[List[Dict], List[float]]:

        if self.index is None:
            raise RuntimeError("FAISS index not initialized.")

        query_embedding = self.model.encode([query])
        query_np = np.asarray(query_embedding, dtype=np.float32)

        self._normalize(query_np)

        scores, indices = self._search(self.index, query_np, k)

        results: List[Dict] = []
        similarities: List[float] = []

        for idx, score in zip(indices[0], scores[0]):
            if idx == -1 or score < threshold:
                continue

            chunk = self.chunks[idx]
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "similarity": float(score),
                }
            )
            similarities.append(float(score))

        return results, similarities
