from typing import Dict
import logging

from .embedding_retriever import EmbeddingRetriever
from .llm_generator import GeminiGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Intent-aware Retrieval-Augmented Generation pipeline for Homer's Odyssey.
    """



    ODYSSEY_CHARACTERS = {
        "odysseus",
        "penelope",
        "telemachus",
        "athena",
        "poseidon",
        "circe",
        "calypso",
    }

    OVERVIEW_PHRASES = {
        "tell me about",
        "what is",
        "who is",
        "describe",
        "overview",
    }



    def __init__(self, chunks_path: str, embeddings_path: str):
        self.retriever = EmbeddingRetriever()
        self.generator = GeminiGenerator()

        self.retriever.load_index(embeddings_path, chunks_path)


    @classmethod
    def is_character_question(cls, question: str) -> bool:
        q = question.lower()
        return any(name in q for name in cls.ODYSSEY_CHARACTERS)

    @classmethod
    def is_overview_question(cls, question: str) -> bool:
        q = question.lower()
        return any(phrase in q for phrase in cls.OVERVIEW_PHRASES)

    @classmethod
    def is_global_odyssey_question(cls, question: str) -> bool:
        q = question.lower().strip()
        return "odyssey" in q and len(q.split()) <= 5


    def answer_question(
        self,
        question: str,
        k: int = 5,
        threshold: float = 0.25,
    ) -> Dict:
        """
        Answer a question using intent-aware routing.
        """

        if (
            self.is_overview_question(question)
            or self.is_character_question(question)
            or self.is_global_odyssey_question(question)
        ):
            answer = self.generator.generate(
                f"Give a clear, concise explanation of {question.strip('?')} "
                "based on Homer's Odyssey."
            )
            return {
                "answer": answer,
                "sources": [],
            }

        retrieved_chunks, similarities = self.retriever.retrieve(
            question, k=k, threshold=threshold
        )

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")

        if not retrieved_chunks:
            return {
                "answer": "This question is not relevant to the provided text.",
                "sources": [],
            }

        context = self.generator.format_context(retrieved_chunks)
        prompt = self.generator.create_prompt(context, question)
        answer = self.generator.generate(prompt)

        if not answer.strip():
            answer = "This question is not relevant to the provided text."

        sources = [
            {
                "chapter": chunk["metadata"]["chapter"],
                "paragraph_range": chunk["metadata"]["paragraph_range"],
                "similarity": chunk["similarity"],
            }
            for chunk in retrieved_chunks
        ]

        return {
            "answer": answer,
            "sources": sources,
        }
