import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from google.generativeai.client import configure
from google.generativeai.generative_models import GenerativeModel

load_dotenv()

class GeminiGenerator:
    """
    Thin wrapper around the Gemini API optimized for RAG-style usage.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not found in environment variables")

        configure(api_key=api_key)

        self.model_name = model_name
        self.model = GenerativeModel(model_name)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        """
        Generate text from Gemini.
        Low temperature is intentional for factual / RAG responses.
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_new_tokens,
                    "temperature": temperature,
                },
            )

            text = getattr(response, "text", None)
            if not text:
                return "I couldn’t generate a meaningful answer from the available information."

            return text.strip()

        except Exception as e:
            return f"Generation error: {e}"


    def format_context(self, retrieved_chunks: List[Dict]) -> str:
        """
        Convert retrieved chunks into a readable context block.
        """
        if not retrieved_chunks:
            return ""

        context_parts: List[str] = []

        for chunk in retrieved_chunks:
            metadata = chunk.get("metadata", {})
            chapter = metadata.get("chapter", "Unknown chapter")
            paragraph_range = metadata.get("paragraph_range", "Unknown")

            text = chunk.get("text", "").strip()
            if not text:
                continue

            context_parts.append(
                f"[{chapter}, paragraphs {paragraph_range}]\n{text}"
            )

        return "\n\n".join(context_parts)

    def create_prompt(self, context: str, question: str) -> str:
        """
        Prompt optimized for:
        - grounded answers when context exists
        - sensible fallback for high-level literary questions
        - zero hallucinated citations
        """
        return (
            "You are a knowledgeable literature assistant answering questions about Homer's *Odyssey*.\n\n"
            "Use the provided context as your primary source.\n"
            "If the context clearly answers the question, base your answer on it.\n"
            "If the context is only partially relevant, you may answer using general knowledge of the Odyssey,\n"
            "but do NOT invent quotes, paragraph numbers, or citations.\n"
            "If the question is unrelated to the Odyssey, say so clearly.\n\n"
            "Context:\n"
            f"{context if context else '(No relevant context retrieved)'}\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Answer:"
        )
