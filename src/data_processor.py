# src/data_processor.py

import json
import re
from typing import List, Dict

from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("punkt", quiet=True)


class OdysseyDataProcessor:
    """
    Extracts and chunks The Odyssey for RAG usage.
    """

    def __init__(self, html_path: str):
        self.html_path = html_path

    # -------------------------
    # Extraction
    # -------------------------

    def extract_text_from_html(self) -> List[Dict]:
        with open(self.html_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        for tag in soup(["style", "script", "meta", "link", "head"]):
            tag.decompose()

        content = soup.body or soup

        paragraphs = []
        current_chapter = "Unknown Chapter"
        paragraph_id = 0

        for el in content.find_all(["h1", "h2", "h3", "h4", "p"]):
            text = el.get_text(" ", strip=True)

            if not text:
                continue

            if el.name.startswith("h") and (
                "book" in text.lower() or "chapter" in text.lower()
            ):
                current_chapter = text
                continue

            if len(text.split()) < 20:
                continue

            paragraphs.append(
                {
                    "id": paragraph_id,
                    "text": text,
                    "chapter": current_chapter,
                }
            )
            paragraph_id += 1

        return paragraphs

    # -------------------------
    # Chunking
    # -------------------------

    def create_chunks(
        self,
        paragraphs: List[Dict],
        max_words: int = 220,
        overlap_sentences: int = 2,
    ) -> List[Dict]:

        chunks: List[Dict] = []
        chunk_id = 0
        buffer: List[str] = []
        buffer_words = 0
        start_para = 0

        for i, para in enumerate(paragraphs):
            sentences = sent_tokenize(para["text"])

            for sent in sentences:
                words = word_tokenize(sent)

                if buffer_words + len(words) > max_words and buffer:
                    chunk_text = " ".join(buffer)

                    chunks.append(
                        {
                            "chunk_id": f"chunk_{chunk_id:04d}",
                            "text": chunk_text,
                            "metadata": {
                                "chapter": paragraphs[start_para]["chapter"],
                                "paragraph_range": f"{start_para}-{i}",
                                "source": "Project Gutenberg – The Odyssey",
                            },
                        }
                    )

                    chunk_id += 1
                    buffer = buffer[-overlap_sentences:]
                    buffer_words = sum(len(word_tokenize(s)) for s in buffer)
                    start_para = i

                buffer.append(sent)
                buffer_words += len(words)

        if buffer:
            chunks.append(
                {
                    "chunk_id": f"chunk_{chunk_id:04d}",
                    "text": " ".join(buffer),
                    "metadata": {
                        "chapter": paragraphs[start_para]["chapter"],
                        "paragraph_range": f"{start_para}-{len(paragraphs) - 1}",
                        "source": "Project Gutenberg – The Odyssey",
                    },
                }
            )

        return chunks

    # -------------------------
    # Pipeline
    # -------------------------

    def process_and_save(self, output_path: str) -> List[Dict]:
        paragraphs = self.extract_text_from_html()
        chunks = self.create_chunks(paragraphs)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        return chunks
