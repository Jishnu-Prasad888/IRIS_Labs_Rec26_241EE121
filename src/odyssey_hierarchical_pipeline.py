# src/odyssey_hierarchical_pipeline.py (UPDATED)
from typing import Dict, List, Optional, Tuple
import logging
from .html_hierarchical_processor import HTMLHierarchicalProcessor
from .simple_hierarchical_retriever import SimpleHierarchicalRetriever, HierarchicalResult
from .llm_generator import GeminiGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OdysseyHierarchicalPipeline:
    """
    Hierarchical RAG pipeline for The Odyssey HTML
    """
    
    def __init__(
        self,
        html_path: str = "data/raw/odyssey.html",
        model_name: str = "gemini-2.5-flash-lite"
    ):
        self.html_path = html_path
        self.processor = HTMLHierarchicalProcessor(html_path)
        self.retriever = SimpleHierarchicalRetriever()
        self.generator = GeminiGenerator(model_name)
        
        # Process the HTML once
        self.nodes = self.processor.process_html()
        self.chunks = self.processor.create_semantic_chunks()
        
        # Create embeddings
        self.retriever.create_embeddings(self.chunks)
        
        # Save hierarchy for reference
        self.processor.save_hierarchy("data/processed/hierarchy.json")
        
        logger.info(f"Pipeline initialized with {len(self.chunks)} semantic chunks")
    
    def answer_question(
        self,
        question: str,
        k: int = 5,
        threshold: float = 0.25,
        strategy: str = "adaptive"  # adaptive, overview, detail, character, structural
    ) -> Dict:
        """
        Answer question with adaptive hierarchical retrieval
        """
        # Analyze question type
        question_type = self._analyze_question(question)
        logger.info(f"Question type: {question_type}")
        
        # Get retrieval parameters based on question type
        retrieval_kwargs = self._get_retrieval_kwargs(question_type, k, threshold)
        
        # Perform hierarchical retrieval
        retrieved_results = self.retriever.retrieve_with_context(
            query=question,
            **retrieval_kwargs
        )
        
        logger.info(f"Retrieved {len(retrieved_results)} results")
        
        if not retrieved_results:
            return {
                "answer": "I couldn't find relevant information about that in The Odyssey.",
                "sources": [],
                "question_type": question_type,
                "strategy": strategy,
                "retrieval_params": retrieval_kwargs
            }
        
        # Format context with hierarchy information
        context = self._format_context_with_hierarchy(retrieved_results, question_type)
        
        # Create prompt
        prompt = self._create_odyssey_prompt(context, question, question_type)
        
        # Generate answer
        answer = self.generator.generate(prompt, max_new_tokens=512)
        
        # Prepare sources for display
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
        
        return {
            "answer": answer.strip(),
            "sources": sources,
            "question_type": question_type,
            "strategy": strategy,
            "retrieval_params": retrieval_kwargs,
            "chunks_retrieved": len(retrieved_results)
        }
    
    def _analyze_question(self, question: str) -> str:
        """Analyze question to determine retrieval strategy"""
        question_lower = question.lower()
        
        # Odyssey character questions
        odyssey_characters = {
            "odysseus", "penelope", "telemachus", "athena", "poseidon",
            "circe", "calypso", "polyphemus", "nestor", "menelaus",
            "agamemnon", "nausicaa", "eumaeus", "antinous", "eurymachus"
        }
        
        # Overview questions
        overview_indicators = {
            "what is", "who is", "describe", "explain", "tell me about",
            "overview", "summary", "introduction", "background"
        }
        
        # Detail questions
        detail_indicators = {
            "how did", "when did", "where did", "why did", "what happened",
            "specific", "exact", "detailed", "specifically"
        }
        
        # Book/chapter questions
        book_indicators = {
            "book", "chapter", "part", "canto", "section"
        }
        
        # Check for character questions
        for character in odyssey_characters:
            if character in question_lower:
                return "character"
        
        # Check for overview questions
        for indicator in overview_indicators:
            if indicator in question_lower:
                return "overview"
        
        # Check for detail questions
        for indicator in detail_indicators:
            if indicator in question_lower:
                return "detail"
        
        # Check for book/chapter questions
        for indicator in book_indicators:
            if indicator in question_lower:
                return "structural"
        
        return "general"
    
    def _get_retrieval_kwargs(self, question_type: str, k: int, threshold: float) -> Dict:
        """Get retrieval keyword arguments based on question type"""
        # Default parameters
        kwargs = {
            "k": k,
            "threshold": threshold,
            "include_parent": True,
            "include_children": False
        }
        
        if question_type == "overview":
            kwargs.update({
                "k": min(k, 3),  # Fewer, broader chunks
                "threshold": threshold * 0.9,  # Slightly lower threshold
                "include_parent": True,
                "include_children": False
            })
        elif question_type == "detail":
            kwargs.update({
                "k": min(k, 7),  # More, detailed chunks
                "threshold": threshold,
                "include_parent": True,
                "include_children": True  # Include relevant children
            })
        elif question_type == "character":
            kwargs.update({
                "k": k,
                "threshold": threshold,
                "include_parent": True,
                "include_children": False
            })
        elif question_type == "structural":
            kwargs.update({
                "k": k,
                "threshold": threshold * 0.8,  # Lower threshold for structural
                "include_parent": False,  # Don't need parent for book/chapter
                "include_children": True  # Include sections
            })
        
        return kwargs
    
    def _format_context_with_hierarchy(self, results: List[HierarchicalResult], question_type: str) -> str:
        """Format retrieved results with hierarchy indicators"""
        if not results:
            return "No relevant context found."
        
        context_parts = []
        
        for i, result in enumerate(results):
            # Add hierarchy indicator
            level = result.metadata.get("level", 4)
            indent = "  " * (level - 1) if level > 1 else ""
            level_label = f"[Level {level}: {result.metadata.get('chunk_type', 'content')}]"
            
            # Add parent context if available and relevant
            if result.parent_text and question_type not in ["structural", "overview"]:
                context_parts.append(f"{indent}Context from parent: {result.parent_text}")
            
            # Add main text
            text_to_show = result.text
            if len(text_to_show) > 500:  # Truncate very long chunks
                text_to_show = text_to_show[:500] + "... [truncated]"
            
            context_parts.append(f"{indent}{level_label}")
            context_parts.append(f"{indent}{text_to_show}")
            
            # Add separator
            if i < len(results) - 1:
                context_parts.append("-" * 40)
        
        return "\n".join(context_parts)
    
    def _create_odyssey_prompt(self, context: str, question: str, question_type: str) -> str:
        """Create prompt tailored for Odyssey questions"""
        
        # Add instructions based on question type
        instructions = ""
        if question_type == "overview":
            instructions = "Provide a comprehensive overview. Focus on main themes and key points."
        elif question_type == "detail":
            instructions = "Provide specific, detailed information. Include exact events and descriptions."
        elif question_type == "character":
            instructions = "Focus on character traits, actions, and significance in the story."
        elif question_type == "structural":
            instructions = "Focus on the structure, organization, and key sections."
        else:
            instructions = "Provide a clear, accurate answer based on the context."
        
        prompt = f"""You are an expert on Homer's Odyssey. Answer the question based on the provided context.

Context (organized by document structure - indentation shows hierarchy):
{context}

Question: {question}

{instructions}

Guidelines:
1. Base your answer primarily on the context provided.
2. If the context doesn't fully answer, you may supplement with general knowledge.
3. Be specific about characters, events, and locations when possible.
4. Do not invent or hallucinate details not present in the context.
5. If unsure, acknowledge the limitations of the available information.

Answer in clear, concise English:"""

        return prompt
    
    def save_system(self, base_path: str = "data/processed"):
        """Save the entire RAG system state"""
        import os
        os.makedirs(base_path, exist_ok=True)
        
        # Save index and chunks
        self.retriever.save_index(
            index_path=f"{base_path}/faiss_index",
            chunks_path=f"{base_path}/semantic_chunks.json",
            hierarchy_path=f"{base_path}/hierarchy_map.json"
        )
        
        print(f"System saved to {base_path}")
    
    def load_system(self, base_path: str = "data/processed"):
        """Load the RAG system state"""
        self.retriever.load_index(
            index_path=f"{base_path}/faiss_index",
            chunks_path=f"{base_path}/semantic_chunks.json",
            hierarchy_path=f"{base_path}/hierarchy_map.json"
        )
        
        print(f"System loaded from {base_path}")
    
    def test_retrieval(self, question: str, k: int = 3):
        """Test retrieval without generation"""
        question_type = self._analyze_question(question)
        kwargs = self._get_retrieval_kwargs(question_type, k, 0.25)
        
        results = self.retriever.retrieve_with_context(query=question, **kwargs)
        
        print(f"\nQuestion: {question}")
        print(f"Type: {question_type}")
        print(f"Retrieval params: {kwargs}")
        print(f"Results: {len(results)}")
        
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"Chunk ID: {result.chunk_id}")
            print(f"Similarity: {result.similarity:.3f}")
            print(f"Level: {result.metadata.get('level')}")
            print(f"Type: {result.metadata.get('chunk_type')}")
            if result.parent_text:
                print(f"Parent context: {result.parent_text}")
            print(f"Text preview: {result.text[:200]}...")