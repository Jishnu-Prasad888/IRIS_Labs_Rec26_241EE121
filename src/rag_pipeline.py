# src/enhanced_rag_pipeline.py
from typing import Dict, List, Optional
import logging
from .hierarchical_retriever import HierarchicalRetriever, RetrievalResult
from .data_processor import AdvancedDataProcessor
from .llm_generator import GeminiGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedRAGPipeline:
    """
    RAG pipeline with logical segmentation and hierarchical retrieval
    """
    
    def __init__(
        self,
        content_path: str,
        content_type: str = "pdf",
        model_name: str = "gemini-2.5-flash-lite"
    ):
        self.processor = AdvancedDataProcessor(content_path, content_type)
        self.retriever = HierarchicalRetriever()
        self.generator = GeminiGenerator(model_name)
        
        # Process content
        logger.info("Processing content with logical segmentation...")
        if content_type.lower() == "pdf":
            nodes = self.processor.process_pdf()
        else:
            nodes = self.processor.process_html()
        
        # Create logical chunks
        self.chunks = self.processor.create_logical_chunks()
        logger.info(f"Created {len(self.chunks)} logical chunks")
        
        # Build hierarchical index
        logger.info("Building hierarchical index...")
        self.retriever.create_multi_level_embeddings(self.chunks)
        
        # Save hierarchy for visualization/debugging
        self.processor.save_hierarchy("data/processed/hierarchy.json")
    
    def answer_question(
        self,
        question: str,
        k: int = 5,
        threshold: float = 0.25,
        retrieval_strategy: str = "hybrid"
    ) -> Dict:
        """
        Answer question using hierarchical retrieval
        """
        # Classify question type
        question_type = self._classify_question(question)
        
        # Adjust retrieval strategy based on question type
        if question_type == "overview":
            strategy = "top_down"
            k = min(k, 3)  # Fewer, higher-level chunks for overview
        elif question_type == "detail":
            strategy = "bottom_up"
            k = min(k, 7)  # More, detailed chunks
        else:
            strategy = retrieval_strategy
        
        # Hierarchical retrieval
        retrieved_results = self.retriever.hierarchical_retrieve(
            question,
            k=k,
            threshold=threshold,
            strategy=strategy
        )
        
        logger.info(f"Retrieved {len(retrieved_results)} chunks using {strategy} strategy")
        
        if not retrieved_results:
            return {
                "answer": "No relevant information found in the document.",
                "sources": [],
                "retrieval_strategy": strategy,
                "question_type": question_type
            }
        
        # Format context with hierarchy information
        context = self._format_hierarchical_context(retrieved_results, question_type)
        
        # Generate answer with enhanced prompt
        prompt = self._create_enhanced_prompt(context, question, question_type)
        answer = self.generator.generate(prompt)
        
        # Extract sources with hierarchy information
        sources = []
        for result in retrieved_results:
            source_info = {
                "chunk_id": result.chunk_id,
                "text_preview": result.text[:200] + "..." if len(result.text) > 200 else result.text,
                "similarity": result.similarity,
                "depth": result.depth,
                "level": result.metadata.get("level", "unknown"),
                "parent_chunks": len(result.parent_chunks),
                "child_chunks": len(result.child_chunks)
            }
            sources.append(source_info)
        
        return {
            "answer": answer,
            "sources": sources,
            "retrieval_strategy": strategy,
            "question_type": question_type,
            "chunks_retrieved": len(retrieved_results)
        }
    
    def _classify_question(self, question: str) -> str:
        """Classify question type to determine retrieval strategy"""
        question_lower = question.lower()
        
        # Overview questions
        overview_keywords = [
            "overview", "summary", "introduction", "what is", "explain",
            "describe", "tell me about", "background"
        ]
        
        # Detail questions
        detail_keywords = [
            "specific", "detail", "exact", "precise", "section",
            "subsection", "clause", "paragraph", "line"
        ]
        
        # Comparison questions
        comparison_keywords = [
            "compare", "difference", "similar", "versus", "vs",
            "contrast", "relationship between"
        ]
        
        for keyword in overview_keywords:
            if keyword in question_lower:
                return "overview"
        
        for keyword in detail_keywords:
            if keyword in question_lower:
                return "detail"
        
        for keyword in comparison_keywords:
            if keyword in question_lower:
                return "comparison"
        
        return "general"
    
    def _format_hierarchical_context(self, results: List[RetrievalResult], question_type: str) -> str:
        """Format retrieved chunks with hierarchy information"""
        context_parts = []
        
        for i, result in enumerate(results):
            # Add hierarchy indicator
            indent = "  " * result.depth
            level_indicator = f"[Level {result.metadata.get('level', '?')}]"
            
            # Include parent context for overview questions
            if question_type == "overview" and result.parent_chunks:
                parent_context = "\n".join([
                    f"  Parent: {p['text'][:100]}..." 
                    for p in result.parent_chunks[:2]
                ])
                context_parts.append(f"{indent}{level_indicator} (Context from parent sections)")
                context_parts.append(parent_context)
            
            # Main chunk text
            context_parts.append(f"{indent}{level_indicator} {result.text}")
            
            # Include relevant children for detail questions
            if question_type == "detail" and result.child_chunks:
                child_context = "\n".join([
                    f"    • {c['text'][:150]}..." 
                    for c in result.child_chunks[:3]
                ])
                context_parts.append(f"{indent}  Relevant details:")
                context_parts.append(child_context)
            
            context_parts.append("")  # Empty line between chunks
        
        return "\n".join(context_parts)
    
    def _create_enhanced_prompt(
        self,
        context: str,
        question: str,
        question_type: str
    ) -> str:
        """Create enhanced prompt based on question type and hierarchy"""
        
        base_prompt = (
            "You are a knowledgeable assistant answering questions about regulatory documents.\n\n"
            "The following context is organized hierarchically (indentation shows document structure):\n"
            f"{context}\n\n"
            "Question type: {question_type}\n"
            "Question: {question}\n\n"
        )
        
        # Add specific instructions based on question type
        if question_type == "overview":
            instructions = (
                "Provide a comprehensive overview based on the higher-level sections. "
                "Focus on main themes, structure, and key points. "
                "Synthesize information from multiple related sections."
            )
        elif question_type == "detail":
            instructions = (
                "Provide precise, detailed information. "
                "Reference specific sections, subsections, or clauses when possible. "
                "Include exact requirements, specifications, or conditions."
            )
        elif question_type == "comparison":
            instructions = (
                "Compare and contrast different sections or requirements. "
                "Highlight similarities and differences clearly. "
                "Organize your answer to show comparisons systematically."
            )
        else:
            instructions = (
                "Answer based on the most relevant sections. "
                "If the context provides clear information, base your answer on it. "
                "Otherwise, acknowledge the limitations."
            )
        
        return base_prompt + instructions + "\n\nAnswer:"