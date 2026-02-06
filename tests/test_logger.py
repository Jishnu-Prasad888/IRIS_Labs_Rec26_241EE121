# tests/test_logger.py
import json
import os
from datetime import datetime
from src.rag_pipeline import RAGPipeline

class TestLogger:
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.test_results = []
        
    def run_tests(self, test_questions_path: str):
        """Run test questions and log results"""
        with open(test_questions_path, 'r') as f:
            test_cases = json.load(f)
        
        for test in test_cases:
            print(f"\nTesting: {test['question']}")
            print(f"Category: {test['category']}")
            
            result = self.pipeline.answer_question(test['question'])
            
            test_result = {
                "timestamp": datetime.now().isoformat(),
                "question": test['question'],
                "category": test['category'],
                "expected_type": test.get("expected_type", "unknown"),
                "retrieved_chunks": result['retrieved_chunks'],
                "similarities": result['similarities'],
                "answer": result['answer'],
                "has_rejection": "not relevant" in result['answer'].lower(),
                "sources_count": len(result['sources'])
            }
            
            self.test_results.append(test_result)
            
            # Print result
            print(f"Retrieved chunks: {len(result['retrieved_chunks'])}")
            print(f"Answer: {result['answer'][:200]}...")
        
        # Save results
        self.save_results()
    
    def save_results(self, output_path: str = "test_results.json"):
        """Save test results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)

# Test questions
test_questions = [
    # Direct factual
    {"question": "Who is the main character of The Odyssey?", "category": "direct"},
    {"question": "What is the name of Odysseus's wife?", "category": "direct"},
    {"question": "Who is Telemachus?", "category": "direct"},
    
    # Indirect
    {"question": "How does Odysseus escape from Polyphemus?", "category": "indirect"},
    {"question": "What happens to Odysseus's men who eat the lotus?", "category": "indirect"},
    
    # Follow-up
    {"question": "What did Circe turn Odysseus's men into?", "category": "follow-up"},
    {"question": "How long was Odysseus away from Ithaca?", "category": "follow-up"},
    
    # Ambiguous
    {"question": "What are the suitors doing in Odysseus's palace?", "category": "ambiguous"},
    {"question": "Describe the character of Penelope.", "category": "ambiguous"},
    
    # Unrelated
    {"question": "Who won the World Series in 2023?", "category": "unrelated"},
    {"question": "Explain quantum mechanics.", "category": "unrelated"},
    {"question": "What is the capital of France?", "category": "unrelated"}
]