import os
import json
import sys
import time
from dotenv import load_dotenv

from src.data_processor import OdysseyDataProcessor
from src.embedding_retriever import EmbeddingRetriever
from src.rag_pipeline import RAGPipeline
from src.llm_generator import GeminiGenerator

load_dotenv()

class OdysseyRAGTester:
    def __init__(self):
        self.odyssey_html = "data/raw/odyssey.html"
        self.chunks_path = "data/processed/chunks.json"
        self.embeddings_path = "data/embeddings"
        self.metrics_file = "test_metrics.json"
        
        self.test_questions = [
            "Who is the main character of the Odyssey?",
            "What is Penelope doing while Odysseus is away?",
            "Who is Telemachus?",
            "What is Odysseus trying to do throughout the story?",
            "How many years does Odysseus wander?",
            "Who are the suitors and what do they want?",
            "Who helps Telemachus?",
            "Where is Odysseus trying to return to?",
            "What challenges does Odysseus face on his journey?",
            "Who is Athena and what role does she play?"
        ]
        
        self.metrics = {}

    def save_metrics(self):
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Metrics saved to {self.metrics_file}")

    def setup_directories(self):
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs(self.embeddings_path, exist_ok=True)

    def check_odyssey_html(self):
        result = {}
        result['exists'] = os.path.exists(self.odyssey_html)
        if result['exists']:
            result['file_size_kb'] = os.path.getsize(self.odyssey_html) / 1024
        else:
            result['file_size_kb'] = 0
        self.metrics['odyssey_html'] = result
        return result['exists']

    def test_api_key(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        self.metrics['api_key_present'] = bool(api_key)
        if not api_key:
            return False, None
        try:
            generator = GeminiGenerator()
            self.metrics['gemini_init_success'] = True
            return True, generator
        except Exception:
            self.metrics['gemini_init_success'] = False
            return False, None

    def process_odyssey_data(self):
        try:
            processor = OdysseyDataProcessor(self.odyssey_html)
            paragraphs = processor.extract_text_from_html()
            chunks = processor.create_chunks(paragraphs)
            with open(self.chunks_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            
            # Metrics
            self.metrics['data_processing'] = {
                'paragraph_count': len(paragraphs),
                'chunk_count': len(chunks),
                'sample_chunk_word_count': chunks[0]['metadata']['word_count'] if chunks else 0
            }
            return True
        except Exception:
            self.metrics['data_processing'] = {'paragraph_count': 0, 'chunk_count': 0, 'sample_chunk_word_count': 0}
            return False

    def create_embeddings(self):
        try:
            with open(self.chunks_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            retriever = EmbeddingRetriever()
            start_time = time.time()
            embeddings = retriever.create_embeddings(chunks)
            elapsed = time.time() - start_time
            retriever.save_index(self.embeddings_path, self.chunks_path)

            self.metrics['embeddings'] = {
                'num_embeddings': len(embeddings),
                'embedding_shape': embeddings.shape[1] if embeddings is not None else 0,
                'time_sec': elapsed
            }
            return True
        except Exception:
            self.metrics['embeddings'] = {'num_embeddings': 0, 'embedding_shape': 0, 'time_sec': 0}
            return False

    def test_rag_pipeline(self):
        try:
            pipeline = RAGPipeline(self.chunks_path, self.embeddings_path)
            question_metrics = []
            for question in self.test_questions[:5]:
                start_time = time.time()
                answer_data = pipeline.answer_question(question, k=3, threshold=0.2)
                elapsed = time.time() - start_time
                top_similarity = answer_data['sources'][0]['similarity'] if answer_data['sources'] else 0
                question_metrics.append({
                    'question': question,
                    'response_time_sec': elapsed,
                    'sources_count': len(answer_data['sources']),
                    'top_similarity': top_similarity
                })
            
            avg_time = sum(q['response_time_sec'] for q in question_metrics)/len(question_metrics)
            avg_sources = sum(q['sources_count'] for q in question_metrics)/len(question_metrics)
            avg_similarity = sum(q['top_similarity'] for q in question_metrics)/len(question_metrics)
            
            self.metrics['rag_pipeline'] = {
                'questions': question_metrics,
                'avg_response_time_sec': avg_time,
                'avg_sources': avg_sources,
                'avg_top_similarity': avg_similarity
            }
            return True
        except Exception:
            self.metrics['rag_pipeline'] = {'questions': [], 'avg_response_time_sec': 0, 'avg_sources': 0, 'avg_top_similarity': 0}
            return False

    def run_comprehensive_test(self):
        self.setup_directories()
        results = {}
        
        results['odyssey_html'] = self.check_odyssey_html()
        results['api_key'] = self.test_api_key()[0]
        results['data_processing'] = self.process_odyssey_data()
        results['embeddings'] = self.create_embeddings()
        results['rag_pipeline'] = self.test_rag_pipeline()
        
        self.metrics['final_summary'] = {
            'total_tests': len(results),
            'passed_tests': sum(1 for k,v in results.items() if v)
        }
        
        self.save_metrics()
        return results

if __name__ == "__main__":
    tester = OdysseyRAGTester()
    tester.run_comprehensive_test()
