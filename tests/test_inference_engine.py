"""Tests for InferenceEngine singleton."""

import unittest
import threading
from agentsim import InferenceEngine
from agentsim.inference_engine import InferenceRequest


class TestInferenceEngine(unittest.TestCase):
    """Test cases for InferenceEngine."""
    
    def setUp(self):
        """Reset singleton before each test."""
        InferenceEngine.reset_instance()
    
    def test_singleton_pattern(self):
        """Test that InferenceEngine follows singleton pattern."""
        engine1 = InferenceEngine()
        engine2 = InferenceEngine()
        
        self.assertIs(engine1, engine2, "InferenceEngine should be a singleton")
        self.assertEqual(id(engine1), id(engine2))
    
    def test_initialization(self):
        """Test that InferenceEngine initializes correctly."""
        engine = InferenceEngine()
        
        self.assertIsNotNone(engine.device)
        self.assertEqual(engine.request_count, 0)
    
    def test_basic_inference(self):
        """Test basic inference functionality."""
        engine = InferenceEngine()
        
        request = InferenceRequest(
            prompt="Hello, how are you?",
            system_prompt="You are a helpful assistant.",
            temperature=0.7,
            max_tokens=256
        )
        
        response = engine.infer(request)
        
        self.assertIsNotNone(response)
        self.assertIsNotNone(response.text)
        self.assertEqual(response.prompt, "Hello, how are you?")
        self.assertEqual(engine.request_count, 1)
    
    def test_multiple_inferences(self):
        """Test multiple inference requests."""
        engine = InferenceEngine()
        
        for i in range(5):
            request = InferenceRequest(
                prompt=f"Request {i}",
                system_prompt="Test system prompt"
            )
            response = engine.infer(request)
            self.assertIsNotNone(response)
        
        self.assertEqual(engine.request_count, 5)
    
    def test_batch_inference(self):
        """Test batch inference functionality."""
        engine = InferenceEngine()
        
        requests = [
            InferenceRequest(prompt=f"Prompt {i}", system_prompt="Test")
            for i in range(3)
        ]
        
        responses = engine.batch_infer(requests)
        
        self.assertEqual(len(responses), 3)
        self.assertEqual(engine.request_count, 3)
    
    def test_thread_safety(self):
        """Test that InferenceEngine is thread-safe."""
        engine = InferenceEngine()
        results = []
        
        def inference_task(task_id):
            request = InferenceRequest(
                prompt=f"Task {task_id}",
                system_prompt="Test"
            )
            response = engine.infer(request)
            results.append(response)
        
        # Create multiple threads
        threads = [
            threading.Thread(target=inference_task, args=(i,))
            for i in range(10)
        ]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all requests completed
        self.assertEqual(len(results), 10)
        self.assertEqual(engine.request_count, 10)
    
    def test_response_metadata(self):
        """Test that response includes metadata."""
        engine = InferenceEngine()
        
        request = InferenceRequest(
            prompt="Test prompt",
            system_prompt="Test system",
            temperature=0.8,
            max_tokens=128
        )
        
        response = engine.infer(request)
        
        self.assertIn('request_count', response.metadata)
        self.assertIn('device', response.metadata)
        self.assertIn('temperature', response.metadata)
        self.assertIn('max_tokens', response.metadata)
        self.assertEqual(response.metadata['temperature'], 0.8)
        self.assertEqual(response.metadata['max_tokens'], 128)


if __name__ == '__main__':
    unittest.main()
