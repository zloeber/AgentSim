"""
InferenceEngine - Singleton class for serial GPU processing.

This module provides a singleton InferenceEngine that ensures all inference
requests are processed serially on a single local GPU, preventing resource
conflicts and managing GPU memory efficiently.
"""

import threading
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class InferenceRequest:
    """Represents a single inference request."""
    prompt: str
    system_prompt: str
    temperature: float = 0.7
    max_tokens: int = 512


@dataclass
class InferenceResponse:
    """Represents an inference response."""
    text: str
    prompt: str
    metadata: Dict[str, Any]


class InferenceEngine:
    """
    Singleton InferenceEngine for serial GPU processing.
    
    This class ensures that all inference operations are processed serially
    on a single GPU, preventing resource conflicts. It uses the singleton
    pattern to ensure only one instance exists throughout the application.
    """
    
    _instance: Optional['InferenceEngine'] = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls):
        """Ensure only one instance is created (singleton pattern)."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the InferenceEngine (only once)."""
        # Check initialization status under lock for thread safety
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        with self._lock:
            # Double-check after acquiring lock
            if hasattr(self, '_initialized') and self._initialized:
                return
                
            self._initialized = True
            self._inference_lock = threading.Lock()
            self._model = None
            self._device = None
            self._request_count = 0
            
            # Mock initialization - in a real implementation, this would
            # initialize the actual GPU model
            self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize the GPU model.
        
        In a real implementation, this would load a model like:
        - transformers library models (GPT, LLaMA, etc.)
        - vLLM for optimized inference
        - Custom GPU inference backend
        
        For now, this is a mock implementation.
        """
        # Mock implementation
        self._device = "cuda:0"  # or "cpu" if no GPU available
        self._model = "mock_model"
        print(f"[InferenceEngine] Initialized on device: {self._device}")
    
    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """
        Process an inference request serially.
        
        Args:
            request: InferenceRequest containing the prompt and parameters
            
        Returns:
            InferenceResponse containing the generated text
            
        Note:
            This method is thread-safe and processes requests serially
            to prevent GPU resource conflicts.
        """
        with self._inference_lock:
            self._request_count += 1
            
            # In a real implementation, this would call the actual model
            # For now, return a mock response
            full_prompt = f"{request.system_prompt}\n\nUser: {request.prompt}"
            
            # Mock inference - in reality, this would be:
            # response = self._model.generate(full_prompt, max_tokens=request.max_tokens)
            mock_response = self._mock_inference(full_prompt, request)
            
            return InferenceResponse(
                text=mock_response,
                prompt=request.prompt,
                metadata={
                    "request_count": self._request_count,
                    "device": self._device,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens
                }
            )
    
    def _mock_inference(self, full_prompt: str, request: InferenceRequest) -> str:
        """
        Mock inference function for demonstration.
        
        In a real implementation, this would be replaced with actual
        model inference on GPU.
        """
        # Simple mock that echoes back a response
        return f"[Mock Response to: '{request.prompt[:50]}...']"
    
    def batch_infer(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """
        Process multiple inference requests serially.
        
        Args:
            requests: List of InferenceRequest objects
            
        Returns:
            List of InferenceResponse objects
        """
        responses = []
        for request in requests:
            response = self.infer(request)
            responses.append(response)
        return responses
    
    @property
    def request_count(self) -> int:
        """Get the total number of processed requests."""
        return self._request_count
    
    @property
    def device(self) -> str:
        """Get the device being used for inference."""
        return self._device
    
    @classmethod
    def reset_instance(cls):
        """
        Reset the singleton instance (primarily for testing).
        
        Warning: This should only be used in test scenarios.
        """
        with cls._lock:
            cls._instance = None
