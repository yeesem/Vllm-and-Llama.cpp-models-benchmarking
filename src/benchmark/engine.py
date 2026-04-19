from abc import ABC, abstractmethod
from typing import AsyncGenerator

class BaseInferenceEngine(ABC):
    """Abstract base class for all LLM inference engines."""
    
    def __init__(self, endpoint_url: str, model_name: str | None = None):
        self.endpoint_url = endpoint_url
        self.model_name = model_name

    @abstractmethod
    async def stream_inference(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        Streams text chunks from the inference engine.
        
        Args:
            prompt: The input text for generation.
            **kwargs: Additional parameters like temperature, max_tokens, etc.
            
        Yields:
            String chunks of generated text.
        """
        pass

    @abstractmethod
    def get_engine_name(self) -> str:
        """Returns the identifier for the engine."""
        pass
