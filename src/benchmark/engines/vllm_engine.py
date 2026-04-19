import json
import httpx
from typing import AsyncGenerator
from ..engine import BaseInferenceEngine

class VLLMEngine(BaseInferenceEngine):
    """vLLM engine implementation using the OpenAI-compatible API."""

    def get_engine_name(self) -> str:
        return "vLLM"

    async def stream_inference(
        self, 
        prompt: str, 
        **kwargs
    ) -> AsyncGenerator[str, None]:
        url = f"{self.endpoint_url}/v1/completions"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.0),
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    raise RuntimeError(f"vLLM API Error {response.status_code}: {error_body.decode()}")

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[len("data: "):]
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            chunk = data["choices"][0].get("text", "")
                            if chunk:
                                yield chunk
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
