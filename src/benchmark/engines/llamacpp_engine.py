import json
import httpx
from typing import AsyncGenerator
from ..engine import BaseInferenceEngine

class LlamaCppEngine(BaseInferenceEngine):
    """llama.cpp engine implementation using the native server API."""

    def get_engine_name(self) -> str:
        return "LlamaCpp"

    async def stream_inference(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        url = f"{self.endpoint_url}/completion"
        payload = {
            "prompt": prompt,
            "stream": True,
            "n_predict": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.0),
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    raise RuntimeError(f"Llama.cpp API Error {response.status_code}: {error_body.decode()}")

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[len("data: "):]
                        try:
                            data = json.loads(data_str)
                            chunk = data.get("content", "")
                            if chunk:
                                yield chunk
                            if data.get("stop", False):
                                break
                        except (json.JSONDecodeError, KeyError):
                            continue
