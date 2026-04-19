import asyncio
import logging
import time
import tiktoken
from typing import List
from .engine import BaseInferenceEngine
from .metrics import RequestMetrics

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """Orchestrates concurrent LLM requests and captures metrics."""

    def __init__(
        self, 
        engine: BaseInferenceEngine, 
        tokenizer_name: str = "o200k_base"
    ):
        self.engine = engine
        self.tokenizer = tiktoken.get_encoding(
            tokenizer_name
        )

    async def _run_single(
        self, 
        prompt: str, 
        **kwargs
    ) -> RequestMetrics:
        start_time = time.perf_counter()
        first_token_time = None
        full_text = ""
        success = False
        error_msg = None

        try:
            async for chunk in self.engine.stream_inference(
                prompt, 
                **kwargs
            ):
                if first_token_time is None and chunk:
                    first_token_time = time.perf_counter()
                full_text += chunk
            success = True
        except Exception as e:
            logger.error(
                f"Request failed for {self.engine.get_engine_name()}: {e}"
            )
            error_msg = str(e)

        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Fallback if no tokens were generated
        ttft = (first_token_time - start_time) if first_token_time else total_time
        
        # Accurate token counting
        tokens = len(self.tokenizer.encode(full_text))
        
        # Generation time is the time spent *after* the first token was received
        generation_time = (end_time - first_token_time) if first_token_time else 0.0
        
        tps = (tokens / generation_time) if (generation_time > 0 and tokens > 0) else 0.0

        return RequestMetrics(
            ttft_s=ttft,
            tps=tps,
            total_time_s=total_time,
            total_tokens=tokens,
            success=success,
            error=error_msg
        )

    async def run_concurrent(
        self, 
        prompt: str, 
        concurrency: int, 
        **kwargs
    ) -> List[RequestMetrics]:
        """Runs the benchmark with the specified number of concurrent users."""
        logger.info(
            f"Launching {concurrency} concurrent requests to {self.engine.get_engine_name()}"
        )
        tasks = [
            self._run_single(prompt, **kwargs) 
            for _ in range(concurrency)
        ]
        
        return await asyncio.gather(*tasks)
