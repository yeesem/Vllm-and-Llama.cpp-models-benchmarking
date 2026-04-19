import asyncio
import logging
import sys
from typing import List
from src.benchmark.engines.vllm_engine import VLLMEngine
from src.benchmark.engines.llamacpp_engine import LlamaCppEngine
from src.benchmark.runner import BenchmarkRunner
from src.benchmark.metrics import aggregate_results, save_to_csv, save_to_json, AggregatedMetrics

# Configuration
CONCURRENCY_LEVELS = [1, 5, 10, 20]
PROMPT = "Write a comprehensive 500-word essay about the history of artificial intelligence."
VLLM_URL = "http://localhost:8000"
VLLM_MODEL = "facebook/opt-125m"
LLAMACPP_URL = "http://localhost:8080"
OUTPUT_CSV = "benchmark_results.csv"
OUTPUT_JSON = "benchmark_results.json"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

async def run_benchmark():
    # Example: Select the engine to test (could be parameter driven)
    # engine = VLLMEngine(endpoint_url=VLLM_URL, model_name=VLLM_MODEL)
    engine = LlamaCppEngine(endpoint_url=LLAMACPP_URL)
    
    runner = BenchmarkRunner(engine)
    all_aggregated_metrics: List[AggregatedMetrics] = []

    logger.info(f"Starting LLM Benchmark Suite for engine: {engine.get_engine_name()}")
    
    for concurrency in CONCURRENCY_LEVELS:
        results = await runner.run_concurrent(PROMPT, concurrency, max_tokens=512)
        
        aggregated = aggregate_results(engine.get_engine_name(), concurrency, results)
        all_aggregated_metrics.append(aggregated)
        
        # Immediate console reporting
        logger.info(f"--- Concurrency: {concurrency} ---")
        logger.info(f"Success Rate: {aggregated.success_rate:.2%}")
        logger.info(f"Avg TTFT: {aggregated.avg_ttft_s:.4f}s")
        logger.info(f"Avg TPS: {aggregated.avg_tps:.2f} tokens/s")
        logger.info(f"P95 TTFT: {aggregated.p95_ttft_s:.4f}s")
        logger.info(f"P95 TPS: {aggregated.p95_tps:.2f} tokens/s")
        logger.info("-" * 30)

    # Save results to files
    save_to_csv(OUTPUT_CSV, all_aggregated_metrics)
    save_to_json(OUTPUT_JSON, all_aggregated_metrics)
    logger.info(f"Benchmark complete. Results saved to {OUTPUT_CSV} and {OUTPUT_JSON}")

if __name__ == "__main__":
    try:
        asyncio.run(run_benchmark())
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user.")
    except Exception as e:
        logger.critical(f"Unhandled exception during benchmark: {e}")
        sys.exit(1)
