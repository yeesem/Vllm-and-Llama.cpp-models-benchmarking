# LLM Inference Benchmark Suite

A production-grade benchmarking tool for measuring the performance of LLM inference engines (vLLM and llama.cpp). It calculates Time To First Token (TTFT) and Tokens Per Second (TPS) across different concurrency levels.

## Key Features
- **Extensible:** Abstract base class `BaseInferenceEngine` allows easy integration of new backends.
- **Accurate:** Uses `tiktoken` for precise token counting and `asyncio` for high-concurrency simulation.
- **Streaming:** Measures performance using streaming responses to capture real-world latency metrics.
- **Reporting:** Exports detailed results to both CSV and JSON.

## Prerequisites
- Python 3.12+
- `httpx`
- `tiktoken`

## Installation
```bash
pip install httpx tiktoken
```

## Usage
1.  **Configure Your Engine:** In `main.py`, update the `VLLM_URL`, `LLAMACPP_URL`, and relevant constants.
2.  **Select Engine:** Choose the backend by uncommenting the desired engine in `main.py`'s `run_benchmark` function.
3.  **Run:**
    ```bash
    python main.py
    ```

## Project Structure
- `src/benchmark/engine.py`: Defines the interface for all engines.
- `src/benchmark/engines/`: Contains concrete implementations for `vLLM` and `llama.cpp`.
- `src/benchmark/runner.py`: The benchmarking orchestrator.
- `src/benchmark/metrics.py`: Data models and reporting logic.
- `main.py`: Main entry point for the CLI.
