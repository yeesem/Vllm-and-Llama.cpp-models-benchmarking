import csv
import json
import logging
import statistics
from dataclasses import dataclass, asdict
from typing import List

logger = logging.getLogger(__name__)

@dataclass
class RequestMetrics:
    ttft_s: float
    tps: float
    total_time_s: float
    total_tokens: int
    success: bool
    error: str | None = None

@dataclass
class AggregatedMetrics:
    engine: str
    concurrency: int
    avg_ttft_s: float
    p50_ttft_s: float
    p95_ttft_s: float
    p99_ttft_s: float
    avg_tps: float
    p50_tps: float
    p95_tps: float
    p99_tps: float
    total_requests: int
    success_rate: float

def aggregate_results(engine: str, concurrency: int, results: List[RequestMetrics]) -> AggregatedMetrics:
    successful_results = [r for r in results if r.success]
    total = len(results)
    success_count = len(successful_results)
    success_rate = (success_count / total) if total > 0 else 0.0

    if not successful_results:
        return AggregatedMetrics(
            engine=engine, concurrency=concurrency,
            avg_ttft_s=0.0, p50_ttft_s=0.0, p95_ttft_s=0.0, p99_ttft_s=0.0,
            avg_tps=0.0, p50_tps=0.0, p95_tps=0.0, p99_tps=0.0,
            total_requests=total, success_rate=success_rate
        )

    ttfts = sorted([r.ttft_s for r in successful_results])
    tps_values = sorted([r.tps for r in successful_results])

    def get_percentile(data, p):
        if not data: return 0.0
        idx = max(0, min(len(data) - 1, int(len(data) * p / 100)))
        return data[idx]

    return AggregatedMetrics(
        engine=engine,
        concurrency=concurrency,
        avg_ttft_s=statistics.mean(ttfts),
        p50_ttft_s=statistics.median(ttfts),
        p95_ttft_s=get_percentile(ttfts, 95),
        p99_ttft_s=get_percentile(ttfts, 99),
        avg_tps=statistics.mean(tps_values),
        p50_tps=statistics.median(tps_values),
        p95_tps=get_percentile(tps_values, 95),
        p99_tps=get_percentile(tps_values, 99),
        total_requests=total,
        success_rate=success_rate
    )

def save_to_csv(filepath: str, metrics: List[AggregatedMetrics]):
    if not metrics:
        return
    keys = metrics[0].__dict__.keys()
    with open(filepath, "w", newline="") as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows([asdict(m) for m in metrics])

def save_to_json(filepath: str, metrics: List[AggregatedMetrics]):
    with open(filepath, "w") as f:
        json.dump([asdict(m) for m in metrics], f, indent=2)
