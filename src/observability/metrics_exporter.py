"""
Prometheus metrics exporter — exposes AI-specific metrics for time-series monitoring.
Run alongside your FastAPI server. Prometheus scrapes :8001/metrics every 15s.
"""

from __future__ import annotations

import time
from typing import Optional

from prometheus_client import (
    Counter, Gauge, Histogram, start_http_server,
)

REQUEST_LATENCY = Histogram(
    "ai_request_latency_seconds",
    "AI pipeline request latency",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    labelnames=["endpoint", "model"],
)
FAITHFULNESS_SCORE   = Gauge("ai_faithfulness_score",   "Rolling faithfulness")
ANSWER_REL_SCORE     = Gauge("ai_answer_relevancy_score", "Rolling answer relevancy")
CONTEXT_RECALL_SCORE = Gauge("ai_context_recall_score",  "Rolling context recall")
COST_PER_QUERY       = Gauge("ai_cost_per_query_usd",    "Avg cost per query USD")
TOTAL_TOKENS         = Counter("ai_total_tokens", "Total tokens", labelnames=["type"])
ERROR_COUNTER        = Counter("ai_errors_total", "Pipeline errors", labelnames=["error_type"])
REQUEST_COUNTER      = Counter("ai_requests_total", "Total requests", labelnames=["query_type", "model"])
ACTIVE_REQUESTS      = Gauge("ai_active_requests", "In-flight requests")
QUALITY_DRIFT_ALERT  = Gauge("ai_quality_drift_alert", "1 if drift, 0 if normal")


class AIMetricsRecorder:
    """Drop into your FastAPI pipeline to record metrics on every request."""

    def record(
        self,
        latency_seconds: float,
        endpoint: str = "query",
        model: str = "gpt-4o-mini",
        query_type: str = "general",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        REQUEST_LATENCY.labels(endpoint=endpoint, model=model).observe(latency_seconds)
        REQUEST_COUNTER.labels(query_type=query_type, model=model).inc()
        if input_tokens:  TOTAL_TOKENS.labels(type="input").inc(input_tokens)
        if output_tokens: TOTAL_TOKENS.labels(type="output").inc(output_tokens)
        if cost_usd:      COST_PER_QUERY.set(cost_usd)
        if error:         ERROR_COUNTER.labels(error_type=error).inc()

    def update_quality(
        self,
        faithfulness: float,
        answer_relevancy: float,
        context_recall: Optional[float] = None,
    ) -> None:
        FAITHFULNESS_SCORE.set(faithfulness)
        ANSWER_REL_SCORE.set(answer_relevancy)
        if context_recall is not None:
            CONTEXT_RECALL_SCORE.set(context_recall)
        degraded = faithfulness < 0.75 or answer_relevancy < 0.70
        QUALITY_DRIFT_ALERT.set(1 if degraded else 0)
        if degraded:
            print(f"  ⚠️  Quality drift alert: faith={faithfulness:.3f}  rel={answer_relevancy:.3f}")

    def track_active(self, increment: bool = True) -> None:
        if increment: ACTIVE_REQUESTS.inc()
        else:         ACTIVE_REQUESTS.dec()


def start_metrics_server(port: int = 8001) -> None:
    start_http_server(port)
    print(f"✅ Prometheus metrics at http://localhost:{port}/metrics")
    print(f"   prometheus.yml → targets: ['localhost:{port}']")


if __name__ == "__main__":
    import random
    recorder = AIMetricsRecorder()
    start_metrics_server(8001)
    print("Simulating 60 requests...")
    for _ in range(60):
        recorder.record(
            latency_seconds=random.uniform(0.5, 3.0),
            model=random.choice(["gpt-4o-mini", "gpt-4o"]),
            query_type=random.choice(["simple", "moderate", "complex"]),
            input_tokens=random.randint(200, 800),
            output_tokens=random.randint(50, 300),
            cost_usd=random.uniform(0.00002, 0.0002),
        )
    recorder.update_quality(
        faithfulness=random.uniform(0.78, 0.92),
        answer_relevancy=random.uniform(0.75, 0.89),
    )
    print("Metrics live for 60s. Open http://localhost:8001/metrics")
    time.sleep(60)
