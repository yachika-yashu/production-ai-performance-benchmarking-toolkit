"""
MLflow benchmark logger — unified experiment tracking for all 12 benchmarking layers.
Every run is searchable, comparable, and auditable across time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.tracking

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class BenchmarkRecord:
    run_name: str
    commit_sha: str          = "local"
    trigger: str             = "manual"
    model_used: str          = "gpt-4o-mini"
    eval_dataset_size: int   = 0

    # Quality
    faithfulness: Optional[float]       = None
    context_recall: Optional[float]     = None
    context_precision: Optional[float]  = None
    answer_relevancy: Optional[float]   = None

    # Latency
    p50_latency_ms: Optional[float]  = None
    p95_latency_ms: Optional[float]  = None
    p99_latency_ms: Optional[float]  = None
    mean_latency_ms: Optional[float] = None

    # Cost
    avg_cost_per_query: Optional[float]    = None
    monthly_projection_usd: Optional[float] = None

    # Retrieval
    retrieval_ndcg: Optional[float]      = None
    retrieval_precision: Optional[float] = None
    retrieval_recall: Optional[float]    = None

    # Load
    max_concurrent_users: Optional[int]  = None

    # Agentic
    avg_trajectory_length: Optional[float]     = None
    avg_trajectory_efficiency: Optional[float] = None
    tool_call_accuracy: Optional[float]        = None
    loop_rate: Optional[float]                 = None

    # Safety
    refusal_rate: Optional[float]      = None
    pii_leakage_rate: Optional[float]  = None

    # Drift
    query_drift_detected: Optional[bool]   = None
    quality_drift_detected: Optional[bool] = None

    # Regression
    gates_passed: Optional[int]              = None
    gates_total: Optional[int]               = None
    regression_check_passed: Optional[bool]  = None

    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def metrics(self) -> dict[str, float]:
        raw = {
            "faithfulness": self.faithfulness,
            "context_recall": self.context_recall,
            "context_precision": self.context_precision,
            "answer_relevancy": self.answer_relevancy,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "mean_latency_ms": self.mean_latency_ms,
            "avg_cost_per_query": self.avg_cost_per_query,
            "monthly_projection_usd": self.monthly_projection_usd,
            "retrieval_ndcg": self.retrieval_ndcg,
            "retrieval_precision": self.retrieval_precision,
            "retrieval_recall": self.retrieval_recall,
            "avg_trajectory_length": self.avg_trajectory_length,
            "avg_trajectory_efficiency": self.avg_trajectory_efficiency,
            "tool_call_accuracy": self.tool_call_accuracy,
            "loop_rate": self.loop_rate,
            "refusal_rate": self.refusal_rate,
            "pii_leakage_rate": self.pii_leakage_rate,
            "gates_passed": self.gates_passed,
            "gates_total": self.gates_total,
            "eval_dataset_size": self.eval_dataset_size,
        }
        return {k: float(v) for k, v in raw.items() if v is not None}

    def composite_scores(self) -> dict[str, float]:
        scores: dict[str, float] = {}
        q_vals = [v for v in [self.faithfulness, self.context_recall, self.answer_relevancy] if v]
        if q_vals:
            scores["quality_composite"] = round(sum(q_vals) / len(q_vals), 3)
        if self.p99_latency_ms is not None:
            p99 = self.p99_latency_ms
            scores["speed_score"] = round(max(0.1, 1.0 - max(0, p99 - 1000) / 8000), 3)
        if self.avg_cost_per_query is not None:
            c = self.avg_cost_per_query
            scores["cost_score"] = round(max(0.0, 1.0 - c * 500), 3)
        if self.refusal_rate is not None:
            pii = self.pii_leakage_rate or 0.0
            scores["safety_score"] = round(self.refusal_rate * 0.7 + (1 - pii) * 0.3, 3)
        cs = [v for v in scores.values()]
        if cs:
            scores["overall_health"] = round(sum(cs) / len(cs), 3)
        return scores

    def params(self) -> dict[str, str]:
        return {
            "commit_sha": self.commit_sha,
            "trigger": self.trigger,
            "model_used": self.model_used,
            "timestamp": self.timestamp,
            "regression_check_passed": str(self.regression_check_passed),
            "query_drift_detected": str(self.query_drift_detected),
            "quality_drift_detected": str(self.quality_drift_detected),
        }


class BenchmarkLogger:

    def __init__(
        self,
        experiment_name: str = "ragops-benchmarking",
        tracking_uri: str = "./mlruns",
    ) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        print(f"✅ MLflow ready: {experiment_name}  (uri={tracking_uri})")

    def log(self, record: BenchmarkRecord) -> str:
        with mlflow.start_run(run_name=record.run_name) as run:
            mlflow.log_metrics({**record.metrics(), **record.composite_scores()})
            mlflow.log_params(record.params())
            run_id = run.info.run_id
        print(f"✅ Logged: {record.run_name}  (run={run_id[:8]})")
        print(f"   Overall health: {record.composite_scores().get('overall_health', 'N/A')}")
        return run_id

    def get_runs(self, limit: int = 20) -> list[dict]:
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(self.experiment_name)
        if not exp:
            return []
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=limit,
        )
        return [
            {
                "run_id": r.info.run_id,
                "run_name": r.info.run_name,
                "timestamp": datetime.fromtimestamp(r.info.start_time / 1000).isoformat(),
                "metrics": r.data.metrics,
                "params": r.data.params,
            }
            for r in runs
        ]

    def latest(self) -> Optional[dict]:
        runs = self.get_runs(1)
        return runs[0] if runs else None
