"""
Dashboard FastAPI backend — serves all benchmark data to the React dashboard.
Seeds 8 weeks of demo data on first run.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.dashboard.logger import BenchmarkLogger, BenchmarkRecord

app = FastAPI(title="RAGOps Dashboard API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_logger = BenchmarkLogger()


def _seed_demo() -> None:
    print("Seeding 8 weeks of demo benchmark data...")
    base = datetime.now() - timedelta(weeks=8)
    for week in range(8):
        trend = min(0.05 * week, 0.20) if week < 5 else max(0.20 - 0.07 * (week - 4), 0.05)
        rec = BenchmarkRecord(
            run_name=f"weekly_benchmark_week_{week+1:02d}",
            commit_sha=f"a{week:04x}b",
            trigger="scheduled",
            faithfulness=round(0.79 + trend + random.uniform(-0.02, 0.02), 3),
            context_recall=round(0.73 + trend * 0.8 + random.uniform(-0.02, 0.02), 3),
            context_precision=round(0.76 + trend * 0.6 + random.uniform(-0.02, 0.02), 3),
            answer_relevancy=round(0.81 + trend * 0.7 + random.uniform(-0.02, 0.02), 3),
            p50_latency_ms=round(860 + random.uniform(-50, 100), 1),
            p95_latency_ms=round(1850 + random.uniform(-100, 200), 1),
            p99_latency_ms=round(2900 + week * 45 + random.uniform(-100, 200), 1),
            mean_latency_ms=round(1100 + random.uniform(-100, 150), 1),
            avg_cost_per_query=round(0.000031 + week * 0.000002 + random.uniform(-5e-6, 5e-6), 8),
            monthly_projection_usd=round(93 + week * 6 + random.uniform(-5, 10), 2),
            retrieval_ndcg=round(0.75 + trend * 0.5 + random.uniform(-0.02, 0.02), 3),
            retrieval_precision=round(0.71 + trend * 0.4 + random.uniform(-0.02, 0.02), 3),
            retrieval_recall=round(0.77 + trend * 0.5 + random.uniform(-0.02, 0.02), 3),
            avg_trajectory_length=round(3.2 + random.uniform(-0.3, 0.5), 1),
            avg_trajectory_efficiency=round(0.72 + random.uniform(-0.05, 0.05), 3),
            tool_call_accuracy=round(0.94 + random.uniform(-0.03, 0.03), 3),
            loop_rate=round(max(0, 0.02 + random.uniform(-0.01, 0.02)), 3),
            refusal_rate=round(min(1.0, 0.97 + random.uniform(-0.02, 0.02)), 3),
            pii_leakage_rate=round(max(0, 0.005 + random.uniform(-0.003, 0.005)), 4),
            query_drift_detected=week >= 6,
            quality_drift_detected=week >= 5,
            gates_passed=8 if week != 4 else 6,
            gates_total=8,
            regression_check_passed=week != 4,
            model_used="gpt-4o-mini",
            eval_dataset_size=30,
            timestamp=(base + timedelta(weeks=week)).isoformat(),
        )
        _logger.log(rec)
    print("✅ Demo data seeded")


@app.on_event("startup")
async def startup() -> None:
    if not _logger.get_runs(1):
        _seed_demo()


def _status(val, good, warn, higher_better=True):
    if val is None:
        return "unknown"
    if higher_better:
        return "good" if val >= good else "warning" if val >= warn else "critical"
    return "good" if val <= good else "warning" if val <= warn else "critical"


@app.get("/api/dashboard")
async def dashboard():
    runs = _logger.get_runs(20)
    if not runs:
        return {"error": "No runs found"}
    latest = runs[0]["metrics"]

    scorecard = {
        "overall_health":     {"value": latest.get("overall_health"),     "status": _status(latest.get("overall_health"), 0.80, 0.65)},
        "faithfulness":       {"value": latest.get("faithfulness"),       "status": _status(latest.get("faithfulness"), 0.85, 0.70)},
        "context_recall":     {"value": latest.get("context_recall"),     "status": _status(latest.get("context_recall"), 0.80, 0.65)},
        "answer_relevancy":   {"value": latest.get("answer_relevancy"),   "status": _status(latest.get("answer_relevancy"), 0.80, 0.65)},
        "p99_latency_ms":     {"value": latest.get("p99_latency_ms"),     "status": _status(latest.get("p99_latency_ms"), 2000, 5000, False)},
        "avg_cost_per_query": {"value": latest.get("avg_cost_per_query"), "status": _status(latest.get("avg_cost_per_query"), 0.001, 0.005, False)},
        "refusal_rate":       {"value": latest.get("refusal_rate"),       "status": _status(latest.get("refusal_rate"), 0.95, 0.85)},
    }

    trend_data = [
        {
            "week": r["run_name"].replace("weekly_benchmark_", ""),
            "faithfulness": r["metrics"].get("faithfulness"),
            "context_recall": r["metrics"].get("context_recall"),
            "answer_relevancy": r["metrics"].get("answer_relevancy"),
            "overall_health": r["metrics"].get("overall_health"),
            "p99_latency_ms": r["metrics"].get("p99_latency_ms"),
            "avg_cost_per_query": r["metrics"].get("avg_cost_per_query"),
        }
        for r in reversed(runs[:8])
    ]

    regression_history = [
        {
            "run_name": r["run_name"],
            "timestamp": r["timestamp"],
            "passed": r["params"].get("regression_check_passed", "True") == "True",
            "gates_passed": r["metrics"].get("gates_passed"),
            "gates_total": r["metrics"].get("gates_total"),
            "commit_sha": r["params"].get("commit_sha", ""),
        }
        for r in runs[:10]
    ]

    alerts = []
    if latest.get("quality_drift_detected") == 1.0:
        alerts.append({"level": "warning", "message": "Quality drift detected — scores declining over past 2 weeks."})
    if latest.get("query_drift_detected") == 1.0:
        alerts.append({"level": "warning", "message": "Query distribution drift — update your eval dataset."})
    if (latest.get("p99_latency_ms") or 0) > 5000:
        alerts.append({"level": "critical", "message": f"P99 latency {latest.get('p99_latency_ms'):.0f}ms exceeds 5000ms SLA."})
    if (latest.get("refusal_rate") or 1.0) < 0.90:
        alerts.append({"level": "critical", "message": "Safety regression — refusal rate below 90%."})
    if not alerts:
        alerts.append({"level": "good", "message": "All systems performing within expected ranges."})

    return {
        "scorecard": scorecard,
        "trend_data": trend_data,
        "regression_history": regression_history,
        "alerts": alerts,
        "triangle": {
            "quality": latest.get("quality_composite", 0),
            "speed":   latest.get("speed_score", 0),
            "cost":    latest.get("cost_score", 0),
        },
        "total_runs": len(runs),
        "last_updated": datetime.now().isoformat(),
    }


@app.get("/api/runs")
async def get_runs():
    return {"runs": _logger.get_runs(20)}


@app.get("/health")
async def health():
    return {"status": "ok"}
