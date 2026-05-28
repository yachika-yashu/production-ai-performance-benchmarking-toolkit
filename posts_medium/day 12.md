Here we go. The final day.

---

# AI Performance Benchmarking: From Zero to Production
### Day 12 of 12 — The Unified Dashboard: Everything In One Place

---

*This is Day 12 — the final day of a 12-part series. Start with **[Day 0](#)** if you're just joining.*

---

## Twelve Days. One Picture.

Look at what you have built.

Day 0 — LangSmith tracing. Visibility into every query.
Day 1 — The quality-speed-cost triangle. The mental model.
Day 2 — Baseline benchmarks. Your stake in the ground.
Day 3 — Latency profiling. P99 truth instead of average lies.
Day 4 — RAGAS quality metrics. Faithfulness, recall, relevancy.
Day 5 — Retrieval benchmarking. Chunking, embeddings, hybrid search.
Day 6 — A/B testing. Prompt changes measured not guessed.
Day 7 — Cost benchmarking. The bill explained and controlled.
Day 8 — Load testing. Five hundred users instead of one.
Day 9 — Agentic benchmarking. Trajectories not just answers.
Day 10 — Regression testing. CI/CD gates on every PR.
Day 11 — Observability. Production drift caught before users feel it.

Twelve separate measurement systems.

Each one powerful on its own.

But right now they live in twelve separate folders producing twelve separate JSON files.

To understand your system's complete health you would need to open twelve files, mentally combine the numbers, and build the picture in your head.

That is not how production systems work.

Production systems need a single source of truth.

One place where every dimension is visible simultaneously.

One place where a degradation in any corner of the triangle is immediately obvious.

One place where you can answer the question *"how is our AI actually performing right now?"* in thirty seconds.

**That is what we build today.**

---

## What The Unified Dashboard Shows

The dashboard has five panels:

---

**Panel 1 — The Health Scorecard**

A single at-a-glance view of every key metric with traffic light status.

Green: performing well.
Yellow: needs attention.
Red: investigate immediately.

This is what you check every morning. Thirty seconds. Complete picture.

---

**Panel 2 — Quality Over Time**

Faithfulness, context recall, and answer relevancy plotted as trend lines.

Not a snapshot — a history.

This is where you see drift happening before it becomes a crisis. A gradual downward slope that would be invisible in a single number becomes obvious as a trend line.

---

**Panel 3 — Performance Triangle**

The quality-speed-cost triangle from Day 1 — now populated with real numbers.

Shows you exactly where your system sits in the tradeoff space and how it has moved since your baseline.

---

**Panel 4 — Regression Gate History**

Every CI/CD run from Day 10 — pass or fail, with which gates triggered.

This is your change management history. You can see exactly which commit caused a regression and which gate caught it.

---

**Panel 5 — Alerts and Recommendations**

Automated analysis of all metrics combined.

Not just individual threshold breaches — pattern recognition across dimensions.

*"Quality dropped 8% this week AND query distribution drifted toward complex analysis queries — your eval dataset needs updating."*

That kind of insight requires seeing everything at once.

---

## MLflow — The Experiment Tracking Layer

MLflow ties everything together.

Every benchmark run — from Day 2 baselines to Day 6 A/B tests to Day 10 regression checks — logs metrics to MLflow.

MLflow stores them as experiments with runs, making them queryable and comparable across time.

The dashboard reads from MLflow to show trends and comparisons.

---

## The Code — Unified Dashboard

Today we build three things:

1. An **MLflow logger** that standardizes how every day's metrics get stored
2. A **dashboard data aggregator** that pulls all metrics into one structure
3. A **React dashboard** — a real interactive UI rendered in your browser

---

### Step 1 — Set up today's folder

```bash
cd ..
mkdir day-12-dashboard
cd day-12-dashboard
```

---

### Step 2 — Install dependencies

```bash
pip install mlflow langchain langchain-openai python-dotenv pandas numpy requests
```

---

### Step 3 — Create the MLflow logger

Right-click `day-12-dashboard` → **New File** → name it:

```
benchmark_logger.py
```

Paste this exactly:

```python
# benchmark_logger.py
# Standardizes how all benchmark results get logged to MLflow
# Every day's metrics flow through this single interface
# This is what the dashboard reads from

from dotenv import load_dotenv
load_dotenv()

import json
import mlflow
import mlflow.tracking
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass


# ─────────────────────────────────────────────
# UNIFIED BENCHMARK RECORD
# One complete snapshot of system performance
# Combines every dimension from every day
# ─────────────────────────────────────────────

@dataclass
class UnifiedBenchmarkRecord:

    # Identity
    run_name: str               # e.g. "weekly_benchmark_2026_W22"
    commit_sha: str = "local"
    trigger: str = "manual"     # manual / ci / scheduled

    # Quality (Day 4)
    faithfulness: Optional[float] = None
    context_recall: Optional[float] = None
    context_precision: Optional[float] = None
    answer_relevancy: Optional[float] = None

    # Latency (Day 3)
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None
    mean_latency_ms: Optional[float] = None
    ttft_ms: Optional[float] = None

    # Cost (Day 7)
    avg_cost_per_query: Optional[float] = None
    monthly_projection_usd: Optional[float] = None

    # Retrieval (Day 5)
    retrieval_precision: Optional[float] = None
    retrieval_recall: Optional[float] = None
    retrieval_ndcg: Optional[float] = None
    retrieval_mrr: Optional[float] = None

    # Load testing (Day 8)
    max_concurrent_users: Optional[int] = None
    rps_at_threshold: Optional[float] = None

    # Agentic (Day 9)
    avg_trajectory_length: Optional[float] = None
    avg_trajectory_efficiency: Optional[float] = None
    tool_call_accuracy: Optional[float] = None
    loop_rate: Optional[float] = None

    # Safety (Day 10)
    refusal_rate: Optional[float] = None
    pii_leakage_rate: Optional[float] = None

    # Drift (Day 11)
    query_drift_detected: Optional[bool] = None
    quality_drift_detected: Optional[bool] = None
    faithfulness_drift: Optional[float] = None

    # Regression gates (Day 10)
    gates_passed: Optional[int] = None
    gates_total: Optional[int] = None
    regression_check_passed: Optional[bool] = None

    # Metadata
    model_used: str = "gpt-4o-mini"
    eval_dataset_size: int = 0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_metrics_dict(self) -> Dict:
        """Converts to MLflow metrics format — only numeric values"""
        metrics = {}
        numeric_fields = {
            "faithfulness": self.faithfulness,
            "context_recall": self.context_recall,
            "context_precision": self.context_precision,
            "answer_relevancy": self.answer_relevancy,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "mean_latency_ms": self.mean_latency_ms,
            "ttft_ms": self.ttft_ms,
            "avg_cost_per_query": self.avg_cost_per_query,
            "monthly_projection_usd": self.monthly_projection_usd,
            "retrieval_precision": self.retrieval_precision,
            "retrieval_recall": self.retrieval_recall,
            "retrieval_ndcg": self.retrieval_ndcg,
            "retrieval_mrr": self.retrieval_mrr,
            "rps_at_threshold": self.rps_at_threshold,
            "avg_trajectory_length": self.avg_trajectory_length,
            "avg_trajectory_efficiency": self.avg_trajectory_efficiency,
            "tool_call_accuracy": self.tool_call_accuracy,
            "loop_rate": self.loop_rate,
            "refusal_rate": self.refusal_rate,
            "pii_leakage_rate": self.pii_leakage_rate,
            "faithfulness_drift": self.faithfulness_drift,
            "gates_passed": self.gates_passed,
            "gates_total": self.gates_total,
            "eval_dataset_size": self.eval_dataset_size,
        }
        # Only include non-None values
        for key, val in numeric_fields.items():
            if val is not None:
                metrics[key] = float(val)
        return metrics

    def to_params_dict(self) -> Dict:
        """Converts to MLflow params format — string values"""
        return {
            "commit_sha": self.commit_sha,
            "trigger": self.trigger,
            "model_used": self.model_used,
            "timestamp": self.timestamp,
            "regression_check_passed": str(
                self.regression_check_passed
            ),
            "query_drift_detected": str(self.query_drift_detected),
            "quality_drift_detected": str(
                self.quality_drift_detected
            )
        }

    def compute_composite_scores(self) -> Dict:
        """
        Computes composite scores across dimensions.
        These are the single numbers that go on the scorecard.
        """
        scores = {}

        # Quality score — weighted average of RAGAS metrics
        quality_vals = [
            v for v in [
                self.faithfulness,
                self.context_recall,
                self.answer_relevancy
            ] if v is not None
        ]
        if quality_vals:
            scores["quality_composite"] = round(
                sum(quality_vals) / len(quality_vals), 3
            )

        # Speed score — normalized against thresholds
        if self.p99_latency_ms is not None:
            p99 = self.p99_latency_ms
            if p99 <= 1000:
                scores["speed_score"] = 1.0
            elif p99 <= 3000:
                scores["speed_score"] = round(
                    1.0 - (p99 - 1000) / 4000, 3
                )
            elif p99 <= 5000:
                scores["speed_score"] = round(
                    0.5 - (p99 - 3000) / 10000, 3
                )
            else:
                scores["speed_score"] = 0.1

        # Cost score — normalized against budget
        if self.avg_cost_per_query is not None:
            cost = self.avg_cost_per_query
            if cost <= 0.001:
                scores["cost_score"] = 1.0
            elif cost <= 0.005:
                scores["cost_score"] = round(
                    1.0 - (cost - 0.001) / 0.008, 3
                )
            else:
                scores["cost_score"] = max(
                    0.0, round(1.0 - cost * 100, 3)
                )

        # Safety score
        if self.refusal_rate is not None:
            pii = self.pii_leakage_rate or 0.0
            scores["safety_score"] = round(
                (self.refusal_rate * 0.7 +
                 (1 - pii) * 0.3), 3
            )

        # Overall health score
        component_scores = [
            v for k, v in scores.items()
            if k.endswith("_score")
        ]
        if component_scores:
            scores["overall_health"] = round(
                sum(component_scores) / len(component_scores), 3
            )

        return scores


# ─────────────────────────────────────────────
# BENCHMARK LOGGER
# ─────────────────────────────────────────────

class BenchmarkLogger:

    def __init__(
        self,
        experiment_name: str = "ai-performance-benchmarking",
        tracking_uri: str = "./mlruns"
    ):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        print(
            f"✅ MLflow initialized: {experiment_name}"
        )
        print(
            f"   Tracking URI: {tracking_uri}"
        )

    def log_benchmark(
        self,
        record: UnifiedBenchmarkRecord
    ) -> str:
        """
        Logs a complete benchmark record to MLflow.
        Returns the MLflow run ID for reference.
        """
        with mlflow.start_run(run_name=record.run_name) as run:
            # Log all numeric metrics
            metrics = record.to_metrics_dict()
            composite = record.compute_composite_scores()
            all_metrics = {**metrics, **composite}

            mlflow.log_metrics(all_metrics)

            # Log string parameters
            mlflow.log_params(record.to_params_dict())

            # Log full record as artifact
            record_dict = record.__dict__.copy()
            with open("benchmark_record.json", "w") as f:
                json.dump(record_dict, f, indent=2, default=str)
            mlflow.log_artifact("benchmark_record.json")

            run_id = run.info.run_id

        print(f"✅ Benchmark logged to MLflow")
        print(f"   Run ID: {run_id}")
        print(
            f"   Overall health: "
            f"{composite.get('overall_health', 'N/A')}"
        )
        return run_id

    def get_all_runs(self) -> List[Dict]:
        """
        Retrieves all benchmark runs for dashboard display.
        Returns sorted by timestamp descending.
        """
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(
            self.experiment_name
        )

        if not experiment:
            return []

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )

        result = []
        for run in runs:
            result.append({
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "timestamp": datetime.fromtimestamp(
                    run.info.start_time / 1000
                ).isoformat(),
                "metrics": run.data.metrics,
                "params": run.data.params
            })

        return result

    def get_latest_run(self) -> Optional[Dict]:
        """Returns the most recent benchmark run"""
        runs = self.get_all_runs()
        return runs[0] if runs else None

    def compare_runs(
        self,
        run_id_a: str,
        run_id_b: str
    ) -> Dict:
        """
        Compares two runs metric by metric.
        Shows delta and direction for each metric.
        """
        client = mlflow.tracking.MlflowClient()
        run_a = client.get_run(run_id_a)
        run_b = client.get_run(run_id_b)

        metrics_a = run_a.data.metrics
        metrics_b = run_b.data.metrics

        comparison = {}
        all_keys = set(metrics_a.keys()) | set(metrics_b.keys())

        for key in all_keys:
            val_a = metrics_a.get(key)
            val_b = metrics_b.get(key)
            if val_a is not None and val_b is not None:
                delta = val_b - val_a
                comparison[key] = {
                    "run_a": round(val_a, 4),
                    "run_b": round(val_b, 4),
                    "delta": round(delta, 4),
                    "improved": delta > 0
                }

        return comparison
```

---

### Step 4 — Create the dashboard data API

Right-click `day-12-dashboard` → **New File** → name it:

```
dashboard_api.py
```

Paste this:

```python
# dashboard_api.py
# FastAPI backend that serves dashboard data
# The React dashboard fetches from this API

from dotenv import load_dotenv
load_dotenv()

import json
import random
import numpy as np
from datetime import datetime, timedelta
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from benchmark_logger import BenchmarkLogger, UnifiedBenchmarkRecord

app = FastAPI(title="AI Benchmarking Dashboard API")

# Allow the dashboard to fetch from this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

logger = BenchmarkLogger()


def seed_demo_data():
    """
    Seeds MLflow with realistic demo data
    so the dashboard has something to display.
    Remove this in production — use your real benchmark runs.
    """
    print("Seeding demo benchmark data...")

    # Simulate 8 weeks of benchmark history
    base_date = datetime.now() - timedelta(weeks=8)

    for week in range(8):
        # Simulate gradual quality improvement over weeks
        # with a regression dip around week 5
        quality_trend = min(
            0.05 * week, 0.25
        ) if week < 5 else max(
            0.25 - 0.08 * (week - 4), 0.10
        )

        record = UnifiedBenchmarkRecord(
            run_name=f"weekly_benchmark_week_{week+1:02d}",
            commit_sha=f"abc{week:04d}",
            trigger="scheduled",
            faithfulness=round(
                0.78 + quality_trend + random.uniform(-0.02, 0.02), 3
            ),
            context_recall=round(
                0.72 + quality_trend * 0.8 +
                random.uniform(-0.02, 0.02), 3
            ),
            context_precision=round(
                0.75 + quality_trend * 0.6 +
                random.uniform(-0.02, 0.02), 3
            ),
            answer_relevancy=round(
                0.80 + quality_trend * 0.7 +
                random.uniform(-0.02, 0.02), 3
            ),
            p50_latency_ms=round(
                850 + random.uniform(-50, 100), 1
            ),
            p95_latency_ms=round(
                1800 + random.uniform(-100, 200), 1
            ),
            p99_latency_ms=round(
                2800 + week * 50 + random.uniform(-100, 200), 1
            ),
            mean_latency_ms=round(
                1100 + random.uniform(-100, 150), 1
            ),
            avg_cost_per_query=round(
                0.000031 + week * 0.000002 +
                random.uniform(-0.000005, 0.000005), 8
            ),
            monthly_projection_usd=round(
                93 + week * 6 + random.uniform(-5, 10), 2
            ),
            retrieval_ndcg=round(
                0.74 + quality_trend * 0.5 +
                random.uniform(-0.02, 0.02), 3
            ),
            retrieval_precision=round(
                0.71 + quality_trend * 0.4 +
                random.uniform(-0.02, 0.02), 3
            ),
            retrieval_recall=round(
                0.76 + quality_trend * 0.5 +
                random.uniform(-0.02, 0.02), 3
            ),
            retrieval_mrr=round(
                0.78 + quality_trend * 0.4 +
                random.uniform(-0.02, 0.02), 3
            ),
            avg_trajectory_length=round(
                3.2 + random.uniform(-0.3, 0.5), 1
            ),
            avg_trajectory_efficiency=round(
                0.71 + random.uniform(-0.05, 0.05), 3
            ),
            tool_call_accuracy=round(
                0.94 + random.uniform(-0.03, 0.03), 3
            ),
            loop_rate=round(
                max(0, 0.02 + random.uniform(-0.01, 0.02)), 3
            ),
            refusal_rate=round(
                0.97 + random.uniform(-0.02, 0.02), 3
            ),
            pii_leakage_rate=round(
                max(0, 0.005 + random.uniform(-0.003, 0.005)), 4
            ),
            query_drift_detected=week >= 6,
            quality_drift_detected=week >= 5,
            faithfulness_drift=round(
                max(0, (week - 4) * 0.015), 3
            ) if week >= 5 else 0.0,
            gates_passed=8 if week != 4 else 6,
            gates_total=8,
            regression_check_passed=week != 4,
            model_used="gpt-4o-mini",
            eval_dataset_size=50,
            timestamp=(
                base_date + timedelta(weeks=week)
            ).isoformat()
        )

        logger.log_benchmark(record)

    print("✅ Demo data seeded — 8 weeks of benchmarks")


@app.on_event("startup")
async def startup():
    """Seed demo data if no runs exist"""
    runs = logger.get_all_runs()
    if not runs:
        seed_demo_data()


@app.get("/api/dashboard")
async def get_dashboard_data():
    """
    Returns all data needed for the dashboard.
    The React frontend calls this endpoint.
    """
    runs = logger.get_all_runs()

    if not runs:
        return {"error": "No benchmark runs found"}

    latest = runs[0]
    latest_metrics = latest["metrics"]

    # Build scorecard
    def score_status(value, good_threshold, warn_threshold,
                     higher_is_better=True):
        if value is None:
            return "unknown"
        if higher_is_better:
            if value >= good_threshold:
                return "good"
            elif value >= warn_threshold:
                return "warning"
            return "critical"
        else:
            if value <= good_threshold:
                return "good"
            elif value <= warn_threshold:
                return "warning"
            return "critical"

    scorecard = {
        "faithfulness": {
            "value": latest_metrics.get("faithfulness"),
            "status": score_status(
                latest_metrics.get("faithfulness"),
                0.85, 0.70
            )
        },
        "context_recall": {
            "value": latest_metrics.get("context_recall"),
            "status": score_status(
                latest_metrics.get("context_recall"),
                0.80, 0.65
            )
        },
        "answer_relevancy": {
            "value": latest_metrics.get("answer_relevancy"),
            "status": score_status(
                latest_metrics.get("answer_relevancy"),
                0.80, 0.70
            )
        },
        "p99_latency_ms": {
            "value": latest_metrics.get("p99_latency_ms"),
            "status": score_status(
                latest_metrics.get("p99_latency_ms"),
                2000, 5000,
                higher_is_better=False
            )
        },
        "avg_cost_per_query": {
            "value": latest_metrics.get("avg_cost_per_query"),
            "status": score_status(
                latest_metrics.get("avg_cost_per_query"),
                0.001, 0.005,
                higher_is_better=False
            )
        },
        "refusal_rate": {
            "value": latest_metrics.get("refusal_rate"),
            "status": score_status(
                latest_metrics.get("refusal_rate"),
                0.95, 0.85
            )
        },
        "overall_health": {
            "value": latest_metrics.get("overall_health"),
            "status": score_status(
                latest_metrics.get("overall_health"),
                0.80, 0.65
            )
        }
    }

    # Build trend data
    trend_data = []
    for run in reversed(runs[-8:]):
        m = run["metrics"]
        trend_data.append({
            "week": run["run_name"].replace(
                "weekly_benchmark_", ""
            ),
            "faithfulness": m.get("faithfulness"),
            "context_recall": m.get("context_recall"),
            "answer_relevancy": m.get("answer_relevancy"),
            "p99_latency_ms": m.get("p99_latency_ms"),
            "avg_cost_per_query": m.get("avg_cost_per_query"),
            "overall_health": m.get("overall_health")
        })

    # Build regression history
    regression_history = []
    for run in runs[:10]:
        m = run["metrics"]
        passed = run["params"].get(
            "regression_check_passed", "True"
        ) == "True"
        regression_history.append({
            "run_name": run["run_name"],
            "timestamp": run["timestamp"],
            "passed": passed,
            "gates_passed": m.get("gates_passed"),
            "gates_total": m.get("gates_total"),
            "commit_sha": run["params"].get("commit_sha", "")
        })

    # Build alerts
    alerts = []
    if latest_metrics.get("quality_drift_detected") == 1.0:
        alerts.append({
            "level": "warning",
            "message": (
                "Quality drift detected — scores declining "
                "over the past 2 weeks. Check recent changes."
            )
        })
    if latest_metrics.get("query_drift_detected") == 1.0:
        alerts.append({
            "level": "warning",
            "message": (
                "Query distribution drift detected — "
                "users are asking different types of questions. "
                "Update your eval dataset."
            )
        })
    if latest_metrics.get("p99_latency_ms", 0) > 5000:
        alerts.append({
            "level": "critical",
            "message": (
                f"P99 latency "
                f"({latest_metrics.get('p99_latency_ms'):.0f}ms) "
                f"exceeds SLA threshold (5000ms)."
            )
        })
    if latest_metrics.get("refusal_rate", 1.0) < 0.90:
        alerts.append({
            "level": "critical",
            "message": (
                "Safety regression — refusal rate below 90%. "
                "Investigate immediately."
            )
        })
    if not alerts:
        alerts.append({
            "level": "good",
            "message": "All systems performing within expected ranges."
        })

    # Triangle data
    triangle = {
        "quality": latest_metrics.get("quality_composite", 0),
        "speed": latest_metrics.get("speed_score", 0),
        "cost": latest_metrics.get("cost_score", 0)
    }

    return {
        "scorecard": scorecard,
        "trend_data": trend_data,
        "regression_history": regression_history,
        "alerts": alerts,
        "triangle": triangle,
        "total_runs": len(runs),
        "last_updated": datetime.now().isoformat()
    }


@app.get("/api/runs")
async def get_runs():
    return {"runs": logger.get_all_runs()[:20]}


@app.get("/api/compare/{run_id_a}/{run_id_b}")
async def compare_runs(run_id_a: str, run_id_b: str):
    comparison = logger.compare_runs(run_id_a, run_id_b)
    return {"comparison": comparison}
```

---

### Step 5 — Create the React dashboard

Right-click `day-12-dashboard` → **New File** → name it:

```
dashboard.jsx
```

Paste this:

```jsx
import { useState, useEffect } from "react";
import {
  LineChart, Line, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from "recharts";

const API_BASE = "http://localhost:8000";

// ─────────────────────────────────────────────
// STATUS COLORS
// ─────────────────────────────────────────────

const STATUS_COLORS = {
  good: "#22c55e",
  warning: "#f59e0b",
  critical: "#ef4444",
  unknown: "#94a3b8"
};

const STATUS_BG = {
  good: "bg-green-950 border-green-700",
  warning: "bg-yellow-950 border-yellow-700",
  critical: "bg-red-950 border-red-700",
  unknown: "bg-slate-800 border-slate-600"
};

const STATUS_ICONS = {
  good: "✅",
  warning: "⚠️",
  critical: "🔴",
  unknown: "⬜"
};


// ─────────────────────────────────────────────
// SCORECARD CARD
// ─────────────────────────────────────────────

function ScorecardCard({ label, value, status, unit = "" }) {
  const formatted = value !== null && value !== undefined
    ? (typeof value === "number" && value < 0.1
      ? value.toFixed(6)
      : typeof value === "number" && value < 10
      ? value.toFixed(3)
      : typeof value === "number"
      ? value.toFixed(0)
      : value)
    : "—";

  return (
    <div className={`
      border rounded-xl p-4 flex flex-col gap-1
      ${STATUS_BG[status] || STATUS_BG.unknown}
    `}>
      <div className="text-slate-400 text-xs font-medium uppercase tracking-wider">
        {label}
      </div>
      <div className="text-white text-2xl font-bold">
        {formatted}{unit}
      </div>
      <div className="flex items-center gap-1 text-xs">
        <span>{STATUS_ICONS[status]}</span>
        <span className="text-slate-300 capitalize">{status}</span>
      </div>
    </div>
  );
}


// ─────────────────────────────────────────────
// ALERT BANNER
// ─────────────────────────────────────────────

function AlertBanner({ alert }) {
  const colors = {
    good: "bg-green-900 border-green-600 text-green-200",
    warning: "bg-yellow-900 border-yellow-600 text-yellow-200",
    critical: "bg-red-900 border-red-600 text-red-200"
  };

  return (
    <div className={`
      border rounded-lg px-4 py-3 text-sm
      ${colors[alert.level] || colors.warning}
    `}>
      {STATUS_ICONS[alert.level]} {alert.message}
    </div>
  );
}


// ─────────────────────────────────────────────
// TRIANGLE CHART
// Performance triangle using RadarChart
// ─────────────────────────────────────────────

function TriangleChart({ data }) {
  const radarData = [
    {
      dimension: "Quality",
      score: Math.round((data.quality || 0) * 100)
    },
    {
      dimension: "Speed",
      score: Math.round((data.speed || 0) * 100)
    },
    {
      dimension: "Cost",
      score: Math.round((data.cost || 0) * 100)
    }
  ];

  return (
    <ResponsiveContainer width="100%" height={240}>
      <RadarChart data={radarData}>
        <PolarGrid stroke="#334155" />
        <PolarAngleAxis
          dataKey="dimension"
          tick={{ fill: "#94a3b8", fontSize: 13 }}
        />
        <PolarRadiusAxis
          domain={[0, 100]}
          tick={{ fill: "#64748b", fontSize: 10 }}
        />
        <Radar
          name="Score"
          dataKey="score"
          stroke="#6366f1"
          fill="#6366f1"
          fillOpacity={0.3}
          strokeWidth={2}
        />
        <Tooltip
          formatter={(v) => [`${v}/100`, "Score"]}
          contentStyle={{
            background: "#1e293b",
            border: "1px solid #334155",
            borderRadius: "8px",
            color: "#f1f5f9"
          }}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
}


// ─────────────────────────────────────────────
// QUALITY TREND CHART
// ─────────────────────────────────────────────

function QualityTrendChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis
          dataKey="week"
          tick={{ fill: "#64748b", fontSize: 11 }}
        />
        <YAxis
          domain={[0.5, 1.0]}
          tick={{ fill: "#64748b", fontSize: 11 }}
        />
        <Tooltip
          contentStyle={{
            background: "#1e293b",
            border: "1px solid #334155",
            borderRadius: "8px",
            color: "#f1f5f9"
          }}
          formatter={(v) => [v?.toFixed(3), ""]}
        />
        <Legend
          wrapperStyle={{ color: "#94a3b8", fontSize: "12px" }}
        />
        <Line
          type="monotone"
          dataKey="faithfulness"
          stroke="#6366f1"
          strokeWidth={2}
          dot={{ r: 3 }}
          name="Faithfulness"
        />
        <Line
          type="monotone"
          dataKey="context_recall"
          stroke="#22c55e"
          strokeWidth={2}
          dot={{ r: 3 }}
          name="Context Recall"
        />
        <Line
          type="monotone"
          dataKey="answer_relevancy"
          stroke="#f59e0b"
          strokeWidth={2}
          dot={{ r: 3 }}
          name="Answer Relevancy"
        />
        <Line
          type="monotone"
          dataKey="overall_health"
          stroke="#e2e8f0"
          strokeWidth={2.5}
          strokeDasharray="5 5"
          dot={{ r: 3 }}
          name="Overall Health"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}


// ─────────────────────────────────────────────
// LATENCY TREND CHART
// ─────────────────────────────────────────────

function LatencyTrendChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis
          dataKey="week"
          tick={{ fill: "#64748b", fontSize: 11 }}
        />
        <YAxis tick={{ fill: "#64748b", fontSize: 11 }} />
        <Tooltip
          contentStyle={{
            background: "#1e293b",
            border: "1px solid #334155",
            borderRadius: "8px",
            color: "#f1f5f9"
          }}
          formatter={(v) => [`${v?.toFixed(0)}ms`, ""]}
        />
        <Legend
          wrapperStyle={{ color: "#94a3b8", fontSize: "12px" }}
        />
        <Line
          type="monotone"
          dataKey="p99_latency_ms"
          stroke="#ef4444"
          strokeWidth={2}
          dot={{ r: 3 }}
          name="P99 Latency"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}


// ─────────────────────────────────────────────
// REGRESSION HISTORY TABLE
// ─────────────────────────────────────────────

function RegressionHistory({ data }) {
  return (
    <div className="overflow-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-slate-400 text-left border-b border-slate-700">
            <th className="pb-2 pr-4">Run</th>
            <th className="pb-2 pr-4">Commit</th>
            <th className="pb-2 pr-4">Gates</th>
            <th className="pb-2">Status</th>
          </tr>
        </thead>
        <tbody>
          {data.map((run, i) => (
            <tr
              key={i}
              className="border-b border-slate-800 hover:bg-slate-800 transition-colors"
            >
              <td className="py-2 pr-4 text-slate-300 font-mono text-xs">
                {run.run_name.replace("weekly_benchmark_", "")}
              </td>
              <td className="py-2 pr-4 text-slate-400 font-mono text-xs">
                {run.commit_sha?.slice(0, 7) || "—"}
              </td>
              <td className="py-2 pr-4 text-slate-300">
                {run.gates_passed !== null
                  ? `${run.gates_passed}/${run.gates_total}`
                  : "—"}
              </td>
              <td className="py-2">
                <span className={`
                  px-2 py-0.5 rounded-full text-xs font-medium
                  ${run.passed
                    ? "bg-green-900 text-green-300"
                    : "bg-red-900 text-red-300"}
                `}>
                  {run.passed ? "✅ Pass" : "❌ Fail"}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}


// ─────────────────────────────────────────────
// MAIN DASHBOARD
// ─────────────────────────────────────────────

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastRefresh, setLastRefresh] = useState(null);

  const fetchData = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/dashboard`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const json = await response.json();
      setData(json);
      setLastRefresh(new Date().toLocaleTimeString());
      setError(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    // Auto-refresh every 60 seconds
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="text-slate-400 text-lg">
          Loading dashboard...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="text-red-400 text-lg">
          ❌ Could not connect to API: {error}
          <div className="text-slate-400 text-sm mt-2">
            Make sure dashboard_api.py is running on port 8000
          </div>
        </div>
      </div>
    );
  }

  const { scorecard, trend_data, regression_history,
          alerts, triangle, total_runs } = data;

  return (
    <div className="min-h-screen bg-slate-900 text-white p-6">

      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white">
            AI Performance Dashboard
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            production-ai-performance-benchmarking-toolkit
            · {total_runs} benchmark runs
          </p>
        </div>
        <div className="text-right">
          <div className="text-slate-400 text-xs">
            Last updated
          </div>
          <div className="text-slate-300 text-sm">
            {lastRefresh}
          </div>
          <button
            onClick={fetchData}
            className="mt-1 text-xs text-indigo-400 hover:text-indigo-300 transition-colors"
          >
            ↻ Refresh
          </button>
        </div>
      </div>

      {/* Alerts */}
      <div className="flex flex-col gap-2 mb-6">
        {alerts.map((alert, i) => (
          <AlertBanner key={i} alert={alert} />
        ))}
      </div>

      {/* Scorecard Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        <ScorecardCard
          label="Overall Health"
          value={scorecard.overall_health?.value}
          status={scorecard.overall_health?.status}
        />
        <ScorecardCard
          label="Faithfulness"
          value={scorecard.faithfulness?.value}
          status={scorecard.faithfulness?.status}
        />
        <ScorecardCard
          label="Context Recall"
          value={scorecard.context_recall?.value}
          status={scorecard.context_recall?.status}
        />
        <ScorecardCard
          label="Answer Relevancy"
          value={scorecard.answer_relevancy?.value}
          status={scorecard.answer_relevancy?.status}
        />
        <ScorecardCard
          label="P99 Latency"
          value={scorecard.p99_latency_ms?.value}
          status={scorecard.p99_latency_ms?.status}
          unit="ms"
        />
        <ScorecardCard
          label="Cost / Query"
          value={scorecard.avg_cost_per_query?.value}
          status={scorecard.avg_cost_per_query?.status}
          unit=" USD"
        />
        <ScorecardCard
          label="Refusal Rate"
          value={scorecard.refusal_rate?.value}
          status={scorecard.refusal_rate?.status}
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">

        {/* Quality Trend */}
        <div className="md:col-span-2 bg-slate-800 rounded-xl p-5 border border-slate-700">
          <div className="text-slate-300 font-semibold mb-4">
            📈 Quality Metrics Over Time
          </div>
          <QualityTrendChart data={trend_data} />
        </div>

        {/* Performance Triangle */}
        <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
          <div className="text-slate-300 font-semibold mb-2">
            🔺 Performance Triangle
          </div>
          <div className="text-slate-500 text-xs mb-3">
            Quality · Speed · Cost (100 = optimal)
          </div>
          <TriangleChart data={triangle} />
          <div className="grid grid-cols-3 gap-2 mt-2 text-center">
            {Object.entries(triangle).map(([dim, val]) => (
              <div key={dim}>
                <div className="text-xs text-slate-400 capitalize">
                  {dim}
                </div>
                <div className="text-sm font-bold text-white">
                  {Math.round((val || 0) * 100)}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

        {/* Latency Trend */}
        <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
          <div className="text-slate-300 font-semibold mb-4">
            ⚡ P99 Latency Trend
          </div>
          <LatencyTrendChart data={trend_data} />
        </div>

        {/* Regression History */}
        <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
          <div className="text-slate-300 font-semibold mb-4">
            🔍 Regression Gate History
          </div>
          <RegressionHistory data={regression_history} />
        </div>
      </div>

      {/* Footer */}
      <div className="mt-6 text-center text-slate-600 text-xs">
        production-ai-performance-benchmarking-toolkit
        · 12-day series · Built with MLflow + LangSmith +
        RAGAS + Evidently
      </div>
    </div>
  );
}
```

---

### Step 6 — Create the final runner

Right-click `day-12-dashboard` → **New File** → name it:

```
run_dashboard.py
```

Paste this:

```python
# run_dashboard.py
# Seeds demo data, logs a fresh benchmark run,
# and starts the dashboard API
# Run this then open the dashboard artifact

from dotenv import load_dotenv
load_dotenv()

import uvicorn
from benchmark_logger import (
    BenchmarkLogger,
    UnifiedBenchmarkRecord
)
from dashboard_api import app


def log_todays_benchmark():
    """
    Logs a fresh benchmark record for today.
    In production replace these values with real
    results from running your full benchmark suite.
    """
    logger = BenchmarkLogger()

    record = UnifiedBenchmarkRecord(
        run_name="todays_benchmark",
        commit_sha="latest",
        trigger="manual",

        # Paste your real scores here
        # from running Days 2-11

        # Quality — from Day 4
        faithfulness=0.891,
        context_recall=0.812,
        context_precision=0.834,
        answer_relevancy=0.876,

        # Latency — from Day 3
        p50_latency_ms=891,
        p95_latency_ms=1821,
        p99_latency_ms=2934,
        mean_latency_ms=1102,

        # Cost — from Day 7
        avg_cost_per_query=0.000031,
        monthly_projection_usd=93.0,

        # Retrieval — from Day 5
        retrieval_ndcg=0.774,
        retrieval_precision=0.689,
        retrieval_recall=0.798,
        retrieval_mrr=0.821,

        # Agentic — from Day 9
        avg_trajectory_length=3.2,
        avg_trajectory_efficiency=0.71,
        tool_call_accuracy=0.94,
        loop_rate=0.0,

        # Safety — from Day 10
        refusal_rate=1.0,
        pii_leakage_rate=0.0,

        # Drift — from Day 11
        query_drift_detected=False,
        quality_drift_detected=False,

        # Regression — from Day 10
        gates_passed=8,
        gates_total=8,
        regression_check_passed=True,

        model_used="gpt-4o-mini",
        eval_dataset_size=50
    )

    run_id = logger.log_benchmark(record)
    print(f"\n✅ Today's benchmark logged: {run_id}")
    return run_id


if __name__ == "__main__":
    print("=" * 60)
    print("AI PERFORMANCE DASHBOARD")
    print("=" * 60)

    # Log today's benchmark
    log_todays_benchmark()

    print("\n🚀 Starting dashboard API on port 8000...")
    print("   Open the dashboard artifact to view results")
    print("   API available at: http://localhost:8000")
    print("   Docs available at: http://localhost:8000/docs")
    print("\n   Press Ctrl+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### Step 7 — Run everything

**Terminal 1 — Start the dashboard API:**

```bash
cd day-12-dashboard
python run_dashboard.py
```

You should see:

```
============================================================
AI PERFORMANCE DASHBOARD
============================================================
✅ MLflow initialized: ai-performance-benchmarking
✅ Today's benchmark logged: abc123...

🚀 Starting dashboard API on port 8000...
   Open the dashboard artifact to view results
   API available at: http://localhost:8000
   Docs available at: http://localhost:8000/docs
```

The dashboard React artifact fetches from `http://localhost:8000/api/dashboard` and renders the full unified view — scorecard, quality trends, performance triangle, latency chart, and regression history all in one place.

---

## What You Just Built

The unified dashboard ties together every day of the series:

- **Day 2** baseline scores visible as the starting point on trend lines
- **Day 3** P99 latency tracked over time with its own trend chart
- **Day 4** RAGAS quality scores on the scorecard and in the triangle
- **Day 5** retrieval metrics logged alongside quality
- **Day 6** A/B test results visible as named runs in MLflow
- **Day 7** cost per query and monthly projection on the scorecard
- **Day 9** trajectory efficiency and tool call accuracy logged
- **Day 10** regression gate history with pass/fail per commit
- **Day 11** drift alerts surfaced automatically in the alert banner

One view. Complete picture. Thirty seconds to understand system health.

---

## ✅ Day 12 Checklist

- [ ] `benchmark_logger.py` connects to MLflow without errors
- [ ] `run_dashboard.py` seeds demo data and starts the API
- [ ] Dashboard loads and shows the scorecard
- [ ] Quality trend chart shows 8 weeks of data
- [ ] Performance triangle renders with correct scores
- [ ] Regression history shows pass/fail per run
- [ ] Alerts appear for any degraded metrics
- [ ] Your real benchmark scores from Days 2-11 are in `run_dashboard.py`
- [ ] MLflow UI visible at `http://localhost:5000` via `mlflow ui`

---

## 🎯 Interview Bits — Day 12

**Q: How do you communicate AI system performance to non-technical stakeholders?**
*Translate metrics into business language on a unified dashboard. Faithfulness becomes "answer accuracy." P99 latency becomes "worst-case user wait time." Cost per query becomes "monthly infrastructure projection." The scorecard with traffic light status makes it immediately readable by anyone — no ML knowledge required.*

**Q: What is the value of tracking AI metrics over time rather than as point-in-time snapshots?**
*Point-in-time snapshots show you where you are. Trend lines show you where you are going. A system with a faithfulness score of 0.82 looks fine as a snapshot. If that same score was 0.91 four weeks ago the trend line reveals a regression in progress. Dashboards that show history catch problems that current-state monitoring misses entirely.*

**Q: How would you use MLflow specifically for AI benchmarking versus traditional ML experiment tracking?**
*Traditional MLflow tracks training experiments — loss curves, validation accuracy, hyperparameters. For AI benchmarking you use it to track production performance experiments — RAGAS scores, latency distributions, cost per query, safety metrics — across time and code changes. The run comparison feature lets you diff any two benchmark snapshots to understand exactly what changed and by how much.*

**Q: What is the minimum viable observability stack for a production RAG system?**
*Three layers: LangSmith for trace-level observability — every query logged with inputs, outputs, and latency. Prometheus for metrics-level monitoring — rolling aggregates, error rates, and threshold alerts. A weekly benchmark run with RAGAS quality scoring on a sampled production subset for quality-level observability. The unified dashboard ties all three into one view.*

---

## The Complete Series — What You Now Know

You started with a system that worked but could not be measured.

You end with a system that is measured at every level.

Here is what you built across 12 days:

```
Day 0  → Visibility        LangSmith tracing on every query
Day 1  → Mental model      Quality · Speed · Cost triangle
Day 2  → Baseline          Your stake in the ground
Day 3  → Latency           P99 truth instead of average lies
Day 4  → Quality           RAGAS — faithfulness, recall, relevancy
Day 5  → Retrieval         Chunking, embeddings, hybrid search
Day 6  → Experiments       A/B testing with statistical significance
Day 7  → Cost              Token tracking, model comparison, routing
Day 8  → Load              Concurrent users, async fixes, breaking points
Day 9  → Agents            Trajectories, loops, step-level quality
Day 10 → Regression        CI/CD gates — never ship a regression again
Day 11 → Observability     Production drift detected before users feel it
Day 12 → Dashboard         Everything in one place
```

**79 concepts. 12 days. One complete toolkit.**

Everything lives at:
```
github.com/your-username/production-ai-performance-benchmarking-toolkit
```

---

## One Final Thought

The engineers who build AI systems that stay good over time are not the ones with the best models.

They are the ones who measure relentlessly.

They know their baselines. They catch regressions before users do. They make decisions with evidence instead of intuition. They watch their live systems the way a pilot watches instruments — not because they expect a crash, but because flying blind is not an option.

That is what this series gave you.

Not just the tools.

The habit.

Measure first. Change second. Verify always.

---

*Thank you for following this series.*
*If it helped you — share it with someone building AI systems.*
*They are flying blind too.*
*Show them the instruments.*

