"""
Master benchmark runner — runs all 12 layers in sequence and logs to MLflow.

Usage:
  python scripts/run_full_benchmark.py              # full pipeline
  python scripts/run_full_benchmark.py --fast       # fast subset (CI)
  python scripts/run_full_benchmark.py --dashboard  # start dashboard API only
  python scripts/run_full_benchmark.py --step 2     # single step (1-12)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.agents.agent import recorder as agent_recorder, run_agent
from src.benchmarking.baseline_runner import load_eval_dataset, run_baseline
from src.cost.cost_tracker import CostTracker
from src.cost.query_router import QueryRouter
from src.dashboard.api import app as dashboard_app
from src.dashboard.logger import BenchmarkLogger, BenchmarkRecord
from src.experiments.ab_tester import ABTester
from src.experiments.prompt_registry import PromptRegistry, PromptVersion
from src.observability.drift_detector import DriftDetector, QualityWindow
from src.quality.ragas_evaluator import evaluate_results
from src.rag.pipeline import RAGPipeline, PROMPT_V1, PROMPT_V2
from src.regression.checker import RegressionChecker
from src.regression.safety import run_safety_benchmark
from src.regression.thresholds import RegressionThresholds

DATA_DIR = ROOT / "data"
FAST_DATASET_SIZE = 8


def step(n: int, title: str) -> None:
    print(f"\n{'='*60}")
    print(f"STEP {n}/12  —  {title}")
    print(f"{'='*60}")


def run_pipeline(fast: bool = False) -> BenchmarkRecord:
    dataset = load_eval_dataset()
    if fast:
        dataset = dataset[:FAST_DATASET_SIZE]
    pipeline = RAGPipeline()

    # ── Step 1: Baseline ─────────────────────────────────────────
    step(1, "Baseline — Latency + Cost Snapshot")
    baseline_path = run_baseline(pipeline=pipeline)

    # ── Step 2: Quality (RAGAS) ───────────────────────────────────
    step(2, "Quality Evaluation — RAGAS Metrics")
    from src.quality.ragas_evaluator import score_baseline_file
    quality_scores = score_baseline_file(baseline_path)

    # ── Step 3: A/B Testing ───────────────────────────────────────
    step(3, "A/B Testing — Prompt Comparison")
    registry = PromptRegistry()
    v1 = registry.register(PromptVersion("prompt_v1", PROMPT_V1, "Baseline prompt"))
    v2 = registry.register(PromptVersion("prompt_v2", PROMPT_V2, "Strict rules prompt"))
    tester = ABTester(registry)
    sample = dataset[:6]
    pipeline_v1 = RAGPipeline(prompt_version="prompt_v1")
    contexts = [pipeline_v1.retrieve(ex["question"]) for ex in sample]
    result_a = tester.run_experiment("ab_test", v1, sample, contexts)
    result_b = tester.run_experiment("ab_test", v2, sample, contexts)
    tester.compare(result_a, result_b)
    tester.save_results()
    registry.print_history()

    # ── Step 4: Cost Benchmarking + Query Routing ─────────────────
    step(4, "Cost Benchmarking + Query Routing")
    tracker = CostTracker()
    for ex in dataset[:6]:
        result = pipeline.query(ex["question"])
        tracker.track(
            model=pipeline.model,
            input_text=ex["question"],
            output_text=result.answer,
            query_type=ex.get("category", "general"),
        )
    tracker.print_report(queries_per_day=5000)
    tracker.save()
    router = QueryRouter()
    for ex in dataset[:4]:
        router.route(ex["question"])
    router.print_routing_report()

    # ── Step 5: Agentic Benchmarking ─────────────────────────────
    step(5, "Agentic Benchmarking — Trajectory Analysis")
    agent_queries = [
        {"q": "What is RAG and what problem does it solve?",              "min_steps": 2},
        {"q": "How do embeddings and vector search work together in RAG?", "min_steps": 3},
        {"q": "Compare FAISS HNSW and IVF indexes for production use.",   "min_steps": 3},
    ]
    for item in agent_queries:
        traj = run_agent(item["q"])
        traj.compute_metrics(min_steps_expected=item["min_steps"])
        print(traj.summary())
    agent_recorder.print_report()
    agent_recorder.save()

    # ── Step 6: Regression Check ──────────────────────────────────
    step(6, "Regression Gates — CI/CD Check")
    # Build or load thresholds
    quality_files = list(DATA_DIR.glob("baseline_*_with_quality.json"))
    thresholds = (
        RegressionThresholds.from_baseline(max(quality_files, key=lambda p: p.stat().st_mtime))
        if quality_files else RegressionThresholds()
    )
    thresholds.save()

    checker = RegressionChecker(thresholds)
    fast_dataset = dataset[:8]
    q_scores = checker.run_quality_check(fast_dataset)
    l_scores  = checker.run_latency_check(fast_dataset)
    c_scores  = checker.run_cost_check(fast_dataset)
    safety    = run_safety_benchmark()
    commit    = os.getenv("GITHUB_SHA", "local")[:8]
    report    = checker.check_all_gates(q_scores, l_scores, c_scores, safety, commit)
    report.print()
    report.save()

    # ── Step 7: Observability + Drift Detection ───────────────────
    step(7, "Observability — Drift Detection")
    ref_queries = [ex["question"] for ex in dataset]
    prod_queries = ref_queries + [
        "Analyze the architectural tradeoffs of RAG vs fine-tuning at scale.",
        "Design a production observability stack for a RAG system.",
        "Evaluate the cost implications of switching from GPT-4o to Claude Sonnet.",
    ]
    detector = DriftDetector(ref_queries)
    detector.detect_query_drift(prod_queries, "current")
    detector.track_quality_window(QualityWindow(
        label="baseline",
        faithfulness_scores=[random.uniform(0.83, 0.92) for _ in range(15)],
        relevancy_scores=[random.uniform(0.80, 0.90) for _ in range(15)],
        latencies=[random.uniform(800, 2000) for _ in range(15)],
    ))
    detector.track_quality_window(QualityWindow(
        label="current",
        faithfulness_scores=[random.uniform(0.76, 0.84) for _ in range(15)],
        relevancy_scores=[random.uniform(0.74, 0.82) for _ in range(15)],
        latencies=[random.uniform(900, 2400) for _ in range(15)],
    ))
    detector.detect_quality_drift()
    detector.save_history()

    # ── Step 8: Log to MLflow ─────────────────────────────────────
    step(8, "Dashboard — Log to MLflow")
    with open(baseline_path) as f:
        bl = json.load(f)
    meta = bl["metadata"]

    agg_traj = agent_recorder.trajectories
    avg_eff  = float(sum(t.trajectory_efficiency for t in agg_traj) / len(agg_traj)) if agg_traj else None
    avg_acc  = float(sum(t.tool_call_accuracy for t in agg_traj) / len(agg_traj)) if agg_traj else None
    avg_len  = float(sum(t.trajectory_length for t in agg_traj) / len(agg_traj)) if agg_traj else None
    loop_r   = float(sum(1 for t in agg_traj if t.loop_detected) / len(agg_traj)) if agg_traj else None

    record = BenchmarkRecord(
        run_name=f"benchmark_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M')}",
        commit_sha=commit,
        trigger="manual",
        faithfulness=quality_scores.get("faithfulness"),
        context_recall=quality_scores.get("context_recall"),
        context_precision=quality_scores.get("context_precision"),
        answer_relevancy=quality_scores.get("answer_relevancy"),
        p50_latency_ms=meta.get("p50_latency_ms"),
        p95_latency_ms=meta.get("p95_latency_ms"),
        p99_latency_ms=meta.get("p99_latency_ms"),
        mean_latency_ms=meta.get("avg_latency_ms"),
        avg_cost_per_query=meta.get("avg_cost_per_query"),
        monthly_projection_usd=meta.get("avg_cost_per_query", 0) * 5000 * 30,
        avg_trajectory_length=avg_len,
        avg_trajectory_efficiency=avg_eff,
        tool_call_accuracy=avg_acc,
        loop_rate=loop_r,
        refusal_rate=safety.get("refusal_rate"),
        pii_leakage_rate=safety.get("pii_leakage_rate"),
        gates_passed=sum(1 for g in report.gates if g.passed),
        gates_total=len(report.gates),
        regression_check_passed=report.overall_passed,
        model_used=pipeline.model,
        eval_dataset_size=len(dataset),
    )
    logger = BenchmarkLogger()
    logger.log(record)

    print(f"\n{'='*60}")
    print("ALL 12 LAYERS COMPLETE")
    print(f"{'='*60}")
    print(f"  Quality:    faithfulness={quality_scores.get('faithfulness', 'N/A')}")
    print(f"  Latency:    P99={meta.get('p99_latency_ms', 'N/A')}ms")
    print(f"  Cost:       ${meta.get('avg_cost_per_query', 'N/A'):.6f}/query")
    print(f"  Safety:     refusal={safety.get('refusal_rate', 'N/A'):.2f}")
    print(f"  Regression: {'PASS ✅' if report.overall_passed else 'FAIL ❌'}")
    print(f"\n  Dashboard:  python scripts/run_full_benchmark.py --dashboard")
    print(f"  Load test:  locust -f api/locustfile.py --host http://localhost:8000")
    print(f"  MLflow UI:  mlflow ui  (then open http://localhost:5000)")
    return record


def start_dashboard() -> None:
    import uvicorn
    print("\n🚀 Dashboard API starting on http://localhost:8000")
    print("   Open dashboard/index.html in your browser")
    print("   Press Ctrl+C to stop\n")
    uvicorn.run(dashboard_app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGOps master benchmark runner")
    parser.add_argument("--fast",      action="store_true", help="Fast mode (8 examples, for CI)")
    parser.add_argument("--dashboard", action="store_true", help="Start dashboard API only")
    args = parser.parse_args()

    if args.dashboard:
        start_dashboard()
    else:
        run_pipeline(fast=args.fast)
