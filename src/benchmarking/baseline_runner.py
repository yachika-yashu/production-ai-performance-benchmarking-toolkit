"""
Baseline runner — measures quality, latency, and cost before any changes.
Run once. Lock the output file. Every future benchmark compares against it.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import tiktoken
from dotenv import load_dotenv

from src.rag.pipeline import RAGPipeline, QueryResult

load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ENCODER = tiktoken.get_encoding("cl100k_base")


def load_eval_dataset(path: Optional[Path] = None) -> list[dict]:
    p = path or DATA_DIR / "eval_dataset.json"
    with open(p) as f:
        return json.load(f)


def run_baseline(
    pipeline: Optional[RAGPipeline] = None,
    dataset_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    pipeline = pipeline or RAGPipeline()
    dataset = load_eval_dataset(dataset_path)
    out_dir = output_dir or DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running baseline on {len(dataset)} examples...")
    print(f"Model: {pipeline.model}  |  Prompt: {pipeline.prompt_version}\n")

    results: list[dict] = []
    latencies: list[float] = []
    costs: list[float] = []

    for i, example in enumerate(dataset, 1):
        print(f"  [{i:02d}/{len(dataset)}] {example['question'][:60]}...", end=" ")
        result: QueryResult = pipeline.query(example["question"])
        latencies.append(result.latency_ms)
        costs.append(result.cost_usd)

        results.append({
            "question": example["question"],
            "ground_truth": example["ground_truth"],
            "answer": result.answer,
            "contexts": result.contexts,
            "latency_ms": result.latency_ms,
            "cost_usd": result.cost_usd,
            "category": example.get("category", "unknown"),
            "difficulty": example.get("difficulty", "unknown"),
            "model": result.model,
            "prompt_version": result.prompt_version,
        })
        print(f"✓ {result.latency_ms:.0f}ms  ${result.cost_usd:.6f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_path = out_dir / f"baseline_{ts}.json"

    payload = {
        "metadata": {
            "model": pipeline.model,
            "prompt_version": pipeline.prompt_version,
            "dataset_size": len(dataset),
            "created_at": datetime.now().isoformat(),
            "avg_latency_ms": round(float(np.mean(latencies)), 2),
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
            "total_cost_usd": round(sum(costs), 6),
            "avg_cost_per_query": round(float(np.mean(costs)), 8),
        },
        "results": results,
    }

    with open(baseline_path, "w") as f:
        json.dump(payload, f, indent=2)

    _print_summary(payload["metadata"])
    print(f"\nBaseline locked → {baseline_path}")
    print("Do not modify this file. Every future benchmark compares against it.\n")
    return baseline_path


def _print_summary(meta: dict) -> None:
    print(f"\n{'━'*55}")
    print("BASELINE COMPLETE")
    print(f"{'━'*55}")
    print(f"  Examples:        {meta['dataset_size']}")
    print(f"  Avg latency:     {meta['avg_latency_ms']:.0f}ms")
    print(f"  P50 latency:     {meta['p50_latency_ms']:.0f}ms")
    print(f"  P95 latency:     {meta['p95_latency_ms']:.0f}ms")
    print(f"  P99 latency:     {meta['p99_latency_ms']:.0f}ms")
    print(f"  Total cost:      ${meta['total_cost_usd']:.4f}")
    print(f"  Cost per query:  ${meta['avg_cost_per_query']:.6f}")
    print(f"  Quality scores:  run src/quality/ragas_evaluator.py next")
    print(f"{'━'*55}")


if __name__ == "__main__":
    run_baseline()
