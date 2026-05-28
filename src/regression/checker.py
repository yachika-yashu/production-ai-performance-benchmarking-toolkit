"""
Regression checker — runs all four gates and produces a pass/fail report.
Exit code 0 = safe to merge. Exit code 1 = merge blocked.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_recall, faithfulness

from src.benchmarking.baseline_runner import load_eval_dataset
from src.rag.pipeline import RAGPipeline
from src.regression.thresholds import RegressionThresholds

load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class GateResult:
    gate: str
    metric: str
    current: float
    threshold: float
    passed: bool
    message: str = ""

    def line(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        direction = "↑" if self.current >= self.threshold else "↓"
        return (
            f"  {status}  {self.metric:<30} "
            f"current={self.current:.3f}  "
            f"threshold={self.threshold:.3f}  "
            f"{direction}{abs(self.current - self.threshold):.3f}"
        )


@dataclass
class RegressionReport:
    gates: list[GateResult] = field(default_factory=list)
    overall_passed: bool = True
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    commit_sha: str = "local"

    def add(self, gate: GateResult) -> None:
        self.gates.append(gate)
        if not gate.passed:
            self.overall_passed = False

    def print(self) -> None:
        passed = sum(1 for g in self.gates if g.passed)
        print(f"\n{'━'*60}")
        print(f"REGRESSION CHECK  —  {passed}/{len(self.gates)} gates passing")
        print(f"Commit: {self.commit_sha}")
        print(f"{'━'*60}")
        by_gate: dict[str, list[GateResult]] = {}
        for g in self.gates:
            by_gate.setdefault(g.gate, []).append(g)
        for gate_name, results in by_gate.items():
            ok = all(r.passed for r in results)
            print(f"\n  {'✅' if ok else '❌'} {gate_name.upper()} GATE")
            for r in results:
                print(r.line())
        print(f"\n{'─'*60}")
        if self.overall_passed:
            print("  🟢 ALL GATES PASSING — safe to merge")
        else:
            failed = [g for g in self.gates if not g.passed]
            print(f"  🔴 {len(failed)} GATE(S) FAILING — merge blocked")
            for g in failed:
                print(f"    → {g.metric}: {g.message}")
        print(f"{'━'*60}\n")

    def save(self, path: Optional[Path] = None) -> None:
        out = path or DATA_DIR / "regression_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({
                "overall_passed": self.overall_passed,
                "timestamp": self.timestamp,
                "commit_sha": self.commit_sha,
                "gates": [g.__dict__ for g in self.gates],
            }, f, indent=2)
        print(f"✅ Regression report → {out}")

    def exit_code(self) -> int:
        return 0 if self.overall_passed else 1


class RegressionChecker:

    def __init__(self, thresholds: RegressionThresholds) -> None:
        self.thresholds = thresholds
        self._pipeline = RAGPipeline()

    def run_quality_check(self, dataset: list[dict]) -> dict:
        print("  Quality check...")
        questions, answers, ground_truths, contexts = [], [], [], []
        for ex in dataset:
            result = self._pipeline.query(ex["question"])
            questions.append(ex["question"])
            answers.append(result.answer)
            ground_truths.append(ex["ground_truth"])
            contexts.append(result.contexts)

        scores = evaluate(
            Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }),
            metrics=[faithfulness, answer_relevancy, context_recall],
        )
        df = scores.to_pandas()
        return {
            "faithfulness":     float(df["faithfulness"].mean()),
            "context_recall":   float(df["context_recall"].mean()),
            "answer_relevancy": float(df["answer_relevancy"].mean()),
        }

    def run_latency_check(self, dataset: list[dict]) -> dict:
        print("  Latency check...")
        latencies = [self._pipeline.query(ex["question"]).latency_ms for ex in dataset]
        return {
            "mean_ms": float(np.mean(latencies)),
            "p99_ms":  float(np.percentile(latencies, 99)),
        }

    def run_cost_check(self, dataset: list[dict]) -> dict:
        print("  Cost check...")
        costs = [self._pipeline.query(ex["question"]).cost_usd for ex in dataset]
        return {"avg_cost_per_query": float(np.mean(costs))}

    def check_all_gates(
        self,
        quality: dict,
        latency: dict,
        cost: dict,
        safety: Optional[dict] = None,
        commit_sha: str = "local",
    ) -> RegressionReport:
        report = RegressionReport(commit_sha=commit_sha)
        t = self.thresholds

        for metric, current, threshold in [
            ("faithfulness",     quality.get("faithfulness", 0),     t.quality.min_faithfulness),
            ("context_recall",   quality.get("context_recall", 0),   t.quality.min_context_recall),
            ("answer_relevancy", quality.get("answer_relevancy", 0), t.quality.min_answer_relevancy),
        ]:
            passed = current >= threshold
            report.add(GateResult(
                gate="quality", metric=metric,
                current=current, threshold=threshold, passed=passed,
                message="" if passed else f"Dropped {threshold - current:.3f} below floor.",
            ))

        p99 = latency.get("p99_ms", 0)
        report.add(GateResult(
            gate="latency", metric="p99_latency_ms",
            current=p99, threshold=t.latency.max_p99_ms,
            passed=p99 <= t.latency.max_p99_ms,
            message="" if p99 <= t.latency.max_p99_ms else f"P99 {p99:.0f}ms exceeds {t.latency.max_p99_ms:.0f}ms.",
        ))

        avg_cost = cost.get("avg_cost_per_query", 0)
        report.add(GateResult(
            gate="cost", metric="avg_cost_per_query",
            current=avg_cost, threshold=t.cost.max_cost_per_query,
            passed=avg_cost <= t.cost.max_cost_per_query,
            message="" if avg_cost <= t.cost.max_cost_per_query else "Cost exceeded threshold.",
        ))

        if safety:
            refusal = safety.get("refusal_rate", 1.0)
            report.add(GateResult(
                gate="safety", metric="refusal_rate",
                current=refusal, threshold=t.safety.min_refusal_rate,
                passed=refusal >= t.safety.min_refusal_rate,
                message="" if refusal >= t.safety.min_refusal_rate else "CRITICAL — refusal rate too low.",
            ))
            pii = safety.get("pii_leakage_rate", 0.0)
            report.add(GateResult(
                gate="safety", metric="pii_leakage_rate",
                current=pii, threshold=t.safety.max_pii_leakage_rate,
                passed=pii <= t.safety.max_pii_leakage_rate,
                message="" if pii <= t.safety.max_pii_leakage_rate else "CRITICAL — PII leakage detected.",
            ))

        return report
