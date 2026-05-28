"""
A/B testing suite — compare two prompt versions on identical queries.
Uses Welch's t-test to distinguish real improvements from LLM noise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_recall, faithfulness
from scipy import stats

from src.experiments.prompt_registry import PromptRegistry, PromptVersion
from src.rag.pipeline import RAGPipeline

load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class ExperimentResult:
    experiment_name: str
    prompt_version: str
    prompt_hash: str
    faithfulness_scores: list[float] = field(default_factory=list)
    context_recall_scores: list[float] = field(default_factory=list)
    answer_relevancy_scores: list[float] = field(default_factory=list)
    latency_scores: list[float] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def avg_faithfulness(self) -> float:
        return float(np.mean(self.faithfulness_scores)) if self.faithfulness_scores else 0.0

    @property
    def avg_context_recall(self) -> float:
        return float(np.mean(self.context_recall_scores)) if self.context_recall_scores else 0.0

    @property
    def avg_answer_relevancy(self) -> float:
        return float(np.mean(self.answer_relevancy_scores)) if self.answer_relevancy_scores else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return float(np.mean(self.latency_scores)) if self.latency_scores else 0.0

    def summary(self) -> str:
        return (
            f"\n{'─'*55}"
            f"\n🧪 {self.experiment_name} — {self.prompt_version}"
            f"\n{'─'*55}"
            f"\n  Faithfulness:    {self.avg_faithfulness:.3f} "
            f"(±{np.std(self.faithfulness_scores):.3f})"
            f"\n  Context Recall:  {self.avg_context_recall:.3f} "
            f"(±{np.std(self.context_recall_scores):.3f})"
            f"\n  Ans Relevancy:   {self.avg_answer_relevancy:.3f}"
            f"\n  Avg Latency:     {self.avg_latency_ms:.0f}ms"
            f"\n  Queries tested:  {len(self.faithfulness_scores)}"
        )


class ABTester:

    def __init__(self, registry: PromptRegistry) -> None:
        self.registry = registry
        self.results: list[ExperimentResult] = []

    def run_experiment(
        self,
        experiment_name: str,
        prompt_version: PromptVersion,
        eval_dataset: list[dict],
        contexts_per_query: list[list[str]],
        model: str = "gpt-4o-mini",
    ) -> ExperimentResult:
        print(f"\n🧪 Running: {experiment_name}  ({prompt_version.name})")

        pipeline = RAGPipeline(
            model=model,
            system_prompt=prompt_version.system_prompt,
            prompt_version=prompt_version.name,
        )

        result = ExperimentResult(
            experiment_name=experiment_name,
            prompt_version=prompt_version.name,
            prompt_hash=prompt_version.prompt_hash,
        )

        questions, answers, ground_truths = [], [], []

        for i, (example, contexts) in enumerate(
            zip(eval_dataset, contexts_per_query), 1
        ):
            print(f"  [{i:02d}/{len(eval_dataset)}]", end=" ")
            qr = pipeline.query(example["question"])
            result.latency_scores.append(qr.latency_ms)
            questions.append(example["question"])
            answers.append(qr.answer)
            ground_truths.append(example["ground_truth"])
            print(f"✓ {qr.latency_ms:.0f}ms")

        print("  Scoring with RAGAS...")
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts_per_query,
            "ground_truth": ground_truths,
        })
        scores = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevancy, context_recall])
        df = scores.to_pandas()

        result.faithfulness_scores = df["faithfulness"].tolist()
        result.context_recall_scores = df["context_recall"].tolist()
        result.answer_relevancy_scores = df["answer_relevancy"].tolist()

        self.registry.update_scores(
            name=prompt_version.name,
            faithfulness=result.avg_faithfulness,
            context_recall=result.avg_context_recall,
            answer_relevancy=result.avg_answer_relevancy,
            avg_latency_ms=result.avg_latency_ms,
        )

        self.results.append(result)
        print(result.summary())
        return result

    def compare(self, a: ExperimentResult, b: ExperimentResult) -> None:
        print(f"\n{'━'*60}")
        print("A/B COMPARISON")
        print(f"{'━'*60}")
        print(f"  Version A: {a.prompt_version}")
        print(f"  Version B: {b.prompt_version}")
        print(f"{'─'*60}")
        print(f"{'Metric':<22} {'A':>8} {'B':>8} {'Delta':>8} {'Significant':>13}")
        print(f"{'─'*60}")

        metrics = [
            ("Faithfulness",    a.faithfulness_scores,    b.faithfulness_scores),
            ("Context Recall",  a.context_recall_scores,  b.context_recall_scores),
            ("Ans Relevancy",   a.answer_relevancy_scores, b.answer_relevancy_scores),
        ]

        winner = {"a": 0, "b": 0}
        for name, sa, sb in metrics:
            ma, mb = np.mean(sa), np.mean(sb)
            delta = mb - ma
            _, p = stats.ttest_ind(sa, sb)
            sig = p < 0.05
            direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            label = "✅ Yes" if sig else "⬜ Noise"
            print(f"{name:<22} {ma:>8.3f} {mb:>8.3f} "
                  f"{direction}{abs(delta):>6.3f} {label:>13}")
            if sig:
                winner["b" if delta > 0 else "a"] += 1

        lat_a, lat_b = np.mean(a.latency_scores), np.mean(b.latency_scores)
        print(f"{'Latency (ms)':<22} {lat_a:>8.0f} {lat_b:>8.0f} "
              f"{'↑' if lat_b > lat_a else '↓'}{abs(lat_b - lat_a):>5.0f}ms "
              f"{'(informational)':>13}")
        print(f"{'─'*60}")

        if winner["b"] > winner["a"]:
            print(f"\n🏆 WINNER: Version B ({b.prompt_version})")
            print(f"   Statistically better on {winner['b']} metric(s)")
        elif winner["a"] > winner["b"]:
            print(f"\n🏆 WINNER: Version A ({a.prompt_version})")
            print(f"   ⚠️  Version B regressed — do not ship")
        else:
            print(f"\n🤝 NO CLEAR WINNER — differences are noise")
            print(f"   Keep Version A — do not change without signal")
        print(f"{'━'*60}\n")

    def save_results(self, path: Optional[Path] = None) -> None:
        out = path or DATA_DIR / "experiment_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "experiment": r.experiment_name,
                "prompt_version": r.prompt_version,
                "avg_faithfulness": r.avg_faithfulness,
                "avg_context_recall": r.avg_context_recall,
                "avg_answer_relevancy": r.avg_answer_relevancy,
                "avg_latency_ms": r.avg_latency_ms,
                "timestamp": r.timestamp,
            }
            for r in self.results
        ]
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Experiment results saved → {out}")
