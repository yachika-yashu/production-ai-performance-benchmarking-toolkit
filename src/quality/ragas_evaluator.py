"""
RAGAS quality evaluator — faithfulness, context recall, answer relevancy, context precision.
Run after baseline_runner.py to add quality scores to any benchmark result file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

METRICS = [faithfulness, context_recall, answer_relevancy, context_precision]


def evaluate_results(results: list[dict]) -> pd.DataFrame:
    """
    Runs RAGAS on a list of result dicts.
    Each dict must have: question, answer, contexts, ground_truth.
    Returns a DataFrame with per-example scores + aggregate row.
    """
    dataset = Dataset.from_dict({
        "question":   [r["question"] for r in results],
        "answer":     [r["answer"] for r in results],
        "contexts":   [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })

    print("  Running RAGAS evaluation (LLM-as-judge — takes 1-3 min)...")
    scores = evaluate(dataset=dataset, metrics=METRICS)
    df = scores.to_pandas()
    return df


def score_baseline_file(
    baseline_path: Path,
    output_path: Optional[Path] = None,
) -> dict:
    """
    Loads a baseline JSON, runs RAGAS, and saves an enriched file with quality scores.
    Returns the aggregate quality scores dict.
    """
    with open(baseline_path) as f:
        baseline = json.load(f)

    results = baseline["results"]
    print(f"\nScoring {len(results)} examples from {baseline_path.name}...")

    df = evaluate_results(results)

    aggregate = {
        "faithfulness":       round(float(df["faithfulness"].mean()), 4),
        "context_recall":     round(float(df["context_recall"].mean()), 4),
        "answer_relevancy":   round(float(df["answer_relevancy"].mean()), 4),
        "context_precision":  round(float(df["context_precision"].mean()), 4),
    }

    # Attach per-example scores back to results
    for i, row in df.iterrows():
        results[i]["faithfulness"]      = round(float(row["faithfulness"]), 4)
        results[i]["context_recall"]    = round(float(row["context_recall"]), 4)
        results[i]["answer_relevancy"]  = round(float(row["answer_relevancy"]), 4)
        results[i]["context_precision"] = round(float(row["context_precision"]), 4)

    baseline["metadata"]["quality_scores"] = aggregate
    baseline["results"] = results

    out = output_path or baseline_path.with_name(
        baseline_path.stem + "_with_quality.json"
    )
    with open(out, "w") as f:
        json.dump(baseline, f, indent=2)

    _print_quality_report(aggregate, len(results))
    print(f"  Quality scores added → {out}\n")
    return aggregate


def _print_quality_report(scores: dict, n: int) -> None:
    print(f"\n{'━'*50}")
    print(f"QUALITY EVALUATION — {n} examples")
    print(f"{'━'*50}")
    status = lambda v, good, warn: "✅" if v >= good else ("⚠️ " if v >= warn else "🔴")
    print(f"  Faithfulness:      {scores['faithfulness']:.3f}  "
          f"{status(scores['faithfulness'], 0.85, 0.70)}")
    print(f"  Context Recall:    {scores['context_recall']:.3f}  "
          f"{status(scores['context_recall'], 0.80, 0.65)}")
    print(f"  Answer Relevancy:  {scores['answer_relevancy']:.3f}  "
          f"{status(scores['answer_relevancy'], 0.80, 0.65)}")
    print(f"  Context Precision: {scores['context_precision']:.3f}  "
          f"{status(scores['context_precision'], 0.75, 0.60)}")
    print(f"{'━'*50}")


if __name__ == "__main__":
    import glob
    files = sorted(glob.glob(str(DATA_DIR / "baseline_*.json")))
    files = [f for f in files if "with_quality" not in f]
    if not files:
        print("No baseline files found. Run src/benchmarking/baseline_runner.py first.")
    else:
        score_baseline_file(Path(files[-1]))
