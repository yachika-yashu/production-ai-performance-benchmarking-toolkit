# AI Performance Benchmarking: From Zero to Production
### Day 4 of 12 — Quality Metrics: How Do You Know Your AI Is Actually Telling The Truth?

---

*This is Day 4 of a 12-part series. If you're just joining, start with **[Day 0](#)** for setup, **[Day 1](#)** for the mental model, **[Day 2](#)** for baselines, and **[Day 3](#)** for latency. Everything here builds on those foundations.*

---

## Fast Is Great. Fast And Wrong Is A Disaster.

Day 3 gave you speed.

You know your P99 latency. You know which stage is the bottleneck. You have a profiler that tells you exactly where the time goes.

Now a different question.

The answers are arriving fast.

But are they right?

This sounds simple. It is not.

Because with AI systems — especially RAG systems — wrong does not always look wrong.

Wrong often looks confident.

Wrong often looks fluent.

Wrong often looks exactly like right — until someone with domain knowledge reads it carefully.

That is the uniquely dangerous failure mode of large language models.

A traditional software bug crashes or returns an error. You know immediately something is broken.

An LLM confidently fabricates a fact, cites a source that does not exist, or answers a question about Patient A using context retrieved for Patient B.

No error. No crash. Just a smooth, well-written, completely wrong answer.

**This is called hallucination. And latency metrics cannot catch it.**

You need quality metrics. And that is what today is about.

---

## The Four Questions Quality Metrics Answer

Before we touch any code, let us build the mental model.

Every quality metric in a RAG system answers one of four questions:

---

**Question 1 — Did the answer stick to the retrieved context?**

This is **Faithfulness**.

Your AI retrieved some documents. Did it base its answer on those documents? Or did it wander off and start inventing things?

A faithful answer uses the retrieved context as its source of truth.
An unfaithful answer ignores the context and generates from the model's parametric memory — which may be outdated, incorrect, or completely fabricated.

Faithfulness score range: 0 to 1. Higher is better.

---

**Question 2 — Did we retrieve the right context in the first place?**

This is **Context Recall**.

Even a perfectly faithful answer is wrong if the retrieved context was wrong.

Context recall measures whether the relevant information was actually present in what you retrieved. If the answer requires knowing X and X was never in the retrieved documents — the system had no chance of being correct, regardless of how faithfully it used what it had.

Context recall range: 0 to 1. Higher is better.

---

**Question 3 — Was the retrieved context actually relevant to the question?**

This is **Context Precision**.

You retrieved 5 documents. How many of them were actually useful for answering this specific question?

Low context precision means you are flooding the LLM with noise. The relevant signal is buried in irrelevant content. This hurts both quality and cost — you are paying for tokens that do not help.

Context precision range: 0 to 1. Higher is better.

---

**Question 4 — Did the answer actually address what was asked?**

This is **Answer Relevancy**.

A faithful, well-grounded answer can still miss the point.

If someone asks "What are the side effects of this medication?" and the answer faithfully discusses the medication's dosage — it is grounded but irrelevant.

Answer relevancy measures whether the response actually addressed the question asked.

Answer relevancy range: 0 to 1. Higher is better.

---

## The Dangerous Combination Nobody Talks About

Here is the insight that changes how you read quality scores.

**High faithfulness + low context recall = confidently wrong.**

The model is faithfully using what it retrieved.
But what it retrieved was incomplete or wrong.
So the answer is fluent, grounded, and incorrect.

This is the most dangerous failure mode in production RAG systems. The answer looks right. The faithfulness score looks right. Only context recall reveals the problem.

This is why you always measure all four. Never just one.

---

## The RAGAS Framework

RAGAS — Retrieval Augmented Generation Assessment — is the framework that implements all four metrics automatically.

It uses an LLM as a judge — evaluating your system's outputs against your ground truth answers and the retrieved context.

Before Day 4 this was our benchmark scaffold:

```python
@dataclass
class BenchmarkResult:
    query: str
    response: str
    latency_ms: float
    cost_usd: float
    faithfulness: Optional[float] = None        # empty
    context_recall: Optional[float] = None      # empty
    answer_relevancy: Optional[float] = None    # empty
    hallucination_rate: Optional[float] = None  # empty
```

After Day 4 every quality field gets filled in.

---

## The Problem Most People Hit

You install RAGAS. You run it. You get an error.

Or worse — you get scores back but you do not understand what to compare them against. What is a good faithfulness score? What does 0.73 mean in practice?

The second problem: RAGAS requires your retrieved context as an input — not just the question and answer. Most tutorials skip this. We will not.

Let us build it properly.

---

## The Code — Quality Metrics With RAGAS

Today we build three things:

1. A **RAGAS evaluator** that measures all four metrics on any dataset
2. A **hallucination detector** that flags high-risk responses
3. An **updated BenchmarkResult** that now includes quality scores

---

### Step 1 — Set up today's folder

In the VS Code terminal:

```bash
cd ..
mkdir day-04-quality
cd day-04-quality
```

---

### Step 2 — Install today's dependencies

```bash
pip install ragas langchain langchain-openai python-dotenv pandas datasets
```

Wait for the success message. RAGAS has several dependencies — this may take a minute.

⚠️ **If you see a conflict error** between ragas and langchain versions:

```bash
pip install ragas==0.1.21 langchain==0.2.16 langchain-openai==0.1.25
```

---

### Step 3 — Create the RAGAS evaluator

Right-click `day-04-quality` → **New File** → name it:

```
quality_evaluator.py
```

Paste this exactly:

```python
# quality_evaluator.py
# Measures quality across all four RAGAS dimensions
# Works with any RAG system — plug in your retriever and LLM

from dotenv import load_dotenv
load_dotenv()

import json
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from langsmith import traceable, Client

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from datasets import Dataset


# ─────────────────────────────────────────────
# QUALITY RESULT — scores for one query
# ─────────────────────────────────────────────

@dataclass
class QualityResult:
    question: str
    answer: str
    ground_truth: str
    contexts: List[str]             # The retrieved documents

    # The four RAGAS scores
    faithfulness: Optional[float] = None
    context_recall: Optional[float] = None
    context_precision: Optional[float] = None
    answer_relevancy: Optional[float] = None

    # Derived
    hallucination_risk: str = "unknown"   # low / medium / high
    timestamp: str = ""

    def __post_init__(self):
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if self.faithfulness is not None:
            self.hallucination_risk = self._assess_hallucination_risk()

    def _assess_hallucination_risk(self) -> str:
        """
        Derives hallucination risk from faithfulness and context recall.
        High faithfulness + high recall = low risk
        Low faithfulness OR low recall = elevated risk
        """
        f = self.faithfulness or 0
        cr = self.context_recall or 0

        if f >= 0.85 and cr >= 0.80:
            return "🟢 low"
        elif f >= 0.65 and cr >= 0.60:
            return "🟡 medium"
        else:
            return "🔴 high"

    def summary(self) -> str:
        return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 QUALITY RESULT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Question:          {self.question[:60]}{'...' if len(self.question) > 60 else ''}
─────────────────────────────────
Faithfulness:      {self.faithfulness:.2f if self.faithfulness else 'N/A'}   ← Did answer stick to context?
Context Recall:    {self.context_recall:.2f if self.context_recall else 'N/A'}   ← Was right context retrieved?
Context Precision: {self.context_precision:.2f if self.context_precision else 'N/A'}   ← Was retrieved context relevant?
Answer Relevancy:  {self.answer_relevancy:.2f if self.answer_relevancy else 'N/A'}   ← Did answer address the question?
─────────────────────────────────
Hallucination Risk: {self.hallucination_risk}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """


# ─────────────────────────────────────────────
# THE RAGAS EVALUATOR
# ─────────────────────────────────────────────

class QualityEvaluator:

    def __init__(self, langsmith_project: str = "ai-benchmarking-series"):
        self.langsmith_project = langsmith_project
        self.results: List[QualityResult] = []

    @traceable
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str
    ) -> QualityResult:
        """
        Evaluates quality for a single question-answer pair.

        Parameters:
        - question:     The user query
        - answer:       What your AI system responded
        - contexts:     List of retrieved document chunks your system used
        - ground_truth: The correct reference answer (from your eval dataset)
        """

        # RAGAS expects a HuggingFace Dataset format
        eval_data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth]
        }

        dataset = Dataset.from_dict(eval_data)

        # Run all four metrics
        scores = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision
            ]
        )

        result = QualityResult(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            contexts=contexts,
            faithfulness=scores["faithfulness"],
            context_recall=scores["context_recall"],
            context_precision=scores["context_precision"],
            answer_relevancy=scores["answer_relevancy"]
        )

        self.results.append(result)
        return result

    @traceable
    def evaluate_dataset(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str]
    ) -> Dict:
        """
        Evaluates quality across an entire dataset at once.
        More efficient than calling evaluate_single in a loop.

        Parameters:
        - questions:     List of user queries
        - answers:       List of AI responses (one per question)
        - contexts:      List of context lists (one list of chunks per question)
        - ground_truths: List of reference answers (from your eval dataset)
        """

        print(f"🎯 Evaluating quality across {len(questions)} examples...")
        print(f"   This uses an LLM as judge — takes 1-3 minutes for 10 examples.\n")

        eval_data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }

        dataset = Dataset.from_dict(eval_data)

        scores = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision
            ]
        )

        # Store individual results
        scores_df = scores.to_pandas()
        for i, row in scores_df.iterrows():
            result = QualityResult(
                question=questions[i],
                answer=answers[i],
                ground_truth=ground_truths[i],
                contexts=contexts[i],
                faithfulness=row.get("faithfulness"),
                context_recall=row.get("context_recall"),
                context_precision=row.get("context_precision"),
                answer_relevancy=row.get("answer_relevancy")
            )
            self.results.append(result)

        # Aggregate scores
        aggregate = {
            "faithfulness":      round(scores_df["faithfulness"].mean(), 3),
            "context_recall":    round(scores_df["context_recall"].mean(), 3),
            "context_precision": round(scores_df["context_precision"].mean(), 3),
            "answer_relevancy":  round(scores_df["answer_relevancy"].mean(), 3),
            "total_evaluated":   len(questions)
        }

        return aggregate

    def print_aggregate_report(self, scores: Dict):
        """Prints a clean aggregate quality report"""

        def score_bar(score: float) -> str:
            filled = int(score * 20)
            return "█" * filled + "░" * (20 - filled)

        def score_label(score: float) -> str:
            if score >= 0.85: return "✅ Strong"
            if score >= 0.70: return "🟡 Acceptable"
            if score >= 0.55: return "🟠 Needs work"
            return "🔴 Critical"

        print(f"""
{'━'*60}
📊 AGGREGATE QUALITY REPORT — {scores['total_evaluated']} examples
{'━'*60}
Faithfulness:      {scores['faithfulness']:.3f}  {score_bar(scores['faithfulness'])}  {score_label(scores['faithfulness'])}
Context Recall:    {scores['context_recall']:.3f}  {score_bar(scores['context_recall'])}  {score_label(scores['context_recall'])}
Context Precision: {scores['context_precision']:.3f}  {score_bar(scores['context_precision'])}  {score_label(scores['context_precision'])}
Answer Relevancy:  {scores['answer_relevancy']:.3f}  {score_bar(scores['answer_relevancy'])}  {score_label(scores['answer_relevancy'])}
{'─'*60}
Hallucination Risk Assessment:
  High Faith + High Recall  → 🟢 Low risk
  Mixed scores              → 🟡 Monitor closely
  Low Faith OR Low Recall   → 🔴 Investigate immediately
{'━'*60}
        """)

        # Flag dangerous combination
        f = scores["faithfulness"]
        cr = scores["context_recall"]

        if f >= 0.85 and cr < 0.65:
            print("⚠️  WARNING: High faithfulness but low context recall.")
            print("   Your model is faithfully using bad context.")
            print("   Fix: improve your retrieval pipeline (see Day 5)\n")
        elif f < 0.70:
            print("⚠️  WARNING: Low faithfulness score.")
            print("   Your model is generating beyond the retrieved context.")
            print("   Fix: tighten your system prompt, reduce temperature\n")

    def save_results(self, filename: str = "quality_results.json"):
        """Saves all quality results for comparison against future runs"""
        data = []
        for r in self.results:
            data.append({
                "question": r.question,
                "answer": r.answer[:200],
                "ground_truth": r.ground_truth[:200],
                "faithfulness": r.faithfulness,
                "context_recall": r.context_recall,
                "context_precision": r.context_precision,
                "answer_relevancy": r.answer_relevancy,
                "hallucination_risk": r.hallucination_risk,
                "timestamp": r.timestamp
            })

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        # Also save CSV for human review
        pd.DataFrame(data).to_csv(
            filename.replace('.json', '.csv'), index=False
        )
        print(f"✅ Quality results saved → {filename}")
```

---

### Step 4 — Create the runner

Right-click `day-04-quality` → **New File** → name it:

```
run_quality_eval.py
```

Paste this:

```python
# run_quality_eval.py
# Runs quality evaluation on your AI system
# Replace the mock functions with your real pipeline

from dotenv import load_dotenv
load_dotenv()

import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from quality_evaluator import QualityEvaluator

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
evaluator = QualityEvaluator()


# ─────────────────────────────────────────────
# YOUR PIPELINE — replace with your real code
# Must return both the answer AND the contexts used
# ─────────────────────────────────────────────

def mock_retriever(query: str) -> List[str]:
    """
    Replace with your real retriever.
    Must return a list of retrieved text chunks.

    Example for LangChain:
        docs = your_retriever.invoke(query)
        return [doc.page_content for doc in docs]
    """
    # Simulated retrieved context
    return [
        "Apache Kafka is a distributed event streaming platform. "
        "It was originally developed at LinkedIn and open-sourced in 2011. "
        "Kafka uses a distributed commit log architecture.",

        "Kafka organizes messages into topics. Topics are divided into "
        "partitions for scalability. Each partition is an ordered sequence "
        "of messages that is continually appended to.",

        "Kafka brokers are servers that store and serve data. A Kafka cluster "
        "consists of multiple brokers for fault tolerance and scalability."
    ]


def your_ai_pipeline(query: str) -> tuple[str, list[str]]:
    """
    Your complete AI pipeline.
    Returns both the answer and the contexts used.

    The contexts are critical — RAGAS needs them to measure faithfulness.
    If you do not return contexts you cannot measure quality properly.
    """
    # Step 1 — Retrieve
    contexts = mock_retriever(query)

    # Step 2 — Generate
    context_text = "\n\n".join(contexts)
    response = llm.invoke([
        SystemMessage(content="""You are a helpful assistant.
Answer the question using ONLY the provided context.
If the context does not contain the answer, say so clearly."""),
        HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {query}")
    ])

    return response.content, contexts


# ─────────────────────────────────────────────
# EVALUATION DATASET
# Use your eval_dataset.json from Day 2
# or use this sample to get started
# ─────────────────────────────────────────────

EVAL_EXAMPLES = [
    {
        "question": "What is Apache Kafka and where was it developed?",
        "ground_truth": "Apache Kafka is a distributed event streaming platform originally developed at LinkedIn and open-sourced in 2011."
    },
    {
        "question": "How does Kafka organize messages?",
        "ground_truth": "Kafka organizes messages into topics which are divided into partitions for scalability."
    },
    {
        "question": "What is a Kafka broker?",
        "ground_truth": "A Kafka broker is a server that stores and serves data. Multiple brokers form a cluster for fault tolerance."
    },
    {
        "question": "What programming language is Kafka written in?",
        "ground_truth": "Kafka is primarily written in Scala and Java."
        # Note: this answer is NOT in the retrieved context above
        # Watch what happens to context_recall for this question
    },
    {
        "question": "How do Kafka partitions help with scalability?",
        "ground_truth": "Kafka partitions allow topics to be split across multiple brokers enabling parallel processing and horizontal scaling."
    }
]


if __name__ == "__main__":
    print("🎯 Running quality evaluation...\n")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    # Run your pipeline on each example
    print("Step 1 — Running your AI pipeline on all examples...")
    for i, example in enumerate(EVAL_EXAMPLES):
        print(f"  Query {i+1}/{len(EVAL_EXAMPLES)}: {example['question'][:50]}...")
        answer, retrieved_contexts = your_ai_pipeline(example["question"])

        questions.append(example["question"])
        answers.append(answer)
        contexts.append(retrieved_contexts)
        ground_truths.append(example["ground_truth"])

    print("\nStep 2 — Scoring quality with RAGAS...")
    print("  (This calls an LLM as judge — takes 1-3 minutes)\n")

    # Evaluate quality
    aggregate_scores = evaluator.evaluate_dataset(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths
    )

    # Print the report
    evaluator.print_aggregate_report(aggregate_scores)

    # Show individual result for the tricky question
    print("📋 Individual result — the question the context couldn't answer:")
    tricky_result = evaluator.results[3]  # "What language is Kafka written in?"
    print(tricky_result.summary())

    # Save everything
    evaluator.save_results("quality_results.json")

    print("\n✅ Check LangSmith — quality eval logged with all scores.")
    print("   Compare these scores against your Day 2 baseline.")
```

Run it:

```bash
python run_quality_eval.py
```

You should see:

```
🎯 Running quality evaluation...

Step 1 — Running your AI pipeline on all examples...
  Query 1/5: What is Apache Kafka and where was it developed?...
  Query 2/5: How does Kafka organize messages?...
  ...

Step 2 — Scoring quality with RAGAS...
  (This calls an LLM as judge — takes 1-3 minutes)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 AGGREGATE QUALITY REPORT — 5 examples
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Faithfulness:      0.891  ████████████████░░░░  ✅ Strong
Context Recall:    0.743  ██████████████░░░░░░  🟡 Acceptable
Context Precision: 0.812  ████████████████░░░░  ✅ Strong
Answer Relevancy:  0.876  █████████████████░░░  ✅ Strong
─────────────────────────────────────────────────────────

📋 Individual result — the question the context couldn't answer:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 QUALITY RESULT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Question: What programming language is Kafka written in?
─────────────────────────────────
Faithfulness:      0.91   ← Did answer stick to context?
Context Recall:    0.21   ← Was right context retrieved?
Context Precision: 0.44   ← Was retrieved context relevant?
Answer Relevancy:  0.83   ← Did answer address the question?
─────────────────────────────────
Hallucination Risk: 🔴 high
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Look at that fourth result carefully.

Faithfulness: 0.91. The model stuck to the context.
Context Recall: 0.21. The context did not contain the answer.
Hallucination Risk: 🔴 high.

The model was faithful to bad context. That is exactly the dangerous combination we talked about.

---

### Step 5 — Connecting to Your Day 2 Baseline

Remember your `baseline_TIMESTAMP.json` from Day 2?

Now we add quality scores to it.

Right-click `day-04-quality` → **New File** → name it:

```
update_baseline_with_quality.py
```

Paste this:

```python
# update_baseline_with_quality.py
# Adds quality scores to your existing baseline
# Run this once to make your Day 2 baseline complete

from dotenv import load_dotenv
load_dotenv()

import json
import glob
import os
from quality_evaluator import QualityEvaluator
from run_quality_eval import your_ai_pipeline

def update_baseline_with_quality(baseline_path: str):
    """
    Takes your Day 2 baseline file and adds RAGAS quality scores.
    Your baseline is now complete across all three dimensions.
    """

    # Load the baseline
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)

    print(f"📂 Loaded baseline: {baseline_path}")
    print(f"   {baseline['metadata']['total_examples']} examples to evaluate\n")

    evaluator = QualityEvaluator()

    questions = [r["question"] for r in baseline["results"]]
    answers = []
    contexts = []
    ground_truths = [r["ground_truth"] for r in baseline["results"]]

    # Re-run your pipeline to get contexts
    # (baseline only stored the answer, not the retrieved context)
    print("Retrieving contexts for quality evaluation...")
    for q in questions:
        answer, retrieved_contexts = your_ai_pipeline(q)
        answers.append(answer)
        contexts.append(retrieved_contexts)

    # Score quality
    print("\nScoring with RAGAS...")
    aggregate = evaluator.evaluate_dataset(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths
    )

    # Add quality scores to baseline metadata
    baseline["metadata"]["quality_scores"] = aggregate
    baseline["metadata"]["quality_evaluated_at"] = \
        __import__('datetime').datetime.now().isoformat()

    # Add individual scores to each result
    for i, result in enumerate(evaluator.results):
        baseline["results"][i]["faithfulness"] = result.faithfulness
        baseline["results"][i]["context_recall"] = result.context_recall
        baseline["results"][i]["context_precision"] = result.context_precision
        baseline["results"][i]["answer_relevancy"] = result.answer_relevancy
        baseline["results"][i]["hallucination_risk"] = result.hallucination_risk

    # Save updated baseline
    updated_path = baseline_path.replace('.json', '_with_quality.json')
    with open(updated_path, 'w') as f:
        json.dump(baseline, f, indent=2)

    print(f"\n✅ Updated baseline saved → {updated_path}")
    evaluator.print_aggregate_report(aggregate)

    return updated_path


if __name__ == "__main__":
    # Find your baseline file from Day 2
    baseline_files = glob.glob('../day-02-baseline/baseline_*.json')

    if not baseline_files:
        print("❌ No baseline file found.")
        print("   Run Day 2 first to create your baseline.")
    else:
        # Use the most recent baseline
        latest = max(baseline_files, key=os.path.getctime)
        print(f"Found baseline: {latest}\n")
        update_baseline_with_quality(latest)
```

Run it:

```bash
python update_baseline_with_quality.py
```

Your baseline now has speed, cost, AND quality scores in one file.

That is your complete Day 2 baseline — fully populated across all three dimensions of the triangle.

---

## What You Just Built

You now have a quality evaluator that:

- Measures all four RAGAS dimensions on any RAG pipeline
- Flags the dangerous high-faithfulness-low-recall combination automatically
- Generates both aggregate reports and individual result breakdowns
- Saves results for comparison against future runs
- Integrates with your Day 2 baseline to complete the picture

---

## ✅ Day 4 Checklist

- [ ] `quality_evaluator.py` imports without errors
- [ ] `run_quality_eval.py` runs and prints the aggregate report
- [ ] You understand what each of the four scores means
- [ ] You can identify the dangerous faithfulness/recall combination in your results
- [ ] Your retriever is returning actual context chunks not just answers
- [ ] `update_baseline_with_quality.py` runs and updates your Day 2 baseline
- [ ] Quality results saved to `quality_results.json`
- [ ] All runs visible in LangSmith

---

## 🎯 Interview Bits — Day 4

**Q: What is the difference between faithfulness and answer relevancy in RAG evaluation?**
*Faithfulness measures whether the answer is grounded in the retrieved context — did the model stick to its sources? Answer relevancy measures whether the answer actually addressed the question asked. A model can be perfectly faithful to its context while still answering the wrong question.*

**Q: Why is high faithfulness with low context recall dangerous?**
*Because the model is faithfully using context that was insufficient or wrong. The answer is grounded — but grounded in bad information. This failure mode is particularly dangerous because it produces fluent, confident, well-structured wrong answers with no visible error signal.*

**Q: What does RAGAS use to score quality metrics?**
*RAGAS uses an LLM as a judge — it sends the question, retrieved contexts, generated answer, and ground truth to a language model which evaluates the quality dimensions. This means RAGAS evaluation itself incurs LLM API costs and takes time proportional to dataset size.*

**Q: What is context precision and why does it matter for cost?**
*Context precision measures what fraction of retrieved documents were actually relevant to answering the question. Low context precision means irrelevant documents are flooding the LLM context window — increasing token usage and cost without improving answer quality.*

**Q: How would you respond if your faithfulness score is high but users are still reporting wrong answers?**
*Check context recall. High faithfulness means the model is using its context correctly — so the problem is upstream in retrieval. The relevant information was either never retrieved or retrieved incompletely. The fix is in the retrieval pipeline not the generation step.*

---

*Tomorrow we go upstream.*
*Quality scores told us the answers are sometimes wrong.*
*Day 5 shows us exactly why — and fixes the retrieval layer that feeds everything else.*
