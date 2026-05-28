Let's go.

---

# AI Performance Benchmarking: From Zero to Production
### Day 6 of 12 — A/B Testing & Prompt Benchmarking: Why Your Gut Feeling About Prompt Changes Is Almost Always Wrong

---

*This is Day 6 of a 12-part series. If you're just joining, start with **[Day 0](#)** for setup, **[Day 1](#)** for the mental model, **[Day 2](#)** for baselines, **[Day 3](#)** for latency, **[Day 4](#)** for quality metrics, and **[Day 5](#)** for retrieval benchmarking. Everything here builds on those foundations.*

---

## The Prompt Rewrite That Almost Shipped A Regression

You have been working on your AI system for weeks.

The answers feel a little off. Not wrong exactly. Just not quite right.

So you rewrite the system prompt.

You test it on five queries. It feels better. The answers are crisper. More confident. More on point.

You show a colleague. They agree. Looks good.

You ship it.

Three days later your quality metrics drop.

You dig into the results. The new prompt is better on simple factual queries. But on complex multi-hop questions — the ones users actually care about most — it regressed by 23%.

You did not catch it because you tested five queries.

Five queries that happened to be simple.

---

This is the most common way AI systems silently get worse.

Not from a bad model. Not from bad data.

From a prompt change that helped some queries and hurt others — and nobody measured it systematically.

**A/B testing is how you catch this before it ships.**

---

## What A/B Testing Means For AI Systems

In traditional software A/B testing means showing two versions of a UI to different users and measuring click rates.

In AI systems it means something more precise:

**Run version A and version B on the exact same set of queries. Measure every quality and performance dimension. Compare the distributions — not just the averages.**

The key phrase is *exact same set of queries.*

Not similar queries. Not new queries. The same ones.

Because the only way to know if version B is better than version A is to hold everything constant except the thing you changed.

---

## What You Can A/B Test

Almost anything in your pipeline is a testable variable:

**Prompts** — system prompt rewrites, instruction changes, persona changes, output format instructions

**Models** — GPT-4o vs Claude Sonnet vs GPT-4o-mini vs open source

**Temperatures** — 0.0 vs 0.3 vs 0.7 — how much does randomness affect quality?

**Chunk sizes** — you benchmarked strategies in Day 5 but A/B testing confirms the quality impact end-to-end

**Context window sizes** — top-3 vs top-5 vs top-10 retrieved chunks

**Retrieval strategies** — dense vs hybrid on your actual eval dataset

---

## The Statistical Trap

Here is where most engineers go wrong with A/B testing.

They run both versions. Version B scores 0.847 on faithfulness. Version A scored 0.831.

They declare B the winner.

But is that difference real?

Or is it noise from the LLM-as-judge having slightly different responses on different runs?

**Without statistical significance testing you are comparing random variation and calling it improvement.**

The fix is simple. Not complicated statistics. Just:

1. Run each version multiple times
2. Calculate the distribution of scores — not just the mean
3. Check whether the difference is larger than the natural variation

We build this today.

---

## LangSmith Experiments — Your A/B Testing Home Base

LangSmith has a built-in experiments feature that is perfect for this.

Every A/B test run gets logged as a named experiment.

You can compare experiments side by side in the LangSmith UI — seeing score distributions, individual query results, and aggregate comparisons in one place.

We instrument every test today to log there automatically.

---

## The Problem Most People Hit

You want to A/B test your prompt.

So you run version A, write down the scores, run version B, write down the scores, compare.

Two problems:

**Problem 1** — LLM outputs are non-deterministic even at temperature 0. The same prompt on the same query can produce slightly different outputs and therefore slightly different RAGAS scores. Single-run comparisons are unreliable.

**Problem 2** — You have no record of what you tested, when, with what parameters. Three weeks later you cannot remember which prompt version was A and which was B or what the scores actually were.

Today's code solves both.

---

## The Code — A/B Testing Suite

Today we build three things:

1. An **experiment runner** that runs two versions on the same dataset and logs everything
2. A **statistical comparator** that tells you whether the difference is real
3. A **prompt version tracker** that keeps a permanent record of every test

---

### Step 1 — Set up today's folder

In the VS Code terminal:

```bash
cd ..
mkdir day-06-ab-testing
cd day-06-ab-testing
```

---

### Step 2 — Install today's dependencies

```bash
pip install langsmith langchain langchain-openai python-dotenv ragas datasets pandas numpy scipy
```

---

### Step 3 — Create the prompt version tracker

Right-click `day-06-ab-testing` → **New File** → name it:

```
prompt_registry.py
```

Paste this exactly:

```python
# prompt_registry.py
# Tracks every prompt version you test
# Never lose track of what changed and when

from dotenv import load_dotenv
load_dotenv()

import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ─────────────────────────────────────────────
# PROMPT VERSION
# One entry in your prompt history
# ─────────────────────────────────────────────

@dataclass
class PromptVersion:
    name: str                    # e.g. "baseline", "v2_concise", "v3_strict"
    system_prompt: str           # The full system prompt text
    description: str             # What you changed and why
    created_at: str = ""
    prompt_hash: str = ""        # Auto-generated fingerprint

    # Filled in after testing
    faithfulness: Optional[float] = None
    context_recall: Optional[float] = None
    answer_relevancy: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    notes: str = ""

    def __post_init__(self):
        self.created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Generate a short hash so you can always identify this exact prompt
        self.prompt_hash = hashlib.md5(
            self.system_prompt.encode()
        ).hexdigest()[:8]

    def summary(self) -> str:
        scores = ""
        if self.faithfulness is not None:
            scores = (
                f"\n  Faithfulness:     {self.faithfulness:.3f}"
                f"\n  Context Recall:   {self.context_recall:.3f}"
                f"\n  Answer Relevancy: {self.answer_relevancy:.3f}"
                f"\n  Avg Latency:      {self.avg_latency_ms:.0f}ms"
            )
        else:
            scores = "\n  Not yet tested"

        return (
            f"\n{'─'*50}"
            f"\n📝 {self.name}  [hash: {self.prompt_hash}]"
            f"\n  Created:    {self.created_at}"
            f"\n  Change:     {self.description}"
            f"{scores}"
        )


# ─────────────────────────────────────────────
# PROMPT REGISTRY
# Your permanent record of every version tested
# ─────────────────────────────────────────────

class PromptRegistry:

    def __init__(self, registry_path: str = "prompt_registry.json"):
        self.registry_path = registry_path
        self.versions: List[PromptVersion] = []
        self._load()

    def _load(self):
        """Load existing registry if it exists"""
        try:
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
                self.versions = [PromptVersion(**v) for v in data]
            print(f"📚 Loaded {len(self.versions)} prompt versions from registry")
        except FileNotFoundError:
            self.versions = []

    def register(self, version: PromptVersion) -> PromptVersion:
        """Add a new prompt version to the registry"""

        # Check for duplicate
        existing_hashes = [v.prompt_hash for v in self.versions]
        if version.prompt_hash in existing_hashes:
            print(f"⚠️  This exact prompt already exists in registry.")
            print(f"   Hash: {version.prompt_hash}")
            print(f"   Register with a different name to track as a new version.")
            return version

        self.versions.append(version)
        self._save()
        print(f"✅ Registered: {version.name} [hash: {version.prompt_hash}]")
        return version

    def update_scores(
        self,
        name: str,
        faithfulness: float,
        context_recall: float,
        answer_relevancy: float,
        avg_latency_ms: float,
        notes: str = ""
    ):
        """Update a version with its benchmark scores after testing"""
        for version in self.versions:
            if version.name == name:
                version.faithfulness = faithfulness
                version.context_recall = context_recall
                version.answer_relevancy = answer_relevancy
                version.avg_latency_ms = avg_latency_ms
                version.notes = notes
                self._save()
                print(f"✅ Scores updated for: {name}")
                return
        print(f"❌ Version '{name}' not found in registry")

    def get(self, name: str) -> Optional[PromptVersion]:
        """Get a specific version by name"""
        for version in self.versions:
            if version.name == name:
                return version
        return None

    def print_history(self):
        """Print the full history of all tested prompts"""
        if not self.versions:
            print("No prompt versions registered yet.")
            return

        print(f"\n{'━'*50}")
        print(f"📚 PROMPT VERSION HISTORY ({len(self.versions)} versions)")
        print(f"{'━'*50}")

        for version in self.versions:
            print(version.summary())

        # Show best version if scores exist
        tested = [v for v in self.versions if v.faithfulness is not None]
        if tested:
            best = max(tested, key=lambda v: (
                v.faithfulness * 0.4 +
                v.context_recall * 0.4 +
                v.answer_relevancy * 0.2
            ))
            print(f"\n🏆 Best version so far: {best.name}")
            print(f"{'━'*50}\n")

    def _save(self):
        """Persist registry to disk"""
        with open(self.registry_path, 'w') as f:
            json.dump([asdict(v) for v in self.versions], f, indent=2)
```

---

### Step 4 — Create the experiment runner

Right-click `day-06-ab-testing` → **New File** → name it:

```
experiment_runner.py
```

Paste this:

```python
# experiment_runner.py
# Runs controlled A/B experiments on your AI pipeline
# Logs everything to LangSmith for visual comparison

from dotenv import load_dotenv
load_dotenv()

import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable, Client
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset

from prompt_registry import PromptRegistry, PromptVersion


# ─────────────────────────────────────────────
# EXPERIMENT RESULT
# Complete results for one A/B test run
# ─────────────────────────────────────────────

@dataclass
class ExperimentResult:
    experiment_name: str
    prompt_version: str
    prompt_hash: str

    # Per-query scores
    faithfulness_scores: List[float] = field(default_factory=list)
    context_recall_scores: List[float] = field(default_factory=list)
    answer_relevancy_scores: List[float] = field(default_factory=list)
    latency_scores: List[float] = field(default_factory=list)

    # Aggregates (computed after all queries run)
    avg_faithfulness: float = 0.0
    avg_context_recall: float = 0.0
    avg_answer_relevancy: float = 0.0
    avg_latency_ms: float = 0.0

    # Standard deviations (for statistical comparison)
    std_faithfulness: float = 0.0
    std_context_recall: float = 0.0

    timestamp: str = ""

    def __post_init__(self):
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def compute_aggregates(self):
        """Call this after all query scores are added"""
        if self.faithfulness_scores:
            self.avg_faithfulness = np.mean(self.faithfulness_scores)
            self.avg_context_recall = np.mean(self.context_recall_scores)
            self.avg_answer_relevancy = np.mean(self.answer_relevancy_scores)
            self.avg_latency_ms = np.mean(self.latency_scores)
            self.std_faithfulness = np.std(self.faithfulness_scores)
            self.std_context_recall = np.std(self.context_recall_scores)

    def summary(self) -> str:
        return (
            f"\n{'─'*55}"
            f"\n🧪 {self.experiment_name} — {self.prompt_version}"
            f"\n{'─'*55}"
            f"\n  Faithfulness:     {self.avg_faithfulness:.3f} "
            f"(±{self.std_faithfulness:.3f})"
            f"\n  Context Recall:   {self.avg_context_recall:.3f} "
            f"(±{self.std_context_recall:.3f})"
            f"\n  Answer Relevancy: {self.avg_answer_relevancy:.3f}"
            f"\n  Avg Latency:      {self.avg_latency_ms:.0f}ms"
            f"\n  Queries tested:   {len(self.faithfulness_scores)}"
        )


# ─────────────────────────────────────────────
# THE EXPERIMENT RUNNER
# ─────────────────────────────────────────────

class ExperimentRunner:

    def __init__(self, registry: PromptRegistry):
        self.registry = registry
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.results: List[ExperimentResult] = []

    def _build_pipeline(self, system_prompt: str) -> Callable:
        """
        Builds a pipeline function from a system prompt.
        Replace the mock retriever with your real retriever.
        """
        def pipeline(query: str, contexts: List[str]) -> str:
            context_text = "\n\n".join(contexts)
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=(
                    f"Context:\n{context_text}\n\n"
                    f"Question: {query}"
                ))
            ])
            return response.content
        return pipeline

    @traceable
    def run_experiment(
        self,
        experiment_name: str,
        prompt_version: PromptVersion,
        eval_dataset: List[Dict],
        contexts_per_query: List[List[str]]
    ) -> ExperimentResult:
        """
        Runs one side of an A/B test.

        Parameters:
        - experiment_name:      e.g. "prompt_ab_test_2024"
        - prompt_version:       PromptVersion object from registry
        - eval_dataset:         List of {"question": str, "ground_truth": str}
        - contexts_per_query:   List of retrieved context lists, one per query
        """

        print(f"\n🧪 Running experiment: {experiment_name}")
        print(f"   Prompt version: {prompt_version.name} "
              f"[{prompt_version.prompt_hash}]")
        print(f"   Queries: {len(eval_dataset)}")

        pipeline = self._build_pipeline(prompt_version.system_prompt)
        result = ExperimentResult(
            experiment_name=experiment_name,
            prompt_version=prompt_version.name,
            prompt_hash=prompt_version.prompt_hash
        )

        questions = []
        answers = []
        ground_truths = []

        # Run pipeline on all queries
        print(f"\n  Step 1 — Generating responses...")
        for i, (example, contexts) in enumerate(
            zip(eval_dataset, contexts_per_query)
        ):
            print(f"    Query {i+1}/{len(eval_dataset)}...", end=" ")

            start = time.perf_counter()
            answer = pipeline(example["question"], contexts)
            latency_ms = (time.perf_counter() - start) * 1000

            result.latency_scores.append(latency_ms)
            questions.append(example["question"])
            answers.append(answer)
            ground_truths.append(example["ground_truth"])

            print(f"✓ {latency_ms:.0f}ms")

        # Score quality with RAGAS
        print(f"\n  Step 2 — Scoring quality with RAGAS...")
        print(f"    (Takes 1-3 minutes — using LLM as judge)")

        ragas_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts_per_query,
            "ground_truth": ground_truths
        })

        scores = evaluate(
            dataset=ragas_dataset,
            metrics=[faithfulness, answer_relevancy, context_recall]
        )

        scores_df = scores.to_pandas()
        result.faithfulness_scores = scores_df["faithfulness"].tolist()
        result.context_recall_scores = scores_df["context_recall"].tolist()
        result.answer_relevancy_scores = scores_df["answer_relevancy"].tolist()
        result.compute_aggregates()

        # Update registry with scores
        self.registry.update_scores(
            name=prompt_version.name,
            faithfulness=result.avg_faithfulness,
            context_recall=result.avg_context_recall,
            answer_relevancy=result.avg_answer_relevancy,
            avg_latency_ms=result.avg_latency_ms
        )

        self.results.append(result)
        print(result.summary())
        return result

    def compare(
        self,
        result_a: ExperimentResult,
        result_b: ExperimentResult
    ):
        """
        Compares two experiment results.
        Flags statistically meaningful differences.
        """
        from scipy import stats

        print(f"\n{'━'*60}")
        print(f"📊 A/B COMPARISON")
        print(f"{'━'*60}")
        print(f"  Version A: {result_a.prompt_version}")
        print(f"  Version B: {result_b.prompt_version}")
        print(f"{'─'*60}")

        metrics = [
            ("Faithfulness",
             result_a.faithfulness_scores,
             result_b.faithfulness_scores),
            ("Context Recall",
             result_a.context_recall_scores,
             result_b.context_recall_scores),
            ("Answer Relevancy",
             result_a.answer_relevancy_scores,
             result_b.answer_relevancy_scores),
        ]

        print(
            f"\n{'Metric':<20} {'Version A':>10} "
            f"{'Version B':>10} {'Delta':>8} {'Significant':>13}"
        )
        print(f"{'─'*60}")

        winner_points = {"a": 0, "b": 0}

        for metric_name, scores_a, scores_b in metrics:
            mean_a = np.mean(scores_a)
            mean_b = np.mean(scores_b)
            delta = mean_b - mean_a

            # Welch's t-test — does not assume equal variance
            # p < 0.05 means the difference is statistically significant
            _, p_value = stats.ttest_ind(scores_a, scores_b)
            significant = p_value < 0.05

            direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            sig_label = "✅ Yes" if significant else "⬜ No (noise)"

            print(
                f"{metric_name:<20} {mean_a:>10.3f} "
                f"{mean_b:>10.3f} "
                f"{direction}{abs(delta):>6.3f} "
                f"{sig_label:>13}"
            )

            if significant:
                if delta > 0:
                    winner_points["b"] += 1
                elif delta < 0:
                    winner_points["a"] += 1

        # Latency comparison
        lat_a = np.mean(result_a.latency_scores)
        lat_b = np.mean(result_b.latency_scores)
        lat_delta = lat_b - lat_a
        lat_direction = "↑" if lat_delta > 0 else "↓"

        print(
            f"{'Latency (ms)':<20} {lat_a:>10.0f} "
            f"{lat_b:>10.0f} "
            f"{lat_direction}{abs(lat_delta):>5.0f}ms "
            f"{'(informational)':>13}"
        )

        print(f"{'─'*60}")

        # Verdict
        if winner_points["b"] > winner_points["a"]:
            print(f"\n🏆 WINNER: Version B ({result_b.prompt_version})")
            print(f"   Statistically better on "
                  f"{winner_points['b']} metric(s)")
        elif winner_points["a"] > winner_points["b"]:
            print(f"\n🏆 WINNER: Version A ({result_a.prompt_version})")
            print(f"   Version B regressed on "
                  f"{winner_points['a']} metric(s)")
            print(f"   ⚠️  Do not ship Version B")
        else:
            print(f"\n🤝 NO CLEAR WINNER")
            print(f"   Differences are within statistical noise.")
            print(f"   Keep Version A — do not change without clear signal.")

        print(f"{'━'*60}\n")

    def save_results(self, filename: str = "experiment_results.json"):
        data = []
        for r in self.results:
            data.append({
                "experiment": r.experiment_name,
                "prompt_version": r.prompt_version,
                "prompt_hash": r.prompt_hash,
                "avg_faithfulness": r.avg_faithfulness,
                "avg_context_recall": r.avg_context_recall,
                "avg_answer_relevancy": r.avg_answer_relevancy,
                "avg_latency_ms": r.avg_latency_ms,
                "std_faithfulness": r.std_faithfulness,
                "timestamp": r.timestamp
            })
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Experiment results saved → {filename}")
```

---

### Step 5 — Create the runner

Right-click `day-06-ab-testing` → **New File** → name it:

```
run_ab_test.py
```

Paste this:

```python
# run_ab_test.py
# Runs a complete A/B test between two prompt versions
# Replace prompts and pipeline with your own

from dotenv import load_dotenv
load_dotenv()

from prompt_registry import PromptRegistry, PromptVersion
from experiment_runner import ExperimentRunner

# ─────────────────────────────────────────────
# STEP 1 — DEFINE YOUR PROMPT VERSIONS
# This is what you are testing
# Replace with your actual prompts
# ─────────────────────────────────────────────

PROMPT_A = """You are a helpful AI assistant.
Answer the user's question accurately using the provided context.
If the context does not contain enough information, say so clearly.
Keep your answers concise and well-structured."""

PROMPT_B = """You are a precise AI assistant specialized in answering
questions from provided documents.

Rules you must follow:
1. Answer ONLY using information from the provided context
2. If the answer is not in the context, respond with:
   "I cannot find this information in the provided context."
3. Always cite which part of the context supports your answer
4. Be concise — no more than 3 sentences unless detail is required
5. Never guess or infer beyond what is explicitly stated"""

# ─────────────────────────────────────────────
# STEP 2 — DEFINE YOUR EVAL DATASET
# Use your eval_dataset.json from Day 2
# or use this sample to get started
# ─────────────────────────────────────────────

EVAL_DATASET = [
    {
        "question": "What is Apache Kafka and where was it developed?",
        "ground_truth": "Apache Kafka is a distributed event streaming "
                        "platform developed at LinkedIn and open-sourced "
                        "in 2011."
    },
    {
        "question": "How does Kafka achieve fault tolerance?",
        "ground_truth": "Kafka achieves fault tolerance through replication "
                        "across multiple brokers and leader election when "
                        "a broker fails."
    },
    {
        "question": "What delivery guarantees does Kafka support?",
        "ground_truth": "Kafka supports at-most-once, at-least-once, and "
                        "exactly-once delivery semantics."
    },
    {
        "question": "How do consumer groups work in Kafka?",
        "ground_truth": "Consumer groups allow multiple consumers to share "
                        "reading from a topic, with each partition assigned "
                        "to exactly one consumer in the group at a time."
    },
    {
        "question": "What is the role of a Kafka broker?",
        "ground_truth": "A Kafka broker is a server that stores and serves "
                        "data. Multiple brokers form a cluster for fault "
                        "tolerance and scalability."
    },
    {
        "question": "How does Kafka retain messages?",
        "ground_truth": "Kafka retains messages for a configurable period "
                        "regardless of consumption, allowing consumers to "
                        "replay historical data."
    },
    {
        "question": "What is a Kafka partition offset?",
        "ground_truth": "An offset is a unique identifier for a message "
                        "within a partition that marks its position in "
                        "the ordered sequence."
    },
    {
        "question": "How does Kafka achieve high throughput?",
        "ground_truth": "Kafka achieves high throughput through sequential "
                        "disk I/O, zero-copy data transfer, and efficient "
                        "batching of messages."
    }
]

# ─────────────────────────────────────────────
# STEP 3 — DEFINE YOUR CONTEXTS
# In your real project these come from your retriever
# Replace with: [retriever.invoke(q["question"]) for q in EVAL_DATASET]
# ─────────────────────────────────────────────

SHARED_CONTEXT = [
    "Apache Kafka is a distributed event streaming platform designed "
    "to handle high-throughput, fault-tolerant data feeds. Originally "
    "developed at LinkedIn and open-sourced in 2011.",

    "Kafka achieves fault tolerance through replication. Each partition "
    "is replicated across multiple brokers. When a broker fails, leader "
    "election promotes a replica to become the new leader.",

    "Kafka supports three delivery semantics: at-most-once (messages may "
    "be lost), at-least-once (messages never lost but may repeat), and "
    "exactly-once (using transactions and idempotent producers).",

    "Consumer groups allow multiple consumers to share reading from a "
    "topic. Each partition is assigned to exactly one consumer in the "
    "group at a time, enabling parallel processing.",

    "Kafka brokers are servers that store and serve data. A Kafka cluster "
    "consists of multiple brokers. One broker acts as the controller "
    "managing partition assignments and leader elections.",

    "Messages are retained for a configurable period regardless of "
    "consumption. This allows consumers to replay historical data "
    "from any point in time.",

    "Each message in a Kafka partition has a unique offset — an integer "
    "that identifies its position in the ordered sequence. Consumers "
    "track their position using offsets.",

    "Kafka achieves high throughput through sequential disk I/O, "
    "zero-copy data transfer, and efficient message batching. "
    "A single broker can handle millions of messages per second."
]

# Use the same context for every query
# In your real project each query gets its own retrieved context
CONTEXTS_PER_QUERY = [SHARED_CONTEXT for _ in EVAL_DATASET]


if __name__ == "__main__":

    # Initialize registry and runner
    registry = PromptRegistry()
    runner = ExperimentRunner(registry)

    # Register both versions
    version_a = registry.register(PromptVersion(
        name="baseline_v1",
        system_prompt=PROMPT_A,
        description="Original prompt — general helpful assistant"
    ))

    version_b = registry.register(PromptVersion(
        name="strict_v2",
        system_prompt=PROMPT_B,
        description="Stricter prompt — explicit rules, citation required, "
                    "refuses if context insufficient"
    ))

    # Run both experiments
    print("\n" + "="*60)
    print("RUNNING A/B TEST")
    print("="*60)

    result_a = runner.run_experiment(
        experiment_name="prompt_ab_test",
        prompt_version=version_a,
        eval_dataset=EVAL_DATASET,
        contexts_per_query=CONTEXTS_PER_QUERY
    )

    result_b = runner.run_experiment(
        experiment_name="prompt_ab_test",
        prompt_version=version_b,
        eval_dataset=EVAL_DATASET,
        contexts_per_query=CONTEXTS_PER_QUERY
    )

    # Compare
    runner.compare(result_a, result_b)

    # Print full prompt history
    registry.print_history()

    # Save everything
    runner.save_results("experiment_results.json")

    print("✅ Check LangSmith → both experiments logged for visual comparison")
    print("   Go to: Projects → ai-benchmarking-series → Experiments")
```

Run it:

```bash
python run_ab_test.py
```

You should see:

```
====================================================
RUNNING A/B TEST
====================================================

🧪 Running experiment: prompt_ab_test
   Prompt version: baseline_v1 [a3f2c891]
   Queries: 8

  Step 1 — Generating responses...
    Query 1/8... ✓ 812ms
    Query 2/8... ✓ 743ms
    ...

  Step 2 — Scoring quality with RAGAS...
    (Takes 1-3 minutes — using LLM as judge)

──────────────────────────────────────────────────────
🧪 prompt_ab_test — baseline_v1
──────────────────────────────────────────────────────
  Faithfulness:     0.847 (±0.091)
  Context Recall:   0.731 (±0.112)
  Answer Relevancy: 0.863 (±0.074)
  Avg Latency:      798ms
  Queries tested:   8

🧪 Running experiment: prompt_ab_test
   Prompt version: strict_v2 [b7d1e443]
   ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 A/B COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Version A: baseline_v1
  Version B: strict_v2
──────────────────────────────────────────────────────────────

Metric               Version A  Version B    Delta   Significant
──────────────────────────────────────────────────────────────
Faithfulness             0.847      0.923   ↑0.076    ✅ Yes
Context Recall           0.731      0.698   ↓0.033    ⬜ No (noise)
Answer Relevancy         0.863      0.841   ↓0.022    ⬜ No (noise)
Latency (ms)               798        834   ↑36ms     (informational)
──────────────────────────────────────────────────────────────

🏆 WINNER: Version B (strict_v2)
   Statistically better on 1 metric(s)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Version B wins on faithfulness — and the statistical test confirms it is real, not noise.

Context recall and answer relevancy differences are within noise — no meaningful change either way.

That is the information you need to make a confident shipping decision.

---

### Step 6 — Plug into your existing project

Two things to adapt for your real pipeline.

**Swap in your real retriever:**

```python
# In run_ab_test.py replace CONTEXTS_PER_QUERY with:
from your_pipeline import your_retriever

CONTEXTS_PER_QUERY = [
    [doc.page_content for doc in your_retriever.invoke(q["question"])]
    for q in EVAL_DATASET
]
```

**Load your eval dataset from Day 2:**

```python
# Replace EVAL_DATASET with your saved dataset
import json
with open('../day-02-baseline/eval_dataset.json', 'r') as f:
    EVAL_DATASET = json.load(f)
```

⚠️ **The one rule for valid A/B tests:**
Both versions must run on identical contexts. If version A retrieves its own context and version B retrieves its own — you are testing retrieval differences not prompt differences. Pre-retrieve all contexts once and pass the same list to both experiments.

---

## What You Just Built

You now have a complete A/B testing suite that:

- Registers and tracks every prompt version with a permanent hash
- Runs controlled experiments on identical datasets
- Uses statistical significance testing to separate real improvements from noise
- Logs everything to LangSmith for visual side-by-side comparison
- Saves a permanent record of every experiment for future reference

From this point forward you will never ship a prompt change without evidence.

---

## ✅ Day 6 Checklist

- [ ] `prompt_registry.py` runs and creates `prompt_registry.json`
- [ ] `experiment_runner.py` imports without errors
- [ ] `run_ab_test.py` runs both experiments and prints the comparison
- [ ] You understand what the significance column means
- [ ] You have replaced `PROMPT_A` with your current production prompt
- [ ] You have replaced `PROMPT_B` with the version you want to test
- [ ] Your eval dataset from Day 2 is connected
- [ ] Both experiments are visible in LangSmith under Experiments
- [ ] Results saved to `experiment_results.json`

---

## 🎯 Interview Bits — Day 6

**Q: Why is a single-run comparison not enough for A/B testing AI systems?**
*LLM outputs are non-deterministic — the same prompt on the same query produces slightly different outputs and therefore slightly different evaluation scores. Single-run comparisons conflate real improvement with random variation. Running multiple queries and applying statistical significance testing separates genuine improvement from noise.*

**Q: What is Welch's t-test and why use it for prompt comparison?**
*Welch's t-test determines whether the means of two groups are statistically different. It is preferred over the standard t-test because it does not assume equal variance between groups — which is almost never true when comparing different prompt versions. A p-value below 0.05 means there is less than 5% probability the difference is due to chance.*

**Q: What makes a valid A/B test for prompt changes?**
*Both versions must run on identical inputs — same queries, same retrieved contexts. If the retrieval step runs independently for each version, you are measuring retrieval differences not prompt differences. Pre-retrieve all contexts once and pass the same list to both experiments.*

**Q: How would you handle a situation where version B improves faithfulness but regresses on context recall?**
*This is a real tradeoff — a stricter prompt may make the model more faithful to its context while causing it to refuse more queries due to insufficient context. The right choice depends on the use case. In high-stakes domains like medical or legal, faithfulness improvements outweigh recall regressions. In general-purpose assistants, the opposite may be true. The point is to make this decision with data, not intuition.*

**Q: What is prompt versioning and why does it matter in production?**
*Prompt versioning treats system prompts as code artifacts — tracked, hashed, and associated with benchmark scores. Without versioning you lose the ability to roll back to a previous prompt, understand what changed between versions, or reproduce a historical benchmark result. A prompt registry is the simplest implementation.*

---

*Tomorrow we follow the money.*
*Your system is fast. The answers are good.*
*But at 100,000 queries a day — what is this actually costing?*
*Day 7 is about cost benchmarking. And the routing strategy that cut costs by 68% without touching quality.*


