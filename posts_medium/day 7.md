Here we go.

---

# AI Performance Benchmarking: From Zero to Production
### Day 7 — Cost Benchmarking: The Bill That Surprises Everyone

---

*This is Day 7 of a 12-part series. Start with **[Day 0](#)** if you're just joining.*

---

## The Number Nobody Checks Until It's Too Late

Your AI system is working.

Latency is good. Quality scores are strong. The A/B tests gave you confidence in your prompt.

Then finance sends you a spreadsheet.

Last month's API bill.

You look at the number.

You look again.

You did the math when you built it. You estimated maybe $200 a month at scale.

The actual number is $1,847.

---

This happens constantly.

Not because engineers are careless. Because cost in AI systems is genuinely hard to reason about intuitively.

A single query feels free. A fraction of a cent. Negligible.

But fractions of cents at 50,000 queries a day compound into thousands of dollars a month.

And most AI systems have a hidden cost structure that makes it worse:

**Not all queries cost the same.**

A simple lookup query — *"what is the patient's blood type?"* — uses maybe 300 tokens.

A complex multi-document synthesis query — *"summarize all lab results from the past 6 months and flag any anomalies"* — uses 4,000 tokens.

If you are running the same model on both you are massively overpaying for the simple queries.

**That is what today fixes.**

---

## The Three Layers Of AI Cost

Before we benchmark anything, let us build the mental model.

AI system costs live in three layers:

---

**Layer 1 — Token costs**

Every LLM API call charges by token.

Input tokens — the prompt, system message, and retrieved context you send.
Output tokens — the response the model generates.

Output tokens cost more than input tokens on most models.

The variables that drive token costs:
- Model choice — GPT-4o costs 15x more per token than GPT-4o-mini
- Context window size — more retrieved chunks = more input tokens
- Response length — verbose responses = more output tokens
- System prompt length — long instructions cost on every single query

---

**Layer 2 — Retrieval costs**

Vector database queries, embedding API calls, reranker calls.

Often overlooked but significant at scale.

Generating embeddings for new documents costs money.
Querying a managed vector database costs money.
Running a reranker costs money.

---

**Layer 3 — Infrastructure costs**

Compute, memory, networking.

Especially relevant if you are self-hosting any models.

We focus on Layers 1 and 2 today since they are where the biggest wins are.

---

## The Model Comparison Reality

Here is the cost landscape as of 2024 for the most common models:

```
MODEL               INPUT (per 1M tokens)   OUTPUT (per 1M tokens)
──────────────────────────────────────────────────────────────────
GPT-4o              $5.00                   $15.00
GPT-4o-mini         $0.15                   $0.60
Claude Sonnet       $3.00                   $15.00
Claude Haiku        $0.25                   $1.25
Llama 3 (self-host) ~$0.10 (compute only)   ~$0.10
──────────────────────────────────────────────────────────────────
```

GPT-4o costs 33x more per input token than GPT-4o-mini.

If GPT-4o-mini handles 80% of your queries with equivalent quality — you just cut your bill by 80% on those queries.

That is the query routing opportunity.

---

## The Query Routing Strategy

Not every query needs your most powerful model.

Simple factual lookups — use a small fast cheap model.
Moderate reasoning — use a mid-tier model.
Complex multi-hop synthesis — use your best model.

The challenge is classifying queries automatically.

Today we build a lightweight classifier that does this — and benchmark the cost-quality tradeoff so you can tune the routing thresholds for your specific use case.

---

## The Problem Most People Hit

You want to track costs precisely.

So you look at your OpenAI dashboard. You see a total. But you cannot see cost per query, cost per query type, cost trend over time, or where the expensive queries are coming from.

The dashboard tells you what you spent.

It does not tell you why or how to reduce it.

Today's code gives you that visibility.

---

## The Code — Cost Benchmarking Suite

Today we build three things:

1. A **token cost tracker** that measures precise cost per query
2. A **model comparison benchmarker** that compares quality vs cost across models
3. A **query router** that automatically sends queries to the right model

---

### Step 1 — Set up today's folder

```bash
cd ..
mkdir day-07-cost
cd day-07-cost
```

---

### Step 2 — Install dependencies

```bash
pip install langchain langchain-openai python-dotenv tiktoken pandas numpy
```

---

### Step 3 — Create the cost tracker

Right-click `day-07-cost` → **New File** → name it:

```
cost_tracker.py
```

Paste this:

```python
# cost_tracker.py
# Tracks precise token-level costs for every query
# Supports multiple models with configurable pricing

from dotenv import load_dotenv
load_dotenv()

import json
import tiktoken
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from langsmith import traceable


# ─────────────────────────────────────────────
# MODEL PRICING TABLE
# Update these when pricing changes
# Prices in USD per 1000 tokens
# ─────────────────────────────────────────────

MODEL_PRICING = {
    "gpt-4o": {
        "input_per_1k":  0.005,
        "output_per_1k": 0.015
    },
    "gpt-4o-mini": {
        "input_per_1k":  0.000150,
        "output_per_1k": 0.000600
    },
    "claude-opus-4-5": {
        "input_per_1k":  0.015,
        "output_per_1k": 0.075
    },
    "claude-sonnet-4-5": {
        "input_per_1k":  0.003,
        "output_per_1k": 0.015
    },
    "claude-haiku-4-5": {
        "input_per_1k":  0.00025,
        "output_per_1k": 0.00125
    }
}


# ─────────────────────────────────────────────
# QUERY COST — one query's complete cost record
# ─────────────────────────────────────────────

@dataclass
class QueryCost:
    query: str
    model: str
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    query_type: str = "unknown"
    timestamp: str = ""

    def __post_init__(self):
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def summary(self) -> str:
        return (
            f"Model: {self.model:<15} "
            f"In: {self.input_tokens:>5} tokens  "
            f"Out: {self.output_tokens:>4} tokens  "
            f"Cost: ${self.total_cost_usd:.6f}"
        )


# ─────────────────────────────────────────────
# COST TRACKER
# ─────────────────────────────────────────────

class CostTracker:

    def __init__(self):
        self.records: List[QueryCost] = []
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in any text string"""
        return len(self.encoder.encode(text))

    def calculate_cost(
        self,
        model: str,
        input_text: str,
        output_text: str,
        query_type: str = "unknown"
    ) -> QueryCost:
        """
        Calculates the exact cost of one query.
        Call this after every LLM response.
        """

        pricing = MODEL_PRICING.get(model, {
            "input_per_1k": 0.001,
            "output_per_1k": 0.002
        })

        input_tokens = self.count_tokens(input_text)
        output_tokens = self.count_tokens(output_text)

        input_cost = (input_tokens / 1000) * pricing["input_per_1k"]
        output_cost = (output_tokens / 1000) * pricing["output_per_1k"]

        record = QueryCost(
            query=input_text[:100],
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_usd=round(input_cost, 8),
            output_cost_usd=round(output_cost, 8),
            total_cost_usd=round(input_cost + output_cost, 8),
            query_type=query_type
        )

        self.records.append(record)
        return record

    def project_monthly_cost(
        self,
        queries_per_day: int
    ) -> Dict:
        """
        Projects monthly cost based on average query cost so far.
        Run this after benchmarking to understand production economics.
        """
        if not self.records:
            return {}

        avg_cost = np.mean([r.total_cost_usd for r in self.records])
        monthly_queries = queries_per_day * 30

        return {
            "avg_cost_per_query": round(avg_cost, 6),
            "queries_per_day": queries_per_day,
            "monthly_queries": monthly_queries,
            "projected_monthly_usd": round(avg_cost * monthly_queries, 2),
            "projected_annual_usd": round(avg_cost * monthly_queries * 12, 2)
        }

    def print_cost_report(self, queries_per_day: int = 1000):
        """Prints a complete cost analysis report"""

        if not self.records:
            print("No cost records yet.")
            return

        df = pd.DataFrame([{
            "model": r.model,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "total_cost_usd": r.total_cost_usd,
            "query_type": r.query_type
        } for r in self.records])

        projection = self.project_monthly_cost(queries_per_day)

        print(f"\n{'━'*60}")
        print(f"💰 COST ANALYSIS REPORT")
        print(f"{'━'*60}")
        print(f"  Queries analyzed:     {len(self.records)}")
        print(f"  Total cost (sample):  "
              f"${df['total_cost_usd'].sum():.6f}")
        print(f"  Avg cost per query:   "
              f"${df['total_cost_usd'].mean():.6f}")
        print(f"  Avg input tokens:     "
              f"{df['input_tokens'].mean():.0f}")
        print(f"  Avg output tokens:    "
              f"{df['output_tokens'].mean():.0f}")
        print(f"{'─'*60}")
        print(f"  📈 PROJECTION ({queries_per_day:,} queries/day)")
        print(f"  Monthly cost:         "
              f"${projection['projected_monthly_usd']:,.2f}")
        print(f"  Annual cost:          "
              f"${projection['projected_annual_usd']:,.2f}")

        # Cost by model
        if df['model'].nunique() > 1:
            print(f"{'─'*60}")
            print(f"  Cost by model:")
            model_costs = df.groupby('model')['total_cost_usd'].agg(
                ['mean', 'sum', 'count']
            ).round(6)
            for model, row in model_costs.iterrows():
                print(f"    {model:<20} "
                      f"avg=${row['mean']:.6f}  "
                      f"count={int(row['count'])}")

        print(f"{'━'*60}\n")

    def save_records(self, filename: str = "cost_records.json"):
        data = [r.__dict__ for r in self.records]
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        pd.DataFrame(data).to_csv(
            filename.replace('.json', '.csv'), index=False
        )
        print(f"✅ Cost records saved → {filename}")
```

---

### Step 4 — Create the model comparison benchmarker

Right-click `day-07-cost` → **New File** → name it:

```
model_benchmarker.py
```

Paste this:

```python
# model_benchmarker.py
# Compares quality AND cost across multiple models
# Helps you find the cheapest model that meets your quality bar

from dotenv import load_dotenv
load_dotenv()

import time
import numpy as np
import pandas as pd
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset
from cost_tracker import CostTracker


SYSTEM_PROMPT = """You are a helpful assistant.
Answer questions accurately using only the provided context.
If the context does not contain the answer, say so clearly."""


class ModelBenchmarker:

    def __init__(self):
        self.cost_tracker = CostTracker()
        self.results = []

    @traceable
    def benchmark_model(
        self,
        model_name: str,
        eval_dataset: List[Dict],
        contexts_per_query: List[List[str]]
    ) -> Dict:
        """
        Benchmarks a single model on quality AND cost.
        Returns combined scores for comparison.
        """

        print(f"\n  Testing model: {model_name}")

        try:
            llm = ChatOpenAI(model=model_name, temperature=0)
        except Exception as e:
            print(f"  ❌ Could not load {model_name}: {e}")
            return {}

        questions = []
        answers = []
        ground_truths = []
        latencies = []

        # Generate responses
        for example, contexts in zip(eval_dataset, contexts_per_query):
            context_text = "\n\n".join(contexts)
            full_prompt = (
                f"Context:\n{context_text}\n\n"
                f"Question: {example['question']}"
            )

            start = time.perf_counter()
            response = llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=full_prompt)
            ])
            latency_ms = (time.perf_counter() - start) * 1000

            answer = response.content
            latencies.append(latency_ms)

            # Track cost
            self.cost_tracker.calculate_cost(
                model=model_name,
                input_text=SYSTEM_PROMPT + full_prompt,
                output_text=answer,
                query_type="benchmark"
            )

            questions.append(example["question"])
            answers.append(answer)
            ground_truths.append(example["ground_truth"])

        # Score quality
        print(f"    Scoring with RAGAS...")
        ragas_data = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts_per_query,
            "ground_truth": ground_truths
        })

        scores = evaluate(
            dataset=ragas_data,
            metrics=[faithfulness, answer_relevancy, context_recall]
        )
        scores_df = scores.to_pandas()

        # Get cost records for this model
        model_costs = [
            r for r in self.cost_tracker.records
            if r.model == model_name
        ]
        avg_cost = np.mean([r.total_cost_usd for r in model_costs])

        result = {
            "model": model_name,
            "faithfulness": round(scores_df["faithfulness"].mean(), 3),
            "context_recall": round(scores_df["context_recall"].mean(), 3),
            "answer_relevancy": round(scores_df["answer_relevancy"].mean(), 3),
            "avg_latency_ms": round(np.mean(latencies), 0),
            "avg_cost_per_query": round(avg_cost, 6),
            "monthly_cost_1k_daily": round(avg_cost * 1000 * 30, 2),
            "quality_score": round(
                scores_df["faithfulness"].mean() * 0.4 +
                scores_df["context_recall"].mean() * 0.4 +
                scores_df["answer_relevancy"].mean() * 0.2,
                3
            )
        }

        self.results.append(result)
        return result

    def print_comparison(self):
        """Prints the full model comparison matrix"""

        if not self.results:
            print("No results yet.")
            return

        df = pd.DataFrame(self.results).sort_values(
            "quality_score", ascending=False
        )

        print(f"\n{'━'*85}")
        print(f"📊 MODEL COMPARISON — Quality vs Cost")
        print(f"{'━'*85}")
        print(
            f"{'Model':<20} "
            f"{'Quality':>9} "
            f"{'Faithful':>9} "
            f"{'Recall':>8} "
            f"{'Latency':>9} "
            f"{'$/query':>9} "
            f"{'$/mo@1k/day':>12}"
        )
        print(f"{'─'*85}")

        best_quality = df['quality_score'].max()
        best_cost = df['avg_cost_per_query'].min()

        for _, row in df.iterrows():
            quality_marker = " ★" if row['quality_score'] == best_quality \
                else ""
            cost_marker = " ✓" if row['avg_cost_per_query'] == best_cost \
                else ""

            print(
                f"{row['model']:<20} "
                f"{row['quality_score']:>9.3f}"
                f"{quality_marker:<2}"
                f"{row['faithfulness']:>9.3f} "
                f"{row['context_recall']:>8.3f} "
                f"{row['avg_latency_ms']:>8.0f}ms "
                f"{row['avg_cost_per_query']:>8.6f}"
                f"{cost_marker:<2}"
                f"{row['monthly_cost_1k_daily']:>11.2f}"
            )

        print(f"{'━'*85}")
        print(f"  ★ = best quality   ✓ = lowest cost")

        # Value recommendation
        df['value_score'] = df['quality_score'] / (
            df['avg_cost_per_query'] * 10000 + 0.001
        )
        best_value = df.loc[df['value_score'].idxmax()]
        print(
            f"\n💡 Best value: {best_value['model']} — "
            f"quality {best_value['quality_score']:.3f} "
            f"at ${best_value['avg_cost_per_query']:.6f}/query\n"
        )

    def save_results(self, filename: str = "model_comparison.json"):
        import json
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        pd.DataFrame(self.results).to_csv(
            filename.replace('.json', '.csv'), index=False
        )
        print(f"✅ Model comparison saved → {filename}")
```

---

### Step 5 — Create the query router

Right-click `day-07-cost` → **New File** → name it:

```
query_router.py
```

Paste this:

```python
# query_router.py
# Automatically routes queries to the right model based on complexity
# Reduces cost while maintaining quality where it matters

from dotenv import load_dotenv
load_dotenv()

import re
import time
from typing import Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from cost_tracker import CostTracker


# ─────────────────────────────────────────────
# ROUTING RULES
# Adjust thresholds based on your use case
# ─────────────────────────────────────────────

ROUTING_RULES = {
    "simple": {
        "model": "gpt-4o-mini",
        "description": "Factual lookups, yes/no questions, definitions",
        "signals": [
            "what is", "what are", "who is", "when did",
            "define", "list", "name the", "how many"
        ]
    },
    "moderate": {
        "model": "gpt-4o-mini",
        "description": "Explanations, comparisons, short summaries",
        "signals": [
            "explain", "compare", "difference between",
            "how does", "why does", "summarize"
        ]
    },
    "complex": {
        "model": "gpt-4o",
        "description": "Multi-hop reasoning, synthesis, analysis",
        "signals": [
            "analyze", "evaluate", "synthesize", "given that",
            "considering", "in light of", "what would happen if",
            "implications", "recommend", "critique"
        ]
    }
}


class QueryRouter:

    def __init__(self):
        self.cost_tracker = CostTracker()
        self.routing_log = []

        # Initialize models
        self.models = {
            "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", temperature=0),
            "gpt-4o": ChatOpenAI(model="gpt-4o", temperature=0)
        }

    def classify_query(self, query: str) -> Tuple[str, str]:
        """
        Classifies query complexity using keyword signals.
        Returns (complexity_level, model_name)

        To replace with an ML classifier:
        - Train a simple classifier on labeled queries
        - Replace this function with: return your_classifier.predict(query)
        """
        query_lower = query.lower()

        # Check complex signals first — they override simpler ones
        for level in ["complex", "moderate", "simple"]:
            signals = ROUTING_RULES[level]["signals"]
            if any(signal in query_lower for signal in signals):
                return level, ROUTING_RULES[level]["model"]

        # Default to moderate if no signals match
        # Adjust this default based on your query distribution
        return "moderate", ROUTING_RULES["moderate"]["model"]

    @traceable
    def route_and_respond(
        self,
        query: str,
        contexts: list,
        system_prompt: str = "Answer using only the provided context."
    ) -> dict:
        """
        Routes the query to the appropriate model and generates a response.
        Returns the response with routing metadata.
        """

        complexity, model_name = self.classify_query(query)
        llm = self.models[model_name]

        context_text = "\n\n".join(contexts)
        full_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

        start = time.perf_counter()
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=full_prompt)
        ])
        latency_ms = (time.perf_counter() - start) * 1000

        answer = response.content

        # Track cost
        cost_record = self.cost_tracker.calculate_cost(
            model=model_name,
            input_text=system_prompt + full_prompt,
            output_text=answer,
            query_type=complexity
        )

        routing_record = {
            "query": query[:80],
            "complexity": complexity,
            "model_used": model_name,
            "latency_ms": round(latency_ms, 2),
            "cost_usd": cost_record.total_cost_usd,
            "answer": answer
        }
        self.routing_log.append(routing_record)

        return routing_record

    def print_routing_report(self):
        """Shows routing distribution and cost savings"""

        if not self.routing_log:
            print("No routing records yet.")
            return

        import pandas as pd
        df = pd.DataFrame(self.routing_log)

        print(f"\n{'━'*60}")
        print(f"🔀 QUERY ROUTING REPORT")
        print(f"{'━'*60}")

        routing_summary = df.groupby(['complexity', 'model_used']).agg(
            count=('query', 'count'),
            avg_cost=('cost_usd', 'mean'),
            avg_latency=('latency_ms', 'mean')
        ).round(6)

        for (complexity, model), row in routing_summary.iterrows():
            print(
                f"  {complexity:<12} → {model:<15} "
                f"count={int(row['count'])}  "
                f"avg_cost=${row['avg_cost']:.6f}  "
                f"avg_latency={row['avg_latency']:.0f}ms"
            )

        # Cost comparison: what if everything used gpt-4o?
        total_actual = df['cost_usd'].sum()
        gpt4o_price = 0.005 / 1000  # per token approximate
        avg_tokens = self.cost_tracker.records[0].input_tokens if \
            self.cost_tracker.records else 500
        total_if_gpt4o = len(df) * avg_tokens * gpt4o_price * 2

        savings_pct = ((total_if_gpt4o - total_actual) /
                       total_if_gpt4o * 100) if total_if_gpt4o > 0 else 0

        print(f"{'─'*60}")
        print(f"  Actual cost (with routing):   ${total_actual:.6f}")
        print(f"  Estimated without routing:    ${total_if_gpt4o:.6f}")
        print(f"  Estimated savings:            {savings_pct:.1f}%")
        print(f"{'━'*60}\n")
```

---

### Step 6 — Create the runner

Right-click `day-07-cost` → **New File** → name it:

```
run_cost_benchmark.py
```

Paste this:

```python
# run_cost_benchmark.py
# Runs the full cost benchmarking suite

from dotenv import load_dotenv
load_dotenv()

from cost_tracker import CostTracker
from model_benchmarker import ModelBenchmarker
from query_router import QueryRouter

EVAL_DATASET = [
    {
        "question": "What is Apache Kafka?",
        "ground_truth": "Apache Kafka is a distributed event streaming "
                        "platform developed at LinkedIn."
    },
    {
        "question": "How do consumer groups work?",
        "ground_truth": "Consumer groups allow multiple consumers to share "
                        "reading from a topic in parallel."
    },
    {
        "question": "Analyze the tradeoffs between Kafka's "
                    "exactly-once semantics and throughput performance.",
        "ground_truth": "Exactly-once semantics add overhead through "
                        "transactions and idempotent producers, reducing "
                        "throughput compared to at-least-once delivery."
    },
    {
        "question": "Compare Kafka's replication strategy with "
                    "traditional database replication.",
        "ground_truth": "Kafka uses partition-level replication across "
                        "brokers with leader election, while traditional "
                        "databases typically use primary-replica replication."
    }
]

SHARED_CONTEXT = [
    "Apache Kafka is a distributed event streaming platform developed "
    "at LinkedIn and open-sourced in 2011. It uses a distributed commit "
    "log architecture.",
    "Consumer groups allow multiple consumers to share reading from a "
    "topic. Each partition is assigned to exactly one consumer in the "
    "group at a time enabling parallel processing.",
    "Exactly-once semantics in Kafka are achieved through transactions "
    "and idempotent producers. This adds overhead that reduces maximum "
    "throughput compared to at-least-once delivery.",
    "Kafka uses partition-level leader-follower replication. When a "
    "leader broker fails, a follower is elected as the new leader. "
    "Traditional databases use primary-replica replication at table level."
]

CONTEXTS = [SHARED_CONTEXT for _ in EVAL_DATASET]


if __name__ == "__main__":

    # ── Part 1: Model comparison ──────────────
    print("=" * 60)
    print("PART 1 — MODEL QUALITY vs COST COMPARISON")
    print("=" * 60)

    benchmarker = ModelBenchmarker()

    for model in ["gpt-4o-mini", "gpt-4o"]:
        benchmarker.benchmark_model(
            model_name=model,
            eval_dataset=EVAL_DATASET,
            contexts_per_query=CONTEXTS
        )

    benchmarker.print_comparison()
    benchmarker.save_results("model_comparison.json")

    # ── Part 2: Query routing ─────────────────
    print("=" * 60)
    print("PART 2 — QUERY ROUTING DEMO")
    print("=" * 60)

    router = QueryRouter()

    test_queries = [
        "What is a Kafka topic?",
        "Explain how consumer offsets work",
        "Analyze the architectural tradeoffs of using Kafka "
        "vs a traditional message queue for a hospital system"
    ]

    print("\nRouting test queries...\n")
    for query in test_queries:
        result = router.route_and_respond(
            query=query,
            contexts=SHARED_CONTEXT
        )
        print(
            f"  Query:      {result['query'][:55]}..."
            f"\n  Complexity: {result['complexity']}"
            f"\n  Model:      {result['model_used']}"
            f"\n  Cost:       ${result['cost_usd']:.6f}"
            f"\n  Latency:    {result['latency_ms']:.0f}ms\n"
        )

    router.print_routing_report()
    router.cost_tracker.print_cost_report(queries_per_day=1000)
    router.cost_tracker.save_records("cost_records.json")

    print("✅ Check LangSmith — all runs logged with cost metadata.")
```

Run it:

```bash
python run_cost_benchmark.py
```

You will see the model comparison matrix and routing report — with exact costs and projected monthly bills.

---

## ✅ Day 7 Checklist

- [ ] `cost_tracker.py` runs and counts tokens accurately
- [ ] `model_benchmarker.py` compares at least two models
- [ ] You have seen the monthly cost projection for your query volume
- [ ] `query_router.py` correctly routes simple vs complex queries
- [ ] Routing signals are customized for your domain
- [ ] Cost records saved to `cost_records.json`
- [ ] All runs visible in LangSmith

---

## 🎯 Interview Bits — Day 7

**Q: How would you reduce LLM API costs without sacrificing quality?**
*Query routing — classify queries by complexity and send simple ones to cheaper smaller models. Only route genuinely complex multi-hop reasoning to expensive models. In most systems 70-80% of queries are simple enough for a mini model with equivalent quality.*

**Q: What is the cost difference between input and output tokens and why does it matter?**
*Output tokens cost 3-5x more than input tokens on most models. This means verbose responses are disproportionately expensive. Prompts that instruct the model to be concise reduce output token usage and cost without affecting retrieval quality.*

**Q: How do you project monthly AI costs accurately?**
*Benchmark a representative sample of real queries — not just simple test cases. Calculate average token usage per query type. Multiply by your expected query volume and current model pricing. Apply a 20% buffer for traffic spikes. Monitor actual spend weekly and compare to projection.*

**Q: What is the quality-cost tradeoff triangle in model selection?**
*Larger models produce higher quality outputs but cost significantly more per token. The optimal choice depends on your quality floor — the minimum acceptable score on your eval dataset. Find the cheapest model that meets that floor. Anything above the floor is money spent on quality you do not need.*

---

*Tomorrow we find out what happens when real traffic hits your system.*
*Single-user latency looked fine.*
*500 concurrent users is a different story entirely.*
*Day 8 — load testing.*

