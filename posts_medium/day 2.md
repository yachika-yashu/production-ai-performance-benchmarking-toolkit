# AI Performance Benchmarking: From Zero to Production
### Day 2 of 12 — The Baseline Problem: You Can't Improve What You Never Measured

---

*This is Day 2 of a 12-part series. If you're just joining, start with **[Day 0](#)** for setup and **[Day 1](#)** for the mental model. Everything here builds on those foundations.*

---

## The Most Skipped Step In AI Development

You make a change to your AI system.

New prompt. Different chunk size. Swapped the model.

It feels better.

The answers seem more relevant. The responses sound more confident.

You ship it.

Two weeks later someone asks:

*"How much did that change actually improve things?"*

And you realize — you have nothing to compare against.

No numbers from before. No record of what "before" even looked like. Just a feeling that it got better.

That feeling is not a benchmark.

**A baseline is.**

A baseline is a snapshot of your system's performance before you change anything. It is the stake in the ground that every future measurement is compared against.

Without it you are not improving your system.

You are just changing it.

---

## What A Baseline Actually Is

Think of it like a blood test.

When you go to a doctor for the first time they run a full panel. Not because something is wrong. Because they need to know what *normal* looks like for you specifically — before anything goes wrong.

Six months later if something changes in your results the doctor doesn't guess. They compare against your baseline.

That comparison is where the insight lives.

Your AI system needs the same thing.

A baseline answers three questions at a specific moment in time:

- **Quality baseline** — how correct and faithful are the answers right now?
- **Speed baseline** — how fast is the system responding right now?
- **Cost baseline** — how much is each query costing right now?

Everything you do from that moment forward gets compared against these numbers.

---

## The Two Things You Need For A Baseline

Here is where most people get confused.

A baseline is not just running your system and recording some numbers.

It requires two things working together:

**1. An evaluation dataset** — a set of questions with known correct answers

**2. A measurement run** — your system answering those questions while everything gets recorded

Most tutorials talk about the measurement run. Almost nobody talks about the evaluation dataset.

That is the hard part. And that is what we are focusing on today.

---

## Building Your Evaluation Dataset

Your evaluation dataset is the most important artifact in your entire benchmarking practice.

Get it right and every future benchmark is meaningful.

Get it wrong and you will be measuring the wrong things with false confidence.

A good evaluation dataset has four properties:

---

**Property 1 — It covers the real distribution of queries**

Not the queries you think users will ask.

The queries users actually ask.

If your system handles customer support, your dataset should have angry users, confused users, edge cases, and completely off-topic questions — not just the clean polite ones you wrote during development.

---

**Property 2 — It includes ground truth answers**

Every question needs a known correct answer.

Not a perfect answer. A reference answer — something a domain expert would consider acceptable. This is what quality metrics compare against.

Without ground truth you cannot measure quality. You can only measure confidence — and confident wrong answers are the most dangerous kind.

---

**Property 3 — It is diverse enough to catch blind spots**

A dataset of 50 similar questions tells you how your system handles one type of query.

A dataset of 50 carefully varied questions tells you where your system actually breaks.

Include:
- Simple factual questions
- Multi-hop reasoning questions
- Edge cases and ambiguous queries
- Questions your system should refuse or flag

---

**Property 4 — It is version controlled**

Your evaluation dataset is code. Treat it like code.

Put it in your repository. Track changes. Never overwrite the original.

Because if your dataset changes between benchmarks you are not comparing like for like — you are comparing apples to slightly different apples.

---

## How Many Examples Do You Actually Need?

This is the question everyone asks.

The honest answer: it depends on what you are measuring.

Here is a practical starting point:

| Use Case | Minimum Dataset Size | Why |
|---|---|---|
| Quick sanity check | 20-30 queries | Catches obvious regressions |
| Development benchmarking | 50-100 queries | Meaningful quality scores |
| Pre-production validation | 200-500 queries | Statistically reliable |
| Production regression testing | 100+ queries per category | Covers all query types |

Start with 50. That is enough to get real signal without spending days curating data.

---

## Where Does The Data Come From?

Three sources. Use all three.

**Source 1 — Real user queries**
If your system is already live, even in a small way, export a sample of real queries. Anonymize anything sensitive. These are gold — they reflect actual usage patterns.

**Source 2 — Expert-crafted queries**
You or a domain expert deliberately write questions that test specific capabilities and edge cases. Slower to produce but gives you control over coverage.

**Source 3 — Synthetic queries**
Use an LLM to generate questions from your source documents. Fast and scalable but needs human review — LLMs tend to generate clean questions that miss the messy edge cases real users produce.

The best datasets combine all three.

---

## The Problem With Synthetic Data

Let's be honest about this because it matters.

Synthetic evaluation datasets are tempting. You can generate 200 questions in minutes.

But they have a hidden flaw.

**The same model that generates your questions will tend to answer them well.**

If you use GPT-4o to generate your eval dataset and then benchmark GPT-4o against it — you are not measuring real quality. You are measuring how well GPT-4o answers its own questions.

The fix: generate synthetically, then review every question manually. Remove anything that feels too clean or too easy. Add the messy edge cases a real user would throw at your system.

---

## The Code — Building Your Evaluation Dataset

Today we build two things:

1. A dataset builder that creates synthetic questions from your documents and structures them properly
2. A baseline runner that measures your system against that dataset and saves the results

---

### Step 1 — Set up today's folder

In the VS Code terminal:

```bash
cd ..
mkdir day-02-baseline
cd day-02-baseline
```

---

### Step 2 — Install today's dependencies

```bash
pip install langsmith langchain langchain-openai python-dotenv pandas
```

---

### Step 3 — Create the dataset builder

Right-click `day-02-baseline` → **New File** → name it:

```
dataset_builder.py
```

Paste this exactly:

```python
# dataset_builder.py
# Builds a structured evaluation dataset from your documents
# Works with any text content — paste your own documents below

from dotenv import load_dotenv
load_dotenv()

import json
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langsmith import traceable

# ─────────────────────────────────────────────
# THE EVALUATION EXAMPLE STRUCTURE
# One row in your dataset
# ─────────────────────────────────────────────

@dataclass
class EvalExample:
    question: str           # The query your AI will receive
    ground_truth: str       # The correct reference answer
    context: str            # The source text this is based on
    category: str           # Type of question (factual, reasoning, edge_case)
    difficulty: str         # easy, medium, hard
    created_at: str = ""

    def __post_init__(self):
        self.created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# ─────────────────────────────────────────────
# THE DATASET BUILDER
# Generates evaluation examples from your text
# ─────────────────────────────────────────────

@traceable
def generate_eval_examples(
    source_text: str,
    num_questions: int = 10,
    category: str = "factual"
) -> List[EvalExample]:
    """
    Takes any source text and generates structured eval examples.
    Replace source_text with your actual documents.
    """

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
You are building an evaluation dataset for an AI system.

Given the following source text, generate exactly {num_questions} evaluation examples.

For each example generate:
1. A realistic question a user might ask about this content
2. A accurate reference answer based only on the source text
3. The difficulty level: easy, medium, or hard

Source text:
{source_text}

Return your response as a JSON array with this exact structure:
[
  {{
    "question": "the question here",
    "ground_truth": "the reference answer here",
    "difficulty": "easy|medium|hard"
  }}
]

Rules:
- Questions should feel like real user queries, not exam questions
- Include a mix of difficulties
- At least 2 questions should be edge cases or ambiguous
- Ground truth answers should be concise but complete
- Return ONLY the JSON array, no other text
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    # Parse the JSON response
    try:
        raw_examples = json.loads(response.content)
    except json.JSONDecodeError:
        # Sometimes the model adds extra text — strip it
        content = response.content
        start = content.find('[')
        end = content.rfind(']') + 1
        raw_examples = json.loads(content[start:end])

    # Convert to EvalExample objects
    examples = []
    for item in raw_examples:
        example = EvalExample(
            question=item["question"],
            ground_truth=item["ground_truth"],
            context=source_text[:500],  # Store first 500 chars as context reference
            category=category,
            difficulty=item["difficulty"]
        )
        examples.append(example)

    return examples


def save_dataset(examples: List[EvalExample], filename: str = "eval_dataset.json"):
    """
    Saves your dataset to JSON and CSV.
    JSON for programmatic use. CSV for human review.
    """

    # Save as JSON
    data = [asdict(e) for e in examples]
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    # Save as CSV for easy human review
    csv_filename = filename.replace('.json', '.csv')
    df = pd.DataFrame(data)
    df.to_csv(csv_filename, index=False)

    print(f"✅ Dataset saved:")
    print(f"   JSON → {filename}")
    print(f"   CSV  → {csv_filename}")
    print(f"   Total examples: {len(examples)}")

    # Print a summary
    df_summary = df.groupby(['category', 'difficulty']).size().reset_index(name='count')
    print(f"\n📊 Dataset breakdown:\n{df_summary.to_string(index=False)}")


# ─────────────────────────────────────────────
# EXAMPLE — replace this with your own content
# ─────────────────────────────────────────────

SAMPLE_DOCUMENT = """
Apache Kafka is a distributed event streaming platform designed to handle
high-throughput, fault-tolerant, real-time data feeds. Originally developed
at LinkedIn and open-sourced in 2011, Kafka is built around the concept of
a distributed commit log.

Core concepts include Topics, which are categories where messages are published.
Partitions divide topics for parallel processing and scalability. Producers
write data to topics while Consumers read from them. Brokers are the Kafka
servers that store and serve data. Consumer Groups allow multiple consumers
to share the work of reading from a topic.

Kafka guarantees at-least-once delivery by default, with exactly-once
semantics available through transactions. Messages are retained for a
configurable period regardless of whether they have been consumed,
allowing consumers to replay historical data.

Common use cases include real-time analytics pipelines, event sourcing
architectures, log aggregation, and stream processing with tools like
Apache Spark or Apache Flink.
"""


if __name__ == "__main__":
    print("🔨 Building your evaluation dataset...\n")

    # Generate examples
    # Replace SAMPLE_DOCUMENT with your own text
    examples = generate_eval_examples(
        source_text=SAMPLE_DOCUMENT,
        num_questions=10,
        category="factual"
    )

    # Save them
    save_dataset(examples, "eval_dataset.json")

    print("\n📋 First 3 examples:")
    for i, ex in enumerate(examples[:3]):
        print(f"\n[{i+1}] Category: {ex.category} | Difficulty: {ex.difficulty}")
        print(f"     Q: {ex.question}")
        print(f"     A: {ex.ground_truth[:100]}...")
```

Run it:

```bash
python dataset_builder.py
```

You should see:

```
🔨 Building your evaluation dataset...

✅ Dataset saved:
   JSON → eval_dataset.json
   CSV  → eval_dataset.csv
   Total examples: 10

📊 Dataset breakdown:
 category difficulty  count
  factual       easy      4
  factual       hard      2
  factual     medium      4

📋 First 3 examples:
[1] Category: factual | Difficulty: easy
     Q: What is Apache Kafka primarily designed for?
     A: Apache Kafka is designed to handle high-throughput, fault-tolerant...
```

Now open `eval_dataset.csv` in VS Code. Review every row. Edit any question that feels too clean or too obvious. This human review step is not optional.

---

### Step 4 — Create the baseline runner

Right-click `day-02-baseline` → **New File** → name it:

```
baseline_runner.py
```

Paste this exactly:

```python
# baseline_runner.py
# Runs your AI system against the eval dataset and saves the baseline
# This is the snapshot everything future will be compared against

from dotenv import load_dotenv
load_dotenv()

import json
import time
import pandas as pd
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

# Import our scaffold from Day 1
import sys
sys.path.append('../day-01-mental-model')
from benchmark_scaffold import BenchmarkResult


# ─────────────────────────────────────────────
# YOUR AI FUNCTION
# Replace this with your actual pipeline
# ─────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@traceable
def your_ai_function(query: str) -> str:
    """
    Replace the body of this function with your real AI pipeline.
    Could be a RAG chain, an agent, a simple LLM call — anything.
    The interface stays the same: string in, string out.
    """
    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant. Answer concisely and accurately."),
        HumanMessage(content=query)
    ])
    return response.content


# ─────────────────────────────────────────────
# THE BASELINE RUNNER
# ─────────────────────────────────────────────

@traceable
def run_baseline(
    dataset_path: str = "eval_dataset.json",
    model_name: str = "gpt-4o-mini",
    cost_per_1k_tokens: float = 0.00015
) -> str:
    """
    Runs your AI function against every example in your eval dataset.
    Saves results as a timestamped baseline file.
    Returns the path to the saved baseline.
    """

    # Load the dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    print(f"📂 Loaded {len(dataset)} examples from {dataset_path}")
    print(f"🚀 Running baseline with model: {model_name}\n")

    results = []
    total_cost = 0
    total_latency = 0

    for i, example in enumerate(dataset):
        print(f"  Running example {i+1}/{len(dataset)}...", end=" ")

        # Measure latency
        start = time.perf_counter()
        response = your_ai_function(example['question'])
        end = time.perf_counter()

        latency_ms = (end - start) * 1000

        # Estimate cost
        tokens = (len(example['question']) + len(response)) / 4
        cost = (tokens / 1000) * cost_per_1k_tokens

        total_cost += cost
        total_latency += latency_ms

        # Store result
        result = {
            "question": example['question'],
            "ground_truth": example['ground_truth'],
            "ai_response": response,
            "latency_ms": round(latency_ms, 2),
            "cost_usd": round(cost, 6),
            "category": example.get('category', 'unknown'),
            "difficulty": example.get('difficulty', 'unknown'),
            "model": model_name,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        results.append(result)
        print(f"✓ {latency_ms:.0f}ms")

    # ── Save the baseline ────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    baseline_path = f"baseline_{timestamp}.json"

    with open(baseline_path, 'w') as f:
        json.dump({
            "metadata": {
                "model": model_name,
                "dataset": dataset_path,
                "total_examples": len(results),
                "created_at": datetime.now().isoformat(),
                "avg_latency_ms": round(total_latency / len(results), 2),
                "total_cost_usd": round(total_cost, 6),
                "avg_cost_per_query": round(total_cost / len(results), 6)
            },
            "results": results
        }, f, indent=2)

    # ── Print the summary ────────────────────
    avg_latency = total_latency / len(results)

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 BASELINE COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Examples run:     {len(results)}
Avg latency:      {avg_latency:.0f}ms
Total cost:       ${total_cost:.4f}
Cost per query:   ${total_cost/len(results):.6f}
─────────────────────────────────
Quality scores:   Not measured yet — see Day 4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Baseline saved → {baseline_path}
    """)

    return baseline_path


if __name__ == "__main__":
    baseline_path = run_baseline(
        dataset_path="eval_dataset.json",
        model_name="gpt-4o-mini",
        cost_per_1k_tokens=0.00015
    )

    print(f"\n🔒 This is your stake in the ground.")
    print(f"   Every future change gets compared against: {baseline_path}")
    print(f"   Do not modify this file. Ever.")
```

Run it:

```bash
python baseline_runner.py
```

You should see each example running with its latency, then a summary:

```
📂 Loaded 10 examples from eval_dataset.json
🚀 Running baseline with model: gpt-4o-mini

  Running example 1/10... ✓ 847ms
  Running example 2/10... ✓ 612ms
  ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 BASELINE COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Examples run:     10
Avg latency:      731ms
Total cost:       $0.0003
Cost per query:   $0.000031
─────────────────────────────────
Quality scores:   Not measured yet — see Day 4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Baseline saved → baseline_20260526_100000.json
```

⚠️ **The one rule about your baseline file:**
Never overwrite it. Never edit it. It is a timestamped snapshot. When you make improvements you run a new measurement and compare the new file against the original baseline. The original stays untouched.

---

### Step 5 — Plug it into your existing project

If you already have a working AI pipeline, one change in `baseline_runner.py` is all you need.

Find this function:

```python
def your_ai_function(query: str) -> str:
    response = llm.invoke([...])
    return response.content
```

Replace the body with your real pipeline. For a RAG system:

```python
# Example — RAG pipeline replacement
# Your retriever and chain already exist — just plug them in here

def your_ai_function(query: str) -> str:
    # Your existing retrieval step
    docs = your_retriever.invoke(query)

    # Your existing generation step
    response = your_rag_chain.invoke({
        "question": query,
        "context": docs
    })
    return response
```

The interface — string in, string out — never changes. Everything else is your code.

---

## What You Just Built

You now have two files that form the foundation of your entire benchmarking practice:

`eval_dataset.json` — your ground truth. The questions and reference answers your system will always be measured against.

`baseline_TIMESTAMP.json` — your stake in the ground. The speed and cost snapshot of your system before you change anything.

From Day 3 onwards every improvement you make gets compared against these two files. When you can say *"after this change latency dropped 40% and cost dropped 30% with no quality regression"* — this is where that proof comes from.

---

## ✅ Day 2 Checklist

- [ ] `dataset_builder.py` runs and generates `eval_dataset.json`
- [ ] You have reviewed every row in `eval_dataset.csv` manually
- [ ] You have replaced `SAMPLE_DOCUMENT` with your own content
- [ ] `baseline_runner.py` runs against your dataset successfully
- [ ] Your baseline JSON file is saved and you have not modified it
- [ ] Your baseline file is committed to version control

---

## 🎯 Interview Bits — Day 2

**Q: What makes a good evaluation dataset for an AI system?**
*A good eval dataset reflects the real distribution of user queries — not just the clean happy path. It includes ground truth reference answers, covers edge cases and ambiguous inputs, is diverse enough to catch blind spots, and is version controlled so comparisons stay valid over time.*

**Q: What is the risk of using synthetic data for evaluation?**
*The model used to generate synthetic questions tends to answer them well — creating an optimistic benchmark that doesn't reflect real user behavior. Synthetic data needs human review and deliberate injection of messy edge cases to be meaningful.*

**Q: Why should a baseline file never be modified?**
*Because it is a timestamped snapshot of system performance at a specific moment. Modifying it invalidates every future comparison made against it. Improvements are measured by running new benchmarks and comparing new files against the original — never by changing the original.*

**Q: How many examples do you need in an evaluation dataset?**
*It depends on what you are measuring. 20-30 examples catch obvious regressions. 50-100 give meaningful quality scores. 200-500 are statistically reliable for pre-production validation. Start with 50 — enough for real signal without days of curation.*

---

*Tomorrow we go deep on latency.*
*Average response time is lying to you. P99 is telling the truth.*
*Day 3 shows you exactly where the time is being lost — and how to get it back.*

---