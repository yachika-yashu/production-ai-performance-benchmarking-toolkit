# AI Performance Benchmarking: From Zero to Production
### Day 3 of 12 — Latency & Throughput: Why Your AI Feels Slow and Exactly Where to Fix It

---

*This is Day 3 of a 12-part series. If you're just joining, start with **[Day 0](#)** for setup, **[Day 1](#)** for the mental model, and **[Day 2](#)** for building your baseline. Everything here builds on those foundations.*

---

## The Number That's Lying To You

You check your AI system's response time.

Average latency: 1.1 seconds.

Not bad. Feels acceptable. You move on.

Then a user complains it took 14 seconds to get an answer.

You check again. Average is still 1.1 seconds.

So who is right?

Both of you.

And that is exactly the problem.

**Averages hide the worst experiences.**

If 99 queries take 0.5 seconds and one query takes 50 seconds — your average is still under 1 second. But that one user who waited 50 seconds? They are not coming back.

This is why averages are the wrong metric for latency.

And this is why today we talk about percentiles.

---

## P50, P95, P99 — The Metrics That Tell The Truth

Forget averages. Here is how to actually read latency.

**P50 — The Median Experience**

50% of your queries complete faster than this number.
50% take longer.

This is your typical user experience. The middle of the road.

---

**P95 — The Stressed Experience**

95% of your queries complete faster than this number.
5% take longer.

This is what your system looks like when things are slightly busy — real traffic, real load, real concurrency. If P95 is painful, most users will feel it eventually.

---

**P99 — The Worst Case Experience**

99% of your queries complete faster than this number.
1% take longer.

This sounds rare. But at 10,000 queries per day that is 100 users per day hitting your worst case. At 100,000 queries it is 1,000 users.

**P99 is the metric that keeps production systems honest.**

---

Here is what a healthy vs unhealthy latency profile looks like:

```
HEALTHY SYSTEM
──────────────
P50:  0.8s   ← typical user waits under 1s
P95:  1.4s   ← 95% of users under 1.5s
P99:  2.1s   ← worst case still acceptable

UNHEALTHY SYSTEM
────────────────
P50:  0.9s   ← looks fine on average
P95:  3.2s   ← already concerning
P99: 14.7s   ← 1 in 100 users waits 15 seconds
```

Same average. Completely different user experience.

---

## Where Is The Time Actually Going?

Here is the second mistake people make with latency.

They measure end-to-end response time and try to optimize the whole thing at once.

That never works.

Because latency in an AI pipeline is not one problem. It is four separate problems hiding inside one number.

```
User sends query
      ↓
[Stage 1: Pre-processing]     ← query cleaning, routing, validation
      ↓
[Stage 2: Retrieval]          ← vector search, reranking, context assembly
      ↓
[Stage 3: LLM Inference]      ← the actual model call
      ↓
[Stage 4: Post-processing]    ← formatting, filtering, safety checks
      ↓
User receives response
```

Each stage has its own latency profile. Each stage has different optimization strategies.

Before you optimize anything you need to know which stage is the bottleneck.

And you cannot know that without stage-level profiling.

---

## Time To First Token — The Perception Problem

There is one more latency concept that matters enormously for user experience.

**Time To First Token (TTFT).**

End-to-end latency measures how long until the complete response arrives.

TTFT measures how long until the very first word appears.

Why does this matter?

Because humans perceive waiting differently when something is happening versus when nothing is happening.

A system that starts streaming a response in 0.3 seconds and finishes in 4 seconds feels faster than a system that delivers the complete response in 2 seconds with no intermediate output.

The total time is longer. The perceived wait is shorter.

**For any AI system with a user interface — TTFT is as important as total latency.**

Streaming is how you fix TTFT. We cover that in the code today.

---

## The Problem Most People Hit

You have a working AI pipeline.

You want to understand where the latency is coming from.

So you add `time.time()` calls around your code.

And then you realize — your pipeline has 8 nested function calls, three external API calls, and two database queries. Instrumenting all of them manually takes hours and the code becomes unreadable.

There is a better way.

We are going to build a **pipeline profiler** that wraps each stage cleanly, measures it automatically, and logs everything to LangSmith so you can see the full breakdown visually.

---

## The Code — Latency Profiler

Today we build three things:

1. A **stage profiler** that measures each part of your pipeline separately
2. A **percentile calculator** that gives you P50, P95, P99 across many runs
3. A **streaming wrapper** that improves TTFT for user-facing systems

---

### Step 1 — Set up today's folder

In the VS Code terminal:

```bash
cd ..
mkdir day-03-latency
cd day-03-latency
```

---

### Step 2 — Install today's dependencies

```bash
pip install langsmith langchain langchain-openai python-dotenv numpy pandas
```

---

### Step 3 — Create the pipeline profiler

Right-click `day-03-latency` → **New File** → name it:

```
pipeline_profiler.py
```

Paste this exactly:

```python
# pipeline_profiler.py
# Measures latency at every stage of your AI pipeline
# Works with any pipeline — RAG, agents, simple LLM chains

from dotenv import load_dotenv
load_dotenv()

import time
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any
from langsmith import traceable

# ─────────────────────────────────────────────
# STAGE RESULT — one measured pipeline stage
# ─────────────────────────────────────────────

@dataclass
class StageResult:
    stage_name: str
    latency_ms: float
    input_size: int       # characters in, helps correlate size with latency
    output_size: int      # characters out
    success: bool = True
    error: str = ""


# ─────────────────────────────────────────────
# PIPELINE PROFILE — the full breakdown
# ─────────────────────────────────────────────

@dataclass
class PipelineProfile:
    query: str
    stages: List[StageResult] = field(default_factory=list)
    total_latency_ms: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def add_stage(self, stage: StageResult):
        self.stages.append(stage)
        self.total_latency_ms = sum(s.latency_ms for s in self.stages)

    def summary(self) -> str:
        lines = [
            f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"⚡ PIPELINE LATENCY BREAKDOWN",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Query: {self.query[:60]}{'...' if len(self.query) > 60 else ''}",
            f"─────────────────────────────────"
        ]

        for stage in self.stages:
            pct = (stage.latency_ms / self.total_latency_ms * 100) if self.total_latency_ms > 0 else 0
            bar = "█" * int(pct / 5)  # Visual bar — each block = 5%
            status = "✓" if stage.success else "✗"
            lines.append(
                f"{status} {stage.stage_name:<20} {stage.latency_ms:>7.0f}ms  {pct:>4.0f}%  {bar}"
            )

        lines.extend([
            f"─────────────────────────────────",
            f"  {'TOTAL':<20} {self.total_latency_ms:>7.0f}ms  100%",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ])

        return "\n".join(lines)


# ─────────────────────────────────────────────
# THE PROFILER
# Wraps any function and measures it cleanly
# ─────────────────────────────────────────────

class PipelineProfiler:

    def __init__(self, pipeline_name: str = "ai-pipeline"):
        self.pipeline_name = pipeline_name
        self.profiles: List[PipelineProfile] = []

    def measure_stage(
        self,
        stage_name: str,
        func: Callable,
        input_data: Any,
        **kwargs
    ) -> tuple[Any, StageResult]:
        """
        Wraps any function call and measures its latency.

        Usage:
            result, stage = profiler.measure_stage("retrieval", retriever.invoke, query)
        """

        input_size = len(str(input_data))

        start = time.perf_counter()
        try:
            result = func(input_data, **kwargs)
            success = True
            error = ""
        except Exception as e:
            result = None
            success = False
            error = str(e)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        output_size = len(str(result)) if result else 0

        stage_result = StageResult(
            stage_name=stage_name,
            latency_ms=latency_ms,
            input_size=input_size,
            output_size=output_size,
            success=success,
            error=error
        )

        return result, stage_result

    @traceable
    def profile_query(self, query: str, pipeline_func: Callable) -> PipelineProfile:
        """
        Profiles a complete query through your pipeline.
        pipeline_func receives the query and the profiler,
        and is responsible for calling measure_stage for each step.
        """
        profile = PipelineProfile(query=query)
        pipeline_func(query, profile)
        self.profiles.append(profile)
        return profile

    def get_percentiles(self, stage_name: str = None) -> Dict:
        """
        Calculates P50, P95, P99 across all profiled queries.
        If stage_name provided, calculates for that stage only.
        Otherwise calculates for total latency.
        """

        if not self.profiles:
            return {}

        if stage_name:
            latencies = []
            for p in self.profiles:
                for s in p.stages:
                    if s.stage_name == stage_name and s.success:
                        latencies.append(s.latency_ms)
        else:
            latencies = [p.total_latency_ms for p in self.profiles]

        if not latencies:
            return {}

        return {
            "count": len(latencies),
            "mean_ms": round(np.mean(latencies), 2),
            "p50_ms": round(np.percentile(latencies, 50), 2),
            "p95_ms": round(np.percentile(latencies, 95), 2),
            "p99_ms": round(np.percentile(latencies, 99), 2),
            "min_ms": round(np.min(latencies), 2),
            "max_ms": round(np.max(latencies), 2),
        }

    def print_percentile_report(self):
        """
        Prints the full percentile report across all stages.
        This is the report you use to find your bottleneck.
        """

        if not self.profiles:
            print("No profiles recorded yet.")
            return

        # Get all unique stage names
        stage_names = []
        for p in self.profiles:
            for s in p.stages:
                if s.stage_name not in stage_names:
                    stage_names.append(s.stage_name)

        print(f"\n{'━'*65}")
        print(f"📊 LATENCY PERCENTILE REPORT — {len(self.profiles)} queries")
        print(f"{'━'*65}")
        print(f"{'Stage':<22} {'P50':>8} {'P95':>8} {'P99':>8} {'Mean':>8} {'Max':>8}")
        print(f"{'─'*65}")

        for stage in stage_names:
            p = self.get_percentiles(stage)
            if p:
                print(
                    f"{stage:<22} "
                    f"{p['p50_ms']:>7.0f}ms "
                    f"{p['p95_ms']:>7.0f}ms "
                    f"{p['p99_ms']:>7.0f}ms "
                    f"{p['mean_ms']:>7.0f}ms "
                    f"{p['max_ms']:>7.0f}ms"
                )

        total = self.get_percentiles()
        print(f"{'─'*65}")
        print(
            f"{'TOTAL':<22} "
            f"{total['p50_ms']:>7.0f}ms "
            f"{total['p95_ms']:>7.0f}ms "
            f"{total['p99_ms']:>7.0f}ms "
            f"{total['mean_ms']:>7.0f}ms "
            f"{total['max_ms']:>7.0f}ms"
        )
        print(f"{'━'*65}")

        # Identify the bottleneck
        stage_p99s = {}
        for stage in stage_names:
            p = self.get_percentiles(stage)
            if p:
                stage_p99s[stage] = p['p99_ms']

        if stage_p99s:
            bottleneck = max(stage_p99s, key=stage_p99s.get)
            print(f"\n🔴 Bottleneck at P99: [{bottleneck}] → {stage_p99s[bottleneck]:.0f}ms")
            print(f"   This is the stage to optimize first.\n")

    def save_report(self, filename: str = "latency_report.csv"):
        """Saves all stage measurements to CSV for further analysis"""
        rows = []
        for profile in self.profiles:
            for stage in profile.stages:
                rows.append({
                    "query": profile.query[:50],
                    "stage": stage.stage_name,
                    "latency_ms": stage.latency_ms,
                    "total_latency_ms": profile.total_latency_ms,
                    "timestamp": profile.timestamp
                })

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"📁 Full latency report saved → {filename}")
```

---

### Step 4 — Create the example pipeline

Right-click `day-03-latency` → **New File** → name it:

```
run_profiler.py
```

Paste this:

```python
# run_profiler.py
# Shows the profiler working on a realistic AI pipeline
# Replace each stage with your real code

from dotenv import load_dotenv
load_dotenv()

import time
import random
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pipeline_profiler import PipelineProfiler

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
profiler = PipelineProfiler(pipeline_name="my-ai-pipeline")


# ─────────────────────────────────────────────
# DEFINE YOUR PIPELINE STAGES
# Replace these with your real functions
# ─────────────────────────────────────────────

def preprocess_query(query: str) -> str:
    """Stage 1 — Clean and validate the query"""
    # Replace with your real preprocessing
    time.sleep(random.uniform(0.01, 0.05))  # Simulates real work
    return query.strip().lower()


def retrieve_context(query: str) -> str:
    """Stage 2 — Retrieve relevant context"""
    # Replace with your real retriever
    # e.g. your_retriever.invoke(query)
    time.sleep(random.uniform(0.2, 0.8))  # Simulates vector search
    return f"[Retrieved context for: {query}]"


def generate_response(query_with_context: str) -> str:
    """Stage 3 — Generate response from LLM"""
    # Replace with your real LLM call
    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant. Be concise."),
        HumanMessage(content=query_with_context)
    ])
    return response.content


def postprocess_response(response: str) -> str:
    """Stage 4 — Format and safety check"""
    # Replace with your real post-processing
    time.sleep(random.uniform(0.01, 0.03))
    return response.strip()


# ─────────────────────────────────────────────
# THE PROFILED PIPELINE
# This is the function that runs all 4 stages
# and measures each one
# ─────────────────────────────────────────────

def my_pipeline(query: str, profile) -> str:
    """
    Your complete AI pipeline — profiled at every stage.

    To adapt this to your project:
    1. Replace each function call with your real function
    2. Keep the measure_stage wrapper around each call
    3. Keep the stage names descriptive
    """

    # Stage 1 — Pre-processing
    clean_query, stage1 = profiler.measure_stage(
        "1_preprocessing",
        preprocess_query,
        query
    )
    profile.add_stage(stage1)

    # Stage 2 — Retrieval
    context, stage2 = profiler.measure_stage(
        "2_retrieval",
        retrieve_context,
        clean_query
    )
    profile.add_stage(stage2)

    # Stage 3 — LLM Generation
    # Combine query and context as your LLM would receive them
    llm_input = f"Question: {clean_query}\n\nContext: {context}"
    response, stage3 = profiler.measure_stage(
        "3_llm_generation",
        generate_response,
        llm_input
    )
    profile.add_stage(stage3)

    # Stage 4 — Post-processing
    final_response, stage4 = profiler.measure_stage(
        "4_postprocessing",
        postprocess_response,
        response
    )
    profile.add_stage(stage4)

    return final_response


# ─────────────────────────────────────────────
# RUN THE PROFILER
# ─────────────────────────────────────────────

TEST_QUERIES = [
    "What is Apache Kafka used for?",
    "Explain the difference between a topic and a partition",
    "How does consumer group rebalancing work?",
    "What happens when a Kafka broker goes down?",
    "How do I configure retention policy in Kafka?",
    "What is the role of ZooKeeper in Kafka?",
    "Explain exactly-once semantics in Kafka",
    "How does Kafka handle backpressure?",
    "What is a Kafka offset and why does it matter?",
    "Compare Kafka with RabbitMQ"
]

if __name__ == "__main__":
    print("⚡ Profiling your pipeline across 10 queries...\n")

    for query in TEST_QUERIES:
        profile = profiler.profile_query(query, my_pipeline)

    # Print the single-query breakdown for the last query
    print(profile.summary())

    # Print the full percentile report across all queries
    profiler.print_percentile_report()

    # Save the full report
    profiler.save_report("latency_report.csv")

    print("\n✅ Check LangSmith — all 10 runs are logged with stage breakdown.")
```

Run it:

```bash
python run_profiler.py
```

You should see:

```
⚡ Profiling your pipeline across 10 queries...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ PIPELINE LATENCY BREAKDOWN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Query: compare kafka with rabbitmq
─────────────────────────────────
✓ 1_preprocessing        23ms     2%  
✓ 2_retrieval           612ms    54%  ████████████
✓ 3_llm_generation      489ms    43%  ████████
✓ 4_postprocessing       18ms     1%  
─────────────────────────────────
  TOTAL                1142ms   100%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 LATENCY PERCENTILE REPORT — 10 queries
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage                       P50      P95      P99     Mean      Max
─────────────────────────────────────────────────────────────────
1_preprocessing             31ms     47ms     51ms    33ms     52ms
2_retrieval                498ms    743ms    891ms   521ms    912ms
3_llm_generation           461ms    698ms    821ms   478ms    834ms
4_postprocessing            19ms     28ms     31ms    21ms     33ms
─────────────────────────────────────────────────────────────────
TOTAL                      991ms   1412ms   1621ms  1053ms   1701ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔴 Bottleneck at P99: [2_retrieval] → 891ms
   This is the stage to optimize first.
```

Now you know exactly where the time is going.

Not a guess. Not an average. A breakdown.

---

### Step 5 — Add Streaming For Better TTFT

If your system has a user interface, streaming dramatically improves perceived speed.

Right-click `day-03-latency` → **New File** → name it:

```
streaming_example.py
```

Paste this:

```python
# streaming_example.py
# Shows the difference between streaming and non-streaming
# and how to measure TTFT properly

from dotenv import load_dotenv
load_dotenv()

import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_streaming = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)


def measure_without_streaming(query: str) -> dict:
    """Standard call — user waits for complete response"""
    start = time.perf_counter()
    response = llm.invoke([HumanMessage(content=query)])
    end = time.perf_counter()

    return {
        "total_latency_ms": round((end - start) * 1000, 2),
        "ttft_ms": round((end - start) * 1000, 2),  # TTFT = total (no streaming)
        "response": response.content
    }


def measure_with_streaming(query: str) -> dict:
    """
    Streaming call — user sees first token immediately.
    TTFT is measured at the first chunk received.
    """
    start = time.perf_counter()
    first_token_time = None
    full_response = ""

    for chunk in llm_streaming.stream([HumanMessage(content=query)]):
        if first_token_time is None and chunk.content:
            first_token_time = time.perf_counter()
        full_response += chunk.content

    end = time.perf_counter()

    return {
        "total_latency_ms": round((end - start) * 1000, 2),
        "ttft_ms": round((first_token_time - start) * 1000, 2) if first_token_time else 0,
        "response": full_response
    }


if __name__ == "__main__":
    query = "Explain how Kafka handles fault tolerance in 3 sentences."

    print("📊 Comparing streaming vs non-streaming latency...\n")

    # Without streaming
    result_no_stream = measure_without_streaming(query)

    # With streaming
    result_stream = measure_with_streaming(query)

    print(f"{'─'*50}")
    print(f"{'Metric':<25} {'No Streaming':>12} {'Streaming':>12}")
    print(f"{'─'*50}")
    print(f"{'Time To First Token':<25} {result_no_stream['ttft_ms']:>10.0f}ms {result_stream['ttft_ms']:>10.0f}ms")
    print(f"{'Total Latency':<25} {result_no_stream['total_latency_ms']:>10.0f}ms {result_stream['total_latency_ms']:>10.0f}ms")
    print(f"{'─'*50}")

    ttft_improvement = result_no_stream['ttft_ms'] - result_stream['ttft_ms']
    print(f"\n✅ Streaming reduces perceived wait by ~{ttft_improvement:.0f}ms")
    print(f"   The total time is similar — but the user experience is completely different.")
```

Run it:

```bash
python streaming_example.py
```

You should see:

```
📊 Comparing streaming vs non-streaming latency...

──────────────────────────────────────────────────
Metric                    No Streaming    Streaming
──────────────────────────────────────────────────
Time To First Token               891ms        187ms
Total Latency                     923ms        941ms
──────────────────────────────────────────────────

✅ Streaming reduces perceived wait by ~704ms
   The total time is similar — but the user experience is completely different.
```

Same total time. TTFT drops from 891ms to 187ms.

That is the difference between a system that feels instant and one that feels frozen.

---

### Step 6 — Plug Into Your Existing Project

If you already have a pipeline, three changes is all it takes.

**Change 1** — Import the profiler at the top of your main file:

```python
from pipeline_profiler import PipelineProfiler
profiler = PipelineProfiler(pipeline_name="your-project-name")
```

**Change 2** — Wrap each stage of your existing pipeline:

```python
# Before — no visibility
docs = your_retriever.invoke(query)
response = your_llm_chain.invoke({"query": query, "context": docs})

# After — fully profiled
docs, stage_retrieval = profiler.measure_stage(
    "retrieval",
    your_retriever.invoke,
    query
)
profile.add_stage(stage_retrieval)

response, stage_llm = profiler.measure_stage(
    "llm_generation",
    your_llm_chain.invoke,
    {"query": query, "context": docs}
)
profile.add_stage(stage_llm)
```

**Change 3** — After running 10+ queries, print your report:

```python
profiler.print_percentile_report()
```

You now know exactly which stage to optimize.

⚠️ **The most common surprise:** Most engineers assume the LLM is the bottleneck. In RAG systems it usually isn't. Retrieval — especially unoptimized vector search on large document stores — is almost always the P99 bottleneck. Profile before you optimize.

---

## What You Just Built

You now have a pipeline profiler that:

- Measures every stage of your AI pipeline individually
- Calculates P50, P95, P99 across any number of queries
- Identifies your bottleneck automatically
- Logs everything to LangSmith for visual inspection
- Measures TTFT separately from total latency

From this point forward you will never guess where your latency is coming from.

The report tells you. The first time, every time.

---

## ✅ Day 3 Checklist

- [ ] `pipeline_profiler.py` created and imports without errors
- [ ] `run_profiler.py` runs and prints the percentile report
- [ ] You can identify the bottleneck stage from the report output
- [ ] `streaming_example.py` runs and shows the TTFT difference
- [ ] Your pipeline stages are replaced with your real functions
- [ ] The latency report CSV is saved and you have reviewed it
- [ ] All runs are visible in LangSmith with stage-level breakdown

---

## 🎯 Interview Bits — Day 3

**Q: Why are average latencies misleading for AI systems?**
*Averages mask the tail experience. A system where 99 queries take 0.5 seconds and one takes 50 seconds has a perfectly acceptable average but a completely unacceptable worst case. P99 latency reveals what your slowest users actually experience.*

**Q: What is P99 latency and why does it matter in production?**
*P99 is the latency threshold that 99% of queries complete within. At 10,000 daily queries, 1% means 100 users hitting your worst case every day. P99 is the metric that exposes whether your system is production-ready or just demo-ready.*

**Q: What is Time To First Token and when does it matter?**
*TTFT measures how long until the first word of a response appears. Humans perceive a system that starts streaming at 200ms and finishes at 4 seconds as faster than one that delivers a complete response at 2 seconds. For any user-facing interface TTFT is as important as total latency.*

**Q: In a RAG pipeline, where is the latency bottleneck most commonly found?**
*Most engineers assume the LLM call is the bottleneck. In practice, retrieval — especially unoptimized vector search on large document stores — is the most common P99 bottleneck. Stage-level profiling reveals this; total latency measurement hides it.*

**Q: How would you approach reducing P99 latency in a production RAG system?**
*First profile each stage to identify where P99 latency lives. Then target that specific stage — for retrieval this usually means hybrid search, approximate nearest neighbor indexing, or reducing chunk count. For LLM generation it means streaming, caching repeated queries, or routing simple queries to smaller faster models.*

---

*Tomorrow we answer the question latency cannot answer.*
*Fast is great. But fast and wrong is worse than slow and right.*
*Day 4 is about quality — and the metrics that tell you whether your AI is actually telling the truth.*

---
