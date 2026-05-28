Here we go.

---

# AI Performance Benchmarking: From Zero to Production
### Day 11 of 12 — Observability: Watching Your AI System While It's Actually Running

---

*This is Day 11 of a 12-part series. Start with **[Day 0](#)** if you're just joining.*

---

## The Test That Passed. The Production System That Didn't.

Every benchmark passed.

Your regression gates are green on every PR.

Quality scores strong. Latency under threshold. Cost within budget. Safety gates solid.

You are confident.

Three weeks into production something starts feeling off.

Users are not complaining loudly. But engagement is dropping. Queries are getting shorter. Some users stopped coming back.

You pull up your benchmark results.

All green.

You run the regression check manually.

All green.

So what is happening?

---

You dig into LangSmith traces from the past week.

And you see it.

A specific category of queries — complex multi-document questions — has been degrading for 11 days.

Not across the board. Just that one category.

Your eval dataset did not have enough examples of that query type to catch it.

Your benchmarks were measuring the right things on the wrong distribution.

Real users were asking different questions than your eval set anticipated.

And because nobody was watching the live system — nobody saw it happening.

---

**This is the gap that observability fills.**

Benchmarking is what you measure before deployment.

Observability is what you watch after.

And the two together — offline benchmarks plus live monitoring — is the complete picture.

---

## The Three Pillars Of AI Observability

---

**Pillar 1 — Tracing**

Every query your system handles in production gets logged as a trace.

Input. Retrieved context. LLM response. Latency. Cost. Every intermediate step.

LangSmith is your tracing layer. You set it up in Day 0. By now it has weeks of production traces sitting in it.

Today we learn to read them systematically — not just look at individual traces but analyze patterns across thousands of them.

---

**Pillar 2 — Metrics**

Aggregated numerical signals about system health — tracked over time.

Not individual traces. Rolling averages. Percentile distributions. Error rates. Volume trends.

Prometheus collects these metrics. You query them to understand how the system is behaving right now versus yesterday versus last week.

---

**Pillar 3 — Drift Detection**

The most subtle and the most dangerous.

Drift is when the distribution of what users are asking changes — and your system's ability to handle the new distribution degrades.

Your eval dataset captured the query distribution at the time you built it.

Production queries drift over time. New topics emerge. New phrasings appear. Edge cases that were rare become common.

Evidently AI detects this drift before it becomes a user-facing problem.

---

## Benchmarking vs Monitoring vs Observability

Before we write a single line let us nail this distinction. It comes up in interviews constantly.

```
BENCHMARKING
─────────────────────────────────────────────────────
Controlled environment. Fixed dataset. Pre-deployment.
"How does the system perform under known conditions?"
Tools: RAGAS, custom eval harnesses, Locust
When: before every merge, before every deployment
─────────────────────────────────────────────────────

MONITORING
─────────────────────────────────────────────────────
Live system. Real traffic. Post-deployment.
"Is the system healthy right now?"
Tools: Prometheus, Grafana, CloudWatch
When: continuously, 24/7
─────────────────────────────────────────────────────

OBSERVABILITY
─────────────────────────────────────────────────────
The full picture. Structured data that lets you ask
any question about system behavior — including
questions you did not anticipate when you built it.
Tools: LangSmith, Evidently AI, distributed tracing
When: continuously, with periodic deep analysis
─────────────────────────────────────────────────────
```

Monitoring tells you something is wrong.

Observability tells you why.

Benchmarking told you it would work.

You need all three.

---

## The Four Things You Must Watch In Production

---

**1. Quality drift**

Are faithfulness and context recall scores degrading on real production queries?

Not on your eval set — on the actual queries users are sending right now.

This requires sampling production traces, running RAGAS on them, and tracking the scores over time.

---

**2. Latency drift**

Is P99 latency creeping up week over week?

A 10% weekly increase is invisible day to day. Compounded over a month it doubles your worst-case response time.

---

**3. Query distribution drift**

Are users asking different types of questions than your eval set anticipated?

If the distribution shifts — new topics, new phrasings, new complexity levels — your offline benchmarks stop being predictive of production performance.

---

**4. Error rate trends**

Are retrieval failures, LLM errors, or tool call failures increasing?

A 0.1% error rate that doubles weekly becomes a 3% error rate in a month. Catch it early.

---

## The Problem Most People Hit

You want to monitor production quality.

But RAGAS requires ground truth answers — and production queries do not come with ground truth.

Nobody labeled the right answer for every query a real user sent.

Two solutions:

**Solution 1 — Sample and manually label**

Take a random sample of production queries each week. Have a domain expert label the correct answers. Run RAGAS on that labeled sample.

Slow. Expensive. But the gold standard.

**Solution 2 — Reference-free metrics**

Some quality dimensions can be measured without ground truth.

Faithfulness — does the answer stick to the retrieved context? No ground truth needed.
Answer relevancy — does the answer address the question? No ground truth needed.
Context precision — was the retrieved context relevant? No ground truth needed.

Only context recall requires ground truth.

Today we implement reference-free production monitoring — the approach that scales.

---

## The Code — Observability Suite

Today we build four things:

1. A **production trace analyzer** — reads LangSmith traces and measures quality on real queries
2. A **Prometheus metrics exporter** — exposes AI metrics for time-series monitoring
3. An **Evidently drift detector** — detects query distribution and quality drift
4. A **monitoring dashboard** — a lightweight unified view

---

### Step 1 — Set up today's folder

```bash
cd ..
mkdir day-11-observability
cd day-11-observability
```

---

### Step 2 — Install dependencies

```bash
pip install langsmith langchain langchain-openai python-dotenv ragas datasets pandas numpy evidently prometheus-client fastapi uvicorn
```

⚠️ **Evidently may take a minute to install — it has several dependencies. Wait for the success message.**

---

### Step 3 — Create the production trace analyzer

Right-click `day-11-observability` → **New File** → name it:

```
trace_analyzer.py
```

Paste this exactly:

```python
# trace_analyzer.py
# Reads production traces from LangSmith
# Runs quality metrics on real user queries
# No ground truth required — uses reference-free metrics

from dotenv import load_dotenv
load_dotenv()

import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langsmith import traceable
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset


# ─────────────────────────────────────────────
# PRODUCTION SAMPLE
# One query from production traces
# ─────────────────────────────────────────────

@dataclass
class ProductionSample:
    trace_id: str
    query: str
    response: str
    retrieved_contexts: List[str]
    latency_ms: float
    timestamp: str
    faithfulness_score: Optional[float] = None
    answer_relevancy_score: Optional[float] = None
    quality_flag: str = "unknown"  # good / warning / critical


# ─────────────────────────────────────────────
# TRACE ANALYZER
# ─────────────────────────────────────────────

class ProductionTraceAnalyzer:

    def __init__(self, project_name: str = "ai-benchmarking-series"):
        self.client = Client()
        self.project_name = project_name
        self.samples: List[ProductionSample] = []

    def fetch_recent_traces(
        self,
        hours: int = 24,
        limit: int = 50
    ) -> List[Dict]:
        """
        Fetches recent traces from LangSmith.

        Parameters:
        - hours: how far back to look
        - limit: maximum traces to fetch
          (keep low — each one gets quality scored)

        Returns raw trace data for processing.
        """
        print(f"\n  Fetching traces from last {hours}h "
              f"(max {limit})...")

        start_time = datetime.utcnow() - timedelta(hours=hours)

        try:
            runs = list(self.client.list_runs(
                project_name=self.project_name,
                start_time=start_time,
                limit=limit,
                run_type="chain"
            ))

            print(f"  Found {len(runs)} traces")
            return runs

        except Exception as e:
            print(f"  ⚠️  Could not fetch from LangSmith: {e}")
            print(f"  Using simulated traces for demonstration")
            return self._get_simulated_traces()

    def _get_simulated_traces(self) -> List[Dict]:
        """
        Simulates production traces for demonstration.
        Replace this with real LangSmith data in production.
        """

        class SimulatedRun:
            def __init__(self, i):
                import random
                query_types = [
                    ("What is Apache Kafka?",
                     "Apache Kafka is a distributed streaming platform."),
                    ("How do partitions work?",
                     "Partitions are ordered message sequences."),
                    ("Explain consumer groups",
                     "Consumer groups enable parallel processing."),
                    ("What is a Kafka offset?",
                     "An offset identifies a message position."),
                    ("How does Kafka scale?",
                     "Kafka scales by adding brokers and partitions.")
                ]
                q, a = random.choice(query_types)
                self.id = f"sim-trace-{i:04d}"
                self.inputs = {"input": q}
                self.outputs = {"output": a}
                self.extra = {
                    "metadata": {
                        "contexts": [
                            "Kafka is a distributed event streaming platform",
                            "Topics are divided into partitions",
                            "Consumer groups allow parallel consumption"
                        ]
                    }
                }
                self.end_time = datetime.utcnow()
                self.start_time = (
                    self.end_time -
                    timedelta(
                        milliseconds=random.uniform(500, 3000)
                    )
                )

        return [SimulatedRun(i) for i in range(20)]

    def extract_samples(self, runs: List) -> List[ProductionSample]:
        """
        Extracts structured samples from raw LangSmith traces.
        Adapts to your trace structure — modify field names
        to match how your pipeline logs data.
        """
        samples = []

        for run in runs:
            try:
                # Extract query
                inputs = run.inputs or {}
                query = (
                    inputs.get("input") or
                    inputs.get("query") or
                    inputs.get("question") or
                    str(inputs)[:200]
                )

                # Extract response
                outputs = run.outputs or {}
                response = (
                    outputs.get("output") or
                    outputs.get("answer") or
                    outputs.get("response") or
                    str(outputs)[:200]
                )

                # Extract contexts
                # Adapt this to match how your pipeline
                # stores retrieved context in traces
                metadata = (run.extra or {}).get("metadata", {})
                contexts = metadata.get("contexts", [
                    "No context available in trace"
                ])

                # Calculate latency
                if run.end_time and run.start_time:
                    latency_ms = (
                        run.end_time - run.start_time
                    ).total_seconds() * 1000
                else:
                    latency_ms = 0.0

                sample = ProductionSample(
                    trace_id=str(run.id),
                    query=str(query),
                    response=str(response),
                    retrieved_contexts=(
                        contexts if isinstance(contexts, list)
                        else [str(contexts)]
                    ),
                    latency_ms=latency_ms,
                    timestamp=str(
                        run.end_time or datetime.utcnow()
                    )
                )
                samples.append(sample)

            except Exception as e:
                print(f"  ⚠️  Skipped trace {run.id}: {e}")
                continue

        self.samples = samples
        print(f"  Extracted {len(samples)} valid samples")
        return samples

    @traceable
    def score_production_quality(
        self,
        samples: List[ProductionSample],
        batch_size: int = 10
    ) -> List[ProductionSample]:
        """
        Runs reference-free quality metrics on production samples.

        Uses only faithfulness and answer_relevancy —
        these do not require ground truth answers.

        Processes in batches to manage API costs.
        """
        print(f"\n  Scoring quality on {len(samples)} "
              f"production samples...")
        print(f"  (Reference-free — no ground truth required)")

        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            print(
                f"  Batch {i//batch_size + 1}/"
                f"{len(samples)//batch_size + 1}..."
            )

            try:
                ragas_data = Dataset.from_dict({
                    "question": [s.query for s in batch],
                    "answer": [s.response for s in batch],
                    "contexts": [
                        s.retrieved_contexts for s in batch
                    ]
                })

                scores = evaluate(
                    dataset=ragas_data,
                    metrics=[faithfulness, answer_relevancy]
                )
                scores_df = scores.to_pandas()

                for j, sample in enumerate(batch):
                    sample.faithfulness_score = float(
                        scores_df["faithfulness"].iloc[j]
                    )
                    sample.answer_relevancy_score = float(
                        scores_df["answer_relevancy"].iloc[j]
                    )

                    # Flag quality issues
                    f_score = sample.faithfulness_score
                    r_score = sample.answer_relevancy_score

                    if f_score >= 0.85 and r_score >= 0.80:
                        sample.quality_flag = "🟢 good"
                    elif f_score >= 0.65 and r_score >= 0.65:
                        sample.quality_flag = "🟡 warning"
                    else:
                        sample.quality_flag = "🔴 critical"

            except Exception as e:
                print(f"  ⚠️  Batch scoring failed: {e}")
                for sample in batch:
                    sample.quality_flag = "⬜ unscored"

        return samples

    def print_quality_summary(self):
        """Prints aggregate quality across all scored samples"""

        scored = [
            s for s in self.samples
            if s.faithfulness_score is not None
        ]

        if not scored:
            print("No scored samples available.")
            return

        faithfulness_scores = [
            s.faithfulness_score for s in scored
        ]
        relevancy_scores = [
            s.answer_relevancy_score for s in scored
            if s.answer_relevancy_score is not None
        ]
        latencies = [s.latency_ms for s in self.samples]

        flags = {
            "🟢 good": sum(
                1 for s in scored if s.quality_flag == "🟢 good"
            ),
            "🟡 warning": sum(
                1 for s in scored if s.quality_flag == "🟡 warning"
            ),
            "🔴 critical": sum(
                1 for s in scored if s.quality_flag == "🔴 critical"
            )
        }

        print(f"\n{'━'*60}")
        print(f"📊 PRODUCTION QUALITY SUMMARY")
        print(f"   {len(scored)} traces analyzed")
        print(f"{'━'*60}")
        print(
            f"  Avg faithfulness:      "
            f"{np.mean(faithfulness_scores):.3f}"
        )
        print(
            f"  Avg answer relevancy:  "
            f"{np.mean(relevancy_scores):.3f}"
            if relevancy_scores else
            f"  Avg answer relevancy:  N/A"
        )
        print(
            f"  Avg latency:           "
            f"{np.mean(latencies):.0f}ms"
        )
        print(
            f"  P99 latency:           "
            f"{np.percentile(latencies, 99):.0f}ms"
        )
        print(f"{'─'*60}")
        print(f"  Quality distribution:")
        for flag, count in flags.items():
            pct = count / len(scored) * 100
            bar = "█" * int(pct / 5)
            print(f"    {flag}     {count:>4} ({pct:>5.1f}%)  {bar}")

        critical = [
            s for s in scored
            if s.quality_flag == "🔴 critical"
        ]
        if critical:
            print(f"\n  ⚠️  Critical quality issues detected:")
            for s in critical[:3]:
                print(
                    f"    Trace: {s.trace_id[:12]}  "
                    f"Faith: {s.faithfulness_score:.2f}  "
                    f"Query: {s.query[:40]}..."
                )

        print(f"{'━'*60}\n")

    def save_analysis(
        self, filename: str = "production_analysis.json"
    ):
        data = [
            {
                "trace_id": s.trace_id,
                "query": s.query[:100],
                "latency_ms": s.latency_ms,
                "faithfulness": s.faithfulness_score,
                "answer_relevancy": s.answer_relevancy_score,
                "quality_flag": s.quality_flag,
                "timestamp": s.timestamp
            }
            for s in self.samples
        ]
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        pd.DataFrame(data).to_csv(
            filename.replace('.json', '.csv'), index=False
        )
        print(f"✅ Production analysis saved → {filename}")
```

---

### Step 4 — Create the Prometheus metrics exporter

Right-click `day-11-observability` → **New File** → name it:

```
metrics_exporter.py
```

Paste this:

```python
# metrics_exporter.py
# Exposes AI system metrics for Prometheus scraping
# Run this alongside your AI API
# Prometheus scrapes it every 15 seconds

from dotenv import load_dotenv
load_dotenv()

import time
import random
import threading
from datetime import datetime
from typing import Optional
from prometheus_client import (
    start_http_server,
    Histogram,
    Counter,
    Gauge,
    Summary
)


# ─────────────────────────────────────────────
# METRIC DEFINITIONS
# These are the AI-specific metrics Prometheus tracks
# ─────────────────────────────────────────────

# Latency histogram — tracks distribution not just average
REQUEST_LATENCY = Histogram(
    'ai_request_latency_seconds',
    'AI pipeline request latency in seconds',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    labelnames=['endpoint', 'model']
)

# Quality gauges — updated from periodic RAGAS sampling
FAITHFULNESS_SCORE = Gauge(
    'ai_faithfulness_score',
    'Rolling average faithfulness score from RAGAS evaluation'
)

ANSWER_RELEVANCY_SCORE = Gauge(
    'ai_answer_relevancy_score',
    'Rolling average answer relevancy score'
)

CONTEXT_RECALL_SCORE = Gauge(
    'ai_context_recall_score',
    'Rolling average context recall score'
)

# Cost tracking
COST_PER_QUERY = Gauge(
    'ai_cost_per_query_usd',
    'Average cost per query in USD'
)

TOTAL_TOKENS = Counter(
    'ai_total_tokens_used',
    'Total tokens consumed',
    labelnames=['token_type']  # input / output
)

# Error tracking
ERROR_COUNTER = Counter(
    'ai_errors_total',
    'Total AI pipeline errors',
    labelnames=['error_type']
)

# Request volume
REQUEST_COUNTER = Counter(
    'ai_requests_total',
    'Total requests processed',
    labelnames=['query_type', 'model']
)

# Active requests
ACTIVE_REQUESTS = Gauge(
    'ai_active_requests',
    'Number of requests currently being processed'
)

# Quality drift alert
QUALITY_DRIFT_ALERT = Gauge(
    'ai_quality_drift_alert',
    '1 if quality drift detected, 0 if normal'
)


# ─────────────────────────────────────────────
# METRICS RECORDER
# Call these functions from your AI pipeline
# ─────────────────────────────────────────────

class AIMetricsRecorder:
    """
    Drop this into your AI pipeline to record metrics.

    Usage in your FastAPI endpoint:
        recorder = AIMetricsRecorder()

        @app.post("/query")
        async def query(request: QueryRequest):
            with recorder.track_request("query", "gpt-4o-mini"):
                response = your_pipeline(request.query)
            return response
    """

    def __init__(self):
        self._quality_buffer = []
        self._buffer_size = 50  # Score after every 50 queries

    def record_request(
        self,
        latency_seconds: float,
        endpoint: str = "query",
        model: str = "gpt-4o-mini",
        query_type: str = "general",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        error: Optional[str] = None
    ):
        """Record metrics for one completed request"""

        REQUEST_LATENCY.labels(
            endpoint=endpoint, model=model
        ).observe(latency_seconds)

        REQUEST_COUNTER.labels(
            query_type=query_type, model=model
        ).inc()

        if input_tokens > 0:
            TOTAL_TOKENS.labels(token_type="input").inc(input_tokens)
        if output_tokens > 0:
            TOTAL_TOKENS.labels(token_type="output").inc(output_tokens)
        if cost_usd > 0:
            COST_PER_QUERY.set(cost_usd)
        if error:
            ERROR_COUNTER.labels(error_type=error).inc()

    def update_quality_scores(
        self,
        faithfulness: float,
        answer_relevancy: float,
        context_recall: Optional[float] = None
    ):
        """
        Update quality metric gauges.
        Call this after running periodic RAGAS sampling
        on production traces.
        """
        FAITHFULNESS_SCORE.set(faithfulness)
        ANSWER_RELEVANCY_SCORE.set(answer_relevancy)

        if context_recall is not None:
            CONTEXT_RECALL_SCORE.set(context_recall)

        # Set drift alert if quality drops below threshold
        if faithfulness < 0.75 or answer_relevancy < 0.70:
            QUALITY_DRIFT_ALERT.set(1)
            print(
                f"  ⚠️  Quality drift alert triggered: "
                f"faithfulness={faithfulness:.3f}  "
                f"relevancy={answer_relevancy:.3f}"
            )
        else:
            QUALITY_DRIFT_ALERT.set(0)

    def track_active_request(self, increment: bool = True):
        """Track concurrent request count"""
        if increment:
            ACTIVE_REQUESTS.inc()
        else:
            ACTIVE_REQUESTS.dec()


# ─────────────────────────────────────────────
# METRICS SERVER
# Starts the Prometheus scrape endpoint
# ─────────────────────────────────────────────

def start_metrics_server(port: int = 8001):
    """
    Starts the Prometheus metrics HTTP server.
    Prometheus scrapes http://localhost:8001/metrics

    Run this at startup alongside your main API.
    """
    start_http_server(port)
    print(f"✅ Metrics server started on port {port}")
    print(f"   Prometheus scrapes: http://localhost:{port}/metrics")
    print(f"   Add this to your prometheus.yml:")
    print(f"   - job_name: 'ai-system'")
    print(f"     static_configs:")
    print(f"     - targets: ['localhost:{port}']")


def simulate_production_metrics(recorder: AIMetricsRecorder):
    """
    Simulates production traffic for demonstration.
    Remove this in production — use your real pipeline.
    """
    print("\n  Simulating production metrics...")
    print("  (Replace with real pipeline instrumentation)")

    query_types = ["simple", "moderate", "complex"]
    models = ["gpt-4o-mini", "gpt-4o-mini", "gpt-4o"]

    for i in range(100):
        query_type = random.choice(query_types)
        model = random.choice(models)

        # Simulate realistic latency distribution
        base_latency = {
            "simple": 0.8, "moderate": 1.4, "complex": 3.2
        }[query_type]
        latency = max(
            0.1, random.gauss(base_latency, base_latency * 0.3)
        )

        # Simulate cost
        cost = random.uniform(0.00002, 0.0002)

        # Simulate occasional errors
        error = None
        if random.random() < 0.02:
            error = random.choice([
                "rate_limit", "timeout", "retrieval_error"
            ])

        recorder.record_request(
            latency_seconds=latency,
            endpoint="query",
            model=model,
            query_type=query_type,
            input_tokens=random.randint(200, 800),
            output_tokens=random.randint(50, 300),
            cost_usd=cost,
            error=error
        )

    # Update quality scores
    recorder.update_quality_scores(
        faithfulness=random.uniform(0.78, 0.92),
        answer_relevancy=random.uniform(0.75, 0.89),
        context_recall=random.uniform(0.70, 0.85)
    )

    print(f"  ✅ Simulated 100 requests")
    print(
        f"  View metrics at: http://localhost:8001/metrics"
    )


if __name__ == "__main__":
    recorder = AIMetricsRecorder()
    start_metrics_server(port=8001)
    simulate_production_metrics(recorder)

    print("\n  Keeping metrics server alive for 60 seconds...")
    print("  Open http://localhost:8001/metrics to see raw metrics")
    time.sleep(60)
```

---

### Step 5 — Create the drift detector

Right-click `day-11-observability` → **New File** → name it:

```
drift_detector.py
```

Paste this:

```python
# drift_detector.py
# Detects when production query distribution drifts
# from your training/eval distribution
# Uses Evidently AI for statistical drift detection

from dotenv import load_dotenv
load_dotenv()

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import (
    DatasetDriftMetric,
    ColumnDriftMetric
)


# ─────────────────────────────────────────────
# QUERY FEATURE EXTRACTOR
# Turns raw queries into measurable features
# for drift detection
# ─────────────────────────────────────────────

def extract_query_features(queries: List[str]) -> pd.DataFrame:
    """
    Extracts numerical features from queries for drift detection.

    These features capture the statistical properties
    of your query distribution — not the content itself.
    When the distribution changes, these numbers change.

    Add more features specific to your domain.
    """
    features = []

    for query in queries:
        words = query.lower().split()
        sentences = query.split('.')

        feature = {
            # Length features
            "query_length": len(query),
            "word_count": len(words),
            "avg_word_length": (
                np.mean([len(w) for w in words])
                if words else 0
            ),

            # Complexity signals
            "question_words": sum(
                1 for w in words
                if w in ["what", "how", "why", "when",
                         "where", "who", "which"]
            ),
            "comparison_words": sum(
                1 for w in words
                if w in ["compare", "difference", "versus",
                         "vs", "better", "worse", "between"]
            ),
            "analysis_words": sum(
                1 for w in words
                if w in ["analyze", "evaluate", "explain",
                         "describe", "summarize", "assess"]
            ),

            # Sentence structure
            "sentence_count": len(
                [s for s in sentences if s.strip()]
            ),

            # Domain signals — customize for your domain
            "technical_terms": sum(
                1 for w in words
                if w in ["kafka", "partition", "broker",
                         "consumer", "producer", "topic",
                         "offset", "cluster", "replication"]
            ),

            # Query type signals
            "is_definition": int(
                any(w in words
                    for w in ["what", "define", "definition"])
            ),
            "is_howto": int(
                any(w in words
                    for w in ["how", "steps", "process", "way"])
            ),
            "is_comparison": int(
                any(w in words
                    for w in ["compare", "difference",
                               "versus", "vs"])
            )
        }
        features.append(feature)

    return pd.DataFrame(features)


# ─────────────────────────────────────────────
# QUALITY DRIFT TRACKER
# Tracks quality metrics over time windows
# ─────────────────────────────────────────────

@dataclass
class QualityWindow:
    window_label: str       # e.g. "2024-week-01"
    faithfulness_scores: List[float]
    relevancy_scores: List[float]
    latencies: List[float]
    sample_count: int

    def to_dict(self) -> Dict:
        return {
            "window": self.window_label,
            "avg_faithfulness": np.mean(self.faithfulness_scores),
            "avg_relevancy": np.mean(self.relevancy_scores),
            "p99_latency": np.percentile(self.latencies, 99),
            "sample_count": self.sample_count
        }


# ─────────────────────────────────────────────
# DRIFT DETECTOR
# ─────────────────────────────────────────────

class DriftDetector:

    def __init__(self, reference_queries: List[str]):
        """
        Initialize with your reference distribution.

        reference_queries: the queries from your eval dataset
        or a sample of early production queries that represent
        the expected distribution.
        """
        self.reference_df = extract_query_features(
            reference_queries
        )
        self.quality_windows: List[QualityWindow] = []
        print(
            f"  Drift detector initialized with "
            f"{len(reference_queries)} reference queries"
        )

    def detect_query_drift(
        self,
        production_queries: List[str],
        window_label: str = "current"
    ) -> Dict:
        """
        Detects statistical drift between reference and
        production query distributions.

        Returns drift report with per-feature analysis.
        """
        print(
            f"\n  Running drift detection on "
            f"{len(production_queries)} production queries..."
        )

        production_df = extract_query_features(production_queries)

        # Build Evidently drift report
        report = Report(metrics=[
            DatasetDriftMetric(),
            ColumnDriftMetric(column_name="query_length"),
            ColumnDriftMetric(column_name="word_count"),
            ColumnDriftMetric(column_name="analysis_words"),
            ColumnDriftMetric(column_name="technical_terms")
        ])

        report.run(
            reference_data=self.reference_df,
            current_data=production_df
        )

        report_dict = report.as_dict()

        # Extract key findings
        dataset_drift = report_dict["metrics"][0]["result"]
        drift_detected = dataset_drift.get(
            "dataset_drift", False
        )
        drift_share = dataset_drift.get("share_of_drifted_columns", 0)

        # Extract per-column drift
        column_drifts = {}
        for metric in report_dict["metrics"][1:]:
            col = metric["result"].get("column_name", "unknown")
            drifted = metric["result"].get("drift_detected", False)
            score = metric["result"].get("drift_score", 0)
            column_drifts[col] = {
                "drifted": drifted,
                "score": score
            }

        result = {
            "window": window_label,
            "drift_detected": drift_detected,
            "drift_share": drift_share,
            "column_drifts": column_drifts,
            "reference_size": len(self.reference_df),
            "production_size": len(production_df),
            "timestamp": datetime.now().isoformat()
        }

        self._print_drift_report(result)

        # Save HTML report
        report_path = f"drift_report_{window_label}.html"
        report.save_html(report_path)
        print(f"  📄 Full drift report saved → {report_path}")

        return result

    def track_quality_window(
        self,
        window: QualityWindow
    ):
        """
        Adds a quality measurement window to the tracker.
        Call this weekly with production RAGAS scores.
        """
        self.quality_windows.append(window)

    def detect_quality_drift(
        self,
        drift_threshold: float = 0.05
    ) -> Dict:
        """
        Detects quality score drift across time windows.
        Alerts when a metric drops more than threshold
        from the first window (baseline).
        """
        if len(self.quality_windows) < 2:
            return {"message": "Need at least 2 windows to detect drift"}

        baseline = self.quality_windows[0]
        latest = self.quality_windows[-1]

        baseline_faith = np.mean(baseline.faithfulness_scores)
        latest_faith = np.mean(latest.faithfulness_scores)
        faith_drop = baseline_faith - latest_faith

        baseline_rel = np.mean(baseline.relevancy_scores)
        latest_rel = np.mean(latest.relevancy_scores)
        rel_drop = baseline_rel - latest_rel

        faith_drift = faith_drop > drift_threshold
        rel_drift = rel_drop > drift_threshold

        result = {
            "faithfulness_drift": faith_drift,
            "faithfulness_drop": round(faith_drop, 3),
            "relevancy_drift": rel_drift,
            "relevancy_drop": round(rel_drop, 3),
            "baseline_window": baseline.window_label,
            "current_window": latest.window_label,
            "alert": faith_drift or rel_drift
        }

        print(f"\n{'━'*55}")
        print(f"📈 QUALITY DRIFT ANALYSIS")
        print(f"   {baseline.window_label} → {latest.window_label}")
        print(f"{'━'*55}")
        print(
            f"  Faithfulness: "
            f"{baseline_faith:.3f} → {latest_faith:.3f}  "
            f"{'⚠️  DRIFT' if faith_drift else '✅ stable'}"
        )
        print(
            f"  Relevancy:    "
            f"{baseline_rel:.3f} → {latest_rel:.3f}  "
            f"{'⚠️  DRIFT' if rel_drift else '✅ stable'}"
        )

        if result["alert"]:
            print(
                f"\n  🔴 Quality drift detected. "
                f"Investigate recent changes."
            )
            print(
                f"     Check: prompt changes, "
                f"retrieval changes, model updates"
            )
        else:
            print(f"\n  🟢 Quality is stable across windows")

        print(f"{'━'*55}\n")

        return result

    def _print_drift_report(self, result: Dict):
        print(f"\n{'━'*55}")
        print(f"🔍 QUERY DISTRIBUTION DRIFT REPORT")
        print(f"{'━'*55}")
        print(
            f"  Drift detected:   "
            f"{'⚠️  YES' if result['drift_detected'] else '✅ NO'}"
        )
        print(
            f"  Drifted features: "
            f"{result['drift_share']*100:.0f}%"
        )
        print(f"\n  Feature breakdown:")

        for col, data in result["column_drifts"].items():
            flag = "⚠️ " if data["drifted"] else "✅"
            print(
                f"    {flag} {col:<25} "
                f"drift score: {data['score']:.3f}"
            )

        if result["drift_detected"]:
            print(
                f"\n  ⚠️  Production queries are drifting from "
                f"your eval distribution."
            )
            print(
                f"     Action: update your eval dataset to "
                f"include new query patterns."
            )
        print(f"{'━'*55}")

    def save_drift_history(
        self, filename: str = "drift_history.json"
    ):
        windows = [w.to_dict() for w in self.quality_windows]
        with open(filename, 'w') as f:
            json.dump(windows, f, indent=2)
        print(f"✅ Drift history saved → {filename}")
```

---

### Step 6 — Create the runner

Right-click `day-11-observability` → **New File** → name it:

```
run_observability.py
```

Paste this:

```python
# run_observability.py
# Runs the complete observability suite
# In production run this as a scheduled job — daily or weekly

from dotenv import load_dotenv
load_dotenv()

import random
import numpy as np
from trace_analyzer import ProductionTraceAnalyzer
from metrics_exporter import (
    AIMetricsRecorder,
    start_metrics_server,
    simulate_production_metrics
)
from drift_detector import DriftDetector, QualityWindow

# ─────────────────────────────────────────────
# REFERENCE QUERIES
# These represent your expected query distribution
# Use your eval dataset queries from Day 2
# ─────────────────────────────────────────────

REFERENCE_QUERIES = [
    "What is Apache Kafka?",
    "How do Kafka partitions work?",
    "What is a consumer group?",
    "Explain Kafka brokers",
    "What is a Kafka offset?",
    "How does Kafka handle fault tolerance?",
    "What delivery guarantees does Kafka support?",
    "How does Kafka achieve high throughput?",
    "What is the role of ZooKeeper in Kafka?",
    "How do producers send messages in Kafka?"
]

# ─────────────────────────────────────────────
# SIMULATED PRODUCTION QUERIES
# In production replace with real queries from LangSmith
# ─────────────────────────────────────────────

PRODUCTION_QUERIES_WEEK1 = REFERENCE_QUERIES + [
    "What is a Kafka topic?",
    "How does Kafka replication work?",
    "What is a Kafka cluster?"
]

# Week 2 queries drift toward more complex analysis
PRODUCTION_QUERIES_WEEK2 = [
    "Analyze the tradeoffs between Kafka and RabbitMQ",
    "Compare Kafka's exactly-once semantics with alternatives",
    "Evaluate Kafka for a real-time fraud detection system",
    "How would you design a Kafka architecture for 1M messages/sec?",
    "What are the limitations of Kafka for small teams?",
    "Compare Kafka Streams with Apache Flink for stream processing",
    "Analyze Kafka's cost at scale for a startup",
    "What monitoring strategy would you recommend for Kafka?",
    "How does Kafka compare to Kinesis for AWS deployments?",
    "Evaluate Kafka for event sourcing in microservices"
]


if __name__ == "__main__":

    print("=" * 60)
    print("OBSERVABILITY SUITE")
    print("=" * 60)

    # ── Part 1: Production Trace Analysis ────
    print("\n" + "─"*60)
    print("PART 1 — PRODUCTION TRACE ANALYSIS")
    print("─"*60)

    analyzer = ProductionTraceAnalyzer(
        project_name="ai-benchmarking-series"
    )

    # Fetch and analyze recent traces
    runs = analyzer.fetch_recent_traces(hours=24, limit=20)
    samples = analyzer.extract_samples(runs)

    if samples:
        scored_samples = analyzer.score_production_quality(
            samples, batch_size=5
        )
        analyzer.print_quality_summary()
        analyzer.save_analysis("production_analysis.json")

    # ── Part 2: Drift Detection ───────────────
    print("\n" + "─"*60)
    print("PART 2 — DRIFT DETECTION")
    print("─"*60)

    detector = DriftDetector(
        reference_queries=REFERENCE_QUERIES
    )

    # Detect query distribution drift
    drift_result_w1 = detector.detect_query_drift(
        production_queries=PRODUCTION_QUERIES_WEEK1,
        window_label="week-1"
    )

    drift_result_w2 = detector.detect_query_drift(
        production_queries=PRODUCTION_QUERIES_WEEK2,
        window_label="week-2"
    )

    # Track quality windows over time
    # In production these scores come from your trace analyzer
    detector.track_quality_window(QualityWindow(
        window_label="week-1",
        faithfulness_scores=[
            random.uniform(0.85, 0.92) for _ in range(20)
        ],
        relevancy_scores=[
            random.uniform(0.82, 0.90) for _ in range(20)
        ],
        latencies=[
            random.uniform(800, 2000) for _ in range(20)
        ],
        sample_count=20
    ))

    detector.track_quality_window(QualityWindow(
        window_label="week-2",
        faithfulness_scores=[
            random.uniform(0.71, 0.79) for _ in range(20)
        ],
        relevancy_scores=[
            random.uniform(0.68, 0.76) for _ in range(20)
        ],
        latencies=[
            random.uniform(1200, 3500) for _ in range(20)
        ],
        sample_count=20
    ))

    quality_drift = detector.detect_quality_drift(
        drift_threshold=0.05
    )
    detector.save_drift_history("drift_history.json")

    # ── Part 3: Prometheus Metrics ────────────
    print("\n" + "─"*60)
    print("PART 3 — PROMETHEUS METRICS")
    print("─"*60)

    recorder = AIMetricsRecorder()
    start_metrics_server(port=8001)
    simulate_production_metrics(recorder)

    print("\n" + "="*60)
    print("OBSERVABILITY SUITE COMPLETE")
    print("="*60)
    print("\n  What you now have:")
    print("  ✅ Production trace quality analysis")
    print("  ✅ Query distribution drift detection")
    print("  ✅ Quality score drift tracking")
    print("  ✅ Prometheus metrics endpoint")
    print("\n  Next steps:")
    print(
        "  1. Schedule run_observability.py as a weekly job"
    )
    print(
        "  2. Set up Prometheus to scrape port 8001"
    )
    print(
        "  3. Build a Grafana dashboard from the metrics"
    )
    print(
        "  4. Set up alerts when drift_detected = True"
    )
    print(
        "\n  All data logged to LangSmith. "
        "Check your project dashboard."
    )
```

Run it:

```bash
python run_observability.py
```

You should see trace analysis, drift detection reports, and the Prometheus metrics server start up — all in one run.

---

## ✅ Day 11 Checklist

- [ ] `trace_analyzer.py` fetches and scores production traces
- [ ] `metrics_exporter.py` starts on port 8001
- [ ] Raw metrics visible at `http://localhost:8001/metrics`
- [ ] `drift_detector.py` runs and generates HTML drift report
- [ ] Quality drift detected between week-1 and week-2 windows
- [ ] `run_observability.py` completes without errors
- [ ] All output files saved: `production_analysis.json`, `drift_history.json`
- [ ] You understand the difference between benchmarking, monitoring, and observability

---

## 🎯 Interview Bits — Day 11

**Q: What is the difference between monitoring and observability for AI systems?**
*Monitoring tracks predefined metrics and alerts when thresholds are breached — it answers "is something wrong?" Observability provides structured data that lets you ask arbitrary questions about system behavior — it answers "why is something wrong?" For AI systems, LangSmith traces provide observability while Prometheus metrics provide monitoring. You need both.*

**Q: How do you measure quality on production queries that have no ground truth?**
*Use reference-free metrics. Faithfulness — does the answer stick to the retrieved context — requires no ground truth. Answer relevancy — does the answer address the question — requires no ground truth. Only context recall requires labeled answers. This lets you monitor 80% of your quality signal on all production traffic.*

**Q: What is query distribution drift and why does it matter?**
*Query distribution drift occurs when production users start asking different types of questions than your eval dataset anticipated. Your offline benchmarks stop being predictive of production performance because they measure the wrong distribution. Evidently AI detects this statistically so you can update your eval dataset before quality silently degrades.*

**Q: What AI-specific metrics should you expose to Prometheus?**
*Beyond standard API metrics — request latency histograms, error rates, and throughput — AI systems need quality gauges updated from periodic RAGAS sampling, token usage counters for cost tracking, per-model latency breakdowns, and a quality drift alert flag that fires when production scores drop below threshold.*

**Q: How often should you run production trace quality analysis?**
*Daily for high-stakes systems, weekly for standard production systems. The cadence depends on traffic volume and business impact. For systems processing medical, legal, or financial queries — run it daily. For general-purpose assistants — weekly is sufficient. Always run it immediately after any deployment.*

---

*One day left.*
*Everything is built.*
*Tomorrow we tie it all together into a single unified dashboard.*
*The complete picture — quality, speed, cost, safety, and drift — in one place.*
*Day 12.*

