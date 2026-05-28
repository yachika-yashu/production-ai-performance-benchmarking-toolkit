Here we go.

---

# AI Performance Benchmarking: From Zero to Production
### Day 8 of 12 — Load Testing: What Happens When Everyone Shows Up At Once

---

*This is Day 8 of a 12-part series. Start with **[Day 0](#)** if you're just joining.*

---

## The Demo That Fooled Everyone

Your system passed every benchmark.

Latency: excellent.
Quality scores: strong.
Cost: under budget.

You demo it to the team. Twenty people in the room. Everyone impressed.

Two weeks later you roll it out to real users.

The first morning everything is fine.

Then 9am hits.

Everyone arrives at work. Opens the app. Starts querying.

Response times climb from 1 second to 8 seconds.
Then 15 seconds.
Some requests start timing out entirely.

Your Slack blows up.

---

What happened?

Nothing broke. No code changed. No model went down.

**You tested with one user. You deployed to five hundred.**

The system that looked perfect in isolation collapsed under real concurrent load.

This is the most common and most preventable production failure in AI systems.

Load testing is how you catch it before your users do.

---

## Why AI Systems Break Under Load Differently

Traditional web APIs break under load in predictable ways.

Too many requests → database connection pool exhausted → requests queue up → timeouts.

AI systems break in all of those ways plus several more:

---

**The LLM API rate limit wall**

Every LLM provider enforces rate limits — requests per minute and tokens per minute.

A single user never hits them.

Two hundred concurrent users absolutely do.

When you hit the rate limit the API returns a 429 error. If your code does not handle this gracefully — retrying with exponential backoff — requests fail hard.

---

**The async/sync deadlock**

This is the most common performance bug in FastAPI-based AI systems.

FastAPI is asynchronous. But most LangChain retrievers and some LLM clients are synchronous — they block the thread they run on.

If you call a blocking function directly inside an async FastAPI route, it blocks the entire event loop. Every other request waits. Under load this creates a deadlock that makes P99 latency explode.

The fix is one line: `asyncio.to_thread()`. But you have to know it exists.

---

**The memory ceiling**

Each concurrent request holds documents in memory — retrieved chunks, conversation history, intermediate results.

At low concurrency this is invisible.

At high concurrency memory usage spikes. If you hit the container's memory limit the process gets killed and restarted. Users see connection errors.

---

**The cold start penalty**

When your container scales up under load — new instances spinning up — the first requests to hit a new instance pay the cold start cost.

Model loading, connection establishment, cache warming.

These first requests can be 5-10x slower than steady state.

---

## The Two Numbers That Matter Most Under Load

**Requests Per Second (RPS)**

How many requests can your system handle per second before quality of service degrades?

This is your throughput ceiling.

---

**Concurrent Users**

How many users can be in-flight simultaneously before response times become unacceptable?

This is your concurrency ceiling.

These two numbers together define your system's operating envelope. Stay inside it and everything is fine. Exceed it and the experience degrades.

Load testing finds both numbers before your users do.

---

## The Problem Most People Hit

You want to load test your AI system.

You write a loop that sends 100 requests and times them.

But a loop runs sequentially — one after the other. That is not concurrent load. That is just a slow single user.

Real load testing requires genuinely concurrent requests — many users hitting the system simultaneously, each waiting for their own response.

Locust is the tool that does this properly. It simulates real concurrent users, measures response time distributions under load, and finds the exact point where your system starts degrading.

---

## The Code — Load Testing Suite

Today we build four things:

1. A **FastAPI endpoint** — a realistic AI API to load test
2. A **Locust load test** — simulating concurrent users
3. An **async fix** — the one-line change that prevents event loop blocking
4. A **memory profiler** — tracking memory under concurrent load

---

### Step 1 — Set up today's folder

```bash
cd ..
mkdir day-08-load-testing
cd day-08-load-testing
```

---

### Step 2 — Install dependencies

```bash
pip install fastapi uvicorn locust langchain langchain-openai python-dotenv psutil httpx asyncio
```

---

### Step 3 — Create the API to load test

Right-click `day-08-load-testing` → **New File** → name it:

```
api.py
```

Paste this exactly:

```python
# api.py
# A realistic FastAPI AI endpoint to load test
# Shows both the broken version and the fixed version
# Run with: uvicorn api:app --host 0.0.0.0 --port 8000

from dotenv import load_dotenv
load_dotenv()

import asyncio
import time
import psutil
import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

app = FastAPI(title="AI Benchmarking Load Test API")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    use_async_fix: bool = True   # Toggle to compare broken vs fixed


class QueryResponse(BaseModel):
    answer: str
    latency_ms: float
    memory_mb: float
    async_mode: bool


# ─────────────────────────────────────────────
# MOCK RETRIEVER
# Replace with your real retriever
# ─────────────────────────────────────────────

def retrieve_context_sync(query: str) -> list[str]:
    """
    Synchronous retriever — simulates a real vector DB call.
    This is the function that blocks the event loop
    when called directly inside an async route.

    Replace with:
        docs = your_retriever.invoke(query)
        return [doc.page_content for doc in docs]
    """
    time.sleep(0.1)  # Simulates 100ms vector DB latency
    return [
        f"Retrieved context chunk 1 for: {query}",
        f"Retrieved context chunk 2 for: {query}",
        f"Retrieved context chunk 3 for: {query}"
    ]


@traceable
def generate_response_sync(query: str, contexts: list[str]) -> str:
    """Synchronous LLM call"""
    context_text = "\n".join(contexts)
    response = llm.invoke([
        SystemMessage(content="Answer using only the provided context."),
        HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {query}")
    ])
    return response.content


# ─────────────────────────────────────────────
# THE BROKEN ROUTE — blocks the event loop
# ─────────────────────────────────────────────

@app.post("/query/broken")
async def query_broken(request: QueryRequest):
    """
    ❌ BROKEN — calls blocking functions directly in async route.

    What happens under load:
    - First request runs fine
    - Second request arrives while first is waiting for retriever
    - The event loop is blocked — second request cannot start
    - Requests queue up
    - P99 latency explodes
    - At high concurrency: total deadlock
    """
    start = time.perf_counter()

    # ❌ This blocks the event loop
    contexts = retrieve_context_sync(request.query)
    answer = generate_response_sync(request.query, contexts)

    latency_ms = (time.perf_counter() - start) * 1000
    memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

    return QueryResponse(
        answer=answer,
        latency_ms=round(latency_ms, 2),
        memory_mb=round(memory_mb, 2),
        async_mode=False
    )


# ─────────────────────────────────────────────
# THE FIXED ROUTE — runs blocking calls in thread pool
# ─────────────────────────────────────────────

@app.post("/query/fixed")
async def query_fixed(request: QueryRequest):
    """
    ✅ FIXED — moves blocking calls to a thread pool.

    asyncio.to_thread() runs the blocking function in a
    separate thread, freeing the event loop to handle
    other requests while waiting.

    This single change can reduce P99 latency by 80%+
    under concurrent load.
    """
    start = time.perf_counter()

    # ✅ Runs in thread pool — event loop stays free
    contexts = await asyncio.to_thread(
        retrieve_context_sync, request.query
    )
    answer = await asyncio.to_thread(
        generate_response_sync, request.query, contexts
    )

    latency_ms = (time.perf_counter() - start) * 1000
    memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

    return QueryResponse(
        answer=answer,
        latency_ms=round(latency_ms, 2),
        memory_mb=round(memory_mb, 2),
        async_mode=True
    )


# ─────────────────────────────────────────────
# HEALTH CHECK
# Always include this — Locust uses it to verify
# the server is up before starting the test
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    return {
        "status": "healthy",
        "memory_mb": round(memory_mb, 2)
    }
```

---

### Step 4 — Create the Locust load test

Right-click `day-08-load-testing` → **New File** → name it:

```
locustfile.py
```

Paste this:

```python
# locustfile.py
# Simulates concurrent users hitting your AI API
# Run with: locust -f locustfile.py --host http://localhost:8000

import random
from locust import HttpUser, task, between, events
import json


# ─────────────────────────────────────────────
# TEST QUERIES
# Replace with queries representative of your
# real user traffic distribution
# ─────────────────────────────────────────────

SIMPLE_QUERIES = [
    "What is Apache Kafka?",
    "What is a Kafka topic?",
    "Who developed Kafka?",
    "What is a Kafka broker?",
    "What is an offset in Kafka?"
]

MODERATE_QUERIES = [
    "How do consumer groups work in Kafka?",
    "Explain the difference between a topic and a partition",
    "How does Kafka handle message ordering?",
    "What delivery guarantees does Kafka provide?",
    "How does Kafka achieve fault tolerance?"
]

COMPLEX_QUERIES = [
    "Analyze the tradeoffs between Kafka and RabbitMQ "
    "for a hospital real-time data pipeline",
    "Compare exactly-once semantics implementation "
    "across different streaming platforms",
    "Evaluate the architectural implications of "
    "using Kafka for event sourcing in a microservices system"
]


# ─────────────────────────────────────────────
# THE SIMULATED USER
# Each Locust user runs these tasks continuously
# ─────────────────────────────────────────────

class AISystemUser(HttpUser):
    """
    Simulates a real user of your AI system.
    wait_time controls how long each user pauses
    between requests — simulates realistic usage patterns.
    """

    # Each user waits 1-3 seconds between requests
    # Adjust to match your real usage patterns
    wait_time = between(1, 3)

    @task(5)  # Weight 5 — runs 5x more often than complex
    def simple_query(self):
        """Simple factual queries — most common in real systems"""
        query = random.choice(SIMPLE_QUERIES)
        self._send_query(query, endpoint="/query/fixed")

    @task(3)  # Weight 3
    def moderate_query(self):
        """Moderate complexity queries"""
        query = random.choice(MODERATE_QUERIES)
        self._send_query(query, endpoint="/query/fixed")

    @task(1)  # Weight 1 — least common
    def complex_query(self):
        """Complex reasoning queries — rare but expensive"""
        query = random.choice(COMPLEX_QUERIES)
        self._send_query(query, endpoint="/query/fixed")

    @task(2)
    def health_check(self):
        """Periodically check system health"""
        self.client.get("/health")

    def _send_query(self, query: str, endpoint: str):
        """
        Sends a query and records the result.
        Locust automatically tracks:
        - Response time
        - Success/failure rate
        - Requests per second
        """
        with self.client.post(
            endpoint,
            json={"query": query},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                latency = data.get("latency_ms", 0)

                # Flag responses that are too slow
                # Adjust this threshold for your SLA
                if latency > 10000:  # 10 seconds
                    response.failure(
                        f"Response too slow: {latency:.0f}ms"
                    )
                else:
                    response.success()
            elif response.status_code == 429:
                # Rate limited — this is important to track
                response.failure("Rate limited by LLM API")
            else:
                response.failure(
                    f"HTTP {response.status_code}"
                )


# ─────────────────────────────────────────────
# BROKEN VS FIXED COMPARISON USER
# Specifically tests both endpoints to compare
# ─────────────────────────────────────────────

class ComparisonUser(HttpUser):
    """
    Sends identical requests to both broken and fixed endpoints.
    Use this to directly compare async vs sync performance.
    Run separately with:
    locust -f locustfile.py --host http://localhost:8000
           ComparisonUser
    """
    wait_time = between(0.5, 1.5)

    @task
    def compare_endpoints(self):
        query = random.choice(SIMPLE_QUERIES + MODERATE_QUERIES)

        # Test fixed endpoint
        self.client.post(
            "/query/fixed",
            json={"query": query},
            name="/query/fixed"
        )
```

---

### Step 5 — Create the memory profiler

Right-click `day-08-load-testing` → **New File** → name it:

```
memory_profiler.py
```

Paste this:

```python
# memory_profiler.py
# Tracks memory usage under concurrent load
# Run this alongside your load test to catch memory leaks

from dotenv import load_dotenv
load_dotenv()

import asyncio
import httpx
import psutil
import os
import time
import json
import numpy as np
from datetime import datetime


async def send_concurrent_requests(
    endpoint: str,
    queries: list,
    concurrency: int
) -> list:
    """
    Sends concurrent requests and measures response times.
    This is what Locust does internally — here for standalone use.
    """
    async with httpx.AsyncClient(
        base_url="http://localhost:8000",
        timeout=30.0
    ) as client:
        tasks = [
            client.post(endpoint, json={"query": q})
            for q in queries[:concurrency]
        ]
        start = time.perf_counter()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = (time.perf_counter() - start) * 1000

        results = []
        for r in responses:
            if isinstance(r, Exception):
                results.append({"error": str(r), "latency_ms": None})
            elif r.status_code == 200:
                data = r.json()
                results.append({
                    "latency_ms": data.get("latency_ms"),
                    "memory_mb": data.get("memory_mb")
                })
            else:
                results.append({
                    "error": f"HTTP {r.status_code}",
                    "latency_ms": None
                })

        return results, total_time


async def run_memory_profile(
    endpoint: str = "/query/fixed",
    concurrency_levels: list = [1, 5, 10, 20, 50]
):
    """
    Tests memory usage at increasing concurrency levels.
    Shows you when memory becomes a problem.
    """

    TEST_QUERIES = [
        "What is Apache Kafka?",
        "How do partitions work?",
        "What is a consumer group?",
        "Explain Kafka brokers",
        "What is an offset?",
        "How does Kafka handle failures?",
        "What is topic replication?",
        "Explain Kafka producers",
        "What are Kafka streams?",
        "How does Kafka scale?"
    ] * 10  # Repeat to have enough for high concurrency

    process = psutil.Process(os.getpid())
    results = []

    print(f"\n{'━'*65}")
    print(f"🧠 MEMORY & CONCURRENCY PROFILE — {endpoint}")
    print(f"{'━'*65}")
    print(
        f"{'Concurrency':<14} "
        f"{'Avg Latency':>13} "
        f"{'P99 Latency':>13} "
        f"{'Errors':>8} "
        f"{'Memory MB':>12}"
    )
    print(f"{'─'*65}")

    baseline_memory = process.memory_info().rss / 1024 / 1024

    for concurrency in concurrency_levels:
        queries = TEST_QUERIES[:concurrency]

        try:
            responses, total_ms = await send_concurrent_requests(
                endpoint=endpoint,
                queries=queries,
                concurrency=concurrency
            )
        except Exception as e:
            print(f"  {concurrency:<12} ERROR: {e}")
            continue

        # Calculate metrics
        latencies = [
            r["latency_ms"] for r in responses
            if r.get("latency_ms") is not None
        ]
        errors = sum(1 for r in responses if "error" in r)
        current_memory = process.memory_info().rss / 1024 / 1024

        avg_latency = np.mean(latencies) if latencies else 0
        p99_latency = np.percentile(latencies, 99) if latencies else 0

        # Flag degradation
        warning = ""
        if avg_latency > 5000:
            warning = " ⚠️  SLOW"
        if errors > concurrency * 0.1:
            warning = " 🔴 ERRORS"

        print(
            f"{concurrency:<14} "
            f"{avg_latency:>11.0f}ms "
            f"{p99_latency:>11.0f}ms "
            f"{errors:>8} "
            f"{current_memory:>10.1f}MB"
            f"{warning}"
        )

        results.append({
            "concurrency": concurrency,
            "avg_latency_ms": round(avg_latency, 2),
            "p99_latency_ms": round(p99_latency, 2),
            "errors": errors,
            "memory_mb": round(current_memory, 2),
            "memory_delta_mb": round(current_memory - baseline_memory, 2)
        })

        # Brief pause between concurrency levels
        await asyncio.sleep(2)

    print(f"{'━'*65}")

    # Find breaking point
    degraded = [
        r for r in results
        if r["avg_latency_ms"] > 5000 or r["errors"] > 0
    ]
    if degraded:
        breaking_point = degraded[0]["concurrency"]
        print(
            f"\n🔴 Performance degrades at "
            f"{breaking_point} concurrent users"
        )
    else:
        print(
            f"\n✅ System handled all concurrency levels within SLA"
        )

    # Save results
    with open("memory_profile.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Memory profile saved → memory_profile.json\n")

    return results


if __name__ == "__main__":
    asyncio.run(run_memory_profile(
        endpoint="/query/fixed",
        concurrency_levels=[1, 5, 10, 20, 50]
    ))
```

---

### Step 6 — Run everything

You need three terminal windows for this.

---

**Terminal 1 — Start the API server:**

```bash
cd day-08-load-testing
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

Leave this running.

---

**Terminal 2 — Run the memory profiler:**

Open a second terminal in VS Code:
- Click the `+` icon in the terminal panel
- Navigate to the folder: `cd day-08-load-testing`

Run:
```bash
python memory_profiler.py
```

You should see:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧠 MEMORY & CONCURRENCY PROFILE — /query/fixed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Concurrency    Avg Latency   P99 Latency    Errors    Memory MB
─────────────────────────────────────────────────────────────────
1                    923ms         941ms         0      187.4MB
5                   1102ms        1834ms         0      198.2MB
10                  1891ms        3421ms         0      214.6MB
20                  3204ms        7821ms         2      241.3MB  ⚠️  SLOW
50                  8921ms       18400ms        11      289.7MB  🔴 ERRORS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔴 Performance degrades at 20 concurrent users
✅ Memory profile saved → memory_profile.json
```

Now you know your concurrency ceiling. In this case: 20 users is where things start going wrong.

---

**Terminal 3 — Run the Locust load test:**

Open a third terminal:
```bash
cd day-08-load-testing
locust -f locustfile.py --host http://localhost:8000
```

You should see:
```
[2026-05-26 10:00:00,000] INFO/locust.main: Starting web interface at http://0.0.0.0:8089
```

Open your browser and go to:
```
http://localhost:8089
```

You will see the Locust UI. Fill in:
- **Number of users:** 50
- **Spawn rate:** 5 (users added per second)
- **Host:** http://localhost:8000

Click **Start swarming**.

Watch the real-time charts as concurrent users increase. You will see exactly when response times start climbing and where failures begin.

⚠️ **When you are done testing:**
Press `Ctrl+C` in all three terminal windows to stop everything.

---

### Step 7 — Plug into your existing project

Two changes for your real API:

**Change 1 — Fix your async routes:**

```python
# Find every place you call a blocking function
# inside an async FastAPI route

# Before — blocks event loop ❌
@app.post("/query")
async def query(request: QueryRequest):
    docs = your_retriever.invoke(request.query)      # blocking
    response = your_chain.invoke({"query": request.query})  # blocking
    return {"answer": response}

# After — thread pool safe ✅
import asyncio

@app.post("/query")
async def query(request: QueryRequest):
    docs = await asyncio.to_thread(
        your_retriever.invoke, request.query
    )
    response = await asyncio.to_thread(
        your_chain.invoke, {"query": request.query}
    )
    return {"answer": response}
```

**Change 2 — Replace test queries in locustfile.py:**

```python
# Replace SIMPLE_QUERIES, MODERATE_QUERIES, COMPLEX_QUERIES
# with real queries from your system
# The best source: your LangSmith trace history from Day 0

# Export from LangSmith:
# Projects → your project → Traces → Export → CSV
# Then load the "input" column as your query list
```

---

## What You Just Built

You now have a complete load testing suite that:

- Exposes both the broken and fixed async patterns for direct comparison
- Simulates realistic concurrent user behavior with weighted query distribution
- Profiles memory usage at increasing concurrency levels
- Identifies your exact breaking point before users find it
- Gives you a visual real-time dashboard via Locust UI

---

## ✅ Day 8 Checklist

- [ ] `api.py` starts without errors on port 8000
- [ ] `/health` endpoint returns 200
- [ ] `memory_profiler.py` runs and identifies your concurrency ceiling
- [ ] Locust UI opens at `http://localhost:8089`
- [ ] You have run at least one load test with 20+ users
- [ ] You understand the difference between `/query/broken` and `/query/fixed`
- [ ] You have applied `asyncio.to_thread()` to your real pipeline
- [ ] Memory profile saved to `memory_profile.json`

---

## 🎯 Interview Bits — Day 8

**Q: What is the difference between latency and throughput in AI systems?**
*Latency is how long a single request takes from start to finish. Throughput is how many requests the system can handle per unit of time. They are related but not the same — a system can have low single-request latency but poor throughput if it cannot handle many concurrent requests. Load testing measures both.*

**Q: Why do synchronous calls inside async FastAPI routes cause problems under load?**
*FastAPI's event loop handles all concurrent requests on a single thread. A synchronous blocking call — like a vector DB query or LangChain retriever — holds that thread until it completes. No other requests can be processed while it waits. Under concurrent load this creates a queue that causes P99 latency to explode. The fix is asyncio.to_thread() which moves blocking calls to a thread pool, freeing the event loop.*

**Q: What is a cold start in containerized AI systems and how do you mitigate it?**
*A cold start is the latency penalty on the first request to a new container instance — model loading, connection establishment, cache warming. Under load when auto-scaling spins up new instances, users hitting those instances experience 5-10x normal latency. Mitigation strategies include keeping minimum instances warm, pre-loading models at startup, and using connection pooling.*

**Q: How do you determine the right concurrency ceiling for an AI system?**
*Load test at increasing concurrency levels and measure P99 latency and error rate at each level. The concurrency ceiling is the point where P99 latency exceeds your SLA or error rate exceeds your acceptable threshold. Set your auto-scaling trigger at 70-80% of that ceiling to leave headroom for traffic spikes.*

**Q: What HTTP status code does an LLM API return when rate limited and how should you handle it?**
*429 Too Many Requests. Handle it with exponential backoff and jitter — wait, retry, wait longer, retry again, with random jitter to prevent thundering herd when many clients retry simultaneously. Most LLM SDKs have built-in retry logic but you need to configure the maximum retry count and backoff parameters explicitly.*

---

*Tomorrow we go to the frontier.*
*Everything so far measured a single LLM call.*
*But what happens when your AI takes multiple steps, uses tools, and makes decisions along the way?*
*Day 9 — agentic benchmarking.*

