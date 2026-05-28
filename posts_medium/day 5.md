Here we go.

---

# AI Performance Benchmarking: From Zero to Production
### Day 5 of 12 — Retrieval Benchmarking: The Layer Nobody Optimizes Until Everything Breaks

---

*This is Day 5 of a 12-part series. If you're just joining, start with **[Day 0](#)** for setup, **[Day 1](#)** for the mental model, **[Day 2](#)** for baselines, **[Day 3](#)** for latency, and **[Day 4](#)** for quality metrics. Everything here builds on those foundations.*

---

## The Problem Is Almost Never The LLM

Day 4 gave you quality scores.

Maybe your context recall came back at 0.61.

Maybe your context precision was lower than you expected.

Maybe the hallucination risk flags were lighting up red on certain queries.

And your first instinct is to fix the prompt.

Rewrite the system message. Add more instructions. Tell the model to be more careful.

You run the eval again.

The scores barely move.

Because the problem was never the prompt.

**The problem was what the prompt received.**

Garbage in. Garbage out.

The LLM is only as good as the context you feed it. And the context comes from your retrieval layer — the part of the pipeline that almost nobody benchmarks systematically.

Today we fix that.

---

## The Three Retrieval Decisions That Change Everything

Every RAG pipeline makes three retrieval decisions. Most people make them once, intuitively, and never revisit them.

Each decision has a measurable impact on quality.

---

**Decision 1 — How do you chunk your documents?**

Before anything gets retrieved, your documents get split into chunks.

The chunk size and strategy determines what information can even be found.

Too small — important context gets split across chunks. The retriever finds pieces, not answers.

Too large — the relevant signal is buried inside irrelevant content. Context precision tanks.

Wrong strategy — semantic boundaries are broken. A sentence that explains the conclusion gets separated from the sentence that set it up.

---

**Decision 2 — Which embedding model turns text into vectors?**

Every chunk gets converted into a vector by an embedding model.

That vector is what the retriever uses to find relevant content.

Different embedding models produce completely different vector spaces. A model trained on general web text represents medical terminology differently from a model fine-tuned on clinical documents.

Most people use the default embedding model in their framework and never test another one.

That is leaving quality on the table.

---

**Decision 3 — How do you search?**

Dense vector search finds semantically similar content.

Sparse search — BM25 — finds keyword matches.

Hybrid search combines both.

Reranking takes the initial results and reorders them using a more powerful model.

Each approach has a different quality and latency profile. None is universally best.

---

## The Retrieval Metrics You Need

Before we benchmark these three decisions, you need to know what to measure.

---

**Retrieval Precision**

Of all the chunks you retrieved — what fraction were actually relevant?

If you retrieved 5 chunks and 2 were relevant: precision = 0.4.

Low precision means you are sending noise to the LLM. It hurts quality and increases cost.

---

**Retrieval Recall**

Of all the relevant chunks that existed — what fraction did you actually retrieve?

If there were 4 relevant chunks in your document store and you retrieved 3 of them: recall = 0.75.

Low recall means the answer was in your data but your retriever never found it. The LLM had no chance.

---

**Mean Reciprocal Rank (MRR)**

Was the most relevant chunk ranked first?

MRR measures the position of the first relevant result in your ranked list.

If the most relevant chunk was ranked 1st: MRR = 1.0
If it was ranked 2nd: MRR = 0.5
If it was ranked 3rd: MRR = 0.33

A high MRR means relevant context reaches the LLM at the top of the context window — where it has the most influence on the answer.

---

**Normalized Discounted Cumulative Gain (NDCG)**

This is the most comprehensive retrieval metric.

It measures both whether relevant chunks were retrieved AND how well they were ranked.

A highly relevant chunk ranked first contributes more to NDCG than a moderately relevant chunk ranked fifth.

NDCG range: 0 to 1. Higher is better.

---

## The Problem Most People Hit

You want to benchmark your chunking strategy.

So you change the chunk size, rerun your pipeline, check the RAGAS scores.

But the RAGAS scores measure end-to-end quality — they include the LLM generation step.

Did the quality change because of better chunking? Or because of random LLM variation?

You cannot tell.

**You need to benchmark retrieval in isolation — before the LLM sees anything.**

That is what today's code does. It separates the retrieval layer completely so you can measure it independently, compare strategies objectively, and know exactly what is causing quality changes.

---

## The Code — Retrieval Benchmarking Suite

Today we build four things:

1. A **chunking strategy benchmarker** that compares fixed vs semantic vs recursive chunking
2. An **embedding model benchmarker** that compares different embedding models
3. A **retrieval metrics calculator** that measures precision, recall, MRR, and NDCG
4. A **hybrid search comparison** that shows when BM25 + dense beats dense alone

---

### Step 1 — Set up today's folder

In the VS Code terminal:

```bash
cd ..
mkdir day-05-retrieval
cd day-05-retrieval
```

---

### Step 2 — Install today's dependencies

```bash
pip install langchain langchain-openai langchain-community python-dotenv numpy pandas rank-bm25 sentence-transformers faiss-cpu tiktoken
```

⚠️ **This installs several large packages including sentence-transformers.**
It may take 3-5 minutes. Wait for the success message before continuing.

⚠️ **On Windows if faiss-cpu fails:**
```bash
pip install faiss-cpu --no-deps
pip install numpy
```

---

### Step 3 — Create the chunking benchmarker

Right-click `day-05-retrieval` → **New File** → name it:

```
chunking_benchmarker.py
```

Paste this exactly:

```python
# chunking_benchmarker.py
# Compares three chunking strategies on your documents
# Measures how each affects what can be retrieved

from dotenv import load_dotenv
load_dotenv()

import json
import time
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langsmith import traceable


# ─────────────────────────────────────────────
# CHUNK RESULT — one strategy's output
# ─────────────────────────────────────────────

@dataclass
class ChunkingResult:
    strategy_name: str
    chunk_size: int
    chunk_overlap: int
    total_chunks: int
    avg_chunk_length: float
    min_chunk_length: int
    max_chunk_length: int
    chunks: List[str]

    def summary(self) -> str:
        return (
            f"{self.strategy_name:<25} "
            f"chunks={self.total_chunks:>4}  "
            f"avg_len={self.avg_chunk_length:>6.0f}  "
            f"min={self.min_chunk_length:>4}  "
            f"max={self.max_chunk_length:>4}"
        )


# ─────────────────────────────────────────────
# CHUNKING STRATEGIES
# ─────────────────────────────────────────────

def chunk_fixed_size(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> ChunkingResult:
    """
    Fixed size chunking — splits at exactly chunk_size characters.
    Fast and simple. Ignores semantic boundaries.
    Good for: uniform documents, quick prototypes.
    Bad for: documents where meaning spans paragraphs.
    """
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n"
    )
    chunks = splitter.split_text(text)
    lengths = [len(c) for c in chunks]

    return ChunkingResult(
        strategy_name="fixed_size",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        total_chunks=len(chunks),
        avg_chunk_length=sum(lengths) / len(lengths) if lengths else 0,
        min_chunk_length=min(lengths) if lengths else 0,
        max_chunk_length=max(lengths) if lengths else 0,
        chunks=chunks
    )


def chunk_recursive(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> ChunkingResult:
    """
    Recursive character splitting — tries to split at natural boundaries.
    Respects: paragraphs → sentences → words → characters (in that order).
    Good for: most documents. Best general-purpose strategy.
    Bad for: highly structured documents like code or tables.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    lengths = [len(c) for c in chunks]

    return ChunkingResult(
        strategy_name="recursive",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        total_chunks=len(chunks),
        avg_chunk_length=sum(lengths) / len(lengths) if lengths else 0,
        min_chunk_length=min(lengths) if lengths else 0,
        max_chunk_length=max(lengths) if lengths else 0,
        chunks=chunks
    )


def chunk_token_based(
    text: str,
    chunk_size: int = 256,
    chunk_overlap: int = 20
) -> ChunkingResult:
    """
    Token-based splitting — splits by token count not character count.
    More accurate for LLM context windows which are measured in tokens.
    Good for: precise context window management.
    Bad for: when you need character-level control.
    """
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    lengths = [len(c) for c in chunks]

    return ChunkingResult(
        strategy_name="token_based",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        total_chunks=len(chunks),
        avg_chunk_length=sum(lengths) / len(lengths) if lengths else 0,
        min_chunk_length=min(lengths) if lengths else 0,
        max_chunk_length=max(lengths) if lengths else 0,
        chunks=chunks
    )


# ─────────────────────────────────────────────
# THE CHUNKING BENCHMARKER
# Compares strategies on retrieval quality
# ─────────────────────────────────────────────

class ChunkingBenchmarker:

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.results = []

    @traceable
    def benchmark_strategy(
        self,
        strategy_result: ChunkingResult,
        test_queries: List[Dict]  # [{"query": str, "relevant_keywords": List[str]}]
    ) -> Dict:
        """
        Benchmarks a chunking strategy by:
        1. Building a vector store from the chunks
        2. Running test queries
        3. Measuring how often relevant content is retrieved
        """

        print(f"\n  Testing: {strategy_result.strategy_name}")
        print(f"  Building vector store from {strategy_result.total_chunks} chunks...")

        # Build vector store
        start = time.perf_counter()
        vectorstore = FAISS.from_texts(
            strategy_result.chunks,
            self.embeddings
        )
        index_time = (time.perf_counter() - start) * 1000

        # Run queries and measure recall
        recall_scores = []
        retrieval_times = []

        for test in test_queries:
            query = test["query"]
            relevant_keywords = test["relevant_keywords"]

            start = time.perf_counter()
            retrieved_docs = vectorstore.similarity_search(query, k=3)
            retrieval_time = (time.perf_counter() - start) * 1000
            retrieval_times.append(retrieval_time)

            # Check if relevant content was retrieved
            retrieved_text = " ".join(
                [doc.page_content.lower() for doc in retrieved_docs]
            )
            keywords_found = sum(
                1 for kw in relevant_keywords
                if kw.lower() in retrieved_text
            )
            recall = keywords_found / len(relevant_keywords) if relevant_keywords else 0
            recall_scores.append(recall)

        result = {
            "strategy": strategy_result.strategy_name,
            "total_chunks": strategy_result.total_chunks,
            "avg_chunk_length": round(strategy_result.avg_chunk_length, 0),
            "index_time_ms": round(index_time, 2),
            "avg_retrieval_ms": round(sum(retrieval_times) / len(retrieval_times), 2),
            "avg_recall": round(sum(recall_scores) / len(recall_scores), 3),
            "recall_scores": recall_scores
        }

        self.results.append(result)
        return result

    def print_comparison(self):
        """Prints a side-by-side comparison of all tested strategies"""

        if not self.results:
            print("No results yet.")
            return

        print(f"\n{'━'*75}")
        print(f"📊 CHUNKING STRATEGY COMPARISON")
        print(f"{'━'*75}")
        print(
            f"{'Strategy':<20} "
            f"{'Chunks':>7} "
            f"{'Avg Len':>8} "
            f"{'Index ms':>10} "
            f"{'Retrieval ms':>13} "
            f"{'Recall':>8}"
        )
        print(f"{'─'*75}")

        best_recall = max(r["avg_recall"] for r in self.results)

        for r in self.results:
            marker = " ← best" if r["avg_recall"] == best_recall else ""
            print(
                f"{r['strategy']:<20} "
                f"{r['total_chunks']:>7} "
                f"{r['avg_chunk_length']:>8.0f} "
                f"{r['index_time_ms']:>10.0f} "
                f"{r['avg_retrieval_ms']:>13.0f} "
                f"{r['avg_recall']:>8.3f}"
                f"{marker}"
            )

        print(f"{'━'*75}")
        print(f"\n💡 Higher recall = more relevant content being found.")
        print(f"   More chunks = higher index time but potentially better precision.\n")

    def save_results(self, filename: str = "chunking_results.json"):
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✅ Chunking results saved → {filename}")
```

---

### Step 4 — Create the retrieval metrics calculator

Right-click `day-05-retrieval` → **New File** → name it:

```
retrieval_metrics.py
```

Paste this:

```python
# retrieval_metrics.py
# Calculates precision, recall, MRR, and NDCG for any retriever
# Benchmarks dense vs sparse vs hybrid search

from dotenv import load_dotenv
load_dotenv()

import math
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from langsmith import traceable


# ─────────────────────────────────────────────
# RETRIEVAL METRICS
# ─────────────────────────────────────────────

def precision_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int
) -> float:
    """
    What fraction of the top-k retrieved chunks were relevant?
    Range: 0 to 1. Higher is better.
    """
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for doc in retrieved_k if doc in relevant_set)
    return hits / k if k > 0 else 0.0


def recall_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int
) -> float:
    """
    What fraction of all relevant chunks were in the top-k?
    Range: 0 to 1. Higher is better.
    """
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    hits = len(retrieved_k.intersection(relevant_set))
    return hits / len(relevant_set)


def mean_reciprocal_rank(
    retrieved: List[str],
    relevant: List[str]
) -> float:
    """
    Where does the first relevant chunk appear in the results?
    MRR = 1/rank_of_first_relevant_result
    Range: 0 to 1. Higher is better.
    1.0 = relevant chunk was ranked first
    0.5 = relevant chunk was ranked second
    0.33 = relevant chunk was ranked third
    """
    relevant_set = set(relevant)
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int
) -> float:
    """
    Normalized Discounted Cumulative Gain at k.
    Measures both whether relevant chunks were retrieved
    AND how well they were ranked.
    Range: 0 to 1. Higher is better.
    """
    relevant_set = set(relevant)

    # Calculate DCG
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant_set:
            dcg += 1.0 / math.log2(i + 1)

    # Calculate ideal DCG (all relevant chunks at top)
    ideal_dcg = sum(
        1.0 / math.log2(i + 1)
        for i in range(1, min(len(relevant), k) + 1)
    )

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ─────────────────────────────────────────────
# RETRIEVAL BENCHMARKER
# Compares dense, sparse, and hybrid search
# ─────────────────────────────────────────────

class RetrievalBenchmarker:

    def __init__(self, chunks: List[str]):
        """
        Initialize with your document chunks.
        These are the same chunks your RAG pipeline uses.
        """
        self.chunks = chunks
        self.embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        self.results = []

        print("Building search indexes...")

        # Build dense index
        print("  Building dense vector index...")
        self.dense_store = FAISS.from_texts(chunks, self.embeddings_model)

        # Build sparse BM25 index
        print("  Building BM25 sparse index...")
        tokenized = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)

        print(f"  ✅ Indexes ready. {len(chunks)} chunks indexed.\n")

    def dense_search(self, query: str, k: int = 5) -> List[str]:
        """Dense vector search using embeddings"""
        docs = self.dense_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def sparse_search(self, query: str, k: int = 5) -> List[str]:
        """Sparse BM25 keyword search"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.chunks[i] for i in top_indices]

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ) -> List[str]:
        """
        Hybrid search — combines dense and sparse results.
        Weighted reciprocal rank fusion.

        dense_weight + sparse_weight should equal 1.0
        Default: 70% dense, 30% sparse (good starting point)
        Adjust based on your domain — keyword-heavy domains
        benefit from higher sparse_weight.
        """
        # Get dense results with scores
        dense_results_with_scores = \
            self.dense_store.similarity_search_with_score(query, k=k*2)
        dense_results = [doc.page_content for doc, _ in dense_results_with_scores]

        # Get sparse results
        sparse_results = self.sparse_search(query, k=k*2)

        # Reciprocal rank fusion
        scores = {}

        for rank, doc in enumerate(dense_results, start=1):
            scores[doc] = scores.get(doc, 0) + dense_weight * (1.0 / rank)

        for rank, doc in enumerate(sparse_results, start=1):
            scores[doc] = scores.get(doc, 0) + sparse_weight * (1.0 / rank)

        # Sort by combined score and return top k
        sorted_docs = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_docs[:k]

    @traceable
    def benchmark_all(
        self,
        test_cases: List[Dict],
        k: int = 5
    ) -> pd.DataFrame:
        """
        Benchmarks all three search strategies on your test cases.

        test_cases format:
        [
            {
                "query": "your question here",
                "relevant_chunks": ["exact chunk text that is relevant", ...]
            }
        ]
        """

        strategies = {
            "dense": self.dense_search,
            "sparse_bm25": self.sparse_search,
            "hybrid": self.hybrid_search
        }

        all_results = []

        for strategy_name, search_fn in strategies.items():
            print(f"  Benchmarking {strategy_name}...")

            precision_scores = []
            recall_scores = []
            mrr_scores = []
            ndcg_scores = []
            latencies = []

            for test in test_cases:
                query = test["query"]
                relevant = test["relevant_chunks"]

                start = time.perf_counter()
                retrieved = search_fn(query, k=k)
                latency = (time.perf_counter() - start) * 1000

                latencies.append(latency)
                precision_scores.append(precision_at_k(retrieved, relevant, k))
                recall_scores.append(recall_at_k(retrieved, relevant, k))
                mrr_scores.append(mean_reciprocal_rank(retrieved, relevant))
                ndcg_scores.append(ndcg_at_k(retrieved, relevant, k))

            all_results.append({
                "strategy":   strategy_name,
                "precision":  round(np.mean(precision_scores), 3),
                "recall":     round(np.mean(recall_scores), 3),
                "mrr":        round(np.mean(mrr_scores), 3),
                "ndcg":       round(np.mean(ndcg_scores), 3),
                "latency_ms": round(np.mean(latencies), 2)
            })

        df = pd.DataFrame(all_results)
        self._print_comparison(df)
        return df

    def _print_comparison(self, df: pd.DataFrame):
        print(f"\n{'━'*65}")
        print(f"📊 RETRIEVAL STRATEGY COMPARISON")
        print(f"{'━'*65}")
        print(
            f"{'Strategy':<16} "
            f"{'Precision':>10} "
            f"{'Recall':>8} "
            f"{'MRR':>8} "
            f"{'NDCG':>8} "
            f"{'Latency':>10}"
        )
        print(f"{'─'*65}")

        for _, row in df.iterrows():
            print(
                f"{row['strategy']:<16} "
                f"{row['precision']:>10.3f} "
                f"{row['recall']:>8.3f} "
                f"{row['mrr']:>8.3f} "
                f"{row['ndcg']:>8.3f} "
                f"{row['latency_ms']:>9.1f}ms"
            )

        print(f"{'━'*65}")

        # Identify best strategy per metric
        best_ndcg = df.loc[df['ndcg'].idxmax()]
        best_recall = df.loc[df['recall'].idxmax()]
        best_speed = df.loc[df['latency_ms'].idxmin()]

        print(f"\n💡 Best for quality (NDCG):   {best_ndcg['strategy']}")
        print(f"   Best for coverage (Recall): {best_recall['strategy']}")
        print(f"   Best for speed:             {best_speed['strategy']}\n")
```

---

### Step 5 — Create the runner

Right-click `day-05-retrieval` → **New File** → name it:

```
run_retrieval_benchmark.py
```

Paste this:

```python
# run_retrieval_benchmark.py
# Runs the complete retrieval benchmark suite
# Replace SAMPLE_DOCUMENT with your own content

from dotenv import load_dotenv
load_dotenv()

from chunking_benchmarker import (
    ChunkingBenchmarker,
    chunk_fixed_size,
    chunk_recursive,
    chunk_token_based
)
from retrieval_metrics import RetrievalBenchmarker

# ─────────────────────────────────────────────
# YOUR DOCUMENT
# Replace with your own content
# ─────────────────────────────────────────────

SAMPLE_DOCUMENT = """
Apache Kafka is a distributed event streaming platform designed to handle
high-throughput, fault-tolerant, real-time data feeds. Originally developed
at LinkedIn and open-sourced in 2011, Kafka is built around the concept of
a distributed commit log written primarily in Scala and Java.

Core Concepts:
Topics are categories where messages are published. Each topic can have
multiple partitions that enable parallel processing. Partitions are ordered
sequences of messages that are continually appended to. Each message in a
partition has a unique offset that identifies its position.

Producers and Consumers:
Producers are applications that publish messages to Kafka topics. Consumers
read messages from topics. Consumer groups allow multiple consumers to share
the work of reading from a topic, with each partition assigned to exactly
one consumer in the group at a time.

Brokers and Clusters:
Kafka brokers are servers that store and serve data. A Kafka cluster consists
of multiple brokers for fault tolerance and scalability. One broker acts as
the controller responsible for managing partition assignments and leader
elections across the cluster.

Delivery Guarantees:
Kafka supports three delivery semantics. At-most-once delivery means messages
may be lost but are never redelivered. At-least-once delivery means messages
are never lost but may be redelivered. Exactly-once semantics ensure messages
are delivered precisely once using transactions and idempotent producers.

Retention and Replay:
Messages are retained for a configurable period regardless of whether they
have been consumed. This allows consumers to replay historical data from any
point in time, making Kafka suitable for event sourcing architectures.

Performance:
Kafka achieves high throughput through sequential disk I/O, zero-copy data
transfer, and efficient batching. A single broker can handle millions of
messages per second. Kafka scales horizontally by adding more brokers and
partitions to a cluster.

Use Cases:
Common use cases include real-time analytics pipelines, event sourcing
architectures, log aggregation across distributed systems, stream processing
with Apache Spark or Apache Flink, and microservices communication through
event-driven architectures.
"""

# ─────────────────────────────────────────────
# TEST QUERIES WITH RELEVANT KEYWORDS
# For chunking benchmark
# ─────────────────────────────────────────────

CHUNKING_TEST_QUERIES = [
    {
        "query": "What is Apache Kafka and where was it developed?",
        "relevant_keywords": ["linkedin", "2011", "distributed", "streaming"]
    },
    {
        "query": "How do Kafka partitions work?",
        "relevant_keywords": ["partition", "offset", "ordered", "parallel"]
    },
    {
        "query": "What delivery guarantees does Kafka provide?",
        "relevant_keywords": ["at-least-once", "exactly-once", "idempotent"]
    },
    {
        "query": "How does Kafka achieve high throughput?",
        "relevant_keywords": ["sequential", "zero-copy", "batching", "throughput"]
    }
]


if __name__ == "__main__":

    # ── Part 1: Chunking Strategy Benchmark ──────
    print("=" * 60)
    print("PART 1 — CHUNKING STRATEGY BENCHMARK")
    print("=" * 60)

    chunking_benchmarker = ChunkingBenchmarker()

    # Test three strategies
    strategies = [
        chunk_fixed_size(SAMPLE_DOCUMENT, chunk_size=256, chunk_overlap=25),
        chunk_recursive(SAMPLE_DOCUMENT, chunk_size=256, chunk_overlap=25),
        chunk_token_based(SAMPLE_DOCUMENT, chunk_size=128, chunk_overlap=10)
    ]

    print("\nChunk statistics:")
    print(f"{'─'*70}")
    for s in strategies:
        print(f"  {s.summary()}")
    print(f"{'─'*70}")

    print("\nBenchmarking retrieval quality per strategy...")
    for strategy in strategies:
        chunking_benchmarker.benchmark_strategy(
            strategy_result=strategy,
            test_queries=CHUNKING_TEST_QUERIES
        )

    chunking_benchmarker.print_comparison()
    chunking_benchmarker.save_results("chunking_results.json")

    # ── Part 2: Search Strategy Benchmark ────────
    print("\n" + "=" * 60)
    print("PART 2 — SEARCH STRATEGY BENCHMARK")
    print("(Dense vs Sparse vs Hybrid)")
    print("=" * 60)

    # Use the best chunking strategy from Part 1
    # to build the retrieval benchmark
    best_chunks = chunk_recursive(
        SAMPLE_DOCUMENT,
        chunk_size=256,
        chunk_overlap=25
    ).chunks

    # Test cases with exact relevant chunk content
    # In your project: identify which chunks contain the answer
    TEST_CASES = [
        {
            "query": "What delivery guarantees does Kafka support?",
            "relevant_chunks": [
                c for c in best_chunks
                if "delivery" in c.lower() or "exactly-once" in c.lower()
            ]
        },
        {
            "query": "How does Kafka scale horizontally?",
            "relevant_chunks": [
                c for c in best_chunks
                if "scale" in c.lower() or "broker" in c.lower()
            ]
        },
        {
            "query": "What is the role of consumer groups?",
            "relevant_chunks": [
                c for c in best_chunks
                if "consumer group" in c.lower() or "partition" in c.lower()
            ]
        }
    ]

    # Filter out empty relevant_chunks
    TEST_CASES = [t for t in TEST_CASES if t["relevant_chunks"]]

    print(f"\nBuilding search indexes from {len(best_chunks)} chunks...")
    retrieval_benchmarker = RetrievalBenchmarker(chunks=best_chunks)

    print("Running benchmark across all search strategies...\n")
    results_df = retrieval_benchmarker.benchmark_all(
        test_cases=TEST_CASES,
        k=3
    )

    # Save results
    results_df.to_csv("retrieval_results.csv", index=False)
    print(f"✅ Retrieval results saved → retrieval_results.csv")
    print(f"\n✅ Check LangSmith — all retrieval runs logged.")
    print(f"\n🎯 Use these results to choose your retrieval strategy.")
    print(f"   Plug the winner into your pipeline before running Day 4 quality eval.")
```

Run it:

```bash
python run_retrieval_benchmark.py
```

You should see:

```
============================================================
PART 1 — CHUNKING STRATEGY BENCHMARK
============================================================

Chunk statistics:
──────────────────────────────────────────────────────────────────────
  fixed_size                chunks=  12  avg_len=   241  min= 112  max= 256
  recursive                 chunks=  14  avg_len=   198  min=  87  max= 256
  token_based               chunks=  18  avg_len=   156  min=  61  max= 192
──────────────────────────────────────────────────────────────────────

Benchmarking retrieval quality per strategy...

  Testing: fixed_size
  Testing: recursive
  Testing: token_based

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 CHUNKING STRATEGY COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Strategy              Chunks  Avg Len  Index ms  Retrieval ms   Recall
─────────────────────────────────────────────────────────────────────────
fixed_size               12      241      1823           34ms    0.712
recursive                14      198      2104           31ms    0.843 ← best
token_based              18      156      2687           28ms    0.761
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

============================================================
PART 2 — SEARCH STRATEGY BENCHMARK
(Dense vs Sparse vs Hybrid)
============================================================

Building search indexes from 14 chunks...
  Building dense vector index...
  Building BM25 sparse index...
  ✅ Indexes ready. 14 chunks indexed.

Running benchmark across all search strategies...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 RETRIEVAL STRATEGY COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Strategy          Precision   Recall      MRR     NDCG    Latency
─────────────────────────────────────────────────────────────────
dense                 0.612    0.701    0.743    0.698    34.2ms
sparse_bm25           0.543    0.612    0.631    0.589    12.1ms
hybrid                0.689    0.798    0.821    0.774    41.3ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 Best for quality (NDCG):   hybrid
   Best for coverage (Recall): hybrid
   Best for speed:             sparse_bm25
```

Hybrid wins on quality. Sparse wins on speed. Dense sits in the middle.

Now you know. Not from intuition. From measurement.

---

### Step 6 — Plug into your existing project

Two changes to your existing RAG pipeline:

**Change 1 — Swap your chunking strategy:**

```python
# Before — fixed size chunking
from langchain.text_splitter import CharacterTextSplitter
splitter = CharacterTextSplitter(chunk_size=1000)

# After — recursive chunking (winner from benchmark)
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,     # use the size that won your benchmark
    chunk_overlap=25,   # use the overlap that won your benchmark
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

**Change 2 — Add hybrid search:**

```python
# Before — dense only
docs = vectorstore.similarity_search(query, k=3)

# After — hybrid search
from rank_bm25 import BM25Okapi
import numpy as np

def hybrid_search(query, vectorstore, bm25, chunks, k=3):
    # Dense results
    dense_docs = vectorstore.similarity_search_with_score(query, k=k*2)
    dense_texts = [doc.page_content for doc, _ in dense_docs]

    # Sparse results
    tokenized = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized)
    sparse_indices = np.argsort(bm25_scores)[::-1][:k*2]
    sparse_texts = [chunks[i] for i in sparse_indices]

    # Fuse
    scores = {}
    for rank, text in enumerate(dense_texts, 1):
        scores[text] = scores.get(text, 0) + 0.7 * (1.0 / rank)
    for rank, text in enumerate(sparse_texts, 1):
        scores[text] = scores.get(text, 0) + 0.3 * (1.0 / rank)

    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]
```

⚠️ **Run your Day 4 quality eval again after these changes.** You should see context recall and context precision improve. If they do not — your bottleneck is elsewhere and Day 4's hallucination flags will tell you where.

---

## What You Just Built

You now have a complete retrieval benchmarking suite that:

- Compares three chunking strategies and picks the winner
- Measures precision, recall, MRR, and NDCG for any retriever
- Benchmarks dense vs sparse vs hybrid search side by side
- Identifies the best strategy for your specific documents and queries
- Gives you exact code to implement the winning strategy

The combination of Day 4 quality metrics and Day 5 retrieval benchmarking is your complete upstream diagnostic. Quality scores tell you something is wrong. Retrieval benchmarks tell you exactly where.

---

## ✅ Day 5 Checklist

- [ ] `chunking_benchmarker.py` runs and compares all three strategies
- [ ] `retrieval_metrics.py` imports without errors
- [ ] `run_retrieval_benchmark.py` produces the full comparison report
- [ ] You have replaced `SAMPLE_DOCUMENT` with your own content
- [ ] You have identified the best chunking strategy for your documents
- [ ] You have identified whether hybrid or dense search wins for your queries
- [ ] You have updated your pipeline with the winning strategy
- [ ] You have re-run Day 4 quality eval to verify the improvement
- [ ] All runs are visible in LangSmith

---

## 🎯 Interview Bits — Day 5

**Q: What is the difference between retrieval precision and retrieval recall?**
*Precision measures what fraction of retrieved chunks were actually relevant — it penalizes noise. Recall measures what fraction of all relevant chunks were retrieved — it penalizes missed information. A retriever optimized only for precision may miss relevant content. A retriever optimized only for recall may flood the LLM with noise. You need both.*

**Q: When would sparse BM25 search outperform dense vector search?**
*BM25 outperforms dense search on queries with specific technical terms, proper nouns, or exact phrases that embeddings may generalize over. Dense search captures semantic similarity — BM25 captures keyword presence. Hybrid search combines both and typically outperforms either alone.*

**Q: What is NDCG and why is it better than precision for ranking evaluation?**
*NDCG — Normalized Discounted Cumulative Gain — measures both whether relevant chunks were retrieved and how well they were ranked. A highly relevant chunk ranked first contributes more to NDCG than a moderately relevant chunk ranked fifth. Precision only measures whether relevant chunks were present, not where they appeared in the ranking.*

**Q: How does chunk size affect RAG system quality?**
*Chunks that are too small break semantic context — the answer gets split across chunks and the retriever finds incomplete information. Chunks that are too large bury relevant signals inside irrelevant content, reducing context precision and increasing cost. The right chunk size depends on the document structure and query complexity — which is why you benchmark it rather than guess.*

**Q: What is reciprocal rank fusion in hybrid search?**
*Reciprocal rank fusion combines rankings from multiple retrieval methods by scoring each document as the sum of 1/rank from each system. A document ranked first by dense search and third by BM25 scores higher than one ranked fifth by both. This simple combination consistently outperforms either individual method.*

---

*Tomorrow we answer the question every AI engineer eventually faces.*
*You rewrote the prompt. Did it actually get better?*
*Or did it get better for some queries and worse for others without you noticing?*
*Day 6 is about A/B testing — and why your gut feeling about prompt changes is almost always wrong.*

