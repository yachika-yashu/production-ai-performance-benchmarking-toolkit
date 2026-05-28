"""
Embedded knowledge base — AI, RAG, and LLM production concepts.
No scraping needed. Replace with your own docs via load_from_directory().
"""

DOCUMENTS = [
    {
        "id": "rag-overview",
        "title": "Retrieval Augmented Generation (RAG)",
        "content": """
Retrieval Augmented Generation (RAG) is an AI framework that enhances large language model
outputs by retrieving relevant information from an external knowledge base before generating
a response. Instead of relying solely on knowledge baked into model weights during training,
RAG systems first search a document store for relevant context, then pass that context to
the LLM alongside the user query.

The RAG pipeline has three core stages: indexing, retrieval, and generation.
During indexing, source documents are split into chunks, converted to vector embeddings,
and stored in a vector database. During retrieval, the user query is embedded and the
nearest matching chunks are fetched using similarity search. During generation, the
retrieved chunks are injected into the prompt as context and the LLM generates a
grounded response.

RAG solves the hallucination problem by grounding LLM responses in retrieved facts.
It also solves the knowledge cutoff problem — the knowledge base can be updated without
retraining the model. RAG is preferred over fine-tuning when the knowledge domain changes
frequently or when transparency and source attribution are required.

Common RAG architectures include naive RAG (basic retrieve-then-generate), advanced RAG
(with re-ranking, query rewriting, and hybrid search), and modular RAG (with routing,
fusion, and iterative retrieval). LangChain and LlamaIndex are the most widely used
frameworks for building RAG pipelines in Python.
        """.strip(),
    },
    {
        "id": "vector-databases",
        "title": "Vector Databases and Similarity Search",
        "content": """
A vector database stores data as high-dimensional numerical vectors (embeddings) and enables
fast approximate nearest-neighbor search. Unlike traditional databases that match exact values,
vector databases find semantically similar content by measuring distance in embedding space.

The most common similarity metrics are cosine similarity, dot product, and Euclidean distance.
Cosine similarity measures the angle between two vectors and is preferred for text embeddings
because it is length-invariant — it captures semantic direction regardless of document length.

Popular vector databases include FAISS (Meta, in-memory, excellent for prototyping and
small-to-medium scale), Pinecone (managed cloud service, production-ready), Weaviate
(open source with hybrid search), Chroma (lightweight, local-first), Qdrant (Rust-based,
high performance), and pgvector (PostgreSQL extension for existing relational databases).

FAISS (Facebook AI Similarity Search) is the most widely used for research and prototyping.
It supports multiple index types: Flat (exact search, slow at scale), IVF (inverted file
index, fast approximate search), and HNSW (hierarchical navigable small world graph,
best speed-recall tradeoff for most use cases).

Vector database performance is measured by recall at k (what fraction of true top-k results
are returned), queries per second (QPS), and index build time. Production systems typically
target recall@10 above 0.95 with QPS above 100 for interactive applications.
        """.strip(),
    },
    {
        "id": "embeddings",
        "title": "Embedding Models and Text Representations",
        "content": """
Text embeddings are dense vector representations of text that capture semantic meaning.
Words, sentences, or documents with similar meanings produce vectors that are close together
in the embedding space, enabling similarity-based retrieval.

The most widely used embedding models are OpenAI text-embedding-3-small (1536 dimensions,
fast, cheap, excellent quality), OpenAI text-embedding-3-large (3072 dimensions, highest
quality for English), and open-source alternatives like sentence-transformers/all-MiniLM-L6-v2
(384 dimensions, fast, free, good quality for many tasks).

Embedding quality directly impacts RAG retrieval quality. A poor embedding model produces
vectors where semantically similar text is not close together, causing retrieval failures
regardless of how good the LLM is. Always benchmark embedding models on your specific
domain before selecting one for production.

Chunking strategy affects embedding quality significantly. Each chunk should be semantically
coherent — a single topic or idea. Common strategies include fixed-size chunking (simple but
can split concepts), sentence-based chunking (respects sentence boundaries), and semantic
chunking (splits on topic changes detected by embedding similarity shifts).

Chunk size affects the retrieval-generation tradeoff. Small chunks (128-256 tokens) retrieve
precisely but may lack context. Large chunks (512-1024 tokens) provide more context but
retrieve noisily. A chunk size of 512 tokens with 50-token overlap is a common starting point.
        """.strip(),
    },
    {
        "id": "ragas-metrics",
        "title": "RAGAS — RAG Evaluation Framework",
        "content": """
RAGAS (Retrieval Augmented Generation Assessment) is an open-source framework for evaluating
RAG pipeline quality. It provides reference-free and reference-based metrics that score
different dimensions of RAG system performance using LLM-as-judge methodology.

Faithfulness measures whether the generated answer is factually grounded in the retrieved
context. A faithfulness score of 1.0 means every claim in the answer can be traced to the
retrieved context. Low faithfulness indicates the model is hallucinating — generating claims
not supported by evidence.

Context Recall measures whether the retrieved context contains all information needed to
answer the question. It requires ground truth answers. Low context recall means the retrieval
step is failing — the right information exists in the knowledge base but is not being found.

Answer Relevancy measures whether the generated answer directly addresses the user question.
A high answer relevancy score with low faithfulness is dangerous — the model is giving relevant
but hallucinated answers. Both scores must be high for a trustworthy system.

Context Precision measures whether the retrieved context is relevant and free of noise. High
context precision means the retriever is fetching focused, relevant chunks. Low precision means
irrelevant chunks are contaminating the context window.

RAGAS scores range from 0 to 1. Production systems should target faithfulness above 0.85,
context recall above 0.80, answer relevancy above 0.80, and context precision above 0.75.
Always measure all four metrics together — optimizing one can degrade another.
        """.strip(),
    },
    {
        "id": "llm-latency",
        "title": "LLM Latency — Measuring and Optimizing Response Time",
        "content": """
LLM latency is the time from sending a request to receiving the complete response. It has
two distinct components: Time to First Token (TTFT) — how long until the model starts
responding — and Time to Last Token (TTLT) — how long until the complete response is received.

For streaming applications, TTFT matters most for user experience. For batch processing,
TTLT determines throughput. Always measure both separately.

Latency should never be summarized as a mean alone. P50 (median) shows typical performance.
P95 shows what 95% of users experience. P99 shows the worst-case experience. The mean is
misleading because LLM latency distributions are right-skewed — a few slow requests pull the
mean far above the typical case.

Factors that drive latency: model size (larger models are slower), output token length (more
tokens = more time), context window size (more input tokens = slower prefill), provider
infrastructure (OpenAI vs Anthropic vs local), and network conditions.

Optimization strategies: use streaming to show partial responses immediately (reduces
perceived latency), cache frequent queries with semantic caching (Redis + embedding similarity),
use smaller models for simple queries via query routing, reduce context window by improving
retrieval precision, and run inference on dedicated hardware for latency-sensitive applications.

P99 latency SLAs for production AI systems vary by use case: interactive chat should target
under 3 seconds P99, background processing can tolerate 10-30 seconds, and batch jobs
are typically unbounded.
        """.strip(),
    },
    {
        "id": "llm-cost",
        "title": "LLM Cost Optimization and Token Economics",
        "content": """
LLM API costs are charged per token — a token is approximately 4 characters or 0.75 words.
Input tokens (prompt + context) and output tokens (generated response) are priced separately,
with output tokens typically costing 3-5x more than input tokens.

As of 2024-2025, representative pricing: GPT-4o costs $5/million input tokens and $15/million
output tokens. GPT-4o-mini costs $0.15/million input and $0.60/million output — approximately
33x cheaper. Claude Haiku 3 costs $0.25/million input and $1.25/million output. These prices
shift frequently; always check provider pricing pages for current rates.

Query routing is the highest-leverage cost optimization. Classify incoming queries by
complexity — simple factual lookups go to cheap small models, complex multi-hop reasoning
goes to expensive large models. In most production systems 70-80% of queries are simple
enough for a mini model with equivalent quality, reducing overall API spend by 50-70%.

Token budgeting strategies: limit retrieved context to only what is necessary (reduces input
tokens), instruct the model to be concise (reduces output tokens), cache responses for
repeated queries (eliminates API calls entirely), and batch process non-time-sensitive requests
(enables bulk pricing discounts on some providers).

Monthly cost projection formula: avg_cost_per_query × daily_query_volume × 30.
Always add a 20% buffer for traffic spikes. Monitor actual vs projected spend weekly and
set billing alerts at 80% of monthly budget.
        """.strip(),
    },
    {
        "id": "prompt-engineering",
        "title": "Prompt Engineering for Production RAG Systems",
        "content": """
Prompt engineering is the practice of designing and optimizing the instructions given to an
LLM to produce better outputs. In RAG systems, the system prompt controls how the model uses
retrieved context to generate answers.

A production RAG system prompt has four components: a role definition (who the model is),
context handling instructions (how to use retrieved documents), output format instructions
(how to structure the response), and uncertainty handling (what to do when context is
insufficient).

Effective context handling instructions should explicitly tell the model to answer ONLY from
the provided context, to cite which part of the context supports each claim, and to clearly
state when the context does not contain enough information rather than guessing.

Output format instructions improve downstream processing: specify response length bounds,
use structured formats (JSON, bullet points) for programmatic consumption, and specify
language and tone for user-facing applications.

Common prompt failure modes: the model ignores context and answers from training data
(hallucination), the model refuses valid questions due to overly strict instructions, the
model produces inconsistent output formats, and the model leaks system prompt contents when
prompted adversarially.

Prompt versioning is essential for production: treat prompts as code artifacts, store them
with version numbers and change descriptions, hash prompt content for deduplication, and
associate every benchmark run with the exact prompt version used. This makes it possible to
roll back a prompt regression like any other code regression.
        """.strip(),
    },
    {
        "id": "ai-agents",
        "title": "AI Agents — Architecture and Benchmarking",
        "content": """
An AI agent is a system where an LLM iteratively decides which actions to take, executes
those actions (via tools), observes the results, and repeats until it can produce a final
answer. Unlike single-call pipelines, agents can handle multi-step tasks that require
planning, tool use, and adaptive reasoning.

LangGraph is the leading framework for building structured agents in Python. It models agents
as directed graphs where nodes are steps (reasoning, tool calls, synthesis) and edges are
transitions. The explicit graph structure makes agents debuggable and measurable in ways that
unstructured agent loops are not.

Agent evaluation requires trajectory-level metrics beyond final answer quality. Trajectory
length is the number of steps taken — shorter is more efficient. Tool call accuracy is the
fraction of tool calls that succeed without errors. Trajectory efficiency is the ratio of
minimum possible steps to actual steps. Loop detection identifies when the agent calls the
same tool repeatedly without progress.

Common agent failure modes: wrong tool selection (agent picks an irrelevant tool and builds
answers on irrelevant output), reasoning loops (agent repeatedly calls the same tool in
slightly different ways without convergence), unnecessary steps (task could be solved in 2
steps but agent takes 6), and tool call formatting errors (incorrect parameter types or
missing required fields).

Loop prevention requires a hard step limit in the routing function. Set MAX_STEPS to 2x the
expected maximum steps for complex queries. When the limit is reached, force the agent to
synthesize from whatever information it has collected. Log all loop detections for analysis.
        """.strip(),
    },
    {
        "id": "regression-testing",
        "title": "Regression Testing and CI/CD for AI Systems",
        "content": """
Regression testing for AI systems verifies that code changes do not degrade performance
dimensions — quality scores, latency, cost, and safety. Unlike traditional software where
tests check deterministic correctness, AI regression tests check statistical distributions
against thresholds.

The four gates every AI pull request should pass: quality gate (RAGAS scores above floor),
latency gate (P99 below ceiling), cost gate (average cost per query below budget), and
safety gate (refusal rate above minimum, PII leakage below maximum). Safety gates are
non-negotiable — any safety regression blocks the merge regardless of other improvements.

Setting thresholds correctly prevents false alarms and missed regressions. Derive thresholds
from baseline measurements, not intuition. Measure score variation across multiple runs on
the same unchanged system. Set floors at baseline_score minus two standard deviations.
This threshold accommodates natural LLM non-determinism while catching genuine regressions.

GitHub Actions wires regression checks into the pull request workflow automatically.
The workflow runs on every PR to main: checkout code, install dependencies, run benchmark
subset, compare against thresholds, post results as PR comment, block merge if any gate fails.

Use two benchmark tiers: a fast tier (10-15 examples, 2-3 minutes) on every PR for obvious
regressions, and a full tier (complete eval dataset, 10-20 minutes) on merges to main for
subtle regressions. The fast tier catches most regressions without slowing development.
        """.strip(),
    },
    {
        "id": "production-observability",
        "title": "Production AI Observability — Monitoring and Drift Detection",
        "content": """
Production AI observability has three pillars: tracing (structured logs of every query),
metrics (aggregated numerical signals over time), and drift detection (statistical comparison
of current vs expected behavior). Benchmarking tells you how the system performed before
deployment. Observability tells you how it is performing after.

LangSmith is the standard tracing tool for LangChain-based systems. Every query produces a
trace with inputs, retrieved contexts, LLM calls, outputs, latency, and token counts.
Traces enable root cause analysis when production quality degrades — you can find the exact
query and exact retrieval result that caused a failure.

Prometheus collects time-series metrics via a scrape endpoint. AI-specific metrics to expose:
request latency histogram, quality score gauges (updated from periodic RAGAS sampling),
token usage counters, error rate by type, and active request count. Grafana visualizes
these metrics with configurable dashboards and alerts.

Query distribution drift occurs when production users ask different types of questions than
your evaluation dataset anticipated. The system appears healthy on benchmarks but silently
degrades on real traffic. Evidently AI detects this statistically by comparing feature
distributions of production queries against the reference distribution.

Reference-free quality monitoring enables production quality measurement without ground truth
labels. Faithfulness and answer relevancy can be scored on any production query without a
labeled correct answer. Run periodic RAGAS sampling on 50-100 production traces weekly and
track scores as time-series metrics. This catches silent quality degradation before users notice.
        """.strip(),
    },
]


def get_all_documents() -> list[dict]:
    return DOCUMENTS


def get_all_texts() -> list[str]:
    return [f"{doc['title']}\n\n{doc['content']}" for doc in DOCUMENTS]


def get_document_by_id(doc_id: str) -> dict | None:
    return next((d for d in DOCUMENTS if d["id"] == doc_id), None)
