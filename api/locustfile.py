"""
Locust load test — simulates concurrent users hitting the production API.
Run: locust -f api/locustfile.py --host http://localhost:8000
Then open: http://localhost:8089
"""

import random
from locust import HttpUser, between, task

SIMPLE = [
    "What is RAG?",
    "What is a vector database?",
    "What are embeddings?",
    "What is faithfulness in RAGAS?",
    "What is P99 latency?",
    "What is TTFT?",
    "What does context recall measure?",
]

MODERATE = [
    "How does cosine similarity work for text retrieval?",
    "Explain the difference between RAG and fine-tuning.",
    "How does query routing reduce LLM costs?",
    "What is prompt versioning and why does it matter?",
    "How do you prevent reasoning loops in LangGraph agents?",
]

COMPLEX = [
    "Analyze the tradeoffs between FAISS IVF and HNSW indexes for production RAG.",
    "Compare reference-free vs reference-based RAGAS metrics and when to use each.",
    "Design a cost optimization strategy for a RAG system handling 100k queries per day.",
    "Evaluate the tradeoffs between chunk size and retrieval quality in RAG systems.",
]


class RAGUser(HttpUser):
    wait_time = between(1, 3)

    @task(6)
    def simple_query(self):
        self._query(random.choice(SIMPLE))

    @task(3)
    def moderate_query(self):
        self._query(random.choice(MODERATE))

    @task(1)
    def complex_query(self):
        self._query(random.choice(COMPLEX))

    @task(2)
    def health_check(self):
        self.client.get("/health")

    def _query(self, text: str):
        with self.client.post(
            "/query",
            json={"query": text},
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("latency_ms", 0) > 10000:
                    response.failure(f"Too slow: {data['latency_ms']:.0f}ms")
                else:
                    response.success()
            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"HTTP {response.status_code}")
