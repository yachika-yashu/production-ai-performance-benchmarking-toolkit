# Production AI Performance Benchmarking Toolkit

This repository contains a toolkit for benchmarking production-ready retrieval-augmented generation (RAG) systems, including quality, cost, regression, safety, observability, agent trajectory analysis, and dashboard logging.

## What it does

- Runs a full RAG benchmarking pipeline with baseline latency/cost measurements
- Evaluates generated answers using RAGAS quality metrics
- Performs A/B prompt experiments and prompt version comparisons
- Tracks query cost and routes traffic based on cost-aware rules
- Benchmarks agentic trajectories and tool-call efficiency
- Runs regression checks, safety benchmarks, and drift detection
- Exposes a production API and Prometheus-compatible metrics endpoint
- Logs benchmark runs into MLflow and provides dashboard support

## Repository layout

- `api/` — production FastAPI server for the RAG query endpoint
- `dashboard/` — static dashboard frontend
- `data/` — evaluation dataset and generated benchmark artifacts
- `scripts/` — launcher scripts such as the master benchmark runner
- `src/` — main Python implementation
  - `agents/` — agent trajectory and agent benchmarking logic
  - `benchmarking/` — baseline runner and benchmark orchestration
  - `cost/` — cost tracking and query routing support
  - `dashboard/` — dashboard API and benchmark logging
  - `experiments/` — A/B testing and prompt registry
  - `observability/` — drift detection and metrics exporter
  - `quality/` — RAGAS evaluation and scoring
  - `rag/` — RAG pipeline implementation and corpus handling
  - `regression/` — regression gate checks and safety benchmarking

## Prerequisites

- Python 3.11+ recommended
- OpenAI API key or supported LLM provider credentials
- `pip` for dependency installation

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Copy the environment template and add your API keys:

```powershell
copy .env.example .env
```

4. Update `.env` with your OpenAI key and any optional provider keys.

## Run the full benchmark

The main runner executes the complete benchmarking pipeline and logs results:

```powershell
python scripts/run_full_benchmark.py
```

Optional flags:

- `--fast` — execute a smaller dataset subset for quick runs
- `--dashboard` — start dashboard API only
- `--step N` — run a single benchmark step

## Run the API server

Start the production API for `/query`, `/health`, and `/metrics`:

```powershell
python api/app.py
```

Then visit:

- `http://127.0.0.1:8000/docs` for OpenAPI docs
- `http://127.0.0.1:8000/health` for health status
- `http://127.0.0.1:8000/query` to POST queries

## Benchmark insights

The benchmark runner includes:

- `baseline_runner` for latency and cost snapshots
- `ragas_evaluator` for faithfulness, relevancy, precision, and recall
- `ABTester` for prompt variant comparison
- `CostTracker` and `QueryRouter` for production cost modeling
- `RegressionChecker` for quality/latency/cost gates
- `DriftDetector` for query and quality drift analysis

## Notes

- The evaluation dataset is located at `data/eval_dataset.json`.
- MLflow integration is used for experiment logging and run tracking.
- Prometheus metrics are exposed via the observability module.

## License

This repository does not include a license file. Add one if you want to publish or share this code publicly.
