# SHL Recommendation Agent

Overview
- Small FastAPI service that recommends SHL assessments from an internal catalog.
- Stateless per-request design: callers provide full conversation `history` in the request body.

Key components
- Retrieval pipeline: BM25 + semantic embeddings (FAISS) hybrid retriever with deterministic boosts.
- Orchestration: deterministic clarify / refuse / recommend logic before any LLM usage.
- LLM usage: Groq client used only for phrasing and lightweight reranking (when `GROQ_API_KEY` present).

Local setup
1. Create virtualenv and install:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Place `data/processed/catalog.json` in `data/processed/` (scraper produces it).
3. Run locally:
```bash
export PRELOAD_EMBEDDINGS=1  # optional
uvicorn src.shl_agent.api:app --host 0.0.0.0 --port 8000
```

API
- `GET /health` — returns {"status":"ok"}
- `POST /chat` — accepts `ChatRequest` JSON and returns `ChatResponse` (see `src/shl_agent/schemas/models.py`).

Deployment
- Dockerfile provided; entrypoint performs env validation. Set `GROQ_API_KEY` for LLM features. Optionally set `PRELOAD_EMBEDDINGS=true` to warm embeddings at startup.

Evaluation & Calibration
- See `src/shl_agent/evaluation/` for probes, calibration and benchmarks. `evaluation/retrieval_calibration.json` contains calibration sweep results.

Design tradeoffs
- Deterministic pre-LLM policies for safety and explainability; LLM used only for phrasing and lightweight reranking to avoid hallucination.
- Catalog completeness prioritized over aggressive ranking; incremental crawling recommended for scale.
SHL Assessment Recommendation Agent

Production-style FastAPI + retrieval + LLM orchestration skeleton for the SHL Labs internship assignment.

Phases are implemented incrementally. This repository focuses on modularity, grounding, and testability.

Quick start

1. Create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the dev server:

```bash
./scripts/run_dev.sh
```

Project layout

- `src/shl_agent/` — application package
- `src/shl_agent/api/app.py` — FastAPI entrypoint
- `src/shl_agent/schemas/` — Pydantic models and strict response schemas
- `src/shl_agent/ingestion/` — scraping & ingestion pipeline
- `src/shl_agent/retrieval/` — BM25 + embeddings + FAISS components
- `src/shl_agent/orchestration/` — controlled LLM orchestration
- `tests/` — unit tests and evaluation harness

Follow-up: run Phase 2 (Catalog scraping).
