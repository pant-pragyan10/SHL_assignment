# Changelog

## v1.0-shl-submission (final)

- **Hybrid retrieval implementation**: BM25 + semantic embeddings with FAISS and reciprocal-rank fusion for robust candidate retrieval.
- **Deterministic orchestration**: Pre-LLM clarifications, refusal, and deterministic ranking boosts to avoid hallucination and ensure repeatable behavior.
- **Evaluation harness**: Behavioral probes, schema validation, and recall proxy metrics for automated verification.
- **Retrieval calibration**: Grid search calibration produced tuned weights and improved Recall@K.
- **Deployment cleanup**: Dockerfile, entrypoint, logging improvements, and `.env` safety handling.
- **Hallucination prevention**: Deterministic pre-checks and strict catalog-only recommendations enforced by the orchestrator.
- **Behavioral probes**: Automated probes to verify clarifications, refusals, and schema compliance.
