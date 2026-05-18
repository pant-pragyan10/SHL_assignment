#!/usr/bin/env bash
set -euo pipefail

echo "Starting entrypoint checks..."

# load .env if present
if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

# ensure Python can import package from /app/src in container runtime
export PYTHONPATH=/app/src

# required env vars (GROQ optional)
if [ -z "${PORT-}" ]; then
  export PORT=8000
fi

if [ -z "${GROQ_API_KEY-}" ]; then
  echo "WARNING: GROQ_API_KEY not set; LLM features disabled"
fi

if [ "${PRELOAD_EMBEDDINGS-}" = "1" ] || [ "${PRELOAD_EMBEDDINGS-}" = "true" ]; then
  echo "PRELOAD_EMBEDDINGS enabled: embeddings will be warmed on startup"
fi

echo "Entrypoint checks complete. Launching server..."

exec "$@"
