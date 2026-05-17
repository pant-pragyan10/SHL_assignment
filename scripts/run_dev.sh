#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
uvicorn shl_agent.api.app:app --reload --host 0.0.0.0 --port 8000