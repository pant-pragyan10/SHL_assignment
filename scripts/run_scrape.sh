#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
python -m shl_agent.ingestion.runner --out data/processed/catalog.json