#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
python - <<'PY'
from shl_agent.eval.evaluator import Evaluator
import os, json

# load catalog if available
catalog_path = os.path.join(os.getcwd(), 'data', 'processed', 'catalog.json')
catalog = []
if os.path.exists(catalog_path):
    catalog = json.load(open(catalog_path))

evaler = Evaluator(retriever=None, orchestrator=None, catalog=catalog)
report = {'catalog_count': len(catalog)}
out = os.path.join('reports', 'eval_report.json')
os.makedirs(os.path.dirname(out), exist_ok=True)
evaler.generate_report(out, report)
print('Wrote', out)
PY