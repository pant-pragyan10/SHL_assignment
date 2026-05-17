#!/usr/bin/env python3
import sys
sys.path.insert(0,'src')
from shl_agent.evaluation.benchmarks import run_retrieval_bench

queries = [
    'cognitive ability test', 'situational judgement test', 'call center simulation',
    'personality questionnaire', 'numerical reasoning test'
]

run_retrieval_bench('data/processed/catalog.json', queries, 'evaluation/retrieval_benchmarks.json')
print('Wrote evaluation/retrieval_benchmarks.json')
