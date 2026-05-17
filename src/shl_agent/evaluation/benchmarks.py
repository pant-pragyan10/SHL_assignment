import time
import json
import os
from typing import List
from shl_agent.retrieval.retriever import HybridRetriever


def run_retrieval_bench(catalog_path: str, queries: List[str], out_path: str):
    with open(catalog_path, 'r', encoding='utf-8') as f:
        catalog = json.load(f)
    ret = HybridRetriever(catalog)
    results = []
    for q in queries:
        t0 = time.time()
        semantic = ret.retrieve(q, top_k=10, debug=True, mode='semantic')
        t_sem = time.time() - t0
        t0 = time.time()
        bm25 = ret.retrieve(q, top_k=10, debug=True, mode='bm25')
        t_bm = time.time() - t0
        t0 = time.time()
        hybrid = ret.retrieve(q, top_k=10, debug=True, mode='hybrid')
        t_h = time.time() - t0

        # assemble benchmark record
        rec = {
            'query': q,
            'semantic_top_k': [h['doc']['url'] for h in semantic],
            'bm25_top_k': [h['doc']['url'] for h in bm25],
            'hybrid_top_k': [h['doc']['url'] for h in hybrid],
            'semantic_time': t_sem,
            'bm25_time': t_bm,
            'hybrid_time': t_h,
            'hybrid_scores': [{'url': h['doc']['url'], 'score': h['score'], 'bm25_rank': h.get('bm25_rank'), 'faiss_rank': h.get('faiss_rank')} for h in hybrid],
        }
        results.append(rec)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'benchmarks': results}, f, indent=2)


if __name__ == '__main__':
    queries = [
        'cognitive ability test', 'situational judgement test', 'call center simulation',
        'personality questionnaire', 'numerical reasoning test'
    ]
    run_retrieval_bench('data/processed/catalog.json', queries, 'evaluation/retrieval_benchmarks.json')
