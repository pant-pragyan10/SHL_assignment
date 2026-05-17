from typing import List, Dict
import json


def recall_approximation(catalog_path: str, queries: List[str], retriever) -> Dict[str, Dict]:
    """Approximate Recall@10 by comparing against the union of methods (proxy gold).

    Returns per-query dict with semantic/bm25/hybrid sets and recall scores against union.
    """
    results = {}
    for q in queries:
        sem = retriever.retrieve(q, top_k=10, mode='semantic')
        bm = retriever.retrieve(q, top_k=10, mode='bm25')
        hy = retriever.retrieve(q, top_k=10, mode='hybrid')
        sem_set = {h['doc']['url'] for h in sem}
        bm_set = {h['doc']['url'] for h in bm}
        hy_set = {h['doc']['url'] for h in hy}
        union = sem_set | bm_set | hy_set
        def recall(s):
            if not union:
                return 0.0
            return len(s & union) / len(union)
        results[q] = {
            'semantic': {'top_k': list(sem_set), 'recall_proxy': recall(sem_set)},
            'bm25': {'top_k': list(bm_set), 'recall_proxy': recall(bm_set)},
            'hybrid': {'top_k': list(hy_set), 'recall_proxy': recall(hy_set)},
            'union_size': len(union)
        }
    return results
