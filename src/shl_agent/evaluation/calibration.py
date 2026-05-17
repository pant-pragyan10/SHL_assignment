import json
import os
import time
import itertools
from typing import List, Dict, Any, Tuple
import math

from shl_agent.retrieval.retriever import HybridRetriever


DEFAULT_QUERIES = None


def load_benchmark_queries(path: str = 'evaluation/retrieval_benchmarks.json') -> List[str]:
    if not os.path.exists(path):
        # fallback small set
        return ['cognitive ability test', 'personality questionnaire', 'call center simulation']
    with open(path, 'r', encoding='utf-8') as f:
        j = json.load(f)
    return [b['query'] for b in j.get('benchmarks', [])]


def _compute_boost(query: str, doc: Dict[str, Any], boost_config: Dict[str, float]) -> float:
    import re
    from urllib.parse import urlparse

    boost = 1.0
    q_tokens = set([t.lower() for t in re.findall(r"\w+", query)])
    doc_skills = set([s.lower() for s in doc.get('skills') or []])
    overlap = len(q_tokens & doc_skills)
    if overlap:
        boost += boost_config.get('skill_overlap_boost', 0.0) * overlap
    path = urlparse(doc.get('url', '')).path
    depth = len([p for p in path.split('/') if p])
    if doc.get('is_specific_assessment'):
        boost += boost_config.get('specific_url_boost', 0.0)
    completeness = 0
    if doc.get('duration'):
        completeness += 1
    if doc.get('skills'):
        completeness += 1
    if doc.get('assessment_type') and doc.get('assessment_type') != 'Unknown':
        completeness += 1
    if completeness:
        boost += completeness * boost_config.get('metadata_completeness_boost', 0.0)
    boost += depth * boost_config.get('depth_boost_factor', 0.0)
    return boost


def _rank_to_score(rank: int) -> float:
    # convert rank (1-based) to score in (0,1]
    return 1.0 / (1 + rank) if rank and rank > 0 else 0.0


def run_grid(catalog: List[Dict[str, Any]], queries: List[str], output_path: str = 'evaluation/retrieval_calibration.json') -> Dict:
    retriever = HybridRetriever(catalog)

    # grid definitions (kept modest)
    bm25_weights = [0.5, 1.0, 2.0]
    semantic_weights = [0.5, 1.0, 2.0]
    skill_boosts = [1.0, 2.0]
    metadata_boosts = [0.5, 1.0]
    specific_boosts = [0.5, 1.0, 1.5]
    depth_boosts = [0.0, 0.1]

    top_k = 10

    configs_results = []
    total = 0
    # iterate
    for bm_w, sem_w, skill_b, meta_b, spec_b, depth_b in itertools.product(bm25_weights, semantic_weights, skill_boosts, metadata_boosts, specific_boosts, depth_boosts):
        total += 1
    idx = 0
    for bm_w, sem_w, skill_b, meta_b, spec_b, depth_b in itertools.product(bm25_weights, semantic_weights, skill_boosts, metadata_boosts, specific_boosts, depth_boosts):
        idx += 1
        cfg = {
            'bm25_weight': bm_w,
            'semantic_weight': sem_w,
            'skill_overlap_boost': skill_b,
            'metadata_completeness_boost': meta_b,
            'specific_url_boost': spec_b,
            'depth_boost_factor': depth_b,
        }
        retriever.boost_config.update({
            'skill_overlap_boost': cfg['skill_overlap_boost'],
            'metadata_completeness_boost': cfg['metadata_completeness_boost'],
            'specific_url_boost': cfg['specific_url_boost'],
            'depth_boost_factor': cfg['depth_boost_factor'],
        })

        # collect per-mode metrics
        mode_metrics = {}
        mode_names = ['bm25', 'semantic', 'hybrid']
        per_query_results = {m: [] for m in mode_names}

        bm25_times = []
        sem_times = []

        for q in queries:
            # BM25 raw
            bm_hits = retriever.bm25.retrieve(q, top_k=top_k * 5)
            bm_rank = {h['index']: i + 1 for i, h in enumerate(bm_hits)}
            # faiss
            q_emb = retriever.embedder.embed([q])[0]
            q_emb = retriever.embedder.normalize(q_emb)
            faiss_hits = retriever.faiss.search(q_emb, top_k=top_k * 5) if retriever.faiss else []
            faiss_rank = {h['index']: i + 1 for i, h in enumerate(faiss_hits)}

            # build scored lists per mode
            def build_mode(mode: str) -> List[Tuple[str, float, float, bool]]:
                # returns list of tuples (url, score, retrieval_time, is_specific)
                scored = []
                indices = set(list(bm_rank.keys()) + list(faiss_rank.keys()))
                for idx_doc in indices:
                    doc = retriever.catalog[idx_doc]
                    b_rank = bm_rank.get(idx_doc)
                    f_rank = faiss_rank.get(idx_doc)
                    b_score = _rank_to_score(b_rank)
                    f_score = _rank_to_score(f_rank)
                    if mode == 'bm25':
                        base = b_score
                    elif mode == 'semantic':
                        base = f_score
                    else:
                        base = bm_w * b_score + sem_w * f_score
                    boost = _compute_boost(q, doc, retriever.boost_config)
                    final = base * boost
                    is_spec = bool(doc.get('is_specific_assessment'))
                    scored.append((doc.get('url'), final, is_spec, b_rank or 0, f_rank or 0))
                scored = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
                return scored

            bm_top = build_mode('bm25')
            sem_top = build_mode('semantic')
            hyb_top = build_mode('hybrid')

            # compute times (approx using retriever timings on retrieval of query)
            # use bm25_hits time and faiss search time by performing a small call
            try:
                t0 = time.time()
                _ = retriever.bm25.retrieve(q, top_k=1)
                bm25_times.append(time.time() - t0)
            except Exception:
                bm25_times.append(0.0)
            try:
                t0 = time.time()
                _ = retriever.faiss.search(retriever.embedder.normalize(retriever.embedder.embed([q])[0]), top_k=1)
                sem_times.append(time.time() - t0)
            except Exception:
                sem_times.append(0.0)

            per_query_results['bm25'].append([u for u, *_ in bm_top])
            per_query_results['semantic'].append([u for u, *_ in sem_top])
            per_query_results['hybrid'].append([u for u, *_ in hyb_top])

        # compute aggregate metrics
        def aggregate(mode: str) -> Dict[str, Any]:
            tops = per_query_results[mode]
            total_items = len(queries) * top_k
            unique = len(set([u for lst in tops for u in lst]))
            diversity = unique / total_items if total_items else 0.0
            # category-page frequency (non-specific)
            non_specific = 0
            specific = 0
            for lst in tops:
                for u in lst:
                    # find doc
                    doc = next((d for d in retriever.catalog if d.get('url') == u), {})
                    if doc.get('is_specific_assessment'):
                        specific += 1
                    else:
                        non_specific += 1
            cat_freq = non_specific / (specific + non_specific) if (specific + non_specific) else 0.0
            topk_specificity = specific / (specific + non_specific) if (specific + non_specific) else 0.0
            # recall proxy: union of bm25 & semantic for this config
            union_sets = []
            for i, q in enumerate(queries):
                sset = set(per_query_results['bm25'][i]) | set(per_query_results['semantic'][i])
                union_sets.append(sset)
            recalls = []
            for i, q in enumerate(queries):
                target = union_sets[i]
                if not target:
                    recalls.append(0.0)
                    continue
                hits = set(per_query_results[mode][i])
                recalls.append(len(hits & target) / len(target))
            avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
            avg_latency = (sum(bm25_times) / len(bm25_times) if bm25_times else 0.0) + (sum(sem_times) / len(sem_times) if sem_times else 0.0)
            return {
                'avg_recall_proxy': avg_recall,
                'diversity': diversity,
                'category_page_frequency': cat_freq,
                'avg_latency': avg_latency,
                'topk_specificity': topk_specificity,
            }

        for m in mode_names:
            mode_metrics[m] = aggregate(m)

        configs_results.append({'config': cfg, 'metrics': mode_metrics})

    # choose best config by hybrid avg_recall_proxy
    best = max(configs_results, key=lambda x: x['metrics']['hybrid']['avg_recall_proxy']) if configs_results else None

    out = {
        'tested_configurations': configs_results,
        'best_configuration': best,
    }
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    return out


def rerun_evaluation_with_best(catalog: List[Dict[str, Any]], best_cfg: Dict[str, Any]) -> Dict[str, Any]:
    # create retriever and set boosts
    from shl_agent.evaluation.run_evaluation import ALL_PROBES, validate_schema, load_catalog_urls, hallucination_rate
    from shl_agent.evaluation.run_evaluation import default_agent_fn_factory
    # reuse default factory but with retriever set
    retriever = HybridRetriever(catalog)
    if best_cfg:
        retriever.boost_config.update({
            'skill_overlap_boost': best_cfg['skill_overlap_boost'],
            'metadata_completeness_boost': best_cfg['metadata_completeness_boost'],
            'specific_url_boost': best_cfg['specific_url_boost'],
            'depth_boost_factor': best_cfg['depth_boost_factor'],
        })

    agent_fn = default_agent_fn_factory(retriever)

    # run probes
    probe_results = {}
    for name, fn in ALL_PROBES:
        t0 = time.time()
        try:
            ok = fn(agent_fn)
        except Exception:
            ok = False
        probe_results[name] = {'pass': bool(ok), 'latency': time.time() - t0}

    sample_queries = ['cognitive ability test','personality questionnaire','call center simulation']
    catalog_urls = [d.get('url') for d in catalog]
    schema_pass = 0
    hallucinations = []
    latencies = []
    from shl_agent.evaluation.schema_checks import validate_schema as schema_validate
    from shl_agent.evaluation.hallucination_checks import hallucination_rate as hr_fn
    for q in sample_queries:
        start = time.time()
        resp = agent_fn(q)
        latencies.append(time.time() - start)
        if schema_validate(resp):
            schema_pass += 1
        hallucinations.append(hr_fn(resp, catalog_urls))

    summary = {
        'probe_results': probe_results,
        'schema_compliance_rate': schema_pass / len(sample_queries),
        'avg_hallucination_rate': sum(hallucinations) / len(hallucinations),
        'avg_latency': sum(latencies) / len(latencies),
    }

    with open('evaluation/e2e_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == '__main__':
    queries = load_benchmark_queries()
    with open('data/processed/catalog.json', 'r', encoding='utf-8') as f:
        catalog = json.load(f)
    print('Running calibration grid on', len(queries), 'queries and', len(catalog), 'documents')
    out = run_grid(catalog, queries)
    import logging
    log = logging.getLogger(__name__)
    log.info('Wrote evaluation/retrieval_calibration.json')
    best = out.get('best_configuration')
    best_cfg = best['config'] if best else None
    log.info('Best config: %s', best_cfg)
    log.info('Re-running evaluation harness with best config...')
    rerun_evaluation_with_best(catalog, best_cfg)
    log.info('Wrote evaluation/e2e_summary.json')
