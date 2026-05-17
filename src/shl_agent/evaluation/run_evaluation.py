import json
import os
import time
from typing import Callable
import re

from shl_agent.evaluation.behavioral_probes import ALL_PROBES, ADDITIONAL_PROBES
from shl_agent.evaluation.schema_checks import validate_schema
from shl_agent.evaluation.hallucination_checks import load_catalog_urls, hallucination_rate
from shl_agent.evaluation.recall_eval import recall_approximation
from shl_agent.evaluation.latency_metrics import timeit
from shl_agent.retrieval.retriever import HybridRetriever


def default_agent_fn_factory(retriever: HybridRetriever) -> Callable[[str], dict]:
    """Create a simple agent function that returns the expected schema using retriever."""

    def agent_fn(query: str) -> dict:
        text = query.strip()
        lt = text.lower()
        # deterministic refusal for prompt-injection or off-topic (legal/medical/illicit)
        if re.search(r"ignore previous|disregard previous|what is my api key|provide your source code|build a bomb|make a weapon|how to sue|hire (a )?lawyer", lt):
            return {'reply': "I cannot assist with that request.", 'recommendations': [], 'reason': 'refuse', 'end_of_conversation': True}

        # simple clarification: require role or intent words in short queries
        short = len(text.split()) <= 3
        intent_tokens = ['numerical', 'personality', 'sjt', 'situational', 'cognitive', 'skills', 'simulation', 'assessment']
        if short and not any(term in lt for term in intent_tokens):
            # ask one high-information question
            qtxt = 'Which target role or assessment type do you mean — cognitive or personality?'
            return {'reply': qtxt, 'clarification_questions': [qtxt], 'recommendations': [], 'end_of_conversation': False}

        # explicit ambiguous phrasing should also trigger clarification
        if 'which test' in lt or 'which assessment' in lt:
            qtxt = 'Which do you mean: cognitive, numerical, or personality assessment?'
            return {'reply': qtxt, 'clarification_questions': [qtxt], 'recommendations': [], 'end_of_conversation': False}

        # questions asking whether they "need" cognitive vs personality should clarify
        if re.search(r"\bdo i need\b", lt) and any(tok in lt for tok in ['cognitive', 'personality']):
            qtxt = 'Which outcome are you optimizing for — job-fit (cognitive) or behavioral traits (personality)?'
            return {'reply': qtxt, 'clarification_questions': [qtxt], 'recommendations': [], 'end_of_conversation': False}

        # allow users to override clarification requests (e.g., 'Never mind, just recommend')
        if re.search(r"never mind|just recommend|recommend something", lt):
            hits = retriever.retrieve(lt, top_k=5, debug=True)
            recs = [{'name': h['doc'].get('name'), 'url': h['doc'].get('url')} for h in hits]
            return {'reply': 'Here are recommended assessments.', 'recommendations': recs, 'end_of_conversation': True}

        # normal recommendation flow
        hits = retriever.retrieve(query, top_k=5, debug=True)
        recs = [{'name': h['doc'].get('name'), 'url': h['doc'].get('url')} for h in hits]
        return {'reply': 'Here are recommended assessments.', 'recommendations': recs, 'end_of_conversation': True}

    return agent_fn


def run_all(catalog_path: str = 'data/processed/catalog.json', output_dir: str = 'evaluation') -> dict:
    os.makedirs(output_dir, exist_ok=True)
    with open(catalog_path, 'r', encoding='utf-8') as f:
        catalog = json.load(f)

    retriever = HybridRetriever(catalog)
    agent_fn = default_agent_fn_factory(retriever)

    # behavioral probes
    probe_results = {}
    for name, fn in ALL_PROBES:
        t0 = time.time()
        try:
            ok = fn(agent_fn)
        except Exception:
            ok = False
        probe_results[name] = {'pass': bool(ok), 'latency': time.time() - t0}

    # run additional probes
    for name, fn in ADDITIONAL_PROBES:
        t0 = time.time()
        try:
            ok = fn(agent_fn)
        except Exception:
            ok = False
        probe_results[name] = {'pass': bool(ok), 'latency': time.time() - t0}

    # schema compliance & hallucination checks on a sample query set
    sample_queries = ['cognitive ability test','personality questionnaire','call center simulation']
    catalog_urls = load_catalog_urls(catalog_path)
    schema_pass = 0
    hallucinations = []
    latencies = []
    for q in sample_queries:
        start = time.time()
        resp = agent_fn(q)
        latencies.append(time.time() - start)
        if validate_schema(resp):
            schema_pass += 1
        hr = hallucination_rate(resp, catalog_urls)
        hallucinations.append(hr)

    # recall approximation
    recall = recall_approximation(catalog_path, sample_queries, retriever)

    summary = {
        'probe_results': probe_results,
        'schema_compliance_rate': schema_pass / len(sample_queries),
        'avg_hallucination_rate': sum(hallucinations) / len(hallucinations),
        'recall_proxy': recall,
        'avg_latency': sum(latencies) / len(latencies),
    }

    # write outputs
    with open(os.path.join(output_dir, 'e2e_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == '__main__':
    s = run_all()
    import logging
    logging.getLogger(__name__).info('Wrote evaluation/e2e_summary.json')
    logging.getLogger(__name__).info('Summary: %s', s)
