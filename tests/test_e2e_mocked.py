import sys
sys.path.insert(0, 'src')
import pytest

from shl_agent.orchestration.agent import Orchestrator
from shl_agent.retrieval.retriever import HybridRetriever
from shl_agent.schemas.models import ChatRequest, ChatMessage


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query, constraints=None, top_k=5, debug=False):
        # return docs with a simple score
        out = []
        for i, d in enumerate(self.docs):
            out.append({"doc": d, "score": 1.0 / (i + 1)})
        return out


class FakeGroq:
    def rerank_candidates(self, candidates, constraints):
        # assign higher score to first candidate, but ensure only provided URLs returned
        return {c['doc']['url']: 1.0 - (i * 0.1) for i, c in enumerate(candidates)}


def test_vague_returns_clarify():
    orch = Orchestrator()
    history = [{"role": "user", "text": "I need an assessment"}]
    res = orch.decide(history, retriever=None, top_k=5)
    assert res["action"] == "clarify"


def test_rerank_does_not_hallucinate():
    docs = [{"name": "A", "url": "https://shl/a", "description": "x", "tags": []}]
    retr = FakeRetriever(docs)
    groq = FakeGroq()
    orch = Orchestrator(llm_client=groq)
    history = [{"role": "user", "text": "Hire a developer"}]
    res = orch.decide(history, retriever=retr, top_k=5)
    assert res["action"] in ("recommend", "clarify")
    if res["action"] == "recommend":
        urls = [d.get('url') for d in res['recommendations']]
        assert all(u in [doc['url'] for doc in docs] for u in urls)


def test_schema_compliance_and_grounding():
    # simulate API flow: rank + format
    docs = [{"name": "A", "url": "https://shl/a", "description": "x", "tags": []}]
    ranked = docs
    from shl_agent.recommendation.formatter import format_chat_response
    resp = format_chat_response(ranked)
    # validate required fields exist
    assert hasattr(resp, 'reply')
    assert isinstance(resp.recommendations, list)
