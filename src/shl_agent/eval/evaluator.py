"""Evaluator for Recall@K, relevance, hallucination checks and reports."""
from typing import List, Dict, Any, Tuple
import json
import os
import statistics
from pydantic import ValidationError

from shl_agent.schemas.models import ChatResponse, ChatRecommendation, ChatRequest, ChatMessage


class Evaluator:
    """Evaluation harness to run retrieval and behavioral tests.

    The evaluator is designed to be data-driven: pass in test cases and a
    retriever + orchestrator to exercise the system and produce metrics.
    """

    def __init__(self, retriever: Any = None, orchestrator: Any = None, catalog: List[Dict[str, Any]] = None):
        self.retriever = retriever
        self.orchestrator = orchestrator
        self.catalog = catalog or []

    def recall_at_k(self, queries: List[str], gold_urls: List[List[str]], k: int = 10) -> float:
        if not self.retriever:
            raise RuntimeError("retriever not configured")
        return self.retriever.evaluate_recall_at_k(queries, gold_urls, k=k)

    def validate_schema(self, response: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            ChatResponse(**response)
            return True, "valid"
        except ValidationError as exc:
            return False, str(exc)

    def detect_hallucination(self, response: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Detect hallucinated recommendations or unsupported claims.

        Strategy:
        - Ensure every recommendation URL exists in the catalog
        - Ensure reply text only references values present in recommended docs
        Returns (is_clean, problems)
        """
        problems: List[str] = []
        recs = response.get("recommendations", [])
        catalog_urls = {c.get("url") for c in self.catalog}
        for r in recs:
            if r.get("url") not in catalog_urls:
                problems.append(f"Recommendation URL not in catalog: {r.get('url')}")

        # Simple claim check: every sentence in reply must contain at least one token present in some recommended doc
        reply = response.get("reply", "")
        sentences = [s.strip() for s in reply.split("\n") if s.strip()]
        rec_text = " ".join([json.dumps(d) for d in recs])
        for s in sentences:
            # if sentence contains a URL or name, allow; else check overlap with rec_text
            words = [w.strip(".,") for w in s.split()][:10]
            if not any(w.lower() in rec_text.lower() for w in words):
                problems.append(f"Possibly unsupported sentence: {s}")
        return (len(problems) == 0), problems

    def run_behavioral_probes(self, probes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run a set of conversational probes (simulated histories) and collect results.

        Each probe is a dict: {"name": str, "history": [ {role,text} ] }
        """
        if not self.orchestrator:
            raise RuntimeError("orchestrator not configured")
        results = []
        for p in probes:
            history = p.get("history", [])
            decision = self.orchestrator.decide(history=history, retriever=self.retriever, top_k=5)
            results.append({"name": p.get("name"), "decision": decision})
        return {"results": results}

    def schema_validation_test(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        ok = 0
        failures = []
        for r in responses:
            valid, msg = self.validate_schema(r)
            if valid:
                ok += 1
            else:
                failures.append({"response": r, "error": msg})
        return {"total": len(responses), "valid": ok, "invalid": len(responses) - ok, "failures": failures}

    def retrieval_diagnostics(self, queries: List[str], top_k: int = 10) -> Dict[str, Any]:
        if not self.retriever:
            raise RuntimeError("retriever not configured")
        stats = {"queries": len(queries), "per_query": []}
        for q in queries:
            hits = self.retriever.retrieve(q, top_k=top_k, debug=True)
            per = {"query": q, "num_hits": len(hits), "top_scores": [h.get("score") for h in hits[:5]]}
            stats["per_query"].append(per)
        # aggregate
        all_counts = [p["num_hits"] for p in stats["per_query"]]
        stats["avg_num_hits"] = statistics.mean(all_counts) if all_counts else 0
        return stats

    def generate_report(self, out_path: str, results: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
