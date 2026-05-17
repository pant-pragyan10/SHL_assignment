from typing import List, Dict, Optional, Sequence, Any
import numpy as np
import logging
import time
import re
from urllib.parse import urlparse

from .embedding import Embedder
from .bm25_retriever import BM25Retriever
from .faiss_index import FaissIndex

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(bm25_ranks: Dict[int, float], dense_ranks: Dict[int, float], k: int = 60) -> Dict[int, float]:
    """Compute Reciprocal Rank Fusion (RRF) combined scores.

    bm25_ranks and dense_ranks map doc index -> rank position (1-based). Returns combined score map.
    """
    scores = {}
    for idx, r in bm25_ranks.items():
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + r)
    for idx, r in dense_ranks.items():
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + r)
    return scores


class HybridRetriever:
    """Hybrid retriever combining BM25 and FAISS dense retrieval.

    Args:
        catalog: sequence of catalog records (dicts) containing at least `name`/`description`/`tags`.
        embedder: optional Embedder instance; if None, one will be created.
        bm25_weight: relative weight for BM25 in final fusion (informational; RRF used by default).
    """

    def __init__(
        self,
        catalog: Sequence[Dict[str, Any]],
        embedder: Optional[Embedder] = None,
        bm25_weight: float = 1.0,
    ) -> None:
        self.catalog = list(catalog)
        self.embedder = embedder or Embedder()
        self.bm25 = BM25Retriever(self.catalog)

        # Precompute embeddings for catalog
        texts = [" ".join(filter(None, [c.get("name", ""), c.get("description", ""), " ".join(c.get("tags", []))])) for c in self.catalog]
        embs = self.embedder.embed(texts)
        # normalize
        norm_embs = [self.embedder.normalize(e) for e in embs]
        dim = norm_embs[0].shape[0] if norm_embs else 0
        self.faiss = FaissIndex(dim=dim) if dim > 0 else None
        if self.faiss and norm_embs:
            self.faiss.build(norm_embs, [dict(c) for c in self.catalog])
        self._norm_embs = norm_embs
        self.bm25_weight = bm25_weight
        # tuning knobs
        self.mode = 'hybrid'  # options: 'hybrid', 'bm25', 'semantic'
        self.boost_config = {
            'skill_overlap_boost': 2.0,
            'specific_url_boost': 1.5,
            'metadata_completeness_boost': 1.2,
            'depth_boost_factor': 0.1,
        }

    def retrieve(self, query: str, constraints: Optional[Dict] = None, top_k: int = 10, debug: bool = False, mode: Optional[str] = None) -> List[Dict]:
        """Retrieve top candidates for `query` optionally applying `constraints`.

        Returns list of dicts with keys: `doc`, `score`, and optional debug fields:
        - `bm25_score`, `faiss_score`, `bm25_rank`, `faiss_rank`
        """
        constraints = constraints or {}
        mode = mode or self.mode

        # BM25 results
        t0 = time.time()
        bm25_hits = self.bm25.retrieve(query, top_k=top_k * 5)
        t_bm25 = time.time() - t0
        bm25_rank = {h["index"]: (i + 1) for i, h in enumerate(bm25_hits)}

        # Dense retrieval
        t0 = time.time()
        q_emb = self.embedder.embed([query])[0]
        q_emb = self.embedder.normalize(q_emb)
        faiss_hits = self.faiss.search(q_emb, top_k=top_k * 5) if self.faiss else []
        t_faiss = time.time() - t0
        faiss_rank = {h["index"]: (i + 1) for i, h in enumerate(faiss_hits)}

        # choose combination mode
        if mode == 'bm25':
            combined = {idx: 1.0 / (1 + r) for idx, r in bm25_rank.items()}
        elif mode == 'semantic':
            combined = {idx: 1.0 / (1 + r) for idx, r in faiss_rank.items()}
        else:
            combined = reciprocal_rank_fusion(bm25_rank, faiss_rank)

        # build scored list
        scored = []
        for idx, score in sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k * 3]:
            doc = self.catalog[idx]
            # apply simple constraint filtering
            if not self._satisfies_constraints(doc, constraints):
                continue
            # deterministic boosting heuristics
            boost = 1.0
            # skill overlap boost: if query tokens overlap tags/skills
            q_tokens = set([t.lower() for t in re.findall(r"\w+", query)])
            doc_skills = set([s.lower() for s in doc.get('skills') or []])
            overlap = len(q_tokens & doc_skills)
            if overlap:
                boost += self.boost_config['skill_overlap_boost'] * overlap
            # specific URL boost
            path = urlparse(doc.get('url','')).path
            depth = len([p for p in path.split('/') if p])
            if doc.get('is_specific_assessment'):
                boost += self.boost_config['specific_url_boost']
            # metadata completeness: count presence of duration/skills/type
            completeness = 0
            if doc.get('duration'):
                completeness += 1
            if doc.get('skills'):
                completeness += 1
            if doc.get('assessment_type') and doc.get('assessment_type') != 'Unknown':
                completeness += 1
            if completeness:
                boost += (completeness * self.boost_config['metadata_completeness_boost'])
            # depth boost
            boost += depth * self.boost_config['depth_boost_factor']

            item = {
                "doc": doc,
                "score": float(score) * float(boost),
                "bm25_rank": bm25_rank.get(idx),
                "faiss_rank": faiss_rank.get(idx),
                "bm25_time": t_bm25,
                "faiss_time": t_faiss,
            }
            if debug:
                item.update({
                    "bm25_score": next((h["score"] for h in bm25_hits if h["index"] == idx), 0.0),
                    "faiss_score": next((h["score"] for h in faiss_hits if h["index"] == idx), 0.0),
                })
            scored.append(item)

        # final sort by combined score
        scored = sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]
        # attach retrieval timings
        for s in scored:
            s.setdefault('retrieval_time', {}).update({'bm25': t_bm25, 'faiss': t_faiss})
        return scored

    def _satisfies_constraints(self, doc: Dict, constraints: Dict) -> bool:
        """Check document-level constraints (e.g., max_duration, remote_only, tags)."""
        if not constraints:
            return True
        # duration example: constraints may include max_minutes integer
        max_duration = constraints.get("max_minutes")
        if max_duration and doc.get("duration"):
            # try to extract first integer from duration
            import re

            m = re.search(r"(\d+)", str(doc.get("duration")))
            if m and int(m.group(1)) > int(max_duration):
                return False
        if constraints.get("remote_only"):
            if not doc.get("remote"):
                return False
        # tag constraints: must include all required tags
        required_tags = constraints.get("tags")
        if required_tags:
            doc_tags = set(doc.get("tags", []))
            if not set(required_tags).issubset(doc_tags):
                return False
        return True

    def evaluate_recall_at_k(self, queries: Sequence[str], gold_urls: Sequence[Sequence[str]], k: int = 10) -> float:
        """Compute Recall@k across a set of queries.

        Args:
            queries: sequence of query strings
            gold_urls: sequence of sequences of relevant document URLs corresponding to queries
        Returns:
            mean recall@k
        """
        if len(queries) != len(gold_urls):
            raise ValueError("queries and gold_urls must be same length")
        recalls = []
        for q, gold in zip(queries, gold_urls):
            hits = self.retrieve(q, top_k=k)
            hit_urls = {h["doc"].get("url") for h in hits}
            if not gold:
                recalls.append(0.0)
                continue
            recalls.append(sum(1 for g in gold if g in hit_urls) / len(gold))
        return float(np.mean(np.array(recalls)))

