"""BM25 keyword retriever using rank_bm25."""
from typing import List, Sequence, Dict
import logging

try:
    from rank_bm25 import BM25Okapi
except Exception as exc:  # pragma: no cover - runtime dependency
    raise ImportError("rank_bm25 is required. Install from requirements.txt") from exc

from shl_agent.ingestion.cleaning import normalize_text

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25 retriever built from catalog documents.

    Expects catalog to be a sequence of dict-like objects with accessible
    `text` content (string) or falls back to concatenating `name` and `description`.
    """

    def __init__(self, docs: Sequence[Dict]):
        self.docs = list(docs)
        self.corpus = [self._doc_to_text(d) for d in self.docs]
        self.tokenized = [self._tokenize(d) for d in self.corpus]
        if any(self.tokenized):
            self.bm25 = BM25Okapi(self.tokenized)
        else:
            logger.warning("BM25 initialized with empty tokenized corpus")
            self.bm25 = None

    def _doc_to_text(self, d: Dict) -> str:
        return normalize_text(" ".join(filter(None, [d.get("name", ""), d.get("description", ""), " ".join(d.get("tags", []))])))

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in text.split() if t]

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """Return top_k docs with BM25 scores.

        Outputs a list of dicts with keys: `index`, `doc`, `score`.
        """
        if not self.bm25:
            return []
        qtok = self._tokenize(normalize_text(query))
        scores = self.bm25.get_scores(qtok)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"index": idx, "doc": self.docs[idx], "score": float(score)} for idx, score in ranked]
