"""FAISS index wrapper for dense retrieval."""
from typing import List, Dict, Optional, Tuple
import numpy as np
import logging

try:
    import faiss
except Exception as exc:  # pragma: no cover - runtime dependency
    raise ImportError("faiss-cpu is required. Install from requirements.txt") from exc

logger = logging.getLogger(__name__)


class FaissIndex:
    """Simple FAISS wrapper that stores vectors and maps ids to documents.

    Uses an inner-product index on L2-normalized vectors to approximate cosine.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.id_to_doc: List[Dict] = []

    def build(self, vectors: List[np.ndarray], docs: List[Dict]) -> None:
        """Build index from a list of L2-normalized vectors and corresponding docs."""
        if len(vectors) != len(docs):
            raise ValueError("vectors and docs length mismatch")
        if vectors:
            arr = np.vstack(vectors).astype("float32")
            if arr.shape[1] != self.dim:
                raise ValueError(f"Vector dimension mismatch: expected {self.dim}, got {arr.shape[1]}")
            self.index.reset()
            self.index.add(arr)
            self.id_to_doc = list(docs)
        else:
            logger.warning("Building FAISS index with empty vectors")

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Dict]:
        """Search the index with a single L2-normalized query vector.

        Returns list of dicts: `{index, doc, score}` where score is inner-product.
        """
        if self.index.ntotal == 0:
            return []
        q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        D, I = self.index.search(q, top_k)
        D = D[0]
        I = I[0]
        out = []
        for score, idx in zip(D.tolist(), I.tolist()):
            if idx < 0:
                continue
            out.append({"index": idx, "doc": self.id_to_doc[idx], "score": float(score)})
        return out
