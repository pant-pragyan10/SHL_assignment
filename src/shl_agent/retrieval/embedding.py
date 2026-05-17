"""Embedding utilities using sentence-transformers."""
from typing import Sequence, List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover - runtime dependency
    raise ImportError("sentence-transformers is required. Install from requirements.txt") from exc


class Embedder:
    """Wrapper around SentenceTransformer for batched embedding generation.

    Args:
        model_name: model id for SentenceTransformer (e.g. 'all-MiniLM-L6-v2')
        device: device string passed to SentenceTransformer
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu") -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: Sequence[str], batch_size: int = 32) -> List[np.ndarray]:
        """Convert a sequence of texts into embeddings.

        Returns a list of numpy arrays (one per text).
        """
        if not texts:
            return []
        embs = self.model.encode(list(texts), batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        # ensure 2D numpy
        if embs.ndim == 1:
            embs = np.expand_dims(embs, 0)
        return [e for e in embs]

    @staticmethod
    def normalize(emb: np.ndarray) -> np.ndarray:
        """L2-normalize a vector for cosine-similarity with inner-product FAISS indexes."""
        norm = np.linalg.norm(emb)
        if norm == 0:
            return emb
        return emb / norm
