"""Data cleaning and normalization utilities for catalog entries."""
from typing import List
import re

# lightweight stopword set for tag extraction
STOPWORDS = {
    "and",
    "or",
    "the",
    "a",
    "an",
    "for",
    "to",
    "with",
    "of",
    "in",
}


def normalize_text(text: str) -> str:
    """Normalize text for indexing and tag extraction.

    - Lowercases
    - Removes extra whitespace
    - Strips common punctuation
    """
    if not text:
        return ""
    text = text.lower()
    # remove punctuation except percent and +
    text = re.sub(r"[\,\.;:\'\"\(\)\[\]\{\}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_tags(text: str, max_tags: int = 20) -> List[str]:
    """Extract simple normalized tags from free text.

    This is intentionally simple and deterministic so downstream evaluation
    is reproducible without an LLM.
    """
    text = normalize_text(text)
    # split on non-word characters
    tokens = re.split(r"[^a-z0-9\+%]+", text)
    tokens = [t for t in tokens if t and t not in STOPWORDS and len(t) > 2]
    # frequency-preserving order: keep first occurrences
    seen = set()
    out = []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_tags:
            break
    return out
