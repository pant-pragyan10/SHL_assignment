"""Additional diagnostics for retrieval and ranking."""
from typing import List, Dict, Any


def tag_coverage(catalog: List[Dict[str, Any]]) -> Dict[str, Any]:
    tag_counts = {}
    for d in catalog:
        for t in d.get("tags", []):
            tag_counts[t] = tag_counts.get(t, 0) + 1
    total = len(catalog)
    return {"total_docs": total, "unique_tags": len(tag_counts), "tag_counts": tag_counts}


def top_tag_stats(catalog: List[Dict[str, Any]], top_n: int = 20) -> List[tuple]:
    tc = tag_coverage(catalog)["tag_counts"]
    items = sorted(tc.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return items
