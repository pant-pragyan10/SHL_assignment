from typing import Dict, Any, List
import json
from urllib.parse import urlparse


def load_catalog_urls(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        catalog = json.load(f)
    return [c.get('url') for c in catalog]


def hallucination_rate(resp: Dict[str, Any], catalog_urls: List[str]) -> float:
    recs = resp.get('recommendations') or []
    if not recs:
        return 0.0
    bad = 0
    for r in recs:
        url = r.get('url') or ''
        if url not in catalog_urls:
            bad += 1
        # domain check
        domain = urlparse(url).netloc
        if 'shl.com' not in domain:
            bad += 1
    return min(1.0, bad / len(recs))
