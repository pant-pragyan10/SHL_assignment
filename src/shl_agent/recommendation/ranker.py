"""Ranking logic for recommendations using grounded features."""
from typing import List, Dict, Any, Optional
import math


def role_relevance_score(role: Optional[str], doc: Dict[str, Any]) -> float:
    if not role:
        return 0.0
    name = (doc.get("name") or "").lower()
    desc = (doc.get("description") or "").lower()
    r = role.lower()
    score = 0.0
    if r in name:
        score += 0.6
    if r in desc:
        score += 0.3
    return min(1.0, score)


def skills_overlap_score(required_skills: List[str], doc: Dict[str, Any]) -> float:
    if not required_skills:
        return 0.0
    doc_tags = set([t.lower() for t in doc.get("tags", [])])
    req = set([s.lower() for s in required_skills])
    if not req:
        return 0.0
    overlap = req.intersection(doc_tags)
    return float(len(overlap)) / float(len(req))


def seniority_score(seniority: Optional[str], doc: Dict[str, Any]) -> float:
    if not seniority:
        return 0.0
    desc = (doc.get("description") or "").lower()
    if seniority.lower() in desc:
        return 1.0
    return 0.0


def testing_need_score(constraints: Dict[str, Any], doc: Dict[str, Any]) -> float:
    # check cognitive and personality needs against assessment_type or tags
    score = 0.0
    assessment_type = (doc.get("assessment_type") or "").lower()
    tags = set([t.lower() for t in doc.get("tags", [])])
    cog = set([c.lower() for c in (constraints.get("cognitive_tests") or [])])
    pers = set([p.lower() for p in (constraints.get("personality_tests") or [])])
    # cognitive match
    if cog:
        if any(c in assessment_type for c in cog) or cog.intersection(tags):
            score += 0.6
    if pers:
        if "personality" in assessment_type or pers.intersection(tags):
            score += 0.6
    return min(1.0, score)


def aggregate_score(doc: Dict[str, Any], constraints: Dict[str, Any], weights: Dict[str, float]) -> float:
    s_role = role_relevance_score(constraints.get("role"), doc)
    s_skills = skills_overlap_score(constraints.get("technical_skills", []), doc)
    s_sen = seniority_score(constraints.get("seniority"), doc)
    s_test = testing_need_score(constraints, doc)
    score = (
        weights.get("role", 0.3) * s_role
        + weights.get("skills", 0.3) * s_skills
        + weights.get("seniority", 0.2) * s_sen
        + weights.get("testing", 0.2) * s_test
    )
    # small length penalty for very long descriptions
    desc_len = len((doc.get("description") or ""))
    penalty = 1.0 - min(0.1, desc_len / 10000.0)
    return float(max(0.0, min(1.0, score * penalty)))


def rank_candidates(candidates: List[Dict[str, Any]], constraints: Dict[str, Any], top_k: int = 10, weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """Score and rank candidate docs. Returns up to top_k items with added `score`.

    Each returned item is the original `doc` augmented with `score` and `confidence`.
    """
    if weights is None:
        weights = {"role": 0.3, "skills": 0.3, "seniority": 0.2, "testing": 0.2}
    scored = []
    for doc in candidates:
        sc = aggregate_score(doc, constraints, weights)
        # confidence is a monotonic transform of score
        confidence = sc
        dcopy = dict(doc)
        dcopy["score"] = sc
        dcopy["confidence"] = confidence
        scored.append(dcopy)
    scored = sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]
    return scored
