"""Format grounded conversational replies and strict schema responses."""
from typing import List, Dict, Any
from shl_agent.schemas.models import ChatRecommendation, ChatResponse


def make_reply_explanation(ranked: List[Dict[str, Any]], constraints: Dict[str, Any]) -> str:
    """Create a grounded, deterministic explanatory reply string.

    This avoids free-form hallucination by referencing fields from catalog entries.
    """
    if not ranked:
        return "I couldn't find suitable SHL assessments matching the provided constraints. Could you clarify the role or required skills?"

    lines = []
    lines.append("I recommend the following SHL assessments based on the catalog:")
    for i, doc in enumerate(ranked[:5], start=1):
        name = doc.get("name")
        url = doc.get("url")
        atype = doc.get("assessment_type") or doc.get("test_type") or "Assessment"
        reasons = []
        if constraints.get("role") and constraints.get("role").lower() in (name or "").lower():
            reasons.append("role match")
        if constraints.get("technical_skills"):
            overlap = set([s.lower() for s in constraints.get("technical_skills")]).intersection(set([t.lower() for t in doc.get("tags", [])]))
            if overlap:
                reasons.append("skills: " + ", ".join(sorted(overlap)))
        if doc.get("duration"):
            reasons.append(f"duration: {doc.get('duration')}")
        lines.append(f"{i}. {name} ({atype}) — {url} — {'; '.join(reasons) if reasons else 'catalog match'}")

    lines.append("Each recommendation is taken directly from the official SHL catalog and linked above.")
    return "\n".join(lines)


def format_chat_response(ranked: List[Dict[str, Any]]) -> ChatResponse:
    recs: List[ChatRecommendation] = []
    for doc in ranked[:10]:
        # ensure required fields
        name = doc.get("name")
        url = doc.get("url")
        test_type = doc.get("assessment_type") or doc.get("test_type") or ""
        if not name or not url:
            continue
        recs.append(ChatRecommendation(name=name, url=url, test_type=test_type))

    reply = make_reply_explanation(ranked, {})
    end = False
    return ChatResponse(reply=reply, recommendations=recs, end_of_conversation=end)
