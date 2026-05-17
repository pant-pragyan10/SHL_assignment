"""Grounded comparison engine for two SHL assessments.

Compares purpose, skills, duration, ideal use cases, and assessment type
using only fields present in catalog entries. Produces a concise textual
comparison and a structured diff object.
"""
from typing import List, Dict, Any, Tuple


def _safe(field: Any) -> str:
    if not field:
        return "(not specified in catalog)"
    if isinstance(field, list):
        return ", ".join(map(str, field)) if field else "(not specified in catalog)"
    return str(field)


def compare_two(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Compare two catalog docs and return (reply_text, structured_comparison).

    The reply is concise and grounded; structured_comparison contains fields
    for programmatic use.
    """
    name_a = a.get("name") or "Assessment A"
    name_b = b.get("name") or "Assessment B"

    purpose_a = _safe(a.get("description"))
    purpose_b = _safe(b.get("description"))

    skills_a = _safe(a.get("tags"))
    skills_b = _safe(b.get("tags"))

    duration_a = _safe(a.get("duration"))
    duration_b = _safe(b.get("duration"))

    type_a = _safe(a.get("assessment_type") or a.get("test_type"))
    type_b = _safe(b.get("assessment_type") or b.get("test_type"))

    target_a = _safe(a.get("target_roles") or a.get("target") )
    target_b = _safe(b.get("target_roles") or b.get("target") )

    # Build concise comparison lines using only catalog-provided values
    lines = []
    lines.append(f"Comparison between '{name_a}' and '{name_b}':")
    lines.append(f"Purpose (catalog):\n - {name_a}: {purpose_a}\n - {name_b}: {purpose_b}")
    lines.append(f"Skills measured (catalog tags):\n - {name_a}: {skills_a}\n - {name_b}: {skills_b}")
    lines.append(f"Duration:\n - {name_a}: {duration_a}\n - {name_b}: {duration_b}")
    lines.append(f"Assessment type:\n - {name_a}: {type_a}\n - {name_b}: {type_b}")
    lines.append(f"Ideal use cases / target roles (catalog):\n - {name_a}: {target_a}\n - {name_b}: {target_b}")

    # Short recommendation: choose based on skills overlap and duration where possible
    short = []
    if skills_a != "(not specified in catalog)" and skills_b != "(not specified in catalog)":
        short.append("Choose the assessment whose measured skills best match your required skills.")
    if duration_a != "(not specified in catalog)" and duration_b != "(not specified in catalog)":
        short.append("Consider shorter duration if candidate experience must be quick.")

    if not short:
        short_text = "See the catalog details above to choose the best fit."
    else:
        short_text = " ".join(short)

    lines.append(short_text)

    reply = "\n".join(lines)

    structured = {
        "a": {"name": name_a, "url": a.get("url"), "purpose": purpose_a, "skills": skills_a, "duration": duration_a, "type": type_a, "target_roles": target_a},
        "b": {"name": name_b, "url": b.get("url"), "purpose": purpose_b, "skills": skills_b, "duration": duration_b, "type": type_b, "target_roles": target_b},
        "short_recommendation": short_text,
    }

    return reply, structured
