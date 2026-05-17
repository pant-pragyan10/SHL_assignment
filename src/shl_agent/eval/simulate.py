"""Conversation simulation utilities for evaluation."""
from typing import List, Dict, Any


def make_history_from_turns(turns: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Ensure history items are in the expected format: {role, text}."""
    out = []
    for t in turns:
        role = t.get("role")
        text = t.get("text")
        if role and text:
            out.append({"role": role, "text": text})
    return out


def simulate_conversation(orchestrator: Any, retriever: Any, turns: List[Dict[str, str]]) -> Dict[str, Any]:
    """Run a deterministic simulation: feed full history to orchestrator and return decision."""
    history = make_history_from_turns(turns)
    decision = orchestrator.decide(history=history, retriever=retriever, top_k=5)
    return {"history": history, "decision": decision}
