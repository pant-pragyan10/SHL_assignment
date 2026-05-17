from typing import Dict, Any, List, Optional
import math
import logging
import re

from shl_agent.conversation.analyzer import ConversationAnalyzer
from shl_agent.schemas.models import ConversationConstraints

logger = logging.getLogger(__name__)


class Orchestrator:
    """Deterministic orchestration layer for conversational decisions.

    This component uses explicit control flow to decide whether to ask
    clarifying questions, recommend assessments, compare candidates, or
    refuse requests. It is intentionally LLM-agnostic by default; an external
    `llm_client` can be provided for phrasing or re-ranking but core logic
    remains deterministic and testable.
    """

    def __init__(self, llm_client: Optional[Any] = None, max_turns: int = 8, ambiguity_threshold: float = 0.45) -> None:
        self.llm = llm_client
        self.max_turns = max_turns
        self.ambiguity_threshold = ambiguity_threshold
        self.analyzer = ConversationAnalyzer()

    def decide(
        self,
        history: List[Dict[str, Any]],
        retriever: Optional[Any] = None,
        last_user_turn_index: Optional[int] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Make a conversational decision based on full message `history`.

        Args:
            history: full message list, each item must include `role` ('user'|'assistant') and `text`.
            retriever: optional retrieval component with `.retrieve(query, constraints, top_k)`.
            last_user_turn_index: index of the last user turn in history (optional).
            top_k: number of recommendations to produce when recommending.

        Returns a dict with keys:
            - `action`: one of 'clarify', 'recommend', 'compare', 'refuse'
            - `clarification_questions`: list[str]
            - `recommendations`: list[dict]
            - `confidence`: float (0..1)
            - `ambiguity_score`: float (0..1)
            - `reason`: textual brief reason
        """
        if not history:
            return self._make_refuse("empty history")

        # reconstruct last user message
        last_user_text = self._get_last_user_text(history)

        # stateless constraint extraction
        constraints = self.analyzer.analyze(history)

        # deterministic safety checks (run before any LLM usage)
        if self._detect_prompt_injection(last_user_text):
            return self._make_refuse("prompt-injection detected")
        if self._detect_off_topic(last_user_text):
            return self._make_refuse("off-topic request detected")

        # ambiguity scoring (0 = clear, 1 = very ambiguous)
        ambiguity = self._ambiguity_score(constraints, last_user_text)

        # Enforce clarification policy: require either a target role or explicit assessment intent
        has_intent = bool(constraints.cognitive_tests or constraints.personality_tests or (constraints.technical_skills and len(constraints.technical_skills) > 0))
        if not constraints.role and not has_intent:
            # treat as ambiguous regardless of numeric ambiguity score
            ambiguity = max(ambiguity, 1.0)

        # detect explicit comparison requests
        if self._is_comparison_request(last_user_text):
            # require at least two candidate references to compare
            if retriever is None:
                return self._make_refuse("no retriever available for comparison")
            hits = retriever.retrieve(last_user_text, constraints=constraints.dict(), top_k=top_k)
            return {
                "action": "compare",
                "clarification_questions": [],
                "recommendations": [h["doc"] for h in hits],
                "confidence": self._recommendation_confidence(hits, constraints),
                "ambiguity_score": ambiguity,
                "reason": "user asked to compare",
            }

        # If ambiguous and we have turns left, ask clarifying questions
        turns_so_far = sum(1 for t in history if t.get("role") in ("user", "assistant"))
        turns_left = max(0, self.max_turns - turns_so_far)
        if ambiguity >= self.ambiguity_threshold and turns_left >= 1:
            # ask ONE high-information clarification question only
            questions = self._build_clarification_questions(constraints, last_user_text, max_questions=1)
            return {
                "action": "clarify",
                "clarification_questions": questions,
                "recommendations": [],
                "confidence": 0.0,
                "ambiguity_score": ambiguity,
                "reason": "ambiguous request; asking one high-info clarification",
            }

        # otherwise attempt to recommend
        if retriever is None:
            return self._make_refuse("retriever not provided")

        hits = retriever.retrieve(last_user_text, constraints=constraints.__dict__, top_k=top_k, debug=True)
        recs = [h["doc"] for h in hits]

        # If an LLM client is available, request a lightweight rerank on the top candidates.
        if self.llm and hits:
            try:
                # Groq client expects the hits list in the same structure
                rerank_map = None
                if hasattr(self.llm, 'rerank_candidates'):
                    # pass query + candidates + constraints for a standardized API
                    rerank_map = self.llm.rerank_candidates(query=last_user_text, candidates=hits, constraints=constraints.__dict__)
                # apply rerank if valid
                if rerank_map:
                    logger.debug("LLM rerank map received: %s", rerank_map)
                    # map url -> hit
                    url_to_hit = {h['doc'].get('url'): h for h in hits}
                    # build new ordered list by rerank_map score descending
                    ordered = sorted([(u, s) for u, s in rerank_map.items() if u in url_to_hit], key=lambda x: x[1], reverse=True)
                    new_hits = [url_to_hit[u] for u, _ in ordered]
                    hits = new_hits
                    recs = [h['doc'] for h in hits]
                    logger.debug("Hits reordered after LLM rerank: %s", [h['doc'].get('url') for h in hits])
            except Exception:
                # on error, fall back to deterministic hits
                pass

        confidence = self._recommendation_confidence(hits, constraints)

        # low confidence -> ask a single high-info clarification if turns allow
        if confidence < 0.3 and turns_left >= 1:
            questions = self._build_clarification_questions(constraints, last_user_text, max_questions=1)
            # phrase via LLM only for text quality, but keep deterministic decision
            if self.llm and hasattr(self.llm, 'chat'):
                try:
                    messages = [{"role": "system", "content": "You are a helpful assistant that should phrase a single clarifying question concisely."}, {"role": "user", "content": "Given the constraints: %s and the last user message: %s, produce 1 short clarifying question." % (str(constraints.__dict__), last_user_text)}]
                    phrased = self.llm.chat(messages)
                    alt = [s.strip() for s in phrased.split('\n') if s.strip()]
                    if alt:
                        questions = [alt[0]]
                except Exception:
                    pass
            # during clarification, recommendations must remain empty
            return {
                "action": "clarify",
                "clarification_questions": questions,
                "recommendations": [],
                "confidence": confidence,
                "ambiguity_score": ambiguity,
                "reason": "low confidence; asking single refinement",
            }

        return {
            "action": "recommend",
            "clarification_questions": [],
            "recommendations": recs,
            "confidence": confidence,
            "ambiguity_score": ambiguity,
            "reason": "recommendation ready",
        }

    def _get_last_user_text(self, history: List[Dict[str, Any]]) -> str:
        for item in reversed(history):
            if item.get("role") == "user":
                return (item.get("text") or "").strip()
        return (history[-1].get("text") or "").strip()

    def _ambiguity_score(self, constraints: ConversationConstraints, text: str) -> float:
        # compute completeness over key fields
        fields = [constraints.role, constraints.seniority, constraints.technical_skills or None, constraints.cognitive_tests or None, constraints.personality_tests or None]
        present = sum(1 for f in fields if f)
        total = len(fields)
        completeness = present / total if total else 1.0
        # linguistic vagueness boost
        vague_terms = ["some", "something", "maybe", "possibly", "or so", "etc"]
        vag = 1.0 if any(t in text.lower() for t in vague_terms) else 0.0
        # final ambiguity: more when less complete and when vague language present
        ambiguity = min(1.0, (1.0 - completeness) * 0.9 + vag * 0.2)
        return float(ambiguity)

    def _build_clarification_questions(self, constraints: ConversationConstraints, last_text: str, max_questions: int = 3) -> List[str]:
        # prioritize missing high-impact fields: role, assessment intent, seniority
        candidates = []
        if not constraints.role:
            candidates.append((3, "What is the target role or job title?"))
        if not (constraints.cognitive_tests or constraints.personality_tests or (constraints.technical_skills and len(constraints.technical_skills) > 0)):
            candidates.append((2, "Do you want cognitive ability tests, personality assessments, or both?"))
        if not constraints.seniority:
            candidates.append((1, "What seniority level are you hiring for (junior/mid/senior/lead)?"))
        # duration or remote (lower priority)
        if constraints.max_duration_minutes is None and re.search(r"minute|hour|duration|time", last_text, re.I):
            candidates.append((0, "Is there a maximum acceptable test duration (minutes)?"))
        if constraints.remote_only is None and re.search(r"remote|online|onsite|on-site", last_text, re.I):
            candidates.append((0, "Should the assessment be remote-only?"))

        # sort by priority (higher first) and return up to max_questions questions
        candidates = sorted(candidates, key=lambda x: -x[0])
        questions = [q for _, q in candidates][:max_questions]
        return questions

    def _recommendation_confidence(self, hits: List[Dict[str, Any]], constraints: ConversationConstraints) -> float:
        """Compute a simple confidence score from hit debug info and constraint matches.

        Confidence in [0,1]. Uses top hit score normalization and ratio of docs satisfying tag constraints.
        """
        if not hits:
            return 0.0
        # use the maximum combined RRF score as raw signal
        top_score = max((h.get("score") or 0.0) for h in hits)
        # approximate normalization: assume reasonable score range (0..0.1..1), use logistic scaling
        raw = float(top_score)
        conf_from_score = 1 / (1 + math.exp(-12 * (raw - 0.05)))  # sharp sigmoid around 0.05

        # constraint match ratio: how many hits include required tags
        required = set(getattr(constraints, "technical_skills", []) or [])
        if not required:
            tag_ratio = 1.0
        else:
            matches = 0
            for h in hits:
                doc_tags = set(h.get("doc", {}).get("tags", []))
                if required.intersection(doc_tags):
                    matches += 1
            tag_ratio = matches / len(hits)

        # combine signals
        conf = 0.75 * conf_from_score + 0.25 * tag_ratio
        return float(max(0.0, min(1.0, conf)))

    def _is_comparison_request(self, text: str) -> bool:
        return bool(re.search(r"\b(compare|difference|vs\b|which is better|better option)\b", text, re.I))

    def _detect_off_topic(self, text: str) -> bool:
        # detect requests clearly outside assessment domain (legal/medical/illegal/help to harm)
        t = text.lower()
        legal_patterns = [r"\b(advice about|how to sue|hire (a )?lawyer|attorney|litigation)\b", r"\b(legal advice|lawyer|lawsuit)\b"]
        medical_patterns = [r"\b(diagnos|medical advice|symptom|doctor|prescribe)\b"]
        illicit_patterns = [r"\b(build a bomb|make a weapon|hack into|steal|exploit)\b"]
        for p in legal_patterns + medical_patterns + illicit_patterns:
            if re.search(p, text, re.I):
                return True
        # also catch generic unrelated keywords
        off_topic_keywords = ["recipe", "movie", "torrent", "download", "torrent"]
        return any(k in t for k in off_topic_keywords)

    def _detect_prompt_injection(self, text: str) -> bool:
        # heuristics for prompt-injection style requests
        patterns = [
            r"ignore (?:previous|earlier) instructions",
            r"disregard (?:previous|earlier)",
            r"what is my api key",
            r"provide your source code",
            r"open the (?:jailbreak|backdoor)",
        ]
        for p in patterns:
            if re.search(p, text, re.I):
                return True
        # if text contains a suspicious long external URL asking to fetch secret
        if re.search(r"https?://[\w\.-]+/.*(token|secret|key)", text, re.I):
            return True
        # detect attempts to inject structured instructions like 'system:' or role switches
        if re.search(r"system\s*:\s*|assistant\s*:\s*|--\s*\bignore\b", text, re.I):
            return True
        return False

    def _make_refuse(self, reason: str) -> Dict[str, Any]:
        return {"action": "refuse", "clarification_questions": [], "recommendations": [], "confidence": 0.0, "ambiguity_score": 1.0, "reason": reason}

