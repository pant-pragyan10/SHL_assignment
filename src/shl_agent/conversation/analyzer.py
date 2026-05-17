"""Conversation analyzer: deterministic parsing of message history into structured constraints."""
from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass

from shl_agent.ingestion.cleaning import normalize_text, extract_tags
from shl_agent.schemas.models import ConversationConstraints


COMMON_SOFT_SKILLS = [
    "communication",
    "teamwork",
    "leadership",
    "problem solving",
    "adaptability",
    "time management",
    "creativity",
    "attention to detail",
]


@dataclass
class ConversationAnalyzer:
    """Analyze full message history (stateless) and produce structured constraints.

    The analyzer operates deterministically using regex and heuristics. It's
    designed to be light-weight and explainable; LLM fallback can be added
    later for ambiguous cases.
    """

    def analyze(self, history: List[Dict[str, Any]]) -> ConversationConstraints:
        """Convert `history` into consolidated `ConversationConstraints`.

        Args:
            history: list of turns, each a dict with keys: `role` (user/assistant), `text` (string)

        Returns:
            ConversationConstraints instance with merged results.
        """
        cons = ConversationConstraints()

        # iterate turns in order; later turns override earlier fields
        for turn in history:
            text = (turn.get("text") or "").strip()
            if not text:
                continue
            tnorm = normalize_text(text)

            # role/title
            role = self._extract_role(tnorm)
            if role:
                cons.role = role

            # seniority
            seniority = self._extract_seniority(tnorm)
            if seniority:
                cons.seniority = seniority

            # skills
            tech = self._extract_technical_skills(tnorm)
            if tech:
                # merge preserving order, later occurrences override by placing later first
                cons.technical_skills = self._merge_lists(cons.technical_skills, tech)

            soft = self._extract_soft_skills(tnorm)
            if soft:
                cons.soft_skills = self._merge_lists(cons.soft_skills, soft)

            # testing needs
            cog = self._extract_cognitive_needs(tnorm)
            if cog:
                cons.cognitive_tests = self._merge_lists(cons.cognitive_tests, cog)

            pers = self._extract_personality_needs(tnorm)
            if pers:
                cons.personality_tests = self._merge_lists(cons.personality_tests, pers)

            # duration constraint
            minutes = self._extract_max_duration_minutes(tnorm)
            if minutes is not None:
                cons.max_duration_minutes = minutes

            # remote preference
            remote = self._extract_remote_preference(tnorm)
            if remote is not None:
                cons.remote_only = remote

            # location
            loc = self._extract_location(tnorm)
            if loc:
                cons.location = loc

            # other constraints: capture e.g. "must be completed within 2 weeks"
            other = self._extract_other_constraints(tnorm)
            if other:
                if cons.other_constraints:
                    cons.other_constraints.update(other)
                else:
                    cons.other_constraints = other

        return cons

    def _merge_lists(self, existing: List[str], new: List[str]) -> List[str]:
        # newer items should appear earlier; preserve uniqueness
        out = list(new) + [x for x in existing if x not in new]
        return out

    def _extract_role(self, text: str) -> Optional[str]:
        # look for patterns like 'hiring a senior backend engineer' or 'need a product manager'
        m = re.search(r"hiring (?:for )?(?:a |an )?([\w\- ]{2,40})", text)
        if m:
            role = m.group(1).strip()
            # drop trailing seniority words if present
            role = re.sub(r"\b(junior|senior|mid|lead)\b", "", role).strip()
            return role
        # alternative: "looking for a\u2026" or "need a\u2026"
        m2 = re.search(r"(?:need|looking for|searching for) (?:a |an )?([\w\- ]{2,40})", text)
        if m2:
            return m2.group(1).strip()
        return None

    def _extract_seniority(self, text: str) -> Optional[str]:
        for level in ["senior", "junior", "mid", "lead", "principal"]:
            if re.search(rf"\b{level}\b", text):
                return level
        return None

    def _extract_technical_skills(self, text: str) -> List[str]:
        # heuristics: look for phrases like 'experience with X, Y' or 'proficient in X'
        skills = []
        m = re.search(r"(experience with|proficient in|familiar with|knowledge of|stack:|tech:)([\s\S]{1,120})", text)
        if m:
            tail = m.group(2)
            # split on common delimiters
            parts = re.split(r"[,:;\\/]| and | or ", tail)
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                # stop at sentence end
                p = re.split(r"[\.\n]", p)[0].strip()
                # filter short tokens
                if len(p) > 1:
                    skills.append(p)
        # fallback: extract tags from text
        if not skills:
            tags = extract_tags(text)
            # heuristically treat tokens with + or common tech words as skills
            common_tech = [t for t in tags if any(ch.isalpha() for ch in t)]
            return common_tech[:10]
        return skills[:20]

    def _extract_soft_skills(self, text: str) -> List[str]:
        found = []
        for s in COMMON_SOFT_SKILLS:
            if re.search(rf"\b{re.escape(s)}\b", text):
                found.append(s)
        return found

    def _extract_cognitive_needs(self, text: str) -> List[str]:
        needs = []
        cognitive_map = {
            "numerical": ["numerical", "quantitative", "math", "numbers"],
            "verbal": ["verbal", "reading", "comprehension"],
            "inductive": ["inductive", "logical", "reasoning"],
            "aptitude": ["aptitude", "ability test", "cognitive"],
        }
        for key, kws in cognitive_map.items():
            for kw in kws:
                if re.search(rf"\b{re.escape(kw)}\b", text):
                    needs.append(key)
                    break
        return needs

    def _extract_personality_needs(self, text: str) -> List[str]:
        needs = []
        for kw in ["personality", "motivation", "values", "behaviour", "behavioral"]:
            if re.search(rf"\b{kw}\b", text):
                needs.append("personality")
                break
        return needs

    def _extract_max_duration_minutes(self, text: str) -> Optional[int]:
        # capture 'no longer than 30 minutes' or 'max 45 mins' etc.
        m = re.search(r"(?:no longer than|max(?:imum)?|under|<=)\s*(\d{1,4})\s*(minutes|mins|hours|hrs)?", text)
        if m:
            val = int(m.group(1))
            unit = m.group(2)
            if unit and re.search(r"hour|hr", unit):
                return val * 60
            return val
        # also capture 'around 30 minutes' as a soft constraint
        m2 = re.search(r"(\d{1,3})\s*(minutes|mins|hours|hrs)", text)
        if m2:
            val = int(m2.group(1))
            unit = m2.group(2)
            if re.search(r"hour|hr", unit):
                return val * 60
            return val
        return None

    def _extract_remote_preference(self, text: str) -> Optional[bool]:
        if re.search(r"\b(remote only|remote-only|remote required|must be remote)\b", text):
            return True
        if re.search(r"\b(not remote|on-site|onsite|not remote preferred)\b", text):
            return False
        # soft preference
        if re.search(r"\b(remote|online) testing\b", text):
            return True
        return None

    def _extract_location(self, text: str) -> Optional[str]:
        # naive capture of 'in <Location>' patterns
        m = re.search(r"\bin ([A-Z][a-zA-Z\s]{1,40})\b", text)
        if m:
            return m.group(1).strip()
        return None

    def _extract_other_constraints(self, text: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        # time window: 'within 2 weeks'
        m = re.search(r"within (\d{1,3}) (days|weeks|months)", text)
        if m:
            out["time_window"] = {"value": int(m.group(1)), "unit": m.group(2)}
        return out
