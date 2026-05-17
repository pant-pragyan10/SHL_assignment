"""Groq OpenAI-compatible client for chat completions and lightweight re-ranking.

This client calls the Groq OpenAI-compatible endpoint at
`https://api.groq.com/openai/v1/chat/completions` using the `GROQ_API_KEY`
environment variable. It provides two primary helpers:
- `chat(messages, model) -> str`: return assistant text
- `rerank_candidates(candidates, constraints) -> dict[url->score]` where scores
  are floats in [0,1]. If the LLM response is invalid, `rerank_candidates`
  returns `None` so callers can fall back to deterministic ranking.

Security note: callers must still validate that recommendations originate from
the retrieved candidates; this client never adds new items on its own.
"""
from typing import List, Dict, Any, Optional
import os
import requests
import json
import logging
import re

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class GroqClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.groq.com/openai/v1") -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not set in environment")
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, payload: dict, timeout: int = 30) -> requests.Response:
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp

    def chat(self, messages: List[Dict[str, str]], model: str = "llama-3.3-70b-versatile", timeout: int = 30) -> str:
        payload = {"model": model, "messages": messages}
        try:
            resp = self._post("/chat/completions", payload, timeout=timeout)
        except Exception as exc:
            logger.exception("Groq chat request failed: %s", exc)
            raise
        try:
            data = resp.json()
            # Expect OpenAI-style response: choices -> message -> content
            choices = data.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                return msg.get("content") or json.dumps(data)
            # fallback to top-level text
            return data.get("text") or json.dumps(data)
        except Exception:
            return resp.text

    def rerank_candidates(self, query: str | None, candidates: List[Dict[str, Any]], constraints: Optional[Dict[str, Any]] = None, model: str = "llama-3.3-70b-versatile", timeout: int = 30) -> Optional[Dict[str, float]]:
        """Request lightweight reranking scores from the LLM.

        Standard interface:
            rerank_candidates(query: Optional[str], candidates: List[dict], constraints: Optional[dict]) -> Optional[dict[url->score]]

        Args:
            query: The original user query or prompt text (may be None).
            candidates: List of hit objects as returned by the retriever. Each
                item is expected to be a dict containing a `doc` mapping with
                at least a `url` key. Malformed candidates will be ignored.
            constraints: Optional extracted constraints/context to include in the prompt.
            model: LLM model name.
            timeout: HTTP request timeout in seconds.

        Returns:
            A mapping of candidate URL -> score (0.0-1.0) or None on failure.

        Behavior:
            - If `candidates` is empty or contains no valid urls, returns None.
            - If the model response cannot be parsed to a safe mapping, returns None.
            - Caller must validate that returned URLs are a subset of provided candidates.
        """
        # Defensive validation
        if not candidates:
            logger.debug("rerank_candidates called with empty candidates list")
            return None

        # Build a compact candidates list to include in the prompt
        cand_texts: List[Dict[str, str]] = []
        candidate_urls = set()
        for c in candidates:
            try:
                doc = c.get("doc", {}) if isinstance(c, dict) else {}
                url = doc.get("url") or doc.get("link") or ""
                if not url:
                    continue
                candidate_urls.add(url)
                name = doc.get("name") or doc.get("title") or ""
                desc = doc.get("description") or ""
                tags = ", ".join(doc.get("tags", []) or [])
                cand_texts.append({"name": name, "url": url, "description": desc, "tags": tags})
            except Exception:
                continue

        if not cand_texts:
            logger.debug("No valid candidate docs to rerank after validation")
            return None

        system = (
            "You are an assistant that helps score retrieved candidate assessments."
            " Only score the candidates provided. Do NOT invent new assessments or URLs."
            " Return valid JSON ONLY: an array of objects with keys 'url' and 'score' (0.0-1.0)."
        )

        user_lines = []
        if query:
            user_lines.append(f"Query: {query}")
        if constraints:
            try:
                user_lines.append("Constraints: %s" % json.dumps(constraints, ensure_ascii=False))
            except Exception:
                user_lines.append("Constraints: (unserializable)")
        user_lines.append("Candidates:")
        for c in cand_texts:
            user_lines.append(json.dumps(c, ensure_ascii=False))

        user_lines.append("\nFor each candidate, provide a numeric score between 0 and 1 (higher=better) reflecting how well it matches the constraints. Output only JSON.")

        messages = [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(user_lines)}]

        try:
            resp_text = self.chat(messages, model=model, timeout=timeout)
        except Exception as exc:
            logger.exception("Groq rerank chat failed: %s", exc)
            return None

        # Extract JSON substring and validate
        try:
            m = re.search(r"(\[\s*\{.*?\}\s*\])", resp_text, re.S)
            json_text = None
            if m:
                json_text = m.group(1)
            else:
                start = resp_text.find('[')
                end = resp_text.rfind(']')
                if start != -1 and end != -1 and end > start:
                    json_text = resp_text[start:end+1]
            if not json_text:
                logger.warning("Could not locate JSON in Groq rerank response: %s", resp_text[:200])
                return None
            parsed = json.loads(json_text)
            out: Dict[str, float] = {}
            for item in parsed:
                url = item.get('url')
                score = item.get('score')
                if not url or url not in candidate_urls:
                    continue
                try:
                    s = float(score)
                except Exception:
                    continue
                s = max(0.0, min(1.0, s))
                out[url] = s
            if not out:
                logger.warning("Parsed rerank JSON contained no valid candidate scores")
                return None
            return out
        except Exception as exc:
            logger.exception("Failed to parse Groq rerank JSON: %s", exc)
            return None
