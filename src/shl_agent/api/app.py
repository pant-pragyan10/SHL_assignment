from fastapi import Request, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import logging
import os
import json

from shl_agent.schemas.models import ChatRequest, ChatResponse
from shl_agent.retrieval.retriever import HybridRetriever
from shl_agent.orchestration.agent import Orchestrator
from shl_agent.llm.groq_client import GroqClient
from shl_agent.recommendation.ranker import rank_candidates
from shl_agent.recommendation.formatter import format_chat_response, make_reply_explanation
from shl_agent.recommendation.comparer import compare_two

logger = logging.getLogger(__name__)



router = APIRouter()


@router.get("/health")
async def health() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({"status": "ok"})


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request) -> JSONResponse:
    """Stateless chat endpoint that returns schema-compliant recommendations.

    Expects a full `history` in the request body so the server can reconstruct
    conversation context without session state.
    """
    try:
        payload = await request.json()
        req = ChatRequest(**payload)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid json payload")

    # load catalog if available
    catalog_path = os.path.join(os.getcwd(), "data", "processed", "catalog.json")
    catalog = []
    if os.path.exists(catalog_path):
        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                catalog = json.load(f)
        except Exception:
            logger.exception("Failed to load catalog.json")

    # build retriever only if catalog present
    retriever = HybridRetriever(catalog) if catalog else None
    # create Groq client when key is available; pass into orchestrator for phrasing and rerank
    groq_client = None
    try:
        groq_client = GroqClient() if os.getenv("GROQ_API_KEY") else None
    except Exception:
        groq_client = None
    orchestrator = Orchestrator(llm_client=groq_client)

    decision = orchestrator.decide(history=[m.dict() for m in req.history], retriever=retriever, top_k=10)

    if decision.get("action") == "refuse":
        reply = decision.get("reason") or "I cannot help with that request."
        resp = ChatResponse(reply=reply, recommendations=[], end_of_conversation=True)
        return JSONResponse(status_code=200, content=resp.dict())

    if decision.get("action") == "clarify":
        qs = decision.get("clarification_questions", [])
        reply = "\n".join(qs) if qs else "Could you clarify your requirements?"
        resp = ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)
        return JSONResponse(status_code=200, content=resp.dict())

    if decision.get("action") == "compare":
        docs = decision.get("recommendations") or []
        # ensure at least two valid docs from catalog
        valid = [d for d in docs if d.get("url") and d.get("name")]
        if len(valid) < 2:
            resp = ChatResponse(reply="Comparison requires two catalog-backed assessments.", recommendations=[], end_of_conversation=False)
            return JSONResponse(status_code=200, content=resp.dict())
        # take top two
        a, b = valid[0], valid[1]
        reply_text, structured = compare_two(a, b)
        # prepare recommendations list with both entries
        from shl_agent.schemas.models import ChatRecommendation

        recs = [ChatRecommendation(name=a.get("name"), url=a.get("url"), test_type=a.get("assessment_type") or a.get("test_type") or ""),
                ChatRecommendation(name=b.get("name"), url=b.get("url"), test_type=b.get("assessment_type") or b.get("test_type") or "")]

        resp = ChatResponse(reply=reply_text, recommendations=recs, end_of_conversation=False)
        return JSONResponse(status_code=200, content=resp.dict())

    # recommend or compare -> use retriever + ranker
    hits = decision.get("recommendations") or []
    # ensure hits originate from catalog (no hallucination)
    valid_docs = [d for d in hits if d.get("url") and d.get("name")]

    if not valid_docs:
        resp = ChatResponse(reply="No catalog-backed recommendations found.", recommendations=[], end_of_conversation=False)
        return JSONResponse(status_code=200, content=resp.dict())

    # rank candidates
    constraints = orchestrator.analyzer.analyze([m.dict() for m in req.history]).dict()
    ranked = rank_candidates(valid_docs, constraints, top_k=10)

    # build reply explanation grounded in doc fields
    reply = make_reply_explanation(ranked, constraints)

    # format strict response
    response = format_chat_response(ranked)
    # replace reply with our generated reply
    response.reply = reply
    return JSONResponse(status_code=200, content=response.dict())
