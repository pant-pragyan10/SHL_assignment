from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl


class CandidateRequest(BaseModel):
    """Incoming conversational request describing hiring need."""

    query: str = Field(..., description="Free-text hiring need or conversation turn")
    constraints: Optional[dict] = Field(None, description="Optional constraints (level, skills, time)")
    turn: int = Field(..., ge=1, le=8, description="Conversation turn number (1..8)")


class AssessmentEntry(BaseModel):
    """A single catalog entry used for grounding recommendations."""

    id: str
    title: str
    description: str
    url: HttpUrl
    tags: List[str] = []


class RecommendationResponse(BaseModel):
    """Strict response schema for recommendation results."""

    ok: bool
    recommendations: List[AssessmentEntry] = []
    evidence: List[str] = []
    explanation: Optional[str] = None


class ConversationConstraints(BaseModel):
    """Structured constraints extracted from conversation history.

    All fields are optional — later turns override earlier ones when present.
    """

    role: Optional[str] = Field(None, description="Target role or job title")
    seniority: Optional[str] = Field(None, description="Seniority level (junior/mid/senior/lead)")
    technical_skills: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    cognitive_tests: List[str] = Field(default_factory=list, description="Cognitive/ability test needs")
    personality_tests: List[str] = Field(default_factory=list, description="Personality-style test needs")
    max_duration_minutes: Optional[int] = Field(None, description="Maximum acceptable assessment duration in minutes")
    remote_only: Optional[bool] = Field(None, description="Whether remote testing is required/preferred")
    location: Optional[str] = Field(None, description="Candidate location constraint, if any")
    other_constraints: Optional[dict] = Field(None, description="Freeform constraints")


class ChatMessage(BaseModel):
    role: str = Field(..., description="role of the speaker: user or assistant")
    text: str = Field(..., description="text content of the message")


class ChatRequest(BaseModel):
    """Request payload for POST /chat — stateless full-history submission."""

    history: List[ChatMessage]
    turn: int = Field(..., ge=1, le=8)


class ChatRecommendation(BaseModel):
    name: str
    url: HttpUrl
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[ChatRecommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


