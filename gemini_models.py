"""
Gemini-specific Pydantic models for structured output.

These models are designed to work with Gemini 2.5 Flash structured output requirements,
avoiding Dict[str, Any] and other unsupported field types.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from models import SearchIntent, CompetitionLevel, ContentFormat


class GeminiEntityAnalysis(BaseModel):
    """Gemini-compatible entity analysis model."""
    central_entity: str
    secondary_entities: List[str]
    confidence_score: float = Field(ge=0.0, le=1.0)


class GeminiCompetitionAnalysis(BaseModel):
    """Gemini-compatible competition analysis model."""
    level: CompetitionLevel
    top_competitors: List[str]
    content_gaps: List[str]
    estimated_difficulty: int = Field(ge=0, le=100)


class GeminiKeywordStrategy(BaseModel):
    """Gemini-compatible keyword strategy model."""
    primary_keywords: List[str]
    secondary_keywords: List[str]
    long_tail_opportunities: List[str]


class GeminiContentSection(BaseModel):
    """Gemini-compatible content section model."""
    title: str
    purpose: str
    keywords: List[str]
    priority: int = Field(ge=1, le=10)


class GeminiContentStructure(BaseModel):
    """Gemini-compatible content structure model (replaces Dict[str, Any])."""
    outline_type: str = Field(default="hierarchical")
    main_sections: List[str]
    estimated_word_count: Optional[int] = None
    content_depth: str = Field(default="comprehensive")  # "brief", "moderate", "comprehensive"


class GeminiAIOptimization(BaseModel):
    """Gemini-compatible AI optimization model."""
    ai_overview_present: bool
    citation_opportunities: List[str]
    featured_snippet_target: Optional[str] = None
    answer_box_strategy: Optional[str] = None
    optimization_tactics: List[str]


class GeminiContentRecommendation(BaseModel):
    """Gemini-compatible content recommendation model."""
    analysis_id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    # Core strategy
    primary_topic: str
    content_angle: str
    unique_value_proposition: str

    # Classifications
    search_intent: SearchIntent
    competition_level: CompetitionLevel
    recommended_format: ContentFormat

    # Detailed analysis
    entity_analysis: GeminiEntityAnalysis
    competition_analysis: GeminiCompetitionAnalysis
    keyword_strategy: GeminiKeywordStrategy

    # Content structure (using structured model instead of Dict[str, Any])
    recommended_sections: List[GeminiContentSection]
    content_structure: GeminiContentStructure

    # Optimization
    ai_optimization: GeminiAIOptimization

    # Business context
    business_context: Optional[str] = None
    target_audience: Optional[str] = None

    # Metadata
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    processing_time: Optional[float] = None


def convert_to_standard_model(gemini_response: GeminiContentRecommendation) -> dict:
    """Convert Gemini response to standard ContentRecommendation format."""
    from uuid import UUID
    from datetime import datetime

    # Convert the structured response back to the original format
    result = gemini_response.model_dump()

    # Convert string UUID back to UUID object
    try:
        result["analysis_id"] = UUID(result["analysis_id"])
    except (ValueError, TypeError):
        result["analysis_id"] = uuid4()

    # Convert string timestamp back to datetime
    try:
        result["timestamp"] = datetime.fromisoformat(result["timestamp"])
    except (ValueError, TypeError):
        result["timestamp"] = datetime.now()

    # Convert content_structure to Dict[str, Any] for compatibility
    if "content_structure" in result:
        content_struct = result["content_structure"]
        result["content_structure"] = {
            "outline_type": content_struct.get("outline_type", "hierarchical"),
            "main_sections": content_struct.get("main_sections", []),
            "estimated_word_count": content_struct.get("estimated_word_count"),
            "content_depth": content_struct.get("content_depth", "comprehensive")
        }

    return result