"""
Pydantic models for SerpData MCP server.

This module defines data models for SERP data parsing, content analysis,
and SEO recommendations using Pydantic for validation and serialization.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DeviceType(str, Enum):
    """Supported device types for SERP queries."""
    DESKTOP = "desktop"
    MOBILE = "mobile"


class SearchIntent(str, Enum):
    """Classification of search intent based on SERP analysis."""
    INFORMATIONAL = "informational"
    COMMERCIAL = "commercial"
    TRANSACTIONAL = "transactional"
    NAVIGATIONAL = "navigational"
    MIXED = "mixed"


class CompetitionLevel(str, Enum):
    """Competition difficulty assessment for ranking."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ContentFormat(str, Enum):
    """Recommended content formats."""
    GUIDE = "guide"
    COMPARISON = "comparison"
    REVIEW = "review"
    TOOL = "tool"
    LIST = "list"
    FAQ = "faq"
    NEWS = "news"
    TUTORIAL = "tutorial"


class LLMProvider(str, Enum):
    """Supported LLM providers for analysis."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


# SERP Data Models

class OrganicResult(BaseModel):
    """Individual organic search result."""
    domain: str
    rank_absolute: int
    rank_inner: Optional[int] = None
    title: str
    url: str
    snippet: Optional[str] = None
    type: str = "standard"


class PAAQuestion(BaseModel):
    """People Also Ask question with answer."""
    question: str
    answer: Optional[str] = None
    ai_generated: bool = False
    source_url: Optional[str] = None


class RelatedSearch(BaseModel):
    """Related search suggestion."""
    query: str
    similarity_score: Optional[float] = None


class AIOverview(BaseModel):
    """AI Overview/Answer from SERP."""
    content: str
    sources: Optional[List[str]] = None
    confidence: Optional[float] = None


class SerpData(BaseModel):
    """Complete SERP data response from SerpData API."""
    query: str
    search_engine: str = "google"
    location: Optional[str] = None
    language: Optional[str] = None
    device: Optional[str] = None
    timestamp: Optional[datetime] = None
    total_results_count: Optional[int] = None

    # SERP features
    organic_results: List[OrganicResult] = []
    people_also_ask: Optional[List[PAAQuestion]] = None
    related_searches: Optional[List[RelatedSearch]] = None
    ai_overview: Optional[AIOverview] = None
    snippets_found: List[str] = []

    # Raw data for fallback
    raw_data: Optional[Dict[str, Any]] = None


# Analysis Models

class EntityAnalysis(BaseModel):
    """Semantic entity extraction and analysis."""
    central_entity: str
    secondary_entities: List[str]
    entity_relationships: Dict[str, List[str]]
    entity_attributes: Dict[str, List[str]]
    confidence_score: Optional[float] = None


class CompetitionAnalysis(BaseModel):
    """Competition assessment for the query."""
    level: CompetitionLevel
    top_competitors: List[str]  # domain names
    domain_authority_range: Optional[Dict[str, int]] = None
    content_gaps: List[str]
    ranking_factors: List[str]
    estimated_difficulty: Optional[int] = Field(None, ge=0, le=100)


class KeywordStrategy(BaseModel):
    """Keyword targeting recommendations."""
    primary_keywords: List[str]
    secondary_keywords: List[str]
    long_tail_opportunities: List[str]
    semantic_variations: Dict[str, List[str]]
    keyword_difficulty: Optional[Dict[str, int]] = None


class ContentSection(BaseModel):
    """Recommended content section."""
    title: str
    purpose: str
    keywords: List[str]
    length_estimate: Optional[int] = None  # word count
    priority: int = Field(ge=1, le=10)


class AIOptimization(BaseModel):
    """AI Overview and citation optimization tactics."""
    ai_overview_present: bool
    citation_opportunities: List[str]
    featured_snippet_target: Optional[str] = None
    answer_box_strategy: Optional[str] = None
    optimization_tactics: List[str]


class ContentRecommendation(BaseModel):
    """Complete content strategy recommendation."""
    analysis_id: UUID = Field(default_factory=uuid4)
    query: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Core strategy
    primary_topic: str
    content_angle: str
    unique_value_proposition: str

    # Classifications
    search_intent: SearchIntent
    competition_level: CompetitionLevel
    recommended_format: ContentFormat

    # Detailed analysis
    entity_analysis: EntityAnalysis
    competition_analysis: CompetitionAnalysis
    keyword_strategy: KeywordStrategy

    # Content structure
    recommended_sections: List[ContentSection]
    content_structure: Dict[str, Any]

    # Optimization
    ai_optimization: AIOptimization

    # Business context
    business_context: Optional[str] = None
    target_audience: Optional[str] = None

    # Metadata
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    processing_time: Optional[float] = None


# Request/Response Models

class SearchParams(BaseModel):
    """Parameters for SERP search request."""
    keyword: str = Field(min_length=1, max_length=200)
    hl: str = "en"  # interface language
    gl: str = "us"  # geographic location
    device: DeviceType = DeviceType.DESKTOP


class AnalysisParams(BaseModel):
    """Parameters for content analysis request."""
    business_context: Optional[str] = Field(None, max_length=1000)
    target_audience: Optional[str] = Field(None, max_length=500)
    custom_prompt: Optional[str] = None
    include_raw_data: bool = False


class SearchAndAnalyzeRequest(BaseModel):
    """Complete request for search + analysis."""
    search_params: SearchParams
    analysis_params: AnalysisParams


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# Configuration Models

class LLMConfig(BaseModel):
    """LLM provider configuration."""
    provider: LLMProvider
    model: Optional[str] = None
    temperature: float = Field(0.3, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, gt=0)
    timeout: int = Field(30, gt=0)


class SerpDataConfig(BaseModel):
    """SerpData API configuration."""
    api_key: str
    base_url: str = "https://api.serpdata.io/v1"
    timeout: int = Field(10, gt=0)
    retry_attempts: int = Field(3, ge=0)


class ServerConfig(BaseModel):
    """Complete server configuration."""
    serpdata: SerpDataConfig
    llm: LLMConfig
    default_language: str = "en"
    default_location: str = "us"
    cache_ttl: Optional[int] = Field(None, gt=0)  # cache TTL in seconds