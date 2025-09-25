"""
SerpData MCP Server with AI-Powered SEO Analysis

A FastMCP server that integrates SerpData API with LLM analysis to provide
intelligent SEO recommendations and content strategies based on SERP data.

Usage:
    # Direct execution (for local development)
    python serpdata_server.py

    # Via FastMCP CLI
    fastmcp run serpdata_server.py

    # Via configuration file
    fastmcp run serpdata.fastmcp.json

Features:
    - Real-time SERP data fetching from SerpData API
    - Multi-provider LLM analysis (OpenAI, Anthropic, Google)
    - Semantic SEO analysis and content recommendations
    - Competition analysis and content gap identification
    - AI Overview optimization strategies
    - Entity relationship mapping
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import aiohttp
from fastmcp import FastMCP

from llm_handler import LLMHandler, LLMError
from models import (
    AnalysisParams,
    ContentRecommendation,
    DeviceType,
    ErrorResponse,
    LLMProvider,
    SearchParams,
    SerpData,
)


class SerpDataError(Exception):
    """Exception for SerpData API related errors."""
    pass


# Initialize FastMCP server
mcp = FastMCP("SerpData SEO Intelligence")

# Environment configuration
SERPDATA_API_KEY = os.getenv("SERPDATA_API_KEY")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GOOGLE_API_KEY")
LLM_PROVIDER_STR = os.getenv("LLM_PROVIDER", "openai").lower()

# Validate required environment variables
if not SERPDATA_API_KEY:
    raise ValueError("SERPDATA_API_KEY environment variable is required")

if not LLM_API_KEY:
    raise ValueError("LLM_API_KEY environment variable is required")

# Convert provider string to enum
try:
    LLM_PROVIDER = LLMProvider(LLM_PROVIDER_STR)
except ValueError:
    available = [p.value for p in LLMProvider]
    raise ValueError(f"Invalid LLM_PROVIDER '{LLM_PROVIDER_STR}'. Available: {available}")

# Initialize LLM handler
llm_handler = LLMHandler.from_env(provider=LLM_PROVIDER)


async def fetch_serpdata(
    keyword: str,
    hl: str = "en",
    gl: str = "us",
    device: str = "desktop"
) -> Dict[str, Any]:
    """
    Fetch SERP data from SerpData API.

    Args:
        keyword: Search query
        hl: Interface language
        gl: Geographic location
        device: Device type (desktop/mobile)

    Returns:
        Raw SERP data as dictionary

    Raises:
        SerpDataError: If the API request fails
    """
    headers = {
        "Authorization": f"Bearer {SERPDATA_API_KEY}",
        "Content-Type": "application/json"
    }

    params = {
        "keyword": keyword,
        "hl": hl,
        "gl": gl,
        "device": device
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                "https://api.serpdata.io/v1/search",
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise SerpDataError(f"SerpData API error {response.status}: {error_text}")

                return await response.json()

        except aiohttp.ClientError as e:
            raise SerpDataError(f"SerpData API connection error: {e}")


def process_serp_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process raw SerpData response into structured format.

    Args:
        raw_data: Raw response from SerpData API

    Returns:
        Processed SERP data with structured fields
    """
    results = raw_data.get("results", {})

    # Extract organic results
    organic_results = []
    for result in results.get("organic_results", []):
        organic_results.append({
            "domain": result.get("domain", ""),
            "rank_absolute": result.get("rank_absolute", 0),
            "rank_inner": result.get("rank_inner"),
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "snippet": result.get("snippet"),
            "type": result.get("type", "standard")
        })

    # Extract People Also Ask
    paa_questions = []
    snippets_data = results.get("snippets_data", {})
    if "people_also_ask" in snippets_data:
        for paa in snippets_data["people_also_ask"]:
            paa_questions.append({
                "question": paa.get("question", ""),
                "answer": paa.get("answer"),
                "ai_generated": "ai_overview" in paa,
                "source_url": paa.get("source_url")
            })

    # Extract related searches
    related_searches = []
    if "related_searches" in snippets_data:
        for related in snippets_data["related_searches"]:
            if isinstance(related, str):
                related_searches.append({"query": related})
            elif isinstance(related, dict):
                related_searches.append({
                    "query": related.get("query", ""),
                    "similarity_score": related.get("similarity_score")
                })

    # Extract AI Overview
    ai_overview = None
    if "ai_overview" in snippets_data:
        ai_data = snippets_data["ai_overview"]
        ai_overview = {
            "content": ai_data.get("content", ""),
            "sources": ai_data.get("sources", []),
            "confidence": ai_data.get("confidence")
        }

    return {
        "query": results.get("query", ""),
        "search_engine": raw_data.get("search_engine", "google"),
        "location": raw_data.get("location"),
        "language": raw_data.get("language"),
        "device": raw_data.get("device"),
        "timestamp": raw_data.get("timestamp"),
        "total_results_count": raw_data.get("total_results_count"),
        "organic_results": organic_results,
        "people_also_ask": paa_questions if paa_questions else None,
        "related_searches": related_searches if related_searches else None,
        "ai_overview": ai_overview,
        "snippets_found": results.get("snippets_found", []),
        "raw_data": raw_data
    }


# MCP Tools

@mcp.tool
async def search_and_analyze(
    keyword: str,
    hl: str = "en",
    gl: str = "us",
    device: DeviceType = DeviceType.DESKTOP,
    business_context: Optional[str] = None,
    analysis_type: str = "full_analysis"
) -> ContentRecommendation:
    """
    Perform Google search and analyze results with AI for comprehensive SEO recommendations.

    This is the main tool that combines SERP data fetching with AI analysis to provide
    actionable content strategies, competition analysis, and optimization recommendations.

    Args:
        keyword: Search query to analyze (required)
        hl: Interface language code (default: "en")
        gl: Geographic location code (default: "us")
        device: Device type for search (desktop/mobile, default: desktop)
        business_context: Optional context about your business/website
        analysis_type: Type of analysis (full_analysis, quick_insights, competition, content_structure, entity_analysis)

    Returns:
        Comprehensive content recommendation with SEO strategy

    Raises:
        Exception: If SERP fetching or analysis fails
    """
    try:
        # Step 1: Fetch SERP data
        raw_serp_data = await fetch_serpdata(keyword, hl, gl, device.value)
        processed_data = process_serp_data(raw_serp_data)

        # Step 2: Analyze with LLM
        analysis_result = await llm_handler.analyze_serp(
            serp_json=processed_data,
            prompt_template=analysis_type,
            business_context=business_context
        )

        # Step 3: Structure the response
        recommendation = ContentRecommendation(
            query=keyword,
            **analysis_result
        )

        return recommendation

    except (SerpDataError, LLMError) as e:
        raise Exception(f"Analysis failed: {str(e)}")


@mcp.tool
async def get_raw_serp_data(
    keyword: str,
    hl: str = "en",
    gl: str = "us",
    device: DeviceType = DeviceType.DESKTOP,
    include_processed: bool = True
) -> Dict[str, Any]:
    """
    Fetch raw SERP data from SerpData API without AI analysis.

    Useful for getting the underlying search data for custom analysis or debugging.

    Args:
        keyword: Search query
        hl: Interface language code (default: "en")
        gl: Geographic location code (default: "us")
        device: Device type (desktop/mobile, default: desktop)
        include_processed: Whether to include processed/structured data

    Returns:
        Raw SERP data dictionary

    Raises:
        Exception: If SERP fetching fails
    """
    try:
        raw_data = await fetch_serpdata(keyword, hl, gl, device.value)

        if include_processed:
            processed_data = process_serp_data(raw_data)
            return {
                "raw": raw_data,
                "processed": processed_data
            }

        return raw_data

    except SerpDataError as e:
        raise Exception(f"SERP data fetch failed: {str(e)}")


@mcp.tool
async def analyze_existing_serp(
    serp_data: Dict[str, Any],
    business_context: Optional[str] = None,
    analysis_type: str = "full_analysis"
) -> Dict[str, Any]:
    """
    Analyze pre-fetched SERP data with AI without making new API calls.

    Useful when you already have SERP data and want to run different types of analysis.

    Args:
        serp_data: Previously fetched SERP data (from get_raw_serp_data)
        business_context: Optional business context for analysis
        analysis_type: Type of analysis to perform

    Returns:
        Analysis results as structured dictionary

    Raises:
        Exception: If analysis fails
    """
    try:
        # Use processed data if available, otherwise use raw data
        data_to_analyze = serp_data.get("processed", serp_data)

        analysis_result = await llm_handler.analyze_serp(
            serp_json=data_to_analyze,
            prompt_template=analysis_type,
            business_context=business_context
        )

        return analysis_result

    except LLMError as e:
        raise Exception(f"SERP analysis failed: {str(e)}")


@mcp.tool
async def quick_serp_insights(
    keyword: str,
    hl: str = "en",
    gl: str = "us",
    business_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get quick SEO insights from SERP data with simplified analysis.

    A faster alternative to full analysis that provides key insights and opportunities.

    Args:
        keyword: Search query to analyze
        hl: Interface language code (default: "en")
        gl: Geographic location code (default: "us")
        business_context: Optional business context

    Returns:
        Quick insights and recommendations

    Raises:
        Exception: If analysis fails
    """
    try:
        # Fetch SERP data
        raw_serp_data = await fetch_serpdata(keyword, hl, gl, "desktop")
        processed_data = process_serp_data(raw_serp_data)

        # Get quick analysis
        insights = await llm_handler.quick_analysis(
            serp_json=processed_data,
            business_context=business_context
        )

        return insights

    except (SerpDataError, LLMError) as e:
        raise Exception(f"Quick analysis failed: {str(e)}")


@mcp.tool
async def competition_analysis(
    keyword: str,
    hl: str = "en",
    gl: str = "us"
) -> Dict[str, Any]:
    """
    Perform focused competition analysis for the given keyword.

    Analyzes top organic results to identify competitors, content gaps, and opportunities.

    Args:
        keyword: Search query to analyze
        hl: Interface language code (default: "en")
        gl: Geographic location code (default: "us")

    Returns:
        Detailed competition analysis

    Raises:
        Exception: If analysis fails
    """
    try:
        # Fetch SERP data
        raw_serp_data = await fetch_serpdata(keyword, hl, gl, "desktop")
        processed_data = process_serp_data(raw_serp_data)

        # Get competition analysis
        analysis = await llm_handler.competition_analysis(serp_json=processed_data)

        return analysis

    except (SerpDataError, LLMError) as e:
        raise Exception(f"Competition analysis failed: {str(e)}")


@mcp.tool
async def get_content_structure(
    keyword: str,
    hl: str = "en",
    gl: str = "us",
    business_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate detailed content structure recommendations based on SERP analysis.

    Provides comprehensive content outline with sections, keywords, and optimization tactics.

    Args:
        keyword: Search query to analyze
        hl: Interface language code (default: "en")
        gl: Geographic location code (default: "us")
        business_context: Optional business context

    Returns:
        Detailed content structure and outline

    Raises:
        Exception: If analysis fails
    """
    try:
        # Fetch SERP data
        raw_serp_data = await fetch_serpdata(keyword, hl, gl, "desktop")
        processed_data = process_serp_data(raw_serp_data)

        # Get content structure analysis
        structure = await llm_handler.content_structure_analysis(
            serp_json=processed_data,
            business_context=business_context
        )

        return structure

    except (SerpDataError, LLMError) as e:
        raise Exception(f"Content structure analysis failed: {str(e)}")


@mcp.tool
async def entity_analysis(
    keyword: str,
    hl: str = "en",
    gl: str = "us"
) -> Dict[str, Any]:
    """
    Perform semantic entity analysis on SERP data.

    Identifies central entities, relationships, and semantic clusters for topic authority building.

    Args:
        keyword: Search query to analyze
        hl: Interface language code (default: "en")
        gl: Geographic location code (default: "us")

    Returns:
        Semantic entity analysis with relationships and opportunities

    Raises:
        Exception: If analysis fails
    """
    try:
        # Fetch SERP data
        raw_serp_data = await fetch_serpdata(keyword, hl, gl, "desktop")
        processed_data = process_serp_data(raw_serp_data)

        # Get entity analysis
        entities = await llm_handler.entity_analysis(serp_json=processed_data)

        return entities

    except (SerpDataError, LLMError) as e:
        raise Exception(f"Entity analysis failed: {str(e)}")


# Resource endpoints

@mcp.resource("serpdata://api-status")
async def api_status() -> Dict[str, Any]:
    """Check the status of SerpData and LLM APIs."""
    status = {
        "serpdata_api": "unknown",
        "llm_api": "unknown",
        "provider": LLM_PROVIDER.value,
        "timestamp": time.time()
    }

    # Test SerpData API
    try:
        await fetch_serpdata("test", "en", "us", "desktop")
        status["serpdata_api"] = "healthy"
    except SerpDataError:
        status["serpdata_api"] = "error"

    # Test LLM API (quick test)
    try:
        test_data = {"query": "test", "organic_results": []}
        await llm_handler.quick_analysis(test_data)
        status["llm_api"] = "healthy"
    except LLMError:
        status["llm_api"] = "error"

    return status


@mcp.resource("serpdata://supported-languages")
def supported_languages() -> Dict[str, List[str]]:
    """Get list of supported languages and locations."""
    return {
        "languages": [
            "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh",
            "ar", "hi", "tr", "pl", "nl", "sv", "no", "da", "fi"
        ],
        "locations": [
            "us", "uk", "ca", "au", "de", "fr", "es", "it", "br", "mx",
            "in", "jp", "kr", "ru", "tr", "pl", "nl", "se", "no", "dk", "fi"
        ],
        "devices": ["desktop", "mobile"]
    }


@mcp.resource("serpdata://analysis-types")
def analysis_types() -> Dict[str, str]:
    """Get available analysis types and their descriptions."""
    return {
        "full_analysis": "Comprehensive SEO analysis with all features",
        "quick_insights": "Fast analysis with key insights and opportunities",
        "competition": "Focused competition analysis and content gaps",
        "content_structure": "Detailed content outline and structure recommendations",
        "entity_analysis": "Semantic entity extraction and relationship mapping"
    }


if __name__ == "__main__":
    # This block is ignored by FastMCP Cloud but useful for local development
    mcp.run()