"""
SEO analysis prompts for LLM processing.

This module contains prompt templates for semantic SEO analysis,
content strategy generation, and SERP intelligence processing.
"""

from string import Template
from typing import Dict, Any, Optional


class PromptTemplate:
    """Base class for prompt templates with variable substitution."""

    def __init__(self, template: str):
        self.template = Template(template)

    def format(self, **kwargs) -> str:
        """Format template with provided variables."""
        return self.template.safe_substitute(**kwargs)


# Main SEO Analysis Prompt (based on prompt.md)
SEO_ANALYSIS_PROMPT = PromptTemplate("""
# Semantic SEO Content Strategist & AI Era Optimizer

You are a specialized Semantic SEO Content Strategist expert in generating data-driven article proposals based on JSON search data analysis. Your expertise combines advanced semantic SEO principles with AI era optimization techniques (GEO, AI Overviews, AI Mode) to create actionable content strategies that maximize organic visibility and AI citation potential.

## 📋 PROCEDURE: SEMANTIC SEO CONTENT STRATEGY FROM JSON DATA

### INPUT DATA:
```json
$serp_json
```

### BUSINESS CONTEXT:
$business_context

### ANALYSIS PHASES:

### PHASE 1: DATA EXTRACTION & ENTITY ANALYSIS

**Tasks:**
1. **Extract Core Query Analysis**
   ✅ Primary query analysis
   ✅ Competition assessment
   ✅ Intent categorization (informational / educational / navigational / transactional / commercial)

2. **Identify Central Entity**
   ✅ What is the main subject/concept?
   ✅ Extract from: query + top organic results + PAA patterns
   ✅ Validate: does it appear consistently across all data points?

3. **Map Secondary Entities & Attributes**
   ✅ List related concepts from organic results
   ✅ Extract attributes from PAA questions
   ✅ Identify entity relationships and hierarchy

4. **AI OVERVIEW VERIFICATION PROCEDURE**

**MANDATORY CHECK:**
1. Check the location of AI Overview in JSON:
   ✅ **CORRECT:** snippets_data.ai_overview.content
   ❌ **NOT AI OVERVIEW:** people_also_ask.questions[X].ai_overview

2. Verify presence:
   * Does snippets_found contain "ai_overview"?
   * Does snippets_data.ai_overview exist?

3. Report precisely:
   ❌ **WRONG:** "AI Overview active"
   ✅ **CORRECT:** "AI answers in PAA, no standalone AI Overview"

### PHASE 2: USER INTENT & PAIN POINT ANALYSIS

**Tasks:**
1. **PAA Question Categorization**
   ✅ Group by intent type
   ✅ Identify recurring concerns/fears
   ✅ Map to user journey stages

2. **Related Searches Pattern Analysis**
   ✅ Cluster by semantic similarity
   ✅ Identify long-tail opportunities
   ✅ Spot trending/emerging topics

3. **Competitive Landscape Assessment**
   ✅ Analyze top 3 organic results
   ✅ Identify content gaps
   ✅ Check for AI Overview presence/competition

### PHASE 3: CONTENT STRATEGY GENERATION

**Required Output Structure:**
Generate comprehensive content recommendations based on analysis.

**RESPONSE FORMAT:**
Return ONLY a valid JSON object with this exact structure:

```json
{
  "primary_topic": "Main content topic based on central entity",
  "content_angle": "Unique approach/perspective for the content",
  "unique_value_proposition": "What makes this content different/better",
  "search_intent": "informational|commercial|transactional|navigational|mixed",
  "competition_level": "low|medium|high|very_high",
  "recommended_format": "guide|comparison|review|tool|list|faq|news|tutorial",
  "entity_analysis": {
    "central_entity": "Primary subject/concept",
    "secondary_entities": ["related", "concepts", "from", "analysis"],
    "entity_relationships": {
      "central_entity_name": ["related_entity_1", "related_entity_2"]
    },
    "entity_attributes": {
      "entity_name": ["attribute1", "attribute2"]
    },
    "confidence_score": 0.85
  },
  "competition_analysis": {
    "level": "medium",
    "top_competitors": ["domain1.com", "domain2.com", "domain3.com"],
    "content_gaps": ["gap1", "gap2", "gap3"],
    "ranking_factors": ["authority", "content_depth", "user_experience"],
    "estimated_difficulty": 65
  },
  "keyword_strategy": {
    "primary_keywords": ["main", "target", "keywords"],
    "secondary_keywords": ["supporting", "keywords"],
    "long_tail_opportunities": ["specific", "long", "tail", "phrases"],
    "semantic_variations": {
      "main_keyword": ["variation1", "variation2"],
      "secondary_keyword": ["variation1", "variation2"]
    }
  },
  "recommended_sections": [
    {
      "title": "Section Title",
      "purpose": "Why this section is important",
      "keywords": ["section", "specific", "keywords"],
      "length_estimate": 500,
      "priority": 1
    }
  ],
  "ai_optimization": {
    "ai_overview_present": false,
    "citation_opportunities": ["opportunity1", "opportunity2"],
    "featured_snippet_target": "Specific question to target for featured snippet",
    "optimization_tactics": ["tactic1", "tactic2", "tactic3"]
  },
  "target_audience": "Primary audience based on search patterns",
  "confidence_score": 0.85
}
```

**IMPORTANT REQUIREMENTS:**
- Analyze ALL provided SERP data thoroughly
- Base recommendations on actual data patterns, not assumptions
- Identify specific content gaps from competitor analysis
- Provide actionable, implementable recommendations
- Focus on AI-era optimization opportunities
- Ensure JSON is properly formatted and complete

Analyze the SERP data and provide comprehensive content strategy recommendations following the exact JSON structure above.
""")


# Simplified Analysis Prompt for Quick Insights
QUICK_ANALYSIS_PROMPT = PromptTemplate("""
Analyze this SERP data and provide quick SEO insights:

## SERP Data:
```json
$serp_json
```

## Business Context:
$business_context

Provide analysis in this JSON format:
```json
{
  "primary_topic": "Main topic opportunity",
  "search_intent": "informational|commercial|transactional|navigational",
  "competition_level": "low|medium|high",
  "key_insights": ["insight1", "insight2", "insight3"],
  "content_gaps": ["gap1", "gap2"],
  "quick_wins": ["opportunity1", "opportunity2"],
  "ai_optimization": {
    "ai_overview_present": false,
    "citation_opportunities": ["opportunity1"]
  }
}
```
""")


# Competition Analysis Focused Prompt
COMPETITION_ANALYSIS_PROMPT = PromptTemplate("""
Focus on competitive analysis for this SERP:

## SERP Data:
```json
$serp_json
```

Analyze the top organic results and provide competitive intelligence:

```json
{
  "competition_analysis": {
    "level": "low|medium|high|very_high",
    "top_competitors": ["domain1.com", "domain2.com"],
    "strength_assessment": {
      "domain1.com": {
        "strengths": ["strength1", "strength2"],
        "weaknesses": ["weakness1", "weakness2"],
        "content_type": "guide|comparison|review|tool"
      }
    },
    "content_gaps": ["specific gaps in competitor content"],
    "differentiation_opportunities": ["how to stand out"],
    "estimated_difficulty": 65
  },
  "ranking_strategy": {
    "recommended_approach": "content|authority|technical|user_experience",
    "success_probability": 0.75,
    "time_estimate": "3-6 months"
  }
}
```
""")


# Content Structure Focused Prompt
CONTENT_STRUCTURE_PROMPT = PromptTemplate("""
Create detailed content structure based on SERP analysis:

## SERP Data:
```json
$serp_json
```

## Business Context:
$business_context

Generate comprehensive content outline:

```json
{
  "content_structure": {
    "title_suggestions": ["Title Option 1", "Title Option 2", "Title Option 3"],
    "meta_description": "SEO-optimized meta description",
    "content_sections": [
      {
        "section_number": 1,
        "title": "Introduction",
        "purpose": "Hook readers and establish context",
        "keywords": ["primary", "keywords"],
        "subtopics": ["subtopic1", "subtopic2"],
        "length_estimate": 300,
        "priority": 1
      }
    ],
    "internal_linking": {
      "hub_page_opportunities": ["topic1", "topic2"],
      "supporting_content": ["content idea 1", "content idea 2"]
    },
    "cta_recommendations": ["primary CTA", "secondary CTA"]
  },
  "keyword_placement": {
    "title": "primary keyword placement strategy",
    "headers": {"h1": "keyword", "h2": ["keyword1", "keyword2"]},
    "body": {"keyword_density": "1-2%", "semantic_keywords": ["related", "terms"]}
  }
}
```
""")


# Entity-Focused Analysis Prompt
ENTITY_ANALYSIS_PROMPT = PromptTemplate("""
Perform semantic entity analysis on this SERP data:

## SERP Data:
```json
$serp_json
```

Focus on entity extraction and relationships:

```json
{
  "entity_analysis": {
    "central_entity": "Primary subject/concept",
    "entity_type": "person|organization|product|concept|location|event",
    "secondary_entities": ["related entities found"],
    "entity_relationships": {
      "central_entity": {
        "related_to": ["entity1", "entity2"],
        "attributes": ["attribute1", "attribute2"],
        "context": "how entities relate in this search context"
      }
    },
    "semantic_clusters": {
      "cluster1_name": ["term1", "term2", "term3"],
      "cluster2_name": ["term1", "term2", "term3"]
    },
    "topic_authority_opportunities": [
      {
        "entity": "entity_name",
        "content_type": "definitive guide|comparison|resource",
        "authority_potential": "high|medium|low"
      }
    ]
  },
  "content_entity_optimization": {
    "schema_markup": ["Product", "Article", "FAQPage"],
    "entity_mentions": ["how to mention entities naturally"],
    "topic_clustering": ["related topics to cover"]
  }
}
```
""")


# Available prompt templates
PROMPT_TEMPLATES = {
    "full_analysis": SEO_ANALYSIS_PROMPT,
    "quick_insights": QUICK_ANALYSIS_PROMPT,
    "competition": COMPETITION_ANALYSIS_PROMPT,
    "content_structure": CONTENT_STRUCTURE_PROMPT,
    "entity_analysis": ENTITY_ANALYSIS_PROMPT,
}


def get_prompt_template(template_name: str) -> PromptTemplate:
    """Get a prompt template by name."""
    if template_name not in PROMPT_TEMPLATES:
        available = ", ".join(PROMPT_TEMPLATES.keys())
        raise ValueError(f"Unknown template '{template_name}'. Available: {available}")
    return PROMPT_TEMPLATES[template_name]


def format_prompt(
    template_name: str,
    serp_json: Dict[str, Any],
    business_context: Optional[str] = None,
    **kwargs
) -> str:
    """Format a prompt template with SERP data and context."""
    template = get_prompt_template(template_name)

    # Convert serp_json to formatted JSON string
    import json
    serp_json_str = json.dumps(serp_json, indent=2, ensure_ascii=False)

    return template.format(
        serp_json=serp_json_str,
        business_context=business_context or "No specific business context provided.",
        **kwargs
    )


def create_custom_prompt(
    base_template: str,
    custom_requirements: str,
    output_format: Optional[str] = None
) -> str:
    """Create a custom prompt based on a base template."""
    prompt = f"{base_template}\n\n## CUSTOM REQUIREMENTS:\n{custom_requirements}"

    if output_format:
        prompt += f"\n\n## REQUIRED OUTPUT FORMAT:\n{output_format}"

    return prompt