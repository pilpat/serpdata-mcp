# SerpData SEO Intelligence MCP Server

A comprehensive FastMCP server that combines real-time SERP data from SerpData API with AI-powered analysis to provide intelligent SEO recommendations and content strategies.

## 🚀 Features

- **Real-time SERP Data**: Fetch live Google search results via SerpData API
- **Multi-Provider LLM Analysis**: Support for OpenAI, Anthropic, and Google AI
- **Semantic SEO Intelligence**: Advanced content strategy recommendations
- **Competition Analysis**: Identify content gaps and opportunities
- **AI Overview Optimization**: Strategies for AI-powered search features
- **Entity Relationship Mapping**: Semantic understanding for topic authority
- **Content Structure Generation**: Detailed outlines and keyword strategies
- **Multi-Language Support**: International SEO analysis capabilities

## 📋 Tools Available

### Core Analysis Tools

1. **`search_and_analyze`** - Main comprehensive tool
   - Fetches SERP data and provides complete AI analysis
   - Returns structured content recommendations
   - Supports all analysis types and business context

2. **`get_raw_serp_data`** - Raw SERP data fetching
   - Direct access to SerpData API results
   - Optional data processing
   - Useful for custom analysis workflows

3. **`analyze_existing_serp`** - Analyze pre-fetched data
   - Process existing SERP data with AI
   - Multiple analysis types supported
   - No additional API calls required

### Specialized Analysis Tools

4. **`quick_serp_insights`** - Fast analysis
   - Simplified analysis for quick insights
   - Key opportunities and recommendations
   - Faster processing time

5. **`competition_analysis`** - Competition focus
   - Detailed competitor analysis
   - Content gap identification
   - Ranking difficulty assessment

6. **`get_content_structure`** - Content planning
   - Detailed content outlines
   - Section recommendations
   - Keyword placement strategies

7. **`entity_analysis`** - Semantic analysis
   - Entity extraction and relationships
   - Topic authority opportunities
   - Semantic clustering

## 🔧 Configuration

### Environment Variables

Required environment variables for deployment:

```bash
# SerpData API (Required)
SERPDATA_API_KEY=your_serpdata_api_key_here

# LLM Provider (Required - choose one)
LLM_API_KEY=your_llm_api_key_here
# OR specific provider keys:
# OPENAI_API_KEY=your_openai_key
# ANTHROPIC_API_KEY=your_anthropic_key
# GOOGLE_API_KEY=your_google_key

# LLM Provider Selection (Optional, default: openai)
LLM_PROVIDER=openai  # options: openai, anthropic, google

# Optional LLM Configuration
LLM_MODEL=gpt-4-turbo-preview  # Provider-specific model
LLM_TEMPERATURE=0.3           # Creativity level (0.0-1.0)
LLM_MAX_TOKENS=4000          # Maximum response tokens
LLM_TIMEOUT=30               # Request timeout in seconds
```

### Supported Providers & Models

| Provider | Default Model | Environment Variable | Features |
|----------|---------------|---------------------|----------|
| OpenAI | gpt-4-turbo-preview | `OPENAI_API_KEY` | JSON mode, structured output |
| Anthropic | claude-3-sonnet-20240229 | `ANTHROPIC_API_KEY` | Long context, detailed analysis |
| Google | gemini-pro | `GOOGLE_API_KEY` | Fast processing, JSON output |

## 🚀 Deployment

### FastMCP Cloud Deployment

1. **Get API Keys**:
   - SerpData API: Visit [serpdata.io](https://serpdata.io) and get your API key
   - LLM Provider: Get API key from OpenAI, Anthropic, or Google

2. **Deploy to FastMCP Cloud**:
   ```bash
   # Create GitHub repository with this code
   git init
   git add .
   git commit -m "Initial SerpData MCP server"
   git remote add origin https://github.com/your-username/serpdata-mcp.git
   git push -u origin main
   ```

3. **Configure FastMCP Cloud Project**:
   - Repository: `your-username/serpdata-mcp`
   - Entrypoint: `serpdata_server.py:mcp`
   - Environment Variables: Set `SERPDATA_API_KEY` and `LLM_API_KEY`

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export SERPDATA_API_KEY="your_api_key"
export LLM_API_KEY="your_llm_key"
export LLM_PROVIDER="openai"

# Run the server
python serpdata_server.py

# Or via FastMCP CLI
fastmcp run serpdata_server.py
```

## 📚 Usage Examples

### Basic Search and Analysis

```python
# Comprehensive SEO analysis for a keyword
result = await search_and_analyze(
    keyword="best pizza new york",
    business_context="Local restaurant review website",
    analysis_type="full_analysis"
)
```

### Quick Insights

```python
# Fast analysis for multiple keywords
insights = await quick_serp_insights(
    keyword="pizza delivery nyc",
    business_context="Food delivery service"
)
```

### Competition Analysis

```python
# Focus on competitor analysis
competition = await competition_analysis(
    keyword="pizza restaurant manhattan"
)
```

### Content Structure Planning

```python
# Get detailed content outline
structure = await get_content_structure(
    keyword="how to make pizza dough",
    business_context="Cooking blog and recipe site"
)
```

### Working with Raw Data

```python
# Fetch raw SERP data
raw_data = await get_raw_serp_data(
    keyword="homemade pizza recipe",
    include_processed=True
)

# Analyze with different approaches
full_analysis = await analyze_existing_serp(
    serp_data=raw_data,
    analysis_type="full_analysis"
)

entity_analysis = await analyze_existing_serp(
    serp_data=raw_data,
    analysis_type="entity_analysis"
)
```

## 🔍 Analysis Types

| Type | Description | Use Case |
|------|-------------|----------|
| `full_analysis` | Comprehensive SEO analysis with all features | Complete content strategy development |
| `quick_insights` | Fast analysis with key insights | Quick keyword research and opportunities |
| `competition` | Focused competitor analysis | Understanding competitive landscape |
| `content_structure` | Detailed content outlines | Content planning and creation |
| `entity_analysis` | Semantic entity extraction | Topic authority and clustering |

## 📊 Response Structure

### ContentRecommendation (from search_and_analyze)

```json
{
  "analysis_id": "uuid",
  "query": "search keyword",
  "timestamp": "2024-01-01T00:00:00",
  "primary_topic": "Main content topic",
  "content_angle": "Unique approach",
  "unique_value_proposition": "Differentiator",
  "search_intent": "informational|commercial|transactional|navigational",
  "competition_level": "low|medium|high|very_high",
  "recommended_format": "guide|comparison|review|tool|list",
  "entity_analysis": {
    "central_entity": "Primary concept",
    "secondary_entities": ["related", "entities"],
    "entity_relationships": {},
    "confidence_score": 0.85
  },
  "competition_analysis": {
    "level": "medium",
    "top_competitors": ["domain1.com", "domain2.com"],
    "content_gaps": ["gap1", "gap2"],
    "estimated_difficulty": 65
  },
  "keyword_strategy": {
    "primary_keywords": ["main", "keywords"],
    "secondary_keywords": ["supporting", "terms"],
    "long_tail_opportunities": ["specific", "phrases"]
  },
  "recommended_sections": [
    {
      "title": "Section Title",
      "purpose": "Section purpose",
      "keywords": ["section", "keywords"],
      "priority": 1
    }
  ],
  "ai_optimization": {
    "ai_overview_present": false,
    "citation_opportunities": ["opportunity1"],
    "optimization_tactics": ["tactic1", "tactic2"]
  }
}
```

## 🌐 Resources

### API Status Monitoring

```bash
# Check API health
curl serpdata://api-status
```

### Supported Languages & Locations

```bash
# Get supported configuration
curl serpdata://supported-languages
curl serpdata://analysis-types
```

## 🔧 Advanced Configuration

### Custom LLM Models

```bash
# Use specific models
export LLM_MODEL="gpt-4-0125-preview"  # OpenAI
export LLM_MODEL="claude-3-opus-20240229"  # Anthropic
export LLM_MODEL="gemini-pro-1.5"  # Google
```

### Analysis Parameters

```bash
# Control analysis creativity
export LLM_TEMPERATURE=0.1  # More focused (0.0-1.0)

# Longer responses
export LLM_MAX_TOKENS=8000

# Extended timeout for complex analysis
export LLM_TIMEOUT=60
```

## 🌍 International SEO

The server supports international SEO analysis with language and location parameters:

```python
# German market analysis
result = await search_and_analyze(
    keyword="beste pizza berlin",
    hl="de",  # German interface
    gl="de",  # Germany location
    business_context="German food delivery service"
)

# Japanese market analysis
result = await search_and_analyze(
    keyword="東京 ピザ 配達",
    hl="ja",  # Japanese interface
    gl="jp",  # Japan location
    device="mobile"  # Mobile-first market
)
```

## 🛠️ Error Handling

The server provides comprehensive error handling:

- **SerpDataError**: Issues with SerpData API (rate limits, invalid keys, etc.)
- **LLMError**: Problems with AI analysis (API issues, invalid responses, etc.)
- **ValidationError**: Invalid input parameters or malformed data

All errors are returned as structured `ErrorResponse` objects with details for debugging.

## 📈 Performance Optimization

- **Async Operations**: All API calls are asynchronous for optimal performance
- **Connection Pooling**: Efficient HTTP connection management
- **Timeout Management**: Configurable timeouts prevent hanging requests
- **Error Retry**: Built-in retry logic for transient failures
- **Response Streaming**: Large responses handled efficiently

## 🔒 Security

- **API Key Protection**: Environment variable-based configuration
- **Input Validation**: All parameters validated with Pydantic
- **Rate Limit Handling**: Graceful handling of API limits
- **Error Sanitization**: Sensitive data not exposed in error messages

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

- **Documentation**: [FastMCP Docs](https://gofastmcp.com)
- **SerpData API**: [SerpData Documentation](https://serpdata.io/docs)
- **Issues**: [GitHub Issues](https://github.com/your-username/serpdata-mcp/issues)

---

Built with [FastMCP](https://gofastmcp.com) • Powered by [SerpData](https://serpdata.io) • AI Analysis by OpenAI, Anthropic & Google