🚀 What is SerpData?
SerpData is a powerful API that delivers structured data from Google search results. Instead of complicated scraping, you get ready-to-use JSON data in seconds. Perfect for SEO analysis, competitor monitoring, market research, and marketing automation.

⚡ Quick Start
Choose your preferred method to start using SerpData API in under 1 minute:

🎮 API Playground - Instant Testing
The fastest way to try SerpData without any setup. Perfect for testing queries and understanding the API response structure.

🤖 MCP - Claude Desktop Integration
Use SerpData directly in Claude Desktop conversations with the official MCP server.

Installation:
Get your API key from serpdata.io (click "Copy to clipboard")
Open Claude Desktop configuration file:
macOS:~/Library/Application Support/Claude/claude_desktop_config.json
Windows:%APPDATA%\Claude\claude_desktop_config.json
Add this configuration:
JSON

{
  "mcpServers": {
    "serpdata": {
      "command": "npx",
      "args": ["-y", "serpdata-mcp"],
      "env": {
        "SERPDATA_API_KEY": "your-api-key-here"
      }
    },
  }
}
Restart Claude Desktop
Start using SerpData in your conversations!
📦 Package:serpdata-mcp on npm

Example usage in Claude:
TEXT

# Use in Claude conversation
"Search for 'pizza new york' using SerpData"
💰 Pricing & Plans
SerpData uses simple, usage-based pricing. Your credits never expire!

💡 Why SerpData?
Speed: Responses in ~2 seconds
Reliability: 99.9% uptime
Scalability: From a few queries to millions monthly
Simplicity: One API call instead of complicated scraping
🔌 API Query
SerpData API uses a single endpoint for all search queries. Send HTTP GET requests to retrieve structured Google search data in JSON format.

Base Endpoint
All requests are sent to this endpoint:

Bash

https://api.serpdata.io/v1/search
Request Structure
A typical API request includes the endpoint URL, query parameters, and authorization header:

Bash

GET https://api.serpdata.io/v1/search?keyword=pizza+warszawa&hl=pl&gl=pl&device=desktop
Authorization: Bearer YOUR_API_KEY
Query Parameters Overview
Customize your search using these parameters:

Bash

# Required parameter
keyword=pizza+warszawa          # Your search term (URL encoded)

# Optional parameters
hl=pl                          # Interface language (Polish)
gl=pl                          # Geographic location (Poland)
device=desktop                 # Device type (desktop/mobile)
# num parameter is temporarily unavailable due to Google API changes