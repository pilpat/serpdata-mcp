"""
LLM handler for multi-provider AI analysis.

This module provides a unified interface for calling different LLM APIs
(OpenAI, Anthropic, Google) to analyze SERP data and generate SEO recommendations.
"""

import json
import os
import time
from typing import Any, Dict, Optional

import aiohttp

from models import LLMConfig, LLMProvider
from prompts import format_prompt


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMHandler:
    """Unified handler for multiple LLM providers."""

    def __init__(
        self,
        provider: LLMProvider,
        api_key: str,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        timeout: int = 30
    ):
        self.provider = provider
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Set default models for each provider
        self.model = model or self._get_default_model(provider)

        # Validate configuration
        if not api_key:
            raise ValueError(f"API key is required for {provider.value}")

    def _get_default_model(self, provider: LLMProvider) -> str:
        """Get default model for each provider."""
        defaults = {
            LLMProvider.OPENAI: "gpt-4-turbo-preview",
            LLMProvider.ANTHROPIC: "claude-3-sonnet-20240229",
            LLMProvider.GOOGLE: "gemini-2.5-flash"
        }
        return defaults[provider]

    async def analyze_serp(
        self,
        serp_json: Dict[str, Any],
        prompt_template: str = "full_analysis",
        business_context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze SERP data using the specified LLM provider.

        Args:
            serp_json: Raw SERP data from SerpData API
            prompt_template: Name of prompt template to use
            business_context: Optional business context for analysis
            **kwargs: Additional parameters for prompt formatting

        Returns:
            Structured analysis results as dictionary

        Raises:
            LLMError: If the analysis fails
        """
        start_time = time.time()

        try:
            # Format the prompt with SERP data
            formatted_prompt = format_prompt(
                template_name=prompt_template,
                serp_json=serp_json,
                business_context=business_context,
                **kwargs
            )

            # Call the appropriate LLM API
            if self.provider == LLMProvider.OPENAI:
                result = await self._call_openai(formatted_prompt)
            elif self.provider == LLMProvider.ANTHROPIC:
                result = await self._call_anthropic(formatted_prompt)
            elif self.provider == LLMProvider.GOOGLE:
                result = await self._call_google(formatted_prompt)
            else:
                raise LLMError(f"Unsupported provider: {self.provider}")

            # Add processing metadata
            result["processing_time"] = time.time() - start_time
            result["provider"] = self.provider.value
            result["model"] = self.model

            return result

        except Exception as e:
            raise LLMError(f"Analysis failed: {str(e)}") from e

    async def _call_openai(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API with structured output."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a semantic SEO expert. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"}
        }

        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMError(f"OpenAI API error {response.status}: {error_text}")

                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]

                    # Parse the JSON response
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError as e:
                        raise LLMError(f"Invalid JSON response from OpenAI: {e}")

            except aiohttp.ClientError as e:
                raise LLMError(f"OpenAI API connection error: {e}")

    async def _call_anthropic(self, prompt: str) -> Dict[str, Any]:
        """Call Anthropic Claude API."""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": f"{prompt}\n\nPlease respond with valid JSON only, following the exact structure specified in the prompt."
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens or 4000
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMError(f"Anthropic API error {response.status}: {error_text}")

                    result = await response.json()
                    content = result["content"][0]["text"]

                    # Extract JSON from response (Claude sometimes adds explanation)
                    try:
                        # Try to find JSON in the response
                        json_start = content.find("{")
                        json_end = content.rfind("}") + 1

                        if json_start != -1 and json_end != -1:
                            json_content = content[json_start:json_end]
                            return json.loads(json_content)
                        else:
                            # If no JSON brackets found, try parsing the whole content
                            return json.loads(content)

                    except json.JSONDecodeError as e:
                        raise LLMError(f"Invalid JSON response from Anthropic: {e}\nContent: {content}")

            except aiohttp.ClientError as e:
                raise LLMError(f"Anthropic API connection error: {e}")

    async def _call_google(self, prompt: str) -> Dict[str, Any]:
        """Call Google Gemini API using structured output."""
        try:
            # Import the new Google GenAI client and Gemini-compatible models
            from google import genai
            from gemini_models import GeminiContentRecommendation, convert_to_standard_model

            # Create async client instance
            client = genai.Client(api_key=self.api_key)

            # For structured output, we need to use the Gemini-compatible model
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": GeminiContentRecommendation,
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens or 4000,
                    # Disable thinking for better JSON compliance
                    "thinking_config": {"thinking_budget": 0}
                }
            )

            # Convert the Gemini response to standard format
            if response.parsed:
                return convert_to_standard_model(response.parsed)
            else:
                # Fallback: parse JSON text and convert
                gemini_data = json.loads(response.text)
                gemini_obj = GeminiContentRecommendation(**gemini_data)
                return convert_to_standard_model(gemini_obj)

        except ImportError:
            # Fallback to aiohttp approach if google-genai not available
            return await self._call_google_fallback(prompt)
        except Exception as e:
            raise LLMError(f"Google API error: {str(e)}")

    async def _call_google_fallback(self, prompt: str) -> Dict[str, Any]:
        """Fallback method using aiohttp for Google API calls."""
        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"{prompt}\n\nIMPORTANT: Respond with valid JSON only, following the exact structure specified."
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens or 4000,
                "responseMimeType": "application/json"
            }
        }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMError(f"Google API error {response.status}: {error_text}")

                    result = await response.json()

                    if "candidates" not in result or not result["candidates"]:
                        raise LLMError("No candidates in Google API response")

                    content = result["candidates"][0]["content"]["parts"][0]["text"]

                    # Enhanced JSON parsing with better error handling
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError as e:
                        # Try to extract JSON from content
                        try:
                            json_start = content.find("{")
                            json_end = content.rfind("}") + 1
                            if json_start != -1 and json_end != -1:
                                json_content = content[json_start:json_end]
                                return json.loads(json_content)
                            else:
                                raise LLMError(f"No valid JSON found in response: {content[:200]}...")
                        except json.JSONDecodeError:
                            raise LLMError(f"Invalid JSON response from Google: {e}")

            except aiohttp.ClientError as e:
                raise LLMError(f"Google API connection error: {e}")

    async def quick_analysis(
        self,
        serp_json: Dict[str, Any],
        business_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform quick SERP analysis with simplified prompt."""
        return await self.analyze_serp(
            serp_json=serp_json,
            prompt_template="quick_insights",
            business_context=business_context
        )

    async def competition_analysis(
        self,
        serp_json: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Focus on competitive analysis only."""
        return await self.analyze_serp(
            serp_json=serp_json,
            prompt_template="competition"
        )

    async def content_structure_analysis(
        self,
        serp_json: Dict[str, Any],
        business_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate detailed content structure recommendations."""
        return await self.analyze_serp(
            serp_json=serp_json,
            prompt_template="content_structure",
            business_context=business_context
        )

    async def entity_analysis(
        self,
        serp_json: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform semantic entity analysis."""
        return await self.analyze_serp(
            serp_json=serp_json,
            prompt_template="entity_analysis"
        )

    @classmethod
    def from_config(cls, config: LLMConfig, api_key: str) -> "LLMHandler":
        """Create LLMHandler from configuration object."""
        return cls(
            provider=config.provider,
            api_key=api_key,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout
        )

    @classmethod
    def from_env(
        cls,
        provider: Optional[LLMProvider] = None,
        api_key_env_var: Optional[str] = None
    ) -> "LLMHandler":
        """Create LLMHandler from environment variables."""
        # Determine provider
        if provider is None:
            provider_str = os.getenv("LLM_PROVIDER", "openai").lower()
            try:
                provider = LLMProvider(provider_str)
            except ValueError:
                raise ValueError(f"Invalid LLM_PROVIDER: {provider_str}")

        # Determine API key environment variable
        if api_key_env_var is None:
            key_vars = {
                LLMProvider.OPENAI: "OPENAI_API_KEY",
                LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
                LLMProvider.GOOGLE: "GOOGLE_API_KEY"
            }
            api_key_env_var = key_vars.get(provider, "LLM_API_KEY")

        # Get API key
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env_var} is required")

        # Get optional configuration
        model = os.getenv("LLM_MODEL")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        max_tokens = os.getenv("LLM_MAX_TOKENS")
        timeout = int(os.getenv("LLM_TIMEOUT", "30"))

        return cls(
            provider=provider,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=int(max_tokens) if max_tokens else None,
            timeout=timeout
        )