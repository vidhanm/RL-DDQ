"""
NVIDIA NIM API Client
Wrapper for NVIDIA NIM API - uses Llama 3.3 Nemotron Super 49B v1.5
Best model for conversational AI and instruction following
"""

import os
import json
import time
from typing import Dict, Optional
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.config import LLMConfig
from .prompts import (
    AGENT_SYSTEM_PROMPT,
    DEBTOR_SYSTEM_PROMPT,
    get_agent_utterance_prompt,
    get_debtor_response_prompt
)


class NVIDIAClient:
    """Wrapper for NVIDIA NIM API with retry logic and semantic caching"""

    def __init__(self, api_key: Optional[str] = None, use_cache: bool = True):
        """
        Initialize NVIDIA NIM client

        Args:
            api_key: NVIDIA API key (if None, reads from environment)
            use_cache: Whether to use semantic caching for responses
        """
        if api_key is None:
            api_key = os.getenv("NVIDIA_API_KEY")

        if not api_key:
            raise ValueError(
                "NVIDIA API key not found. Set NVIDIA_API_KEY environment variable or pass api_key parameter."
            )

        # Initialize OpenAI client pointed at NVIDIA endpoint
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )

        # Using Meta Llama 3.1 70B Instruct
        # - Excellent for conversations without thinking mode
        # - Fast and reliable
        # Note: Llama 3.3 Nemotron has <think> tags that cause issues with max_tokens
        self.model = "qwen/qwen3-235b-a22b"

        # Statistics
        self.total_calls = 0
        self.total_tokens_prompt = 0
        self.total_tokens_completion = 0
        self.failed_calls = 0
        
        # Semantic cache for response caching
        self.cache = None
        if use_cache:
            try:
                from .semantic_cache import SemanticCache
                self.cache = SemanticCache(
                    threshold=0.85,
                    max_size=10000,
                    cache_dir="data/llm_cache"
                )
                print("[OK] Semantic cache enabled")
            except Exception as e:
                print(f"[WARN] Semantic cache disabled: {e}")


    def generate_agent_utterance(
        self,
        strategy: str,
        conversation_history: str,
        turn: int
    ) -> str:
        """
        Generate agent utterance for given strategy

        Args:
            strategy: Strategy name (e.g., "empathetic_listening")
            conversation_history: Previous conversation text
            turn: Current turn number

        Returns:
            Generated utterance string
        """
        prompt = get_agent_utterance_prompt(strategy, conversation_history, turn)

        try:
            response = self._call_api(
                system_prompt=AGENT_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=LLMConfig.TEMPERATURE_AGENT
            )
            return response.strip()

        except Exception as e:
            print(f"Error generating agent utterance: {e}")
            # Fallback
            return f"I'd like to discuss your account with you."

    def generate_debtor_response(
        self,
        debtor_context: str,
        agent_utterance: str,
        conversation_history: str
    ) -> Dict:
        """
        Generate debtor response with updated state

        Args:
            debtor_context: Debtor persona and state info
            agent_utterance: What agent just said
            conversation_history: Previous conversation

        Returns:
            Dictionary with response and updated state
        """
        prompt = get_debtor_response_prompt(
            debtor_context,
            agent_utterance,
            conversation_history
        )

        try:
            response_text = self._call_api(
                system_prompt=DEBTOR_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=LLMConfig.TEMPERATURE_DEBTOR
            )

            # Parse JSON response
            response_data = self._parse_json_response(response_text)

            # Validate required fields
            required_fields = ["response", "new_sentiment", "new_cooperation", "new_engagement"]
            for field in required_fields:
                if field not in response_data:
                    print(f"Warning: Missing field '{field}' in debtor response")
                    response_data[field] = 0.0 if field != "response" else "..."

            return response_data

        except Exception as e:
            print(f"Error generating debtor response: {e}")
            # Fallback response
            return {
                "response": "I understand. Let me think about this.",
                "new_sentiment": 0.0,
                "new_cooperation": 0.3,
                "new_engagement": 0.5,
                "shared_situation": False,
                "feels_understood": False,
                "reasoning": "Error occurred, using fallback"
            }

    def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7
    ) -> str:
        """
        Make API call with retry logic and semantic caching

        Args:
            system_prompt: System message
            user_prompt: User message
            temperature: Sampling temperature

        Returns:
            Response text
        """
        # Create cache key from prompts
        cache_key = f"{system_prompt}|||{user_prompt}"
        
        # Check cache first
        if self.cache:
            cached_response = self.cache.get(cache_key)
            if cached_response:
                return cached_response
        
        for attempt in range(LLMConfig.MAX_RETRIES):
            try:
                # Add instruction to disable thinking mode for Qwen and other models
                enhanced_system = system_prompt + "\n\nIMPORTANT: Provide your response directly. Do NOT use <think> tags or show your reasoning process. Output ONLY the final answer."

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": enhanced_system},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=LLMConfig.MAX_TOKENS,
                    timeout=LLMConfig.API_TIMEOUT
                )

                # Update statistics
                self.total_calls += 1
                if hasattr(response, 'usage'):
                    self.total_tokens_prompt += response.usage.prompt_tokens
                    self.total_tokens_completion += response.usage.completion_tokens

                # Get content and strip <think> tags (some models output these)
                content = response.choices[0].message.content
                
                # Handle None content
                if content is None:
                    raise ValueError("API returned empty content")
                
                content = self._strip_think_tags(content)
                
                # Store in cache for future similar prompts
                if self.cache:
                    self.cache.store(cache_key, content)

                return content

            except RateLimitError as e:
                print(f"Rate limit hit. Waiting before retry... (attempt {attempt + 1}/{LLMConfig.MAX_RETRIES})")
                time.sleep(2 ** attempt)  # Exponential backoff
                if attempt == LLMConfig.MAX_RETRIES - 1:
                    self.failed_calls += 1
                    raise

            except APIConnectionError as e:
                print(f"Connection error. Retrying... (attempt {attempt + 1}/{LLMConfig.MAX_RETRIES})")
                time.sleep(1)
                if attempt == LLMConfig.MAX_RETRIES - 1:
                    self.failed_calls += 1
                    raise

            except APIError as e:
                print(f"API error: {e}")
                self.failed_calls += 1
                raise

            except Exception as e:
                print(f"Unexpected error: {e}")
                self.failed_calls += 1
                raise

    def _strip_think_tags(self, text: str) -> str:
        """
        Remove <think> tags that Llama Nemotron outputs

        Llama Nemotron sometimes wraps reasoning in <think> tags, followed by the actual answer.
        We want to keep only the content after </think>.

        Args:
            text: Raw text with possible <think> tags

        Returns:
            Cleaned text
        """
        import re

        # If there's a closing </think> tag, take everything after it
        if '</think>' in text:
            parts = text.split('</think>', 1)
            if len(parts) > 1:
                return parts[1].strip()

        # Otherwise, try to remove <think>...</think> blocks but keep other content
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # If nothing left after cleaning, return original (model might not be using think tags)
        if cleaned.strip():
            return cleaned.strip()

        return text.strip()

    def _parse_json_response(self, response_text: str) -> Dict:
        """
        Parse JSON from response text

        Args:
            response_text: Raw response text

        Returns:
            Parsed dictionary
        """
        import re
        
        # Try direct parsing
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Try finding JSON object
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = response_text[start:end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # FALLBACK: Try to extract "response" field from truncated JSON
        # Pattern: "response": "some text..."
        response_match = re.search(r'"response"\s*:\s*"([^"]+)', response_text)
        if response_match:
            extracted_response = response_match.group(1)
            # Return a valid dict with defaults for other fields
            return {
                "response": extracted_response,
                "new_sentiment": 0.0,
                "new_cooperation": 0.3,
                "new_engagement": 0.5,
                "shared_situation": False,
                "feels_understood": False,
                "reasoning": "Extracted from truncated JSON"
            }

        # Failed to parse
        raise ValueError(f"Could not parse JSON from response: {response_text[:100]}...")

    def get_statistics(self) -> Dict:
        """Get usage statistics"""
        return {
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "success_rate": (self.total_calls - self.failed_calls) / max(1, self.total_calls),
            "total_tokens_prompt": self.total_tokens_prompt,
            "total_tokens_completion": self.total_tokens_completion,
            "total_tokens": self.total_tokens_prompt + self.total_tokens_completion,
            "model": self.model
        }

    def print_statistics(self):
        """Print usage statistics"""
        stats = self.get_statistics()
        print("\n" + "="*50)
        print("NVIDIA NIM API Usage Statistics")
        print("="*50)
        print(f"Model: {stats['model']}")
        print(f"Total API calls: {stats['total_calls']}")
        print(f"Failed calls: {stats['failed_calls']}")
        print(f"Success rate: {stats['success_rate']:.1%}")
        print(f"Total tokens (prompt): {stats['total_tokens_prompt']:,}")
        print(f"Total tokens (completion): {stats['total_tokens_completion']:,}")
        print(f"Total tokens: {stats['total_tokens']:,}")

        # NVIDIA pricing
        print(f"Cost:(NVIDIA beta)")
        print("="*50)
        
        # Print cache statistics if enabled
        if self.cache:
            self.cache.print_stats()
        print("")

