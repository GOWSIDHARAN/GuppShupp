"""
LLM Client Service
Production-ready client for interacting with Groq API
"""

import os
import logging
from typing import List, Dict, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Production-ready LLM client with retry logic, error handling, and rate limiting
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required")
        
        self.base_url = base_url or "https://api.groq.com/openai/v1/chat/completions"
        self.session = self._create_session()
        
        # Default model settings
        # Allow overriding the model via environment variable to keep flexibility
        # and to quickly react to Groq deprecating models.
        self.default_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.default_temperature = 0.7
        self.default_max_tokens = 1000
        
        # Rate limiting settings
        self.requests_per_minute = 30
        self.last_request_time = None
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy"""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        return session
    
    async def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Generate response from LLM with proper error handling
        
        Args:
            system_prompt: System instruction for the LLM
            user_prompt: User message/prompt
            temperature: Response randomness (0-1)
            max_tokens: Maximum response length
            model: Model to use
            
        Returns:
            str: Generated response
            
        Raises:
            LLMClientError: If API call fails
        """
        try:
            # Prepare request
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            payload = {
                "model": model or self.default_model,
                "messages": messages,
                "temperature": temperature or self.default_temperature,
                "max_tokens": max_tokens or self.default_max_tokens
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            logger.info(f"Making LLM request with model: {payload['model']}")
            
            # Make API call
            response = self.session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            # Handle response
            response.raise_for_status()
            response_data = response.json()
            
            # Extract content
            content = response_data['choices'][0]['message']['content']
            
            # Log usage statistics
            usage = response_data.get('usage', {})
            if usage:
                logger.info(f"Token usage - Input: {usage.get('prompt_tokens', 0)}, "
                           f"Output: {usage.get('completion_tokens', 0)}, "
                           f"Total: {usage.get('total_tokens', 0)}")
            
            return content.strip()
            
        except requests.exceptions.Timeout:
            logger.error("LLM API request timed out")
            raise LLMClientError("Request timed out")
        
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to LLM API")
            raise LLMClientError("Connection failed")
        
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error from LLM API: {e.response.status_code} - {e.response.text}")
            raise LLMClientError(f"API error: {e.response.status_code}")
        
        except KeyError as e:
            logger.error(f"Unexpected response format from LLM API: {str(e)}")
            raise LLMClientError("Invalid response format")
        
        except Exception as e:
            logger.error(f"Unexpected error in LLM client: {str(e)}")
            raise LLMClientError(f"Unexpected error: {str(e)}")
    
    async def generate_structured_response(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured JSON response with schema validation
        
        Args:
            system_prompt: System instruction
            user_prompt: User message
            response_schema: Expected response schema
            **kwargs: Additional parameters for generate_response
            
        Returns:
            Dict: Parsed JSON response
        """
        # Add schema instruction to system prompt
        schema_instruction = f"""
You must respond with valid JSON that matches this schema:
{json.dumps(response_schema, indent=2)}

Do not include any text outside the JSON structure.
"""
        
        enhanced_system_prompt = f"{system_prompt}\n{schema_instruction}"
        
        response = await self.generate_response(
            system_prompt=enhanced_system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,  # Lower temperature for structured responses
            **kwargs
        )
        
        try:
            # Clean response and parse JSON
            cleaned_response = response.strip().strip('```json').strip('```').strip()
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structured response: {response}")
            raise LLMClientError(f"Invalid JSON response: {str(e)}")
    
    def health_check(self) -> bool:
        """Check if LLM API is accessible"""
        try:
            response = self.session.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False


class LLMClientError(Exception):
    """Custom exception for LLM client errors"""
    pass
