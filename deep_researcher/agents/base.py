"""Base agent class wrapping the Google GenAI SDK with async support."""
import os
import json
from typing import Type, TypeVar, Optional, Any
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

DEPRECATED_MODEL_MAP = {
    "gemini-2.5-pro": "gemini-3.1-pro-preview",
    "gemini-1.5-pro": "gemini-3.1-pro-preview",
    "gemini-2.5-flash": "gemini-3.8-flash",
    "gemini-2.5-flash-lite": "gemini-3.8-flash",
    "models/gemini-2.5-pro": "gemini-3.1-pro-preview",
    "models/gemini-1.5-pro": "gemini-3.1-pro-preview",
    "models/gemini-2.5-flash": "gemini-3.8-flash",
}


def resolve_model(model_name: str) -> str:
    """Automatically maps retired or deprecated Gemini models to their current replacements."""
    return DEPRECATED_MODEL_MAP.get(model_name, model_name)


T = TypeVar("T", bound=BaseModel)


class BaseAgent:
    """Base class for all specialized agents in the deep researcher team."""

    def __init__(self, name: str, default_model: str = "gemini-3.8-flash"):
        self.name = name
        self.default_model = resolve_model(default_model)
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Please configure it in your .env file or environment."
            )
        if genai is None:
            raise ImportError(
                "The 'google-genai' library is required. Install it with: pip install google-genai"
            )
        http_options = {"api_version": "v1beta"}
        self.client = genai.Client(api_key=api_key, http_options=http_options)

    async def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> T:
        """Generates a structured Pydantic response asynchronously."""
        target_model = resolve_model(model or self.default_model)
        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=response_schema,
            system_instruction=system_instruction,
        )

        response = await self.client.aio.models.generate_content(
            model=target_model,
            contents=prompt,
            config=config,
        )

        # The SDK automatically parses schema or provides JSON text
        raw_text = response.text
        try:
            return response_schema.model_validate_json(raw_text)
        except Exception:
            # Fallback in case of markdown formatting
            cleaned = raw_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            return response_schema.model_validate_json(cleaned.strip())

    async def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
    ) -> str:
        """Generates raw text / markdown output asynchronously."""
        target_model = resolve_model(model or self.default_model)
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
        )

        response = await self.client.aio.models.generate_content(
            model=target_model,
            contents=prompt,
            config=config,
        )
        return response.text or ""
