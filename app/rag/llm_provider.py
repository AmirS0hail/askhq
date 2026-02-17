"""LLM provider abstraction.

Defines the LLMProvider protocol and a factory function that returns the
active provider based on the LLM_PROVIDER environment variable.

Supported providers:
    - "ollama"      → local Ollama server (default)
    - "huggingface" → Hugging Face Inference API (cloud)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Protocol, runtime_checkable

from app.config import get_settings

logger = logging.getLogger(__name__)


# ── Base exception ───────────────────────────────────────────


class LLMError(Exception):
    """Raised when an LLM provider call fails after retries."""


# ── Provider protocol ────────────────────────────────────────


@runtime_checkable
class LLMProvider(Protocol):
    """Interface that every LLM backend must implement."""

    async def generate(self, prompt: str) -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt: Full prompt string (system + context + question).

        Returns:
            Generated text from the model.

        Raises:
            LLMError: If the request fails after retries.
        """
        ...

    async def check_health(self) -> bool:
        """Check if the LLM backend is reachable.

        Returns:
            True if the backend responds normally, False otherwise.
        """
        ...


# ── Factory ──────────────────────────────────────────────────


@lru_cache
def get_llm_provider() -> LLMProvider:
    """Return the configured LLM provider singleton.

    Reads ``LLM_PROVIDER`` from settings and instantiates the matching
    implementation. Defaults to Ollama if the value is unrecognised.
    """
    settings = get_settings()
    provider_name = settings.llm_provider.lower().strip()

    if provider_name == "huggingface":
        from app.rag.providers.huggingface import HuggingFaceProvider

        if not settings.hf_api_key:
            raise ValueError(
                "HF_API_KEY must be set when LLM_PROVIDER=huggingface"
            )
        logger.info(
            "LLM provider: HuggingFace (model=%s)", settings.hf_model_id
        )
        return HuggingFaceProvider(
            api_key=settings.hf_api_key,
            model_id=settings.hf_model_id,
            timeout=settings.hf_timeout,
            temperature=settings.temperature,
        )

    # Default: Ollama
    if provider_name != "ollama":
        logger.warning(
            "Unknown LLM_PROVIDER '%s', falling back to ollama",
            provider_name,
        )

    from app.rag.providers.ollama import OllamaProvider

    logger.info(
        "LLM provider: Ollama (url=%s, model=%s)",
        settings.ollama_url,
        settings.ollama_model,
    )
    return OllamaProvider(
        url=settings.ollama_url,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout,
        temperature=settings.temperature,
    )
