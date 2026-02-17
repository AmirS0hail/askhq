"""LLM client — backward-compatible shim.

Delegates to the active LLM provider (Ollama or HuggingFace) selected
via the LLM_PROVIDER environment variable. Preserves the old module-level
``generate()`` and ``check_health()`` functions so existing imports keep
working during the transition.

New code should import from ``app.rag.llm_provider`` directly.
"""

from app.rag.llm_provider import LLMError, get_llm_provider

# Backward-compatible alias — old code catches OllamaError
OllamaError = LLMError


async def generate(prompt: str) -> str:
    """Delegate to the active provider's generate method."""
    provider = get_llm_provider()
    return await provider.generate(prompt)


async def check_health() -> bool:
    """Delegate to the active provider's check_health method."""
    provider = get_llm_provider()
    return await provider.check_health()

