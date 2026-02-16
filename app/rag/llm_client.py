"""Ollama LLM HTTP client.

Calls the local Ollama /api/generate endpoint with retry logic and
configurable timeouts. Returns generated text or raises descriptive errors.
"""

import logging
import time

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

# Retry config
MAX_RETRIES = 2
RETRY_BACKOFF = 1.0  # seconds, doubled each retry


class OllamaError(Exception):
    """Raised when Ollama call fails after retries."""


async def generate(prompt: str) -> str:
    """Call Ollama /api/generate and return the full response text.

    Uses non-streaming mode for simplicity. Retries on transient failures.

    Args:
        prompt: Full prompt string (system + context + question).

    Returns:
        Generated text from the model.

    Raises:
        OllamaError: If all retries are exhausted or model is not found.
    """
    settings = get_settings()
    url = f"{settings.ollama_url}/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": settings.temperature,
            "top_p": 0.9,
        },
    }

    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            start = time.perf_counter()
            async with httpx.AsyncClient(timeout=httpx.Timeout(
                connect=5.0,
                read=float(settings.ollama_timeout),
                write=5.0,
                pool=5.0,
            )) as client:
                response = await client.post(url, json=payload)

            elapsed = time.perf_counter() - start

            if response.status_code == 404:
                raise OllamaError(
                    f"Model '{settings.ollama_model}' not found. "
                    f"Run: ollama pull {settings.ollama_model}"
                )

            response.raise_for_status()

            data = response.json()
            answer = data.get("response", "").strip()
            logger.info(
                "Ollama response: model=%s tokens=%s latency=%.1fs",
                settings.ollama_model,
                data.get("eval_count", "?"),
                elapsed,
            )
            return answer

        except OllamaError:
            raise  # don't retry model-not-found
        except httpx.ConnectError as e:
            last_error = e
            logger.warning(
                "Ollama connection failed (attempt %d/%d): %s",
                attempt + 1,
                MAX_RETRIES + 1,
                e,
            )
        except httpx.ReadTimeout as e:
            last_error = e
            logger.warning(
                "Ollama read timeout after %ds (attempt %d/%d)",
                settings.ollama_timeout,
                attempt + 1,
                MAX_RETRIES + 1,
            )
        except Exception as e:
            last_error = e
            logger.warning(
                "Ollama unexpected error (attempt %d/%d): %s",
                attempt + 1,
                MAX_RETRIES + 1,
                e,
            )

        if attempt < MAX_RETRIES:
            backoff = RETRY_BACKOFF * (2**attempt)
            logger.info("Retrying in %.1fs...", backoff)
            time.sleep(backoff)  # blocking sleep OK for retry backoff

    raise OllamaError(
        f"Ollama unavailable after {MAX_RETRIES + 1} attempts: {last_error}"
    )


async def check_health() -> bool:
    """Check if Ollama is reachable.

    Returns:
        True if Ollama /api/tags responds with 200, False otherwise.
    """
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{settings.ollama_url}/api/tags")
            return r.status_code == 200
    except Exception:
        return False
