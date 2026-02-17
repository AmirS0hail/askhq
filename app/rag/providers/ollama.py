"""Ollama LLM provider.

Calls the local Ollama /api/generate endpoint with retry logic and
configurable timeouts. Extracted from the original llm_client module.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from app.rag.llm_provider import LLMError

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_BACKOFF = 1.0  # seconds, doubled each retry

# asdasd

class OllamaProvider:
    """LLM provider backed by a local Ollama server."""

    def __init__(
        self,
        url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout: int = 60,
        temperature: float = 0.1,
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    async def generate(self, prompt: str) -> str:
        """Call Ollama /api/generate and return the full response text.

        Uses non-streaming mode for simplicity. Retries on transient failures.

        Raises:
            LLMError: If all retries are exhausted or model is not found.
        """
        endpoint = f"{self.url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": 0.9,
            },
        }

        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                start = time.perf_counter()
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        connect=5.0,
                        read=float(self.timeout),
                        write=5.0,
                        pool=5.0,
                    )
                ) as client:
                    response = await client.post(endpoint, json=payload)

                elapsed = time.perf_counter() - start

                if response.status_code == 404:
                    raise LLMError(
                        f"Ollama model '{self.model}' not found. "
                        f"Run: ollama pull {self.model}"
                    )

                response.raise_for_status()

                data = response.json()
                answer = data.get("response", "").strip()
                logger.info(
                    "Ollama response: model=%s tokens=%s latency=%.1fs",
                    self.model,
                    data.get("eval_count", "?"),
                    elapsed,
                )
                return answer

            except LLMError:
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
                    self.timeout,
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
                await asyncio.sleep(backoff)

        raise LLMError(
            f"Ollama unavailable after {MAX_RETRIES + 1} attempts: {last_error}"
        )

    async def check_health(self) -> bool:
        """Check if Ollama is reachable.

        Returns:
            True if Ollama /api/tags responds with 200, False otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{self.url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False
