"""Hugging Face Inference API provider.

Uses the HF router's OpenAI-compatible chat completions endpoint.
Handles 503 cold-start retries (model loading) automatically.

As of early 2026, HF migrated from api-inference.huggingface.co to
router.huggingface.co with an OpenAI-compatible /v1/chat/completions
format. The old ``inputs`` text-generation endpoint is 410 Gone.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from app.rag.llm_provider import LLMError

logger = logging.getLogger(__name__)

HF_API_BASE = "https://router.huggingface.co"
MAX_COLD_START_WAIT = 30  # seconds cap for 503 retry


class HuggingFaceProvider:
    """LLM provider backed by Hugging Face Inference API (chat completions)."""

    def __init__(
        self,
        api_key: str,
        model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
        timeout: int = 120,
        temperature: float = 0.1,
    ) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.timeout = timeout
        self.temperature = temperature
        self.chat_endpoint = f"{HF_API_BASE}/v1/chat/completions"
        self.health_endpoint = (
            f"https://huggingface.co/api/models/{self.model_id}"
        )

    async def generate(self, prompt: str) -> str:
        """Send prompt to HF chat completions API and return generated text.

        Uses the OpenAI-compatible format: the full prompt (system + context
        + question) is sent as a single user message. The model responds in
        the ``choices[0].message.content`` field.

        Retries once on HTTP 503 (model loading / cold start), using
        the ``estimated_time`` from the response body if available.

        Raises:
            LLMError: If the request fails or the model is unavailable.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": 512,
        }

        last_error: Exception | None = None

        for attempt in range(2):  # initial + 1 retry on 503
            try:
                start = time.perf_counter()
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        connect=10.0,
                        read=float(self.timeout),
                        write=10.0,
                        pool=10.0,
                    )
                ) as client:
                    response = await client.post(
                        self.chat_endpoint, json=payload, headers=headers
                    )

                elapsed = time.perf_counter() - start

                # 503 = model loading (cold start on free tier)
                if response.status_code == 503 and attempt == 0:
                    try:
                        body = response.json()
                        wait_time = min(
                            body.get("estimated_time", 20),
                            MAX_COLD_START_WAIT,
                        )
                    except Exception:
                        wait_time = 20
                    logger.warning(
                        "HuggingFace model loading, retrying in %.0fs...",
                        wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue

                if response.status_code == 401:
                    raise LLMError(
                        "HuggingFace API: Invalid or missing API key. "
                        "Check HF_API_KEY in your .env file."
                    )

                if response.status_code == 404:
                    raise LLMError(
                        f"HuggingFace model '{self.model_id}' not found."
                    )

                if response.status_code not in (200, 201):
                    error_text = response.text[:300]
                    raise LLMError(
                        f"HuggingFace API error {response.status_code}: "
                        f"{error_text}"
                    )

                data = response.json()

                # OpenAI-compatible: {"choices": [{"message": {"content": "..."}}]}
                try:
                    answer = (
                        data["choices"][0]["message"]["content"].strip()
                    )
                except (KeyError, IndexError, TypeError):
                    # Fallback: try old text-generation format
                    if isinstance(data, list) and len(data) > 0:
                        answer = data[0].get("generated_text", "").strip()
                    elif isinstance(data, dict):
                        answer = data.get("generated_text", "").strip()
                    else:
                        raise LLMError(
                            f"Unexpected HuggingFace response format: "
                            f"{str(data)[:200]}"
                        )

                if not answer:
                    raise LLMError("HuggingFace returned empty response")

                usage = data.get("usage", {}) if isinstance(data, dict) else {}
                tokens = usage.get("completion_tokens", "?")
                logger.info(
                    "HuggingFace response: model=%s tokens=%s latency=%.1fs",
                    self.model_id,
                    tokens,
                    elapsed,
                )
                return answer

            except LLMError:
                raise
            except httpx.ConnectError as e:
                last_error = e
                logger.warning("HuggingFace connection failed: %s", e)
            except httpx.ReadTimeout as e:
                last_error = e
                logger.warning(
                    "HuggingFace read timeout after %ds", self.timeout
                )
            except Exception as e:
                last_error = e
                logger.warning("HuggingFace unexpected error: %s", e)
            break  # don't retry on non-503 errors

        raise LLMError(
            f"HuggingFace unavailable: {last_error}"
        )

    async def check_health(self) -> bool:
        """Check if HF Inference API is reachable.

        Pings the HF Hub model info endpoint to confirm the model exists
        and the API key is valid.

        Returns:
            True if the model info endpoint responds 200,
            False on connection failure or auth error.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(self.health_endpoint, headers=headers)
                return r.status_code == 200
        except Exception:
            return False
