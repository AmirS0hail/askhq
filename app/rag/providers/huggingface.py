"""Hugging Face Inference API provider.

Calls the public HF Inference API for text generation.
Handles 503 cold-start retries (model loading) automatically.
"""

import asyncio
import logging
import time

import httpx

from app.rag.llm_provider import LLMError

logger = logging.getLogger(__name__)

HF_API_BASE = "https://api-inference.huggingface.co/models"
MAX_COLD_START_WAIT = 30  # seconds cap for 503 retry


class HuggingFaceProvider:
    """LLM provider backed by Hugging Face Inference API."""

    def __init__(
        self,
        api_key: str,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
        timeout: int = 120,
        temperature: float = 0.1,
    ) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.timeout = timeout
        self.temperature = temperature
        self.endpoint = f"{HF_API_BASE}/{model_id}"

    async def generate(self, prompt: str) -> str:
        """Send prompt to HF Inference API and return generated text.

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
            "inputs": prompt,
            "parameters": {
                "temperature": self.temperature,
                "max_new_tokens": 512,
                "return_full_text": False,
            },
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
                        self.endpoint, json=payload, headers=headers
                    )

                elapsed = time.perf_counter() - start

                # 503 = model loading (cold start on free tier)
                if response.status_code == 503 and attempt == 0:
                    body = response.json()
                    wait_time = min(
                        body.get("estimated_time", 20),
                        MAX_COLD_START_WAIT,
                    )
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

                if response.status_code != 200:
                    error_text = response.text[:200]
                    raise LLMError(
                        f"HuggingFace API error {response.status_code}: "
                        f"{error_text}"
                    )

                data = response.json()

                # HF returns [{"generated_text": "..."}] for text-generation
                if isinstance(data, list) and len(data) > 0:
                    answer = data[0].get("generated_text", "").strip()
                elif isinstance(data, dict):
                    answer = data.get("generated_text", "").strip()
                else:
                    raise LLMError(
                        f"Unexpected HuggingFace response format: {type(data)}"
                    )

                if not answer:
                    raise LLMError("HuggingFace returned empty response")

                logger.info(
                    "HuggingFace response: model=%s chars=%d latency=%.1fs",
                    self.model_id,
                    len(answer),
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
        """Check if HF Inference API is reachable for this model.

        Returns:
            True if the model endpoint responds (200 or 503 loading),
            False on connection failure or auth error.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(self.endpoint, headers=headers)
                # 200 = ready, 503 = loading (still reachable)
                return r.status_code in (200, 503)
        except Exception:
            return False
