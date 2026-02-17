"""Unit tests for LLM providers.

Tests the provider factory, OllamaProvider, and HuggingFaceProvider
with mocked HTTP calls (no real LLM needed).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.rag.llm_provider import LLMError, get_llm_provider
from app.rag.providers.huggingface import HuggingFaceProvider
from app.rag.providers.ollama import OllamaProvider


# ── Factory tests ────────────────────────────────────────────


class TestGetLLMProvider:
    def test_returns_ollama_by_default(self):
        get_llm_provider.cache_clear()
        mock_settings = MagicMock()
        mock_settings.llm_provider = "ollama"
        mock_settings.ollama_url = "http://localhost:11434"
        mock_settings.ollama_model = "llama3.1:8b"
        mock_settings.ollama_timeout = 60
        mock_settings.temperature = 0.1

        with patch("app.rag.llm_provider.get_settings", return_value=mock_settings):
            provider = get_llm_provider()

        assert isinstance(provider, OllamaProvider)
        get_llm_provider.cache_clear()

    def test_returns_huggingface_when_configured(self):
        get_llm_provider.cache_clear()
        mock_settings = MagicMock()
        mock_settings.llm_provider = "huggingface"
        mock_settings.hf_api_key = "hf_test_key"
        mock_settings.hf_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        mock_settings.hf_timeout = 120
        mock_settings.temperature = 0.1

        with patch("app.rag.llm_provider.get_settings", return_value=mock_settings):
            provider = get_llm_provider()

        assert isinstance(provider, HuggingFaceProvider)
        get_llm_provider.cache_clear()

    def test_huggingface_requires_api_key(self):
        get_llm_provider.cache_clear()
        mock_settings = MagicMock()
        mock_settings.llm_provider = "huggingface"
        mock_settings.hf_api_key = ""

        with patch("app.rag.llm_provider.get_settings", return_value=mock_settings):
            with pytest.raises(ValueError, match="HF_API_KEY"):
                get_llm_provider()
        get_llm_provider.cache_clear()

    def test_unknown_provider_falls_back_to_ollama(self):
        get_llm_provider.cache_clear()
        mock_settings = MagicMock()
        mock_settings.llm_provider = "openai"
        mock_settings.ollama_url = "http://localhost:11434"
        mock_settings.ollama_model = "llama3.1:8b"
        mock_settings.ollama_timeout = 60
        mock_settings.temperature = 0.1

        with patch("app.rag.llm_provider.get_settings", return_value=mock_settings):
            provider = get_llm_provider()

        assert isinstance(provider, OllamaProvider)
        get_llm_provider.cache_clear()


# ── OllamaProvider tests ────────────────────────────────────


class TestOllamaProvider:
    @pytest.fixture
    def provider(self):
        return OllamaProvider(
            url="http://localhost:11434",
            model="llama3.1:8b",
            timeout=60,
            temperature=0.1,
        )

    @pytest.mark.asyncio
    async def test_generate_success(self, provider):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "3 casual leave days per year.",
            "eval_count": 15,
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("app.rag.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            result = await provider.generate("test prompt")

        assert result == "3 casual leave days per year."

    @pytest.mark.asyncio
    async def test_generate_model_not_found_raises(self, provider):
        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("app.rag.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(LLMError, match="not found"):
                await provider.generate("test prompt")

    @pytest.mark.asyncio
    async def test_check_health_returns_true(self, provider):
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("app.rag.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            assert await provider.check_health() is True

    @pytest.mark.asyncio
    async def test_check_health_returns_false_on_error(self, provider):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with patch("app.rag.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            assert await provider.check_health() is False


# ── HuggingFaceProvider tests ────────────────────────────────


class TestHuggingFaceProvider:
    @pytest.fixture
    def provider(self):
        return HuggingFaceProvider(
            api_key="hf_test_key",
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            timeout=120,
            temperature=0.1,
        )

    @staticmethod
    def _chat_response(text):
        """Build a mock OpenAI-compatible chat completions response."""
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {"completion_tokens": 15},
        }

    @pytest.mark.asyncio
    async def test_generate_success(self, provider):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self._chat_response(
            "You get 3 casual leave days per year."
        )

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("app.rag.providers.huggingface.httpx.AsyncClient", return_value=mock_client):
            result = await provider.generate("test prompt")

        assert result == "You get 3 casual leave days per year."

    @pytest.mark.asyncio
    async def test_generate_legacy_list_format(self, provider):
        """Fallback: handle old [{'generated_text': ...}] format."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"generated_text": "Answer from legacy format."}
        ]

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("app.rag.providers.huggingface.httpx.AsyncClient", return_value=mock_client):
            result = await provider.generate("test prompt")

        assert result == "Answer from legacy format."

    @pytest.mark.asyncio
    async def test_generate_retries_on_503(self, provider):
        """503 (model loading) should retry once after waiting."""
        loading_response = MagicMock()
        loading_response.status_code = 503
        loading_response.json.return_value = {"estimated_time": 0.1}

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = self._chat_response(
            "Answer after cold start."
        )

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=[loading_response, success_response])

        mock_sleep = AsyncMock()

        with patch("app.rag.providers.huggingface.httpx.AsyncClient", return_value=mock_client), \
             patch("app.rag.providers.huggingface.asyncio.sleep", mock_sleep):
            result = await provider.generate("test prompt")

        assert result == "Answer after cold start."
        assert mock_client.post.call_count == 2
        mock_sleep.assert_awaited_once_with(0.1)

    @pytest.mark.asyncio
    async def test_generate_invalid_api_key_raises(self, provider):
        mock_response = MagicMock()
        mock_response.status_code = 401

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("app.rag.providers.huggingface.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(LLMError, match="Invalid or missing API key"):
                await provider.generate("test prompt")

    @pytest.mark.asyncio
    async def test_generate_model_not_found_raises(self, provider):
        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("app.rag.providers.huggingface.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(LLMError, match="not found"):
                await provider.generate("test prompt")

    @pytest.mark.asyncio
    async def test_check_health_returns_true(self, provider):
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("app.rag.providers.huggingface.httpx.AsyncClient", return_value=mock_client):
            assert await provider.check_health() is True

    @pytest.mark.asyncio
    async def test_check_health_false_on_503(self, provider):
        """503 during health check means service has issues."""
        mock_response = MagicMock()
        mock_response.status_code = 503

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("app.rag.providers.huggingface.httpx.AsyncClient", return_value=mock_client):
            assert await provider.check_health() is False

    @pytest.mark.asyncio
    async def test_check_health_false_on_connection_error(self, provider):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with patch("app.rag.providers.huggingface.httpx.AsyncClient", return_value=mock_client):
            assert await provider.check_health() is False
