"""Application configuration via environment variables.

Uses pydantic-settings to load from .env file with type validation.
No env prefix — plain variable names for hackathon simplicity.
"""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── API ──────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    log_level: str = "INFO"

    # ── LLM Provider ─────────────────────────────────────────
    llm_provider: str = "ollama"  # "ollama" | "huggingface"

    # ── Ollama (local) ───────────────────────────────────────
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout: int = 60
    temperature: float = 0.1

    # ── Hugging Face Inference API (cloud) ───────────────────
    hf_api_key: str = ""
    hf_model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    hf_timeout: int = 120

    # ── Vector DB ────────────────────────────────────────────
    chroma_dir: str = "./data/chroma"
    collection_name: str = "alfabolt_policies"
    top_k: int = 5
    min_similarity: float = 0.4

    # ── Embedding ────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Paths ────────────────────────────────────────────────
    knowledge_dir: str = "./knowledge"

    # ── Security ─────────────────────────────────────────────
    reindex_token: str

    @property
    def knowledge_path(self) -> Path:
        return Path(self.knowledge_dir).resolve()

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_dir).resolve()


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
