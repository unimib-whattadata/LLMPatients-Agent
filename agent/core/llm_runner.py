import logging
import os

from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Import bases, helpers and constants from the base provider module
from agent.core.llm_provider_base import (
    LLMRunnerBase,
    STOP_SEQUENCES,
    _env_positive_int,
    _env_non_negative_float,
    _parse_optional_positive_int,
    _parse_float,
)

# Expose concrete provider implementations to preserve module-level backwards compatibility
from agent.core.llm_provider_local import LocalLLMRunner
from agent.core.llm_provider_vertex import VertexLLMRunner
from agent.core.llm_provider_ollama import OllamaLLMRunner

# === Configure Logging ===
logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = "local"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 512

# === Load Environment ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
for env_file in (".env", "config/.env"):
    load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, env_file))

rel_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "config/vertex-ai-api-key.json")
abs_path = os.path.join(PROJECT_ROOT, rel_path)

# Set the final environment variable for GCP auth
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = abs_path


@dataclass(frozen=True)
class LLMRunnerConfig:
    """Resolved provider configuration used to instantiate an LLM runner."""

    provider: str
    model_id: Optional[str]
    temperature: float
    max_tokens: int
    cache_path: Optional[str] = None
    max_model_len: Optional[int] = None


def _resolve_runner_config(
    *,
    provider: Optional[str],
    model_id: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
    max_model_len: Optional[int],
    cache_path: Optional[str],
) -> LLMRunnerConfig:
    """Merge explicit overrides with environment variables into one typed config."""
    resolved_provider = (provider or os.getenv("model_provider", DEFAULT_PROVIDER)).strip().lower()
    resolved_model_id = model_id if model_id is not None else os.getenv("model_id")
    resolved_temperature = (
        temperature
        if temperature is not None
        else _parse_float("temperature", os.getenv("temperature"), DEFAULT_TEMPERATURE)
    )
    resolved_max_tokens = max_tokens if max_tokens is not None else _env_positive_int("max_tokens", DEFAULT_MAX_TOKENS)
    resolved_max_model_len = (
        max_model_len
        if max_model_len is not None
        else _parse_optional_positive_int("max_model_len", os.getenv("max_model_len"))
    )
    resolved_cache_path = cache_path if cache_path is not None else os.getenv("cache_path")
    return LLMRunnerConfig(
        provider=resolved_provider,
        model_id=resolved_model_id,
        temperature=resolved_temperature,
        max_tokens=resolved_max_tokens,
        cache_path=resolved_cache_path,
        max_model_len=resolved_max_model_len,
    )


# === Factory Function to Create LLM Runner ===
def create_llm_runner(
    *,
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_model_len: Optional[int] = None,
    cache_path: Optional[str] = None,
) -> LLMRunnerBase:
    """Factory that instantiates the correct runner based on environment configuration."""
    config = _resolve_runner_config(
        provider=provider,
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        cache_path=cache_path,
    )

    if config.provider == "local":
        return LocalLLMRunner(
            config.model_id,
            config.cache_path,
            config.temperature,
            config.max_tokens,
            max_model_len=config.max_model_len,
        )
    if config.provider == "vertex_ai":
        return VertexLLMRunner(config.model_id, config.temperature, config.max_tokens)
    if config.provider == "ollama":
        base_url = os.getenv("ollama_base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).strip()
        return OllamaLLMRunner(
            model_id=config.model_id,
            base_url=base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    raise ValueError(f"Unsupported model provider: {config.provider}")
