import logging
import os
import random
import threading
import time

from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from abc import ABC, abstractmethod
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, SafetySetting
    from vertexai.generative_models import HarmCategory, HarmBlockThreshold
    VERTEX_AI_AVAILABLE = True
except ImportError:
    vertexai = None
    GenerativeModel = None
    SafetySetting = None
    HarmCategory = None
    HarmBlockThreshold = None
    VERTEX_AI_AVAILABLE = False
try:
    from google.api_core.exceptions import (
        DeadlineExceeded,
        GatewayTimeout,
        InternalServerError,
        ResourceExhausted,
        ServiceUnavailable,
        TooManyRequests,
    )
    GOOGLE_API_RETRYABLE_ERRORS = (
        DeadlineExceeded,
        GatewayTimeout,
        ResourceExhausted,
        TooManyRequests,
        ServiceUnavailable,
        InternalServerError,
    )
    GOOGLE_API_RATE_LIMIT_ERRORS = (ResourceExhausted, TooManyRequests)
except ImportError:
    GOOGLE_API_RETRYABLE_ERRORS = ()
    GOOGLE_API_RATE_LIMIT_ERRORS = ()
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None  # Placeholder for type hints
    SamplingParams = None
try:
    import torch
except ImportError:
    torch = None

# === Configure Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_VERTEX_MAX_ATTEMPTS = 5
DEFAULT_VERTEX_RETRY_BASE_DELAY_SECONDS = 1.0
DEFAULT_VERTEX_RETRY_MAX_DELAY_SECONDS = 60.0
DEFAULT_VERTEX_RATE_LIMIT_COOLDOWN_SECONDS = 15.0
DEFAULT_VERTEX_MIN_REQUEST_INTERVAL_SECONDS = 0.0
VERTEX_NO_TEXT_RECOVERY_MIN_TOKENS = 256
VERTEX_NO_TEXT_RECOVERY_MAX_TOKENS = 512
DEFAULT_PROVIDER = "local"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 512
DEFAULT_VERTEX_LOCATION = "us-central1"
DEFAULT_HF_CACHE_DIR = "~/.cache/huggingface"
STOP_SEQUENCES = ["\nTherapist:", "Therapist:"]
VERTEX_TOP_P = 0.95
VERTEX_TOP_K = 40

# === Load Environment ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
for env_file in (".env", "config/.env"):
    load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, env_file))

rel_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "config/vertex-ai-api-key.json")
abs_path = os.path.join(PROJECT_ROOT, rel_path)

# Set the final environment variable for GCP auth
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = abs_path

_VERTEX_RATE_LIMIT_LOCK = threading.Lock()
_VERTEX_NEXT_REQUEST_AT = 0.0
_VERTEX_LAST_REQUEST_AT = 0.0


@dataclass(frozen=True)
class LLMRunnerConfig:
    """Resolved provider configuration used to instantiate an LLM runner."""

    provider: str
    model_id: Optional[str]
    temperature: float
    max_tokens: int
    cache_path: Optional[str] = None
    max_model_len: Optional[int] = None


def _env_positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        logger.warning("Ignoring invalid %s=%r. Using %s.", name, raw_value, default)
        return default


def _env_non_negative_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
        if value < 0:
            raise ValueError
        return value
    except ValueError:
        logger.warning("Ignoring invalid %s=%r. Using %.2f.", name, raw_value, default)
        return default


def _parse_optional_positive_int(name: str, raw_value: Optional[str]) -> Optional[int]:
    """Parse an optional positive integer, logging and ignoring invalid values."""
    if not raw_value:
        return None
    try:
        value = int(raw_value)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        logger.warning("Ignoring invalid %s=%r. Use a positive integer.", name, raw_value)
        return None


def _parse_float(name: str, raw_value: Optional[str], default: float) -> float:
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r. Using %.2f.", name, raw_value, default)
        return default


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


def _preferred_local_device() -> tuple[int, str]:
    """Return tensor parallel count and vLLM device for the available accelerator."""
    if torch is not None and hasattr(torch, "xpu") and torch.xpu.is_available():
        n_devices = torch.xpu.device_count()
        logger.info("Intel XPU detected. Using %s XPU(s).", n_devices)
        return n_devices, "xpu"

    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    n_devices = 1 if not cuda_devices else cuda_devices.count(",") + 1
    logger.info("Using %s GPU(s) (CUDA_VISIBLE_DEVICES=%s)", n_devices, cuda_devices)
    return n_devices, "auto"

# === Base LLM Runner ===
class LLMRunnerBase(ABC):
    """Minimal interface all backing LLM providers must implement."""
    def __init__(self, temperature: float, max_tokens: int):
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Return a text completion for the provided prompt."""


# === Local vLLM Runner ===
class LocalLLMRunner(LLMRunnerBase):
    """Adapter that executes prompts against a local vLLM engine."""
    def __init__(
        self,
        model_id: str,
        cache_path: Optional[str],
        temperature: float,
        max_tokens: int,
        max_model_len: Optional[int] = None,
    ):
        if not VLLM_AVAILABLE:
            raise ImportError(
                "Local provider requires the `vllm` package plus compatible `torch`. "
                "Install a matching local stack, use the Docker image, or set model_provider=vertex_ai."
            )
        
        super().__init__(temperature, max_tokens)

        if not model_id:
            raise ValueError("Missing `model_id` for LocalLLMRunner")

        self.model_id = model_id
        self.cache_path = cache_path
        # Keep generation length separate from the model context window.
        inferred_context_len = max(self.max_tokens * 4, 4096)
        self.max_model_len = max_model_len if max_model_len is not None else inferred_context_len
        if self.max_model_len < self.max_tokens:
            logger.warning(
                "max_model_len (%s) is lower than max_tokens (%s); raising it to max_tokens.",
                self.max_model_len,
                self.max_tokens,
            )
            self.max_model_len = self.max_tokens
        self.llm = self._build_llm()


    def _build_llm(self) -> LLM:
        """Instantiate the vLLM object with sane defaults and logging."""
        try:
            download_dir = (
                self.cache_path if self.cache_path and os.path.isdir(self.cache_path)
                else os.getenv("HF_HOME", os.path.expanduser(DEFAULT_HF_CACHE_DIR))
            )
            os.makedirs(download_dir, exist_ok=True)

            n_gpus, device = _preferred_local_device()
            logger.info(f"Download dir: {download_dir}")
            logger.info(f"Loading model: {self.model_id}")

            try:
                return LLM(
                    model=self.model_id,
                    tokenizer_mode="auto",
                    trust_remote_code=True,
                    enable_prefix_caching=True,
                    max_model_len=self.max_model_len,
                    download_dir=download_dir,
                    tensor_parallel_size=n_gpus,
                    device=device,
                )
            except TypeError as te:
                if "device" in str(te):
                    logger.warning("vLLM does not accept 'device' argument. Retrying without it.")
                    return LLM(
                        model=self.model_id,
                        tokenizer_mode="auto",
                        trust_remote_code=True,
                        enable_prefix_caching=True,
                        max_model_len=self.max_model_len,
                        download_dir=download_dir,
                        tensor_parallel_size=n_gpus,
                    )
                else:
                    raise
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise

    def generate(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Generate text locally, trimming to stop tokens and handling transient failures."""
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        try:
            sampling_params = SamplingParams(
                temperature=temp,
                max_tokens=max_tok,
                stop=STOP_SEQUENCES,
            )
            outputs = self.llm.generate(prompt, sampling_params=sampling_params)
            return outputs[0].outputs[0].text.strip() if outputs and outputs[0].outputs else "[NO RESPONSE]"
        except Exception as e:
            logger.error(f"Local vLLM generation error: {e}")
            return "[ERROR] Local vLLM failed to generate response."
        
# === Vertex AI Runner ===
class VertexLLMRunner(LLMRunnerBase):
    """Adapter for Google Vertex AI's text-generation APIs with safety tuning."""
    def __init__(self, model_id: str, temperature: float, max_tokens: int):
        super().__init__(temperature, max_tokens)
        self.model_id = model_id

        if not VERTEX_AI_AVAILABLE:
            raise ImportError(
                "Vertex AI support requires the `vertexai` package (google-cloud-aiplatform). "
                "Install it or switch `model_provider` to `local`."
            )

        project = os.getenv("GCP_PROJECT")
        location = os.getenv("GCP_LOCATION", DEFAULT_VERTEX_LOCATION)

        if not model_id or not project:
            raise ValueError("Missing required Vertex AI configuration.")

        vertexai.init(project=project, location=location)
        logger.info(f"Initialized Vertex AI (project={project}, location={location})")

        self.model = GenerativeModel(model_id)
        self.retry_base_delay_seconds = _env_non_negative_float(
            "VERTEX_RETRY_BASE_DELAY_SECONDS",
            DEFAULT_VERTEX_RETRY_BASE_DELAY_SECONDS,
        )
        self.retry_max_delay_seconds = _env_non_negative_float(
            "VERTEX_RETRY_MAX_DELAY_SECONDS",
            DEFAULT_VERTEX_RETRY_MAX_DELAY_SECONDS,
        )
        self.rate_limit_cooldown_seconds = _env_non_negative_float(
            "VERTEX_RATE_LIMIT_COOLDOWN_SECONDS",
            DEFAULT_VERTEX_RATE_LIMIT_COOLDOWN_SECONDS,
        )
        self.min_request_interval_seconds = _env_non_negative_float(
            "VERTEX_MIN_REQUEST_INTERVAL_SECONDS",
            DEFAULT_VERTEX_MIN_REQUEST_INTERVAL_SECONDS,
        )
        self.max_attempts = _env_positive_int("VERTEX_MAX_ATTEMPTS", DEFAULT_VERTEX_MAX_ATTEMPTS)
        
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: HarmBlockThreshold.BLOCK_NONE,
        }

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        if GOOGLE_API_RETRYABLE_ERRORS and isinstance(exc, GOOGLE_API_RETRYABLE_ERRORS):
            return True

        message = str(exc).lower()
        retryable_markers = (
            "408",
            "429",
            "500",
            "502",
            "503",
            "504",
            "resource exhausted",
            "rate limit",
            "too many requests",
            "service unavailable",
            "temporarily unavailable",
            "deadline exceeded",
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
        )
        return any(marker in message for marker in retryable_markers)

    def _retry_delay_seconds(self, attempt: int) -> float:
        backoff = min(
            self.retry_base_delay_seconds * (2 ** max(attempt - 1, 0)),
            self.retry_max_delay_seconds,
        )
        jitter = random.uniform(0.0, min(1.0, backoff * 0.25))
        return backoff + jitter

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> float:
        for attr_name in ("response", "http_response"):
            response = getattr(exc, attr_name, None)
            headers = getattr(response, "headers", None)
            if not headers:
                continue
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            if not retry_after:
                continue
            try:
                return max(float(retry_after), 0.0)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    @staticmethod
    def _is_rate_limited_error(exc: Exception) -> bool:
        if GOOGLE_API_RATE_LIMIT_ERRORS and isinstance(exc, GOOGLE_API_RATE_LIMIT_ERRORS):
            return True

        message = str(exc).lower()
        rate_limit_markers = (
            "429",
            "resource exhausted",
            "rate limit",
            "too many requests",
            "quota exceeded",
        )
        return any(marker in message for marker in rate_limit_markers)

    def _wait_for_request_slot(self) -> None:
        global _VERTEX_LAST_REQUEST_AT
        global _VERTEX_NEXT_REQUEST_AT

        delay_seconds = 0.0
        with _VERTEX_RATE_LIMIT_LOCK:
            now = time.monotonic()
            next_allowed_at = max(
                _VERTEX_NEXT_REQUEST_AT,
                _VERTEX_LAST_REQUEST_AT + self.min_request_interval_seconds,
            )
            scheduled_at = max(now, next_allowed_at)
            delay_seconds = max(0.0, scheduled_at - now)
            _VERTEX_LAST_REQUEST_AT = scheduled_at

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    def _apply_shared_cooldown(self, delay_seconds: float) -> None:
        global _VERTEX_NEXT_REQUEST_AT

        if delay_seconds <= 0:
            return

        with _VERTEX_RATE_LIMIT_LOCK:
            _VERTEX_NEXT_REQUEST_AT = max(_VERTEX_NEXT_REQUEST_AT, time.monotonic() + delay_seconds)

    @staticmethod
    def _candidate_finish_reasons(response) -> list[str]:
        reasons: list[str] = []
        for candidate in getattr(response, "candidates", []) or []:
            finish_reason = getattr(candidate, "finish_reason", None)
            if finish_reason:
                reasons.append(str(finish_reason))
        return reasons

    @staticmethod
    def _candidate_has_visible_parts(response) -> bool:
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            if parts:
                return True
        return False

    def _recovery_max_tokens(self, current_max_tokens: int) -> int:
        return min(
            max(current_max_tokens * 2, VERTEX_NO_TEXT_RECOVERY_MIN_TOKENS),
            max(VERTEX_NO_TEXT_RECOVERY_MAX_TOKENS, current_max_tokens),
        )

    @staticmethod
    def _extract_text_from_candidates(response) -> str:
        text_chunks = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                value = getattr(part, "text", None)
                if value:
                    text_chunks.append(value)
        return "\n".join(text_chunks).strip()

    def generate(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Proxy prompt execution to Vertex AI with consistent config and error handling."""
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        current_max_tok = max_tok

        for attempt in range(1, self.max_attempts + 1):
            try:
                self._wait_for_request_slot()
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temp,
                        "max_output_tokens": current_max_tok,
                        "stop_sequences": STOP_SEQUENCES,
                        "top_p": VERTEX_TOP_P,
                        "top_k": VERTEX_TOP_K,
                    },
                    safety_settings=self.safety_settings
                )
                try:
                    return response.text.strip()
                except Exception as text_error:
                    extracted = self._extract_text_from_candidates(response)
                    if extracted:
                        logger.warning(
                            "Falling back to candidate-part extraction after response.text error: %s",
                            text_error,
                        )
                        return extracted
                    finish_reasons = self._candidate_finish_reasons(response)
                    if (
                        "MAX_TOKENS" in finish_reasons
                        and not self._candidate_has_visible_parts(response)
                        and attempt < self.max_attempts
                    ):
                        next_max_tok = self._recovery_max_tokens(current_max_tok)
                        if next_max_tok > current_max_tok:
                            logger.warning(
                                "Vertex AI returned no visible text and stopped with MAX_TOKENS on attempt %s/%s. "
                                "Retrying with max_output_tokens=%s (was %s).",
                                attempt,
                                self.max_attempts,
                                next_max_tok,
                                current_max_tok,
                            )
                            current_max_tok = next_max_tok
                            continue
                    if "MAX_TOKENS" in finish_reasons and not self._candidate_has_visible_parts(response):
                        logger.error(
                            "Vertex AI stopped with MAX_TOKENS before emitting visible text. "
                            "finish_reasons=%s model=%s requested_max_output_tokens=%s",
                            finish_reasons,
                            self.model_id,
                            current_max_tok,
                        )
                    raise text_error
            except Exception as e:
                retryable = self._is_retryable_error(e)
                if retryable and attempt < self.max_attempts:
                    delay_seconds = max(
                        self._retry_delay_seconds(attempt),
                        self._retry_after_seconds(e),
                    )
                    shared_cooldown_seconds = delay_seconds
                    if self._is_rate_limited_error(e):
                        shared_cooldown_seconds = max(shared_cooldown_seconds, self.rate_limit_cooldown_seconds)
                    self._apply_shared_cooldown(shared_cooldown_seconds)
                    logger.warning(
                        "Vertex AI transient generation error on attempt %s/%s: %s. Retrying in %.2fs.",
                        attempt,
                        self.max_attempts,
                        e,
                        delay_seconds,
                    )
                    time.sleep(delay_seconds)
                    continue

                if retryable:
                    final_cooldown_seconds = max(
                        self._retry_after_seconds(e),
                        self.rate_limit_cooldown_seconds if self._is_rate_limited_error(e) else 0.0,
                    )
                    self._apply_shared_cooldown(final_cooldown_seconds)
                    logger.error(
                        "Vertex AI (Gemini) generation failed after %s attempts: %s. "
                        "If this keeps happening on Standard/PAYG, try GCP_LOCATION=global "
                        "and/or lower max_tokens in .env.",
                        self.max_attempts,
                        e,
                    )
                else:
                    logger.error(f"Vertex AI (Gemini) generation error: {e}")
                return ""
        

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
    raise ValueError(f"Unsupported model provider: {config.provider}")
