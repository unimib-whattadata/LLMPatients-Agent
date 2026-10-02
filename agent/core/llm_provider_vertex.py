import logging
import os
import random
import re
from typing import Optional

from agent.core.llm_provider_base import (
    LLMRunnerBase,
    STOP_SEQUENCES,
    _env_positive_int,
    _env_non_negative_float,
)
from agent.core.vertex_rate_limit import VertexRateLimitError, VertexRateLimiter, retry_after_seconds

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

# === Configure Logging ===
logger = logging.getLogger(__name__)

DEFAULT_VERTEX_MAX_ATTEMPTS = 5
DEFAULT_VERTEX_RETRY_BASE_DELAY_SECONDS = 1.0
DEFAULT_VERTEX_RETRY_MAX_DELAY_SECONDS = 60.0
VERTEX_RECOVERY_MIN_TOKENS = 1024
DEFAULT_VERTEX_MAX_RECOVERY_TOKENS = 8192
DEFAULT_VERTEX_LOCATION = "us-central1"
VERTEX_TOP_P = 0.95
VERTEX_TOP_K = 40

_RETRY_JITTER = random.SystemRandom()


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
        self.rate_limiter = VertexRateLimiter.from_env(project, location, model_id)
        self.max_attempts = _env_positive_int("VERTEX_MAX_ATTEMPTS", DEFAULT_VERTEX_MAX_ATTEMPTS)
        self.max_recovery_tokens = _env_positive_int(
            "VERTEX_MAX_RECOVERY_TOKENS", DEFAULT_VERTEX_MAX_RECOVERY_TOKENS
        )
        
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
        return bool(re.search(r"\b(?:408|429|500|502|503|504)\b", message)) or any(
            marker in message for marker in retryable_markers
        )

    def _retry_delay_seconds(self, attempt: int) -> float:
        backoff = min(
            self.retry_base_delay_seconds * (2 ** max(attempt - 1, 0)),
            self.retry_max_delay_seconds,
        )
        jitter = _RETRY_JITTER.uniform(0.0, min(1.0, backoff * 0.25))
        return min(self.retry_max_delay_seconds, backoff + jitter)

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> float:
        return retry_after_seconds(exc)

    @staticmethod
    def _is_rate_limited_error(exc: Exception) -> bool:
        if GOOGLE_API_RATE_LIMIT_ERRORS and isinstance(exc, GOOGLE_API_RATE_LIMIT_ERRORS):
            return True

        message = str(exc).lower()
        rate_limit_markers = (
            "resource exhausted",
            "rate limit",
            "too many requests",
            "quota exceeded",
        )
        return bool(re.search(r"\b429\b", message)) or any(marker in message for marker in rate_limit_markers)

    def _wait_for_request_slot(self) -> int:
        return self.rate_limiter.acquire()

    def _apply_shared_cooldown(self, delay_seconds: float) -> None:
        self.rate_limiter.defer(delay_seconds)

    @staticmethod
    def _candidate_finish_reasons(response) -> list[str]:
        reasons: list[str] = []
        for candidate in getattr(response, "candidates", []) or []:
            finish_reason = getattr(candidate, "finish_reason", None)
            if finish_reason:
                # The SDK exposes an IntEnum: str(MAX_TOKENS) is "2", not
                # "MAX_TOKENS". Preserve names for both SDK and test responses.
                reasons.append(getattr(finish_reason, "name", str(finish_reason)))
        return reasons

    @staticmethod
    def _usage_counts(response) -> dict[str, Optional[int]]:
        usage = getattr(response, "usage_metadata", None)
        return {
            name: getattr(usage, name, None)
            for name in ("prompt_token_count", "candidates_token_count", "thoughts_token_count")
        }

    def _recovery_max_tokens(self, current_max_tokens: int) -> int:
        return min(
            max(current_max_tokens * 2, VERTEX_RECOVERY_MIN_TOKENS),
            max(self.max_recovery_tokens, current_max_tokens),
        )

    @staticmethod
    def _extract_text_from_candidates(response) -> str:
        text_chunks = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                if getattr(part, "thought", False):
                    continue
                value = getattr(part, "text", None)
                if value:
                    text_chunks.append(value)
        return "\n".join(text_chunks).strip()

    def generate(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                 *, thinking_budget: Optional[int] = None) -> str:
        """Proxy prompt execution to Vertex AI with consistent config and error handling."""
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        current_max_tok = max_tok
        if thinking_budget is not None and (
            not isinstance(thinking_budget, int) or isinstance(thinking_budget, bool)
            or thinking_budget < 128 or thinking_budget >= max_tok
        ):
            raise ValueError("thinking_budget must be an integer >= 128 and smaller than max_tokens")

        for attempt in range(1, self.max_attempts + 1):
            try:
                generation = self._wait_for_request_slot()
                config = {
                    "temperature": temp,
                    "max_output_tokens": current_max_tok,
                    "stop_sequences": STOP_SEQUENCES,
                    "top_p": VERTEX_TOP_P,
                    "top_k": VERTEX_TOP_K,
                }
                if thinking_budget is not None:
                    config["thinking_config"] = {"thinking_budget": thinking_budget}
                response = self.model.generate_content(
                    prompt,
                    generation_config=config,
                    safety_settings=self.safety_settings
                )
                self.rate_limiter.record_success(generation)
                finish_reasons = self._candidate_finish_reasons(response)
                logger.info(
                    "Vertex generation: model=%s finish_reasons=%s max_output_tokens=%s usage=%s",
                    self.model_id, finish_reasons, current_max_tok, self._usage_counts(response),
                )
                # Never accept a partial completion, even when response.text
                # exists. Thinking tokens share the model's output budget.
                if "MAX_TOKENS" in finish_reasons:
                    next_max_tok = self._recovery_max_tokens(current_max_tok)
                    if attempt < self.max_attempts and next_max_tok > current_max_tok:
                        logger.warning(
                            "Vertex AI stopped with MAX_TOKENS on attempt %s/%s. "
                            "Discarding partial output and retrying with max_output_tokens=%s (was %s).",
                            attempt, self.max_attempts, next_max_tok, current_max_tok,
                        )
                        current_max_tok = next_max_tok
                        continue
                    logger.error(
                        "Vertex AI output remained incomplete at max_output_tokens=%s after %s attempts; "
                        "discarding it. model=%s",
                        current_max_tok, attempt, self.model_id,
                    )
                    return ""
                if any(reason != "STOP" for reason in finish_reasons):
                    logger.warning("Vertex AI returned no completed candidate: finish_reasons=%s", finish_reasons)
                    return ""
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
                    raise text_error
            except VertexRateLimitError:
                # Do not turn an outage into empty patient content. The caller
                # must stop this operation and retain resumable progress.
                raise
            except Exception as e:
                if self._is_rate_limited_error(e):
                    backoff = self.rate_limiter.record_rate_limit(self._retry_after_seconds(e))
                    logger.warning(
                        "Vertex rate limit: attempt=%s/%s shared_failures=%s wait=%.2fs circuit_open=%s",
                        attempt, self.max_attempts, backoff.failures, backoff.delay_seconds, backoff.circuit_open,
                    )
                    if backoff.circuit_open or attempt == self.max_attempts:
                        raise VertexRateLimitError(
                            backoff.delay_seconds, failures=backoff.failures,
                            reason="repeated 429 responses" if backoff.circuit_open else "429 retry budget exhausted",
                        ) from e
                    # The shared gate owns this wait, including any later
                    # cooldown extension from another process.
                    continue
                retryable = self._is_retryable_error(e)
                if retryable and attempt < self.max_attempts:
                    delay_seconds = max(
                        self._retry_delay_seconds(attempt),
                        self._retry_after_seconds(e),
                    )
                    self._apply_shared_cooldown(delay_seconds)
                    logger.warning(
                        "Vertex AI transient generation error on attempt %s/%s: %s. Retrying in %.2fs.",
                        attempt,
                        self.max_attempts,
                        e,
                        delay_seconds,
                    )
                    continue

                if retryable:
                    final_cooldown_seconds = self._retry_after_seconds(e)
                    self._apply_shared_cooldown(final_cooldown_seconds)
                    logger.error(
                        "Vertex AI (Gemini) generation failed after %s attempts: %s. "
                        "Inspect the provider's capacity and service status before retrying.",
                        self.max_attempts,
                        e,
                    )
                else:
                    logger.error(f"Vertex AI (Gemini) generation error: {e}")
                return ""
