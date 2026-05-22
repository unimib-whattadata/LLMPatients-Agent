import logging
import os
import random
import threading
import time
from typing import Optional

from agent.core.llm_provider_base import (
    LLMRunnerBase,
    STOP_SEQUENCES,
    _env_positive_int,
    _env_non_negative_float,
)

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
DEFAULT_VERTEX_RATE_LIMIT_COOLDOWN_SECONDS = 15.0
DEFAULT_VERTEX_MIN_REQUEST_INTERVAL_SECONDS = 0.0
VERTEX_NO_TEXT_RECOVERY_MIN_TOKENS = 256
VERTEX_NO_TEXT_RECOVERY_MAX_TOKENS = 512
DEFAULT_VERTEX_LOCATION = "us-central1"
VERTEX_TOP_P = 0.95
VERTEX_TOP_K = 40

_VERTEX_RATE_LIMIT_LOCK = threading.Lock()
_VERTEX_NEXT_REQUEST_AT = 0.0
_VERTEX_LAST_REQUEST_AT = 0.0


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
