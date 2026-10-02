import logging
import math
import os
from abc import ABC, abstractmethod
from typing import Optional

# === Configure Logging ===
logger = logging.getLogger(__name__)

STOP_SEQUENCES = ["\nTherapist:", "Therapist:"]


class LLMRunnerBase(ABC):
    """Minimal interface all backing LLM providers must implement."""
    def __init__(self, temperature: float, max_tokens: int):
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Return a text completion for the provided prompt."""


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
        if not math.isfinite(value) or value < 0:
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
