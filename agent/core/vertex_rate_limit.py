"""Persistent Vertex request pacing and capacity backoff shared by local processes.

Only scheduling metadata is stored; prompts, responses and credentials never enter
this database. SQLite transactions claim slots when requests actually start, so
waiters observe cooldowns imposed after they began waiting.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import sqlite3
import time

from agent.core.llm_provider_base import _env_non_negative_float, _env_positive_int

logger = logging.getLogger(__name__)
_JITTER = random.SystemRandom()


class VertexRateLimitError(RuntimeError):
    """Capacity is unavailable; callers must preserve progress and stop retrying."""

    def __init__(self, retry_after_seconds: float, *, reason: str, failures: int = 0):
        self.retry_after_seconds = max(0.0, retry_after_seconds)
        self.consecutive_failures = failures
        super().__init__(f"Vertex capacity unavailable: {reason}. Retry after {self.retry_after_seconds:.1f}s.")


def retry_after_seconds(exc: Exception, *, now=None) -> float:
    """Read numeric/date Retry-After and Google RPC RetryInfo, ignoring bad hints."""
    now = time.time() if now is None else now
    delays = [0.0]
    for name in ("response", "http_response"):
        headers = getattr(getattr(exc, name, None), "headers", None)
        if not headers:
            continue
        value = headers.get("retry-after") or headers.get("Retry-After")
        if value is None:
            continue
        try:
            delay = float(value)
        except (TypeError, ValueError):
            try:
                date = parsedate_to_datetime(str(value))
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                delay = date.timestamp() - now
            except (TypeError, ValueError, OverflowError):
                continue
        if math.isfinite(delay):
            delays.append(delay)
    details = getattr(exc, "details", ())
    if isinstance(details, (list, tuple)):
        for detail in details:
            duration = getattr(detail, "retry_delay", None)
            if duration is not None:
                delay = duration.seconds + duration.nanos / 1_000_000_000
                if math.isfinite(delay):
                    delays.append(delay)
            elif isinstance(detail, dict) and str(detail.get("@type", "")).endswith("google.rpc.RetryInfo"):
                try:
                    delay = float(str(detail.get("retryDelay", "")).removesuffix("s"))
                    if math.isfinite(delay):
                        delays.append(delay)
                except (ValueError, TypeError):
                    pass
    return max(delays)


@dataclass(frozen=True)
class CapacityBackoff:
    delay_seconds: float
    failures: int
    circuit_open: bool


class VertexRateLimiter:
    def __init__(
        self, path: Path, key: str, *, interval_seconds: float = 5.0,
        cooldown_seconds: float = 15.0, max_backoff_seconds: float = 60.0,
        failure_threshold: int = 3, circuit_seconds: float = 300.0,
        max_wait_seconds: float = 120.0, probe_seconds: float = 300.0,
        clock=time.time, monotonic=time.monotonic, sleep=time.sleep,
        jitter=None,
    ):
        values = (interval_seconds, cooldown_seconds, max_backoff_seconds,
                  circuit_seconds, max_wait_seconds, probe_seconds)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Rate-limit durations must be finite and non-negative.")
        if failure_threshold < 1 or circuit_seconds <= 0 or probe_seconds <= 0:
            raise ValueError("The circuit threshold and circuit/probe durations must be positive.")
        self.path, self.key = Path(path), key
        self.interval_seconds = interval_seconds
        self.cooldown_seconds = cooldown_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self.failure_threshold = failure_threshold
        self.circuit_seconds = circuit_seconds
        self.max_wait_seconds = max_wait_seconds
        self.probe_seconds = probe_seconds
        self.clock, self.monotonic, self.sleep = clock, monotonic, sleep
        self.jitter = jitter or (lambda delay: _JITTER.uniform(0, min(5.0, delay * .25)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A new connection per transaction also works across threads and forks.
        with sqlite3.connect(self.path, timeout=5) as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS vertex_limits (
                bucket TEXT PRIMARY KEY, next_request REAL NOT NULL DEFAULT 0,
                cooldown_until REAL NOT NULL DEFAULT 0, failures INTEGER NOT NULL DEFAULT 0,
                circuit_until REAL NOT NULL DEFAULT 0, probe_until REAL NOT NULL DEFAULT 0,
                generation INTEGER NOT NULL DEFAULT 0)""")

    @classmethod
    def from_env(cls, project: str, location: str, model_id: str):
        cache = Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        configured_path = os.getenv("VERTEX_RATE_LIMIT_STATE_PATH")
        path = Path(configured_path).expanduser() if configured_path else cache / "llmpatients" / "vertex-rate-limit.sqlite3"
        key = hashlib.sha256(json.dumps([project, location, model_id]).encode()).hexdigest()
        rpm = _env_positive_int("VERTEX_REQUESTS_PER_MINUTE", 12)
        interval = max(60.0 / rpm, _env_non_negative_float("VERTEX_MIN_REQUEST_INTERVAL_SECONDS", 0.0))
        return cls(
            path, key, interval_seconds=interval,
            cooldown_seconds=_env_non_negative_float("VERTEX_RATE_LIMIT_COOLDOWN_SECONDS", 15.0),
            max_backoff_seconds=_env_non_negative_float("VERTEX_RETRY_MAX_DELAY_SECONDS", 60.0),
            failure_threshold=_env_positive_int("VERTEX_RATE_LIMIT_FAILURE_THRESHOLD", 3),
            circuit_seconds=_env_positive_int("VERTEX_RATE_LIMIT_CIRCUIT_SECONDS", 300),
            max_wait_seconds=_env_non_negative_float("VERTEX_RATE_LIMIT_MAX_WAIT_SECONDS", 120.0),
            probe_seconds=_env_positive_int("VERTEX_RATE_LIMIT_PROBE_SECONDS", 300),
        )

    @contextmanager
    def _state(self):
        connection = None
        try:
            connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("INSERT OR IGNORE INTO vertex_limits (bucket) VALUES (?)", (self.key,))
            state = dict(connection.execute("SELECT * FROM vertex_limits WHERE bucket = ?", (self.key,)).fetchone())
            yield state
            connection.execute("""UPDATE vertex_limits SET
                next_request=:next_request, cooldown_until=:cooldown_until, failures=:failures,
                circuit_until=:circuit_until, probe_until=:probe_until, generation=:generation
                WHERE bucket=:bucket""", state)
            connection.commit()
        except sqlite3.Error as exc:
            # A broken scheduler must not silently allow unthrottled requests.
            raise VertexRateLimitError(5.0, reason="shared rate-limit state is unavailable") from exc
        finally:
            if connection is not None:
                connection.close()

    def acquire(self) -> int:
        """Claim a paced start, or stop on an open circuit / bounded wait timeout."""
        deadline = self.monotonic() + self.max_wait_seconds
        announced = False
        while True:
            with self._state() as state:
                now = self.clock()
                blocked_until = max(state["circuit_until"], state["probe_until"])
                if blocked_until > now:
                    raise VertexRateLimitError(blocked_until - now, reason="shared capacity circuit is open", failures=state["failures"])
                delay = max(state["cooldown_until"], state["next_request"]) - now
                if delay <= 0:
                    if state["failures"] >= self.failure_threshold:
                        # Permit one recovery probe; other processes fail fast.
                        # A crashed probe releases itself when this lease expires.
                        state["probe_until"] = now + self.probe_seconds
                        state["generation"] += 1
                    state["next_request"] = now + self.interval_seconds
                    return state["generation"]
                remaining = deadline - self.monotonic()
                if remaining <= 0 or delay > remaining:
                    raise VertexRateLimitError(delay, reason="shared request wait budget exhausted", failures=state["failures"])
            if not announced:
                logger.info("Vertex request pacing: waiting at least %.2fs for a shared slot.", delay)
                announced = True
            # Re-read shared state: a 429 elsewhere can extend this wait.
            self.sleep(min(delay, remaining, 1.0))

    def record_rate_limit(self, retry_after: float = 0.0) -> CapacityBackoff:
        with self._state() as state:
            now = self.clock()
            state["failures"] += 1
            state["generation"] += 1
            backoff = min(self.cooldown_seconds * (2 ** min(state["failures"] - 1, 30)), self.max_backoff_seconds)
            delay = max(self.cooldown_seconds, min(self.max_backoff_seconds, backoff + self.jitter(backoff)), retry_after)
            state["cooldown_until"] = max(state["cooldown_until"], now + delay)
            opened = state["failures"] >= self.failure_threshold
            if opened:
                state["circuit_until"] = max(state["circuit_until"], now + self.circuit_seconds, state["cooldown_until"])
                state["probe_until"] = 0.0
            until = max(state["cooldown_until"], state["circuit_until"])
            return CapacityBackoff(until - now, state["failures"], opened)

    def defer(self, seconds: float) -> None:
        """Share transient non-429 backoff without increasing the 429 streak."""
        if seconds <= 0:
            return
        with self._state() as state:
            state["cooldown_until"] = max(state["cooldown_until"], self.clock() + seconds)
            state["generation"] += 1

    def record_success(self, generation: int) -> None:
        with self._state() as state:
            # A response started before a newer 429 must not clear its cooldown.
            if state["generation"] == generation:
                state.update(failures=0, cooldown_until=0.0, circuit_until=0.0, probe_until=0.0)

    def snapshot(self) -> dict:
        with self._state() as state:
            return dict(state)
