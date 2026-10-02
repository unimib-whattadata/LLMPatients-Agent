"""Offline rate-limit regression tests, including independent worker processes."""
from datetime import datetime, timezone
from email.utils import format_datetime
import json
import multiprocessing
import os
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent.core.llm_provider_base import _env_non_negative_float
from agent.core.questionnaire_runner import QuestionnaireRunner
from agent.core.vertex_rate_limit import VertexRateLimiter, VertexRateLimitError, retry_after_seconds
from agent.test_vertex_generation import response, runner


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.after_sleep = None

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        if self.after_sleep is not None:
            hook, self.after_sleep = self.after_sleep, None
            hook()


def process_request(path, ready, start, results):
    limiter = VertexRateLimiter(Path(path), "shared", interval_seconds=.2, max_wait_seconds=10)
    ready.put(True)
    if not start.wait(15):
        raise RuntimeError("Test start was not signalled")
    limiter.acquire()
    results.put(time.time())


class TestSharedVertexRateLimiter(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "limit.sqlite3"
        self.clock = FakeClock()

    def limiter(self, key="shared", **overrides):
        options = dict(clock=self.clock.time, monotonic=self.clock.time,
                       sleep=self.clock.sleep, jitter=lambda delay: 0)
        options.update(overrides)
        return VertexRateLimiter(self.path, key, **options)

    def test_instances_share_request_spacing(self):
        first, second = self.limiter(), self.limiter()
        first.acquire()
        second.acquire()
        first.acquire()
        self.assertEqual(self.clock.now, 1010.0)

    def test_waiter_rechecks_cooldown_applied_after_wait_started(self):
        first, waiting = self.limiter(), self.limiter()
        first.acquire()
        self.clock.after_sleep = first.record_rate_limit
        waiting.acquire()
        self.assertEqual(self.clock.now, 1016.0)

    def test_exponential_backoff_and_circuit_are_shared_across_instances(self):
        first, second = self.limiter(), self.limiter()
        self.assertEqual(first.record_rate_limit().delay_seconds, 15)
        self.assertEqual(second.record_rate_limit().delay_seconds, 30)
        opened = first.record_rate_limit()
        self.assertTrue(opened.circuit_open)
        self.assertEqual(opened.delay_seconds, 300)
        with self.assertRaises(VertexRateLimitError) as error:
            second.acquire()
        self.assertEqual(error.exception.consecutive_failures, 3)
        self.assertEqual(self.clock.now, 1000.0)

    def test_only_one_recovery_probe_then_success_reopens_gate(self):
        first, second = self.limiter(failure_threshold=1), self.limiter(failure_threshold=1)
        first.record_rate_limit()
        self.clock.sleep(300)
        probe = first.acquire()
        with self.assertRaises(VertexRateLimitError):
            second.acquire()
        first.record_success(probe)
        second.acquire()
        self.assertEqual(first.snapshot()["failures"], 0)
        self.assertEqual(self.clock.now, 1305)

    def test_crashed_probe_lease_expires(self):
        limiter = self.limiter(failure_threshold=1, probe_seconds=10)
        limiter.record_rate_limit()
        self.clock.sleep(300)
        abandoned_probe = limiter.acquire()
        self.clock.sleep(10)
        next_probe = self.limiter(failure_threshold=1, probe_seconds=10).acquire()
        self.assertGreater(next_probe, abandoned_probe)
        limiter.record_success(abandoned_probe)
        self.assertGreater(limiter.snapshot()["probe_until"], self.clock.now)

    def test_old_success_does_not_erase_new_429_cooldown(self):
        limiter = self.limiter()
        old_request = limiter.acquire()
        limiter.record_rate_limit()
        limiter.record_success(old_request)
        self.assertEqual(limiter.snapshot()["failures"], 1)
        self.assertEqual(limiter.snapshot()["cooldown_until"], 1015)

    def test_server_delay_is_not_truncated_to_local_backoff_cap(self):
        limiter = self.limiter(failure_threshold=1, max_backoff_seconds=60)
        result = limiter.record_rate_limit(retry_after=900)
        self.assertEqual(result.delay_seconds, 900)
        with self.assertRaises(VertexRateLimitError) as error:
            limiter.acquire()
        self.assertEqual(error.exception.retry_after_seconds, 900)

    def test_wait_budget_stops_without_claiming_a_future_slot(self):
        limiter = self.limiter(max_wait_seconds=2)
        limiter.defer(30)
        with self.assertRaises(VertexRateLimitError):
            limiter.acquire()
        self.assertEqual(limiter.snapshot()["next_request"], 0)
        self.assertEqual(self.clock.now, 1000)

    def test_rate_state_is_isolated_by_project_region_and_model(self):
        with patch.dict(os.environ, {"VERTEX_RATE_LIMIT_STATE_PATH": str(self.path)}, clear=True):
            one = VertexRateLimiter.from_env("p1", "global", "gemini-one")
            same = VertexRateLimiter.from_env("p1", "global", "gemini-one")
            different = [VertexRateLimiter.from_env(*key) for key in [
                ("p2", "global", "gemini-one"), ("p1", "us-central1", "gemini-one"),
                ("p1", "global", "gemini-two"),
            ]]
        one.record_rate_limit()
        self.assertEqual(same.snapshot()["failures"], 1)
        self.assertTrue(all(item.snapshot()["failures"] == 0 for item in different))
        self.assertEqual(one.interval_seconds, 5)

    def test_state_failure_stops_instead_of_bypassing_limit(self):
        limiter = self.limiter()
        self.path.write_bytes(b"not a sqlite database")
        with self.assertRaises(VertexRateLimitError):
            limiter.acquire()

    def test_jitter_does_not_consume_scientific_random_seed(self):
        limiter = VertexRateLimiter(self.path, "shared")
        before = random.getstate()
        limiter.record_rate_limit()
        self.assertEqual(random.getstate(), before)

    def test_processes_coordinate_using_the_same_file(self):
        context = multiprocessing.get_context("spawn")
        ready, results, start = context.Queue(), context.Queue(), context.Event()
        processes = [context.Process(target=process_request, args=(str(self.path), ready, start, results)) for _ in range(3)]
        try:
            for process in processes:
                process.start()
            for _ in processes:
                self.assertTrue(ready.get(timeout=25))
            start.set()
            timestamps = sorted(results.get(timeout=15) for _ in processes)
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
            self.assertTrue(all(b - a >= .15 for a, b in zip(timestamps, timestamps[1:])), timestamps)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
            ready.close()
            results.close()


class TestRetryHints(unittest.TestCase):
    def test_seconds_and_http_date(self):
        date = format_datetime(datetime.fromtimestamp(1065, timezone.utc), usegmt=True)
        exc = SimpleNamespace(response=SimpleNamespace(headers={"Retry-After": "20"}),
                              http_response=SimpleNamespace(headers={"retry-after": date}))
        self.assertEqual(retry_after_seconds(exc, now=1000), 65)

    def test_rpc_retry_info(self):
        from google.rpc.error_details_pb2 import RetryInfo
        info = RetryInfo()
        info.retry_delay.seconds = 31
        info.retry_delay.nanos = 500_000_000
        self.assertEqual(retry_after_seconds(SimpleNamespace(details=[info])), 31.5)
        self.assertEqual(retry_after_seconds(SimpleNamespace(details=[
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "42.5s"},
        ])), 42.5)

    def test_malformed_negative_and_nonfinite_hints_are_ignored(self):
        for value in ("bad", "-5", "nan", "inf"):
            with self.subTest(value=value):
                self.assertEqual(retry_after_seconds(SimpleNamespace(response=SimpleNamespace(headers={"Retry-After": value}))), 0)

    def test_nonfinite_environment_cannot_disable_pacing(self):
        for value in ("nan", "inf", "-1"):
            with patch.dict(os.environ, {"VERTEX_MIN_REQUEST_INTERVAL_SECONDS": value}):
                self.assertEqual(_env_non_negative_float("VERTEX_MIN_REQUEST_INTERVAL_SECONDS", 5), 5)


class TestProviderCapacityHandling(unittest.TestCase):
    setUp = TestSharedVertexRateLimiter.setUp
    limiter = TestSharedVertexRateLimiter.limiter

    def llm(self, responses, *, attempts=5, **options):
        llm = runner(responses, attempts=attempts)
        llm.rate_limiter = self.limiter(**options)
        llm._wait_for_request_slot = lambda: llm.rate_limiter.acquire()
        llm._apply_shared_cooldown = llm.rate_limiter.defer
        return llm

    def test_429_recovery_preserves_prompt_and_generation_settings(self):
        llm = self.llm([RuntimeError("429 Resource exhausted"), RuntimeError("429 Resource exhausted"), response("Complete")])
        self.assertEqual(llm.generate("original prompt", temperature=.1, max_tokens=220), "Complete")
        calls = llm.model.generate_content.call_args_list
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(calls[1], calls[2])
        self.assertEqual(self.clock.now, 1045)
        self.assertEqual(llm.rate_limiter.snapshot()["failures"], 0)

    def test_repeated_429_raises_and_other_runner_makes_no_api_calls(self):
        llm = self.llm([RuntimeError("429 Resource exhausted")] * 5)
        with self.assertRaises(VertexRateLimitError):
            llm.generate("prompt")
        self.assertEqual(llm.model.generate_content.call_count, 3)
        other = self.llm([response("must not be called")])
        with self.assertRaises(VertexRateLimitError):
            other.generate("prompt")
        other.model.generate_content.assert_not_called()

    def test_final_allowed_attempt_still_records_cooldown(self):
        llm = self.llm([RuntimeError("429 Resource exhausted")], attempts=1)
        with self.assertRaises(VertexRateLimitError):
            llm.generate("prompt")
        self.assertEqual(llm.rate_limiter.snapshot()["failures"], 1)
        self.assertEqual(llm.rate_limiter.snapshot()["cooldown_until"], 1015)

    def test_questionnaire_keeps_saved_answers_and_does_not_multiply_retries(self):
        directory = Path(self.directory.name)
        questionnaire = QuestionnaireRunner.__new__(QuestionnaireRunner)
        questionnaire.force = False
        questionnaire.result_path = directory / "phq9.json"
        questionnaire.partial_path = directory / "phq9.partial.json"
        questionnaire.partial_path.write_text(json.dumps({"answers": {"1": 2}, "last_item": 1}))
        previous = questionnaire.partial_path.read_bytes()
        questionnaire.profile = SimpleNamespace(name="Synthetic patient")
        questionnaire.q_def = {"scale": {"type": "integer", "min": 0, "max": 3},
                               "items": [{"id": 1}, {"id": 2}], "batch_size": 1}
        questionnaire._build_patient_context = Mock(return_value="context")
        questionnaire._build_single_prompt = Mock(return_value="prompt")
        questionnaire._print_run_header = Mock()
        questionnaire.llm = self.llm([RuntimeError("429 Resource exhausted")] * 5)
        with self.assertRaises(VertexRateLimitError):
            questionnaire.run()
        self.assertEqual(questionnaire.llm.model.generate_content.call_count, 3)
        self.assertEqual(questionnaire.partial_path.read_bytes(), previous)
        self.assertFalse(questionnaire.result_path.exists())


if __name__ == "__main__":
    unittest.main()
