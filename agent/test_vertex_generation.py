"""Regression tests for truncated Vertex output using real SDK response enums."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from vertexai.generative_models import GenerationResponse

from agent.core.llm_provider_base import STOP_SEQUENCES
from agent.core.llm_provider_vertex import VertexLLMRunner


def response(text, finish_reason="STOP"):
    parts = [] if text is None else [{"text": text}]
    return GenerationResponse.from_dict({
        "candidates": [{"finish_reason": finish_reason,
                        "content": {"role": "model", "parts": parts}}],
        "usage_metadata": {"prompt_token_count": 20, "candidates_token_count": 4,
                           "thoughts_token_count": 316, "total_token_count": 340},
    })


def runner(responses, *, attempts=5, recovery_limit=8192):
    result = VertexLLMRunner.__new__(VertexLLMRunner)
    result.model_id = "gemini-2.5-pro"
    result.temperature = 0.7
    result.max_tokens = 4096
    result.max_attempts = attempts
    result.max_recovery_tokens = recovery_limit
    result.safety_settings = {"unchanged": "policy"}
    result.model = Mock()
    result.model.generate_content.side_effect = responses
    result._wait_for_request_slot = Mock()
    result._apply_shared_cooldown = Mock()
    result._retry_delay_seconds = Mock(return_value=0)
    result.rate_limiter = Mock()
    result.rate_limiter.record_rate_limit.return_value = SimpleNamespace(
        failures=1, delay_seconds=0, circuit_open=False,
    )
    return result


class TestVertexGeneration(unittest.TestCase):
    def test_explicit_thinking_budget_is_forwarded_and_retained_on_recovery(self):
        llm = runner([response("partial", "MAX_TOKENS"), response('{"facts": []}')])
        self.assertEqual(llm.generate("extract", max_tokens=4096, thinking_budget=1024), '{"facts": []}')
        for call in llm.model.generate_content.call_args_list:
            config = call.kwargs["generation_config"]
            self.assertEqual(config["thinking_config"], {"thinking_budget": 1024})
            # Validate the actual SDK's mapping instead of testing only a mock.
            from google.cloud.aiplatform_v1.types import GenerationConfig
            self.assertEqual(GenerationConfig(config).thinking_config.thinking_budget, 1024)

    def test_invalid_thinking_budget_is_rejected_before_a_provider_call(self):
        for budget in [0, True, 127, 4096, -1, "1024"]:
            llm = runner([response("should not run")])
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                llm.generate("extract", max_tokens=4096, thinking_budget=budget)
            llm.model.generate_content.assert_not_called()

    def test_visible_partial_output_is_discarded_and_retried(self):
        llm = runner([response("I remember the", "MAX_TOKENS"), response("I remember the whole session.")])
        self.assertEqual(llm.generate("original prompt", max_tokens=320), "I remember the whole session.")
        calls = llm.model.generate_content.call_args_list
        self.assertEqual([call.kwargs["generation_config"]["max_output_tokens"] for call in calls], [320, 1024])
        for call in calls:
            self.assertEqual(call.args, ("original prompt",))
            self.assertEqual(call.kwargs["safety_settings"], llm.safety_settings)
            config = call.kwargs["generation_config"]
            self.assertEqual(config["temperature"], 0.7)
            self.assertEqual(config["stop_sequences"], STOP_SEQUENCES)
            self.assertEqual(config["top_p"], 0.95)
            self.assertEqual(config["top_k"], 40)

    def test_sdk_integer_enum_is_normalized_before_text_extraction(self):
        truncated = response(None, "MAX_TOKENS")
        self.assertEqual(str(truncated.candidates[0].finish_reason), "2")
        llm = runner([truncated, response("A complete reflection.")])
        self.assertEqual(llm.generate("reflect", max_tokens=320), "A complete reflection.")
        self.assertEqual(llm.model.generate_content.call_count, 2)

    def test_repeated_truncation_is_bounded_and_never_returned(self):
        llm = runner([response("partial", "MAX_TOKENS") for _ in range(5)])
        self.assertEqual(llm.generate("reflect", max_tokens=320), "")
        budgets = [c.kwargs["generation_config"]["max_output_tokens"] for c in llm.model.generate_content.call_args_list]
        self.assertEqual(budgets, [320, 1024, 2048, 4096, 8192])

    def test_recovery_cap_stops_further_calls(self):
        llm = runner([response("partial", "MAX_TOKENS")], attempts=8, recovery_limit=4096)
        self.assertEqual(llm.generate("reflect"), "")
        self.assertEqual(llm.model.generate_content.call_count, 1)

    def test_blocked_candidate_text_is_not_returned_or_retried(self):
        llm = runner([response("blocked partial text", "SAFETY")])
        self.assertEqual(llm.generate("reflect"), "")
        self.assertEqual(llm.model.generate_content.call_count, 1)

    def test_transient_retry_keeps_budget_and_settings(self):
        llm = runner([RuntimeError("429 Resource exhausted"), response("Complete.")])
        self.assertEqual(llm.generate("reflect", temperature=0.1), "Complete.")
        calls = llm.model.generate_content.call_args_list
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(calls[1].kwargs["generation_config"]["temperature"], 0.1)

    def test_candidate_fallback_excludes_thought_parts(self):
        class TextAccessorFailure:
            candidates = [SimpleNamespace(
                finish_reason="STOP",
                content=SimpleNamespace(parts=[
                    SimpleNamespace(text="private reasoning", thought=True),
                    SimpleNamespace(text="Completed visible reply.", thought=False),
                ]),
            )]
            @property
            def text(self):
                raise ValueError("accessor failed")
        llm = runner([TextAccessorFailure()])
        self.assertEqual(llm.generate("reflect"), "Completed visible reply.")


if __name__ == "__main__":
    unittest.main()
