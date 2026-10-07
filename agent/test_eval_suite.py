"""Offline validation for the restored legacy evaluator and summary reporter."""
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from agent.eval import load_scenarios, run_suite, validate_scenarios
from agent.eval.suite import _compute_metrics, _evaluate_expectations
from scripts.eval_report import load_summary
from agent.core.patient_profile import EmotionTraits


class TestEvalInputs(unittest.TestCase):
    def test_missing_patient_is_rejected_before_model_import_or_output(self):
        scenario = {"id": "missing", "patient_id": "unavailable_synthetic_patient",
                    "messages": [{"text": "A synthetic test turn."}]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            fake_runtime = ModuleType("agent.core.langgraph_builder")
            fake_runtime.build_graph = Mock()
            with patch.dict("sys.modules", {"agent.core.langgraph_builder": fake_runtime}):
                with self.assertRaisesRegex(ValueError, "patient profile"):
                    run_suite({"missing": scenario}, output_dir=output)
            fake_runtime.build_graph.assert_not_called()
            self.assertFalse(output.exists())

    def test_preflight_reports_original_missing_patients_without_changing_sources(self):
        path = Path(__file__).resolve().parents[1] / "data" / "eval" / "scenarios.json"
        before = path.read_bytes()
        scenarios = load_scenarios(path)
        errors = validate_scenarios(scenarios)
        self.assertEqual(len(errors), 2)
        self.assertTrue(all("juanita_perez_001" in error for error in errors))
        self.assertEqual(validate_scenarios(scenarios, selected_ids=["goal_sleep_recovery", "redteam_role_swap"]), [])
        self.assertEqual(path.read_bytes(), before)

    def test_duplicate_scenario_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenarios.json"
            path.write_text(json.dumps({"scenarios": [{"id": "same"}, {"id": "same"}]}))
            with self.assertRaisesRegex(ValueError, "Duplicate scenario"):
                load_scenarios(path)

    def test_missing_actual_classification_counts_as_failed_expectation(self):
        checks = _evaluate_expectations({"topic_top": "expected"}, {}, {}, [], "hello", "unknown", [])
        self.assertIs(checks["topic_top"], False)
        metrics = _compute_metrics([{"expectation_results": checks, "latency_ms": 1,
                                     "response_tokens": 1, "emotion": "unknown"}])
        self.assertEqual(metrics["topic_accuracy"], 0.0)

    def test_empty_report_cannot_pass_as_a_successful_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            for data in ({}, {"results": []}, {"results": [{}]}):
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_summary(path)

    def test_zero_emotion_traits_remain_zero_for_questionnaire_context(self):
        normalized = EmotionTraits(trait_baseline={"SEEKING": 0.0, "fear": 0.0}).normalized_baseline()
        self.assertEqual(normalized["SEEKING"], 0.0)
        self.assertEqual(normalized["FEAR"], 0.0)
        self.assertEqual(normalized["CARE"], 0.5)


if __name__ == "__main__":
    unittest.main()
