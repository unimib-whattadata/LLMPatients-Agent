import os
import unittest
from unittest import mock

from agent.core.questionnaire_catalog import (
    load_questionnaire_definition,
    questionnaire_is_runnable,
    questionnaire_non_runnable_reason,
)
from agent.core.questionnaire_runner import (
    questionnaire_inter_batch_delay_seconds,
    questionnaire_llm_overrides,
)


class TestQuestionnaireCatalog(unittest.TestCase):
    def test_standard_questionnaire_is_runnable_by_default(self):
        questionnaire_def = load_questionnaire_definition("phq9")
        self.assertTrue(questionnaire_is_runnable(questionnaire_def))

    def test_scid_extracts_are_marked_non_runnable(self):
        for questionnaire_id in ("scid5_cv_questions", "scid5_pd_questions"):
            questionnaire_def = load_questionnaire_definition(questionnaire_id)
            self.assertFalse(questionnaire_is_runnable(questionnaire_def))
            self.assertIn("not runnable self-report questionnaires", questionnaire_non_runnable_reason(questionnaire_def))

    def test_questionnaire_inter_batch_delay_env_override(self):
        with mock.patch.dict("os.environ", {"QUESTIONNAIRE_INTER_BATCH_DELAY_SECONDS": "2.5"}, clear=False):
            self.assertEqual(questionnaire_inter_batch_delay_seconds(), 2.5)

    def test_questionnaire_inter_batch_delay_defaults_higher_for_vertex(self):
        original_value = os.environ.pop("QUESTIONNAIRE_INTER_BATCH_DELAY_SECONDS", None)
        try:
            with mock.patch.dict("os.environ", {"model_provider": "vertex_ai"}, clear=False):
                self.assertEqual(questionnaire_inter_batch_delay_seconds(), 3.0)
        finally:
            if original_value is not None:
                os.environ["QUESTIONNAIRE_INTER_BATCH_DELAY_SECONDS"] = original_value

    def test_questionnaire_llm_overrides_reads_questionnaire_specific_env(self):
        with mock.patch.dict(
            "os.environ",
            {
                "QUESTIONNAIRE_MODEL_PROVIDER": "vertex_ai",
                "QUESTIONNAIRE_MODEL_ID": "gemini-2.5-flash-lite",
                "QUESTIONNAIRE_TEMPERATURE": "0.1",
                "QUESTIONNAIRE_MAX_TOKENS": "768",
            },
            clear=False,
        ):
            self.assertEqual(
                questionnaire_llm_overrides(),
                {
                    "provider": "vertex_ai",
                    "model_id": "gemini-2.5-flash-lite",
                    "temperature": 0.1,
                    "max_tokens": 768,
                },
            )


if __name__ == "__main__":
    unittest.main()
