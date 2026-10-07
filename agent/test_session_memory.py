"""Check memory finalization without network calls or model downloads."""
import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

# The local .env may enable LangSmith tracing. Offline tests must not export
# even synthetic graph traces, independently of installed SDK defaults.
for _tracing_flag in ("LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING"):
    os.environ[_tracing_flag] = "false"

from agent.core.llm_provider_vertex import VertexLLMRunner
from agent.core.memory_store import JsonlMemoryStore
from agent.core.vertex_rate_limit import VertexRateLimitError
from agent.core.factual_memory import EvidenceMemory

_runner = VertexLLMRunner.__new__(VertexLLMRunner)
_runner.model_id = "gemini-2.5-pro"
_runner.max_tokens = 4096
_encoder = Mock()
_encoder.get_sentence_embedding_dimension.return_value = 3
# These tests never use a real embedding model. Stub the optional ML package
# before import so an offline test run does not load Torch/SciPy or fetch models.
_sentence_transformers = ModuleType("sentence_transformers")
_sentence_transformers.SentenceTransformer = Mock(return_value=_encoder)
_original_sentence_transformers = sys.modules.get("sentence_transformers")
sys.modules["sentence_transformers"] = _sentence_transformers
try:
    with patch("agent.core.llm_runner.create_llm_runner", return_value=_runner):
        from agent.core import langgraph_builder as builder
finally:
    if _original_sentence_transformers is None:
        sys.modules.pop("sentence_transformers", None)
    else:
        sys.modules["sentence_transformers"] = _original_sentence_transformers

# RunLogger migrates relative legacy files on import; isolate that import too.
with tempfile.TemporaryDirectory() as _import_dir:
    _cwd = Path.cwd()
    try:
        os.chdir(_import_dir)
        from agent.api import app as api
    finally:
        os.chdir(_cwd)


class TestSessionMemory(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = JsonlMemoryStore(Path(self.directory.name))
        self.llm = Mock()
        for name, value in [
            ("MEMORY_STORE", self.store), ("LONG_TERM_STORE", Mock()),
            ("llm_runner", self.llm), ("MEMORY_CACHE_LOADED", {}),
            ("LATEST_SUMMARY_CACHE", {}), ("LATEST_REFLECTION_CACHE", {}),
        ]:
            patcher = patch.object(builder, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.state = {
            "patient_id": "patient", "therapist_id": "therapist", "session_id": "session",
            "history": [{"therapist": "What did we agree?", "patient": "A walk on Friday."}],
            "summary": "Previous completed memory.",
        }

    def test_failed_summary_preserves_memory_and_session_state(self):
        builder.persist_long_term_summary(
            patient_id="patient", therapist_id="therapist", summary_text="Previous completed memory.",
        )
        path = self.store.file_path("patient", "therapist")
        before = path.read_bytes()
        self.llm.generate.side_effect = ["A complete reflection.", ""]
        with self.assertRaises(builder.SessionMemoryError):
            builder.finalize_session_memory(self.state)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.state["summary"], "Previous completed memory.")
        self.assertNotIn("session_reflection", self.state)

    def test_failed_reflection_is_not_persisted(self):
        self.llm.generate.return_value = ""
        with self.assertRaises(builder.SessionMemoryError):
            builder.finalize_session_memory(self.state)
        self.assertEqual(self.llm.generate.call_count, 1)
        self.assertEqual(list(self.store.iter_records("patient", "therapist")), [])

    def test_api_keeps_session_open_when_memory_generation_fails(self):
        logger = Mock()
        key = ("therapist", "patient", "session")
        sessions = {key: {"logger": logger, "latest_state": self.state}}
        self.llm.generate.return_value = ""
        with patch.object(api, "session_loggers", sessions):
            with self.assertRaises(builder.SessionMemoryError):
                asyncio.run(api.end_session(api.SessionEndRequest(
                    external_patient_id="patient", therapist_id="therapist", session_id="session",
                )))
        self.assertIn(key, sessions)
        logger.finalize.assert_not_called()

    def test_complete_memory_is_persisted_and_reloaded_without_caches(self):
        reflection = "I agreed to go for a walk on Friday. I felt heard."
        summary = "I have agreed to go for a walk on Friday. I want to discuss how it goes."
        self.llm.generate.side_effect = [reflection, summary]
        result = builder.finalize_session_memory(self.state)
        records = list(JsonlMemoryStore(Path(self.directory.name)).iter_records("patient", "therapist"))
        self.assertEqual([r["type"] for r in records], ["session_reflection", "long_term_summary"])
        self.assertEqual([r["text"] for r in records], [reflection, summary])
        builder.LATEST_SUMMARY_CACHE.clear()
        builder.LATEST_REFLECTION_CACHE.clear()
        builder.MEMORY_CACHE_LOADED.clear()
        self.assertEqual(builder.load_long_term_summary("patient", "therapist"), summary)
        self.assertEqual(result["session_reflection"], reflection)
        self.assertEqual(result["summary"], summary)

    def test_reflection_includes_turns_after_existing_episode(self):
        self.llm.generate.return_value = "A complete reflection."
        builder._generate_session_reflection(["An earlier disagreement."], self.state["history"])
        prompt = self.llm.generate.call_args.kwargs["prompt"]
        self.assertIn("An earlier disagreement.", prompt)
        self.assertIn("A walk on Friday.", prompt)

    def test_internal_budgets_cover_vertex_gemini_reasoning_only(self):
        with patch.object(builder, "llm_runner", _runner):
            for legacy in (192, 256, 320, 512):
                self.assertEqual(builder._internal_generation_budget(legacy), 4096)
        with patch.object(builder, "llm_runner", SimpleNamespace(model_id="local-model")):
            self.assertEqual(builder._internal_generation_budget(192), 192)

    def test_rate_limit_propagates_without_classification_fallback(self):
        self.llm.generate.side_effect = VertexRateLimitError(300, reason="capacity circuit open")
        with self.assertRaises(VertexRateLimitError):
            builder.classify_topic_and_emotion("How are you?", "I feel tired.")
        self.llm.generate.assert_called_once()

    def test_rate_limit_does_not_create_a_fallback_patient_reply(self):
        self.llm.generate.side_effect = VertexRateLimitError(300, reason="capacity circuit open")
        with self.assertRaises(VertexRateLimitError):
            builder.generate_response(SimpleNamespace(prompt="Respond as the synthetic patient."))
        self.llm.generate.assert_called_once()

    def test_rate_limit_propagates_through_classification_node(self):
        self.llm.generate.side_effect = VertexRateLimitError(300, reason="capacity circuit open")
        state = SimpleNamespace(safe_user_input="How are you?", user_input="How are you?", history=[])
        with patch.object(builder, "_build_topic_text", return_value="How are you?"):
            with self.assertRaises(VertexRateLimitError):
                builder.classify_topic_and_emotion_pre(state)
        self.llm.generate.assert_called_once()

    def test_rate_limit_during_summary_preserves_memory_and_open_session(self):
        builder.persist_long_term_summary(
            patient_id="patient", therapist_id="therapist", summary_text="Previous completed memory.",
        )
        path = self.store.file_path("patient", "therapist")
        previous = path.read_bytes()
        self.llm.generate.side_effect = ["A complete reflection.", VertexRateLimitError(300, reason="capacity circuit open")]
        logger = Mock()
        key = ("therapist", "patient", "session")
        sessions = {key: {"logger": logger, "latest_state": self.state}}
        with patch.object(api, "session_loggers", sessions):
            with self.assertRaises(VertexRateLimitError):
                asyncio.run(api.end_session(api.SessionEndRequest(
                    external_patient_id="patient", therapist_id="therapist", session_id="session",
                )))
        self.assertIn(key, sessions)
        logger.finalize.assert_not_called()
        self.assertEqual(path.read_bytes(), previous)
        self.assertNotIn("session_reflection", self.state)

    def test_api_reports_capacity_as_503_with_retry_after(self):
        from fastapi.testclient import TestClient
        key = ("therapist", "patient", "session")
        logger = Mock()
        sessions = {key: {"logger": logger, "latest_state": self.state}}
        self.llm.generate.side_effect = VertexRateLimitError(31.2, reason="capacity circuit open")
        with patch.object(api, "session_loggers", sessions), TestClient(api.app) as client:
            result = client.post("/session-end", json={
                "external_patient_id": "patient", "therapist_id": "therapist", "session_id": "session",
            })
        self.assertEqual(result.status_code, 503)
        self.assertEqual(result.headers["Retry-After"], "32")
        self.assertEqual(result.json()["detail"]["code"], "vertex_rate_limited")
        self.assertIn(key, sessions)
        logger.finalize.assert_not_called()

    def test_raw_turns_survive_short_term_window_eviction(self):
        state = builder.State(patient_id="patient", therapist_id="therapist", session_id="session")
        with patch.object(builder, "_schedule_episode_job"):
            for index in range(8):
                state.user_input = "The library pass is Amber Moon (second floor)." if index == 0 else f"Ordinary conversation {index}."
                state.response = "Understood."
                builder.update_memory(state)
        self.assertEqual(len(state.history), 5)
        reopened = EvidenceMemory(JsonlMemoryStore(Path(self.directory.name)))
        retrieved = reopened.retrieve(patient_id="patient", therapist_id="therapist", query="What is the library pass?")
        self.assertTrue(any("Amber Moon (second floor)" in item["quote"] for item in retrieved))

    def test_unknown_topic_preserves_source_evidence_and_parentheses_in_prompt(self):
        memory = EvidenceMemory(self.store)
        memory.record_turn(patient_id="patient", therapist_id="therapist", session_id="prior", turn_index=1,
                           therapist_text="The library pass is Amber Moon (second floor).", patient_text="Understood.")
        state = builder.State(patient_id="patient", therapist_id="therapist", session_id="next",
                              user_input="What is the library pass?", intent_topic={"top": "unknown"},
                              episodic_context=["The pass (Amber Moon) was discussed."],
                              evidence_context=memory.retrieve(patient_id="patient", therapist_id="therapist", query="library pass"))
        prompt = builder.build_prompt(state)["prompt"]
        self.assertIn("Amber Moon (second floor)", prompt)
        self.assertIn("The pass (Amber Moon)", prompt)
        self.assertIn("session=prior", prompt)

    def test_invalid_fact_payload_preserves_sources_and_finalizes_partial_memory(self):
        EvidenceMemory(self.store).record_turn(patient_id="patient", therapist_id="therapist", session_id="session",
                                              turn_index=1, therapist_text="The pass is Amber Moon.", patient_text="Understood.")
        before = self.store.file_path("patient", "therapist").read_bytes()
        self.llm.generate.side_effect = ["A complete reflection.", "A complete summary.", '{"facts": "broken"}']
        builder.finalize_session_memory(self.state)
        self.assertTrue(self.store.file_path("patient", "therapist").read_bytes().startswith(before))
        self.assertEqual(self.state["summary"], "A complete summary.")
        self.assertEqual(self.state["memory_consolidation"], {
            "status": "partial", "source_turns": 1, "processed_sources": 1,
            "validated_facts": 0, "rejected_facts": 0, "invalid_batches": 1,
        })
        records = list(self.store.iter_records("patient", "therapist"))
        self.assertEqual([r["type"] for r in records], [
            "conversation_turn", "fact_batch", "session_reflection", "long_term_summary",
        ])
        self.assertEqual(records[1]["facts"], [])
        self.assertTrue(records[1]["validation_error"])
        self.assertEqual(self.llm.generate.call_count, 3)

    def test_api_exposes_partial_memory_and_persists_health_in_ledger_snapshot(self):
        source = EvidenceMemory(self.store).record_turn(
            patient_id="patient", therapist_id="therapist", session_id="session", turn_index=1,
            therapist_text="The notebook is Birch Pages.",
            patient_text="Keeping a notebook sounds fine.",
        )
        good = dict(source_id=source["id"], speaker="therapist", quote="The notebook is Birch Pages.",
                    entity="notebook", attribute="name", value="Birch Pages", status="reported")
        rejected = dict(source_id=source["id"], speaker="patient", quote="Keeping a notebook sounds fine.",
                        entity="activity", attribute="agreement", value="Keeping a notebook", status="agreed")
        self.llm.generate.side_effect = ["A complete reflection.", "A complete summary.",
                                         json.dumps({"facts": [good, rejected]})]
        logger = Mock()
        key = ("therapist", "patient", "session")
        sessions = {key: {"logger": logger, "latest_state": self.state}}
        with patch.object(api, "session_loggers", sessions):
            response = asyncio.run(api.end_session(api.SessionEndRequest(
                external_patient_id="patient", therapist_id="therapist", session_id="session",
            )))
        self.assertEqual(response.status, "finalized")
        self.assertEqual(response.memory_status, "partial")
        self.assertTrue(response.memory_warnings)
        self.assertNotIn(key, sessions)
        logger.finalize.assert_called_once_with(self.state)
        from agent.utils.run_logger import _state_snapshot, _hydrate_snapshot
        restored = _hydrate_snapshot(json.loads(json.dumps(_state_snapshot(self.state))))
        self.assertEqual(restored["memory_consolidation"], self.state["memory_consolidation"])
        reopened = EvidenceMemory(JsonlMemoryStore(Path(self.directory.name)))
        self.assertEqual([f["value"] for f in reopened.current_facts("patient", "therapist")], ["Birch Pages"])
        self.assertEqual(reopened.consolidation_status("patient", "therapist", "session")["rejected_facts"], 1)

    def test_rate_limit_during_extraction_keeps_session_open_and_memory_unchanged(self):
        EvidenceMemory(self.store).record_turn(
            patient_id="patient", therapist_id="therapist", session_id="session", turn_index=1,
            therapist_text="The notebook is Birch Pages.", patient_text="Understood.",
        )
        path = self.store.file_path("patient", "therapist")
        before = path.read_bytes()
        self.llm.generate.side_effect = ["Reflection.", "Summary.",
                                         VertexRateLimitError(60, reason="synthetic capacity limit")]
        logger = Mock()
        key = ("therapist", "patient", "session")
        sessions = {key: {"logger": logger, "latest_state": self.state}}
        with patch.object(api, "session_loggers", sessions), self.assertRaises(VertexRateLimitError):
            asyncio.run(api.end_session(api.SessionEndRequest(
                external_patient_id="patient", therapist_id="therapist", session_id="session",
            )))
        self.assertIn(key, sessions)
        logger.finalize.assert_not_called()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.state["summary"], "Previous completed memory.")

    def test_empty_extraction_still_prevents_finalization(self):
        EvidenceMemory(self.store).record_turn(
            patient_id="patient", therapist_id="therapist", session_id="session", turn_index=1,
            therapist_text="The notebook is Birch Pages.", patient_text="Understood.",
        )
        self.llm.generate.side_effect = ["Reflection.", "Summary.", ""]
        with self.assertRaises(builder.SessionMemoryError):
            builder.finalize_session_memory(self.state)
        self.assertEqual([r["type"] for r in self.store.iter_records("patient", "therapist")], ["conversation_turn"])
        self.assertEqual(self.state["summary"], "Previous completed memory.")

    def test_successful_finalization_consolidates_literal_facts(self):
        source = EvidenceMemory(self.store).record_turn(patient_id="patient", therapist_id="therapist", session_id="session",
                                                       turn_index=1, therapist_text="The pass is Amber Moon.", patient_text="Understood.")
        fact = dict(source_id=source["id"], speaker="therapist", quote="The pass is Amber Moon.",
                    entity="library pass", attribute="name", value="Amber Moon", status="reported")
        self.llm.generate.side_effect = ["A complete reflection.", "A complete summary.", json.dumps({"facts": [fact]})]
        builder.finalize_session_memory(self.state)
        reopened = EvidenceMemory(JsonlMemoryStore(Path(self.directory.name)))
        self.assertEqual(reopened.current_facts("patient", "therapist")[0]["value"], "Amber Moon")
        self.assertEqual(self.llm.generate.call_count, 3)

    def test_retrieval_searches_old_reflections_without_a_hard_topic_filter(self):
        item = SimpleNamespace(score=0.9, value={"text": "The pass is Amber Moon.", "type": "session_reflection"})
        builder.LONG_TERM_STORE.search.side_effect = [[], [item], []]
        found = builder.fetch_relevant_episodic_memories(patient_id="patient", therapist_id="therapist",
                                                        query="pass", topic={"top": "unrelated", "sub": "other"})
        self.assertEqual(found, ["The pass is Amber Moon."])
        self.assertTrue(all("topic_key" not in call.kwargs["filter"] for call in builder.LONG_TERM_STORE.search.call_args_list))


if __name__ == "__main__":
    unittest.main()
