"""Offline checks for durable extraction quarantine and session accounting."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from agent.core.factual_memory import EvidenceMemory, MemoryExtractionError, render_evidence
from agent.core.memory_store import JsonlMemoryStore


class TestMemoryQuarantine(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = JsonlMemoryStore(Path(self.directory.name))
        self.memory = EvidenceMemory(self.store)
        self.patient = "quarantine-patient"
        self.therapist = "quarantine-therapist"
        self.session = "quarantine-session"

    def record(self, therapist_text, patient_text="Understood.", **overrides):
        options = dict(
            patient_id=self.patient, therapist_id=self.therapist,
            session_id=self.session, turn_index=1,
            therapist_text=therapist_text, patient_text=patient_text,
        )
        options.update(overrides)
        return self.memory.record_turn(**options)

    @staticmethod
    def fact(turn, quote, value, **overrides):
        proposal = dict(
            source_id=turn["id"], speaker="therapist", quote=quote,
            entity="journal", attribute="title", value=value, status="reported",
        )
        proposal.update(overrides)
        return proposal

    def consolidate(self, response, **overrides):
        generate = Mock(return_value=response)
        options = dict(
            patient_id=self.patient, therapist_id=self.therapist,
            session_id=self.session, generate=generate, invalid_policy="quarantine",
        )
        options.update(overrides)
        return self.memory.consolidate_session(**options), generate

    def status(self, session_id=None):
        return self.memory.consolidation_status(
            self.patient, self.therapist, session_id or self.session,
        )

    def assert_status(self, **expected):
        actual = self.status()
        for field, value in expected.items():
            self.assertEqual(actual[field], value, field)

    def raw_bytes(self):
        return self.store.file_path(self.patient, self.therapist).read_bytes()

    def reopen(self):
        self.store = JsonlMemoryStore(Path(self.directory.name))
        self.memory = EvidenceMemory(self.store)

    def evidence(self, query):
        return render_evidence(self.memory.retrieve(
            patient_id=self.patient, therapist_id=self.therapist, query=query,
        ))

    def test_mixed_batch_keeps_only_valid_facts_and_records_exact_rejections(self):
        quote = "The journal is called Lantern Notes."
        turn = self.record(quote)
        valid = self.fact(turn, quote, "Lantern Notes")
        invalid = self.fact(turn, "The journal is called Invented Ledger.", "Invented Ledger")
        response = "\n" + json.dumps({"facts": [valid, invalid]}, indent=2) + "\n"

        batch, generate = self.consolidate(response)
        generate.assert_called_once()
        self.assertEqual(batch["extraction_response"], response)
        self.assertEqual([f["value"] for f in batch["facts"]], ["Lantern Notes"])
        self.assertEqual(len(batch["rejected_facts"]), 1)
        rejected = batch["rejected_facts"][0]
        self.assertEqual(rejected["index"], 1)
        self.assertEqual(rejected["proposal"], invalid)
        self.assertIsInstance(rejected["reason"], str)
        self.assertTrue(rejected["reason"])
        self.assertFalse(batch.get("validation_error"))
        self.assert_status(
            status="partial", source_turns=1, processed_sources=1,
            validated_facts=1, rejected_facts=1, invalid_batches=0,
        )

        self.reopen()
        self.assertEqual(
            [f["value"] for f in self.memory.current_facts(self.patient, self.therapist)],
            ["Lantern Notes"],
        )
        self.assertNotIn("Invented Ledger", self.evidence("What is the journal title?"))
        self.assertIn(quote, self.evidence("What is the journal title?"))
        _, repeated = self.consolidate(response)
        repeated.assert_not_called()
        self.assert_status(processed_sources=1, rejected_facts=1)

    def test_colloquial_agreement_is_rejected_but_source_is_processed(self):
        answer = "Sure, Tuesday works."
        turn = self.record("Would Tuesday suit you?", answer)
        invalid = self.fact(
            turn, answer, "Tuesday", speaker="patient", entity="appointment",
            attribute="day", status="agreed",
        )
        batch, _ = self.consolidate(json.dumps({"facts": [invalid]}))
        self.assertEqual(batch["facts"], [])
        self.assertEqual(batch["rejected_facts"][0]["proposal"], invalid)
        self.assertEqual(self.memory.current_facts(self.patient, self.therapist), [])
        self.assert_status(
            status="partial", source_turns=1, processed_sources=1,
            validated_facts=0, rejected_facts=1, invalid_batches=0,
        )
        self.assertIn(answer, self.evidence("Tuesday appointment"))

        # A session-finalization loop can finish without treating the rejected
        # output as an accepted agreement or requesting it indefinitely.
        next_batch, generate = self.consolidate("must not be generated again")
        self.assertIsNone(next_batch)
        generate.assert_not_called()

    def test_quarantine_preserves_quote_source_status_and_recall_guards(self):
        quote = "The journal is called Lantern Notes."
        turn = self.record(quote, turn_index=1)
        recall_answer = "The journal is called Guessed Ledger."
        recall = self.record(
            "Do you remember the journal title?", recall_answer, turn_index=2,
        )
        foreign = self.record(quote, patient_id="different-patient")
        valid = self.fact(turn, quote, "Lantern Notes")
        invalid = [
            self.fact(turn, "This sentence was never said.", "never said"),
            self.fact(foreign, quote, "Lantern Notes"),
            {**valid, "status": "certain"},
            self.fact(recall, recall_answer, "Guessed Ledger", speaker="patient"),
        ]
        batch, _ = self.consolidate(json.dumps({"facts": [valid, *invalid]}))
        self.assertEqual([f["value"] for f in batch["facts"]], ["Lantern Notes"])
        self.assertEqual([r["index"] for r in batch["rejected_facts"]], [1, 2, 3, 4])
        self.assertEqual([r["proposal"] for r in batch["rejected_facts"]], invalid)
        self.assertTrue(all(r["reason"] for r in batch["rejected_facts"]))
        self.assertNotIn("Guessed Ledger", self.evidence("journal title"))
        self.assert_status(
            status="partial", source_turns=2, processed_sources=2,
            validated_facts=1, rejected_facts=4, invalid_batches=0,
        )

    def test_default_and_explicit_strict_policy_remain_atomic(self):
        quote = "The journal is called Lantern Notes."
        turn = self.record(quote)
        valid = self.fact(turn, quote, "Lantern Notes")
        invalid = {**valid, "quote": "A fabricated sentence."}
        response = json.dumps({"facts": [valid, invalid]})
        before = self.raw_bytes()
        for options in ({}, {"invalid_policy": "raise"}):
            with self.subTest(options=options), self.assertRaises(MemoryExtractionError):
                self.memory.consolidate_session(
                    patient_id=self.patient, therapist_id=self.therapist,
                    session_id=self.session, generate=Mock(return_value=response), **options,
                )
            self.assertEqual(self.raw_bytes(), before)
            self.assertEqual(self.memory.current_facts(self.patient, self.therapist), [])
        self.assert_status(
            status="partial", source_turns=1, processed_sources=0,
            validated_facts=0, rejected_facts=0, invalid_batches=0,
        )

    def test_malformed_nonempty_json_is_durable_and_idempotent_after_restart(self):
        quote = "The worksheet is called Amber Window."
        self.record(quote)
        response = '{"facts": [{"source_id": '
        batch, _ = self.consolidate(response)
        self.assertEqual(batch["facts"], [])
        self.assertEqual(batch["extraction_response"], response)
        self.assertIsInstance(batch["validation_error"], str)
        self.assertTrue(batch["validation_error"])
        self.assert_status(
            status="partial", source_turns=1, processed_sources=1,
            validated_facts=0, rejected_facts=0, invalid_batches=1,
        )
        before = self.raw_bytes()

        self.reopen()
        self.assertIn(quote, self.evidence("worksheet"))
        self.assertEqual(self.memory.current_facts(self.patient, self.therapist), [])
        next_batch, generate = self.consolidate("must not be generated again")
        self.assertIsNone(next_batch)
        generate.assert_not_called()
        self.assertEqual(self.raw_bytes(), before)

    def test_wrong_json_schema_is_quarantined_per_session(self):
        for index, response in enumerate(("[]", "null", '{"facts": "wrong"}', '{"other": []}')):
            session = f"schema-session-{index}"
            with self.subTest(response=response):
                self.record("The worksheet is called Amber Window.", session_id=session)
                batch, _ = self.consolidate(response, session_id=session)
                self.assertEqual(batch["facts"], [])
                self.assertEqual(batch["extraction_response"], response)
                self.assertTrue(batch["validation_error"])
                status = self.status(session)
                self.assertEqual(status["status"], "partial")
                self.assertEqual(status["source_turns"], 1)
                self.assertEqual(status["processed_sources"], 1)
                self.assertEqual(status["invalid_batches"], 1)

    def test_empty_or_nonstring_output_is_not_consumed_as_quarantine(self):
        self.record("The worksheet is called Amber Window.")
        before = self.raw_bytes()
        for response in ("", " \n\t", None, {"facts": []}, []):
            with self.subTest(response=response), self.assertRaises(MemoryExtractionError):
                self.consolidate(response)
            self.assertEqual(self.raw_bytes(), before)
            self.assert_status(processed_sources=0, invalid_batches=0, rejected_facts=0)

    def test_provider_exception_propagates_without_consuming_sources(self):
        self.record("The worksheet is called Amber Window.")
        before = self.raw_bytes()
        unavailable = RuntimeError("Synthetic provider unavailable")
        generate = Mock(side_effect=unavailable)
        with self.assertRaises(RuntimeError) as raised:
            self.consolidate("unused", generate=generate)
        self.assertIs(raised.exception, unavailable)
        generate.assert_called_once()
        self.assertEqual(self.raw_bytes(), before)
        self.assert_status(
            status="partial", source_turns=1, processed_sources=0,
            validated_facts=0, rejected_facts=0, invalid_batches=0,
        )

    def test_failed_batch_write_propagates_and_retry_can_process_the_source(self):
        quote = "The worksheet is called Amber Window."
        self.record(quote)
        response = '{"facts": ['
        before = self.raw_bytes()
        unavailable = OSError("Synthetic store write failure")
        with patch.object(self.store, "append", side_effect=unavailable):
            with self.assertRaises(OSError) as raised:
                self.consolidate(response)
        self.assertIs(raised.exception, unavailable)
        self.assertEqual(self.raw_bytes(), before)
        self.assert_status(processed_sources=0, invalid_batches=0)

        self.reopen()
        batch, generate = self.consolidate(response)
        generate.assert_called_once()
        self.assertTrue(batch["validation_error"])
        self.assert_status(processed_sources=1, invalid_batches=1)
        self.assertIn(quote, self.evidence("worksheet"))

    def test_rejected_first_batch_does_not_block_later_turns(self):
        turns = []
        for index in range(1, 7):
            text = f"The activity entry {index} is Copper Workshop {index}."
            turns.append((self.record(text, turn_index=index), text))

        first_batch, _ = self.consolidate("Malformed first batch")
        self.assertEqual(len(first_batch["source_ids"]), 5)
        self.assertTrue(first_batch["validation_error"])
        self.assert_status(source_turns=6, processed_sources=5, invalid_batches=1)

        last_turn, last_quote = turns[-1]
        response = json.dumps({"facts": [self.fact(
            last_turn, last_quote, "Copper Workshop 6", entity="activity entry 6",
        )]})
        second_batch, generate = self.consolidate(response)
        generate.assert_called_once()
        self.assertEqual(second_batch["source_ids"], [last_turn["id"]])
        self.assertEqual([f["value"] for f in second_batch["facts"]], ["Copper Workshop 6"])
        self.assert_status(
            status="partial", source_turns=6, processed_sources=6,
            validated_facts=1, rejected_facts=0, invalid_batches=1,
        )
        self.reopen()
        final_batch, repeated = self.consolidate("must not be generated again")
        self.assertIsNone(final_batch)
        repeated.assert_not_called()
        self.assertIn(turns[0][1], self.evidence("activity entry 1"))

    def test_all_valid_facts_produce_complete_status(self):
        quote = "The journal is called Lantern Notes."
        turn = self.record(quote)
        batch, _ = self.consolidate(json.dumps({"facts": [
            self.fact(turn, quote, "Lantern Notes"),
        ]}))
        self.assertEqual(len(batch["facts"]), 1)
        self.assertFalse(batch.get("rejected_facts"))
        self.assertFalse(batch.get("validation_error"))
        self.assert_status(
            status="complete", source_turns=1, processed_sources=1,
            validated_facts=1, rejected_facts=0, invalid_batches=0,
        )

    def test_valid_empty_fact_list_is_complete_without_validation_errors(self):
        self.record("Hello.", "Hello.")
        batch, _ = self.consolidate('{"facts": []}')
        self.assertEqual(batch["facts"], [])
        self.assertFalse(batch.get("rejected_facts"))
        self.assertFalse(batch.get("validation_error"))
        self.assert_status(
            status="complete", source_turns=1, processed_sources=1,
            validated_facts=0, rejected_facts=0, invalid_batches=0,
        )


if __name__ == "__main__":
    unittest.main()
