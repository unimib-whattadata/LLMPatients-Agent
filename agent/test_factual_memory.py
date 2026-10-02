"""Offline behavioral checks for evidence-backed factual memory.

All dialogue and extraction outputs are synthetic fixtures. These tests do not
load a model or call a service; every store lives in a temporary directory.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from agent.core.factual_memory import (
    EvidenceMemory,
    MemoryExtractionError,
    render_evidence,
)
from agent.core.memory_store import JsonlMemoryStore


class TestFactualMemory(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = JsonlMemoryStore(Path(self.directory.name))
        self.memory = EvidenceMemory(self.store)
        self.patient = "synthetic-patient-a"
        self.therapist = "therapist-a"
        self.session = "session-a"

    def record(self, therapist_text, patient_text="Understood.", **overrides):
        options = dict(
            patient_id=self.patient,
            therapist_id=self.therapist,
            session_id=self.session,
            turn_index=1,
            therapist_text=therapist_text,
            patient_text=patient_text,
            topic=None,
            usable=True,
        )
        options.update(overrides)
        return self.memory.record_turn(**options)

    @staticmethod
    def fact(turn, quote, value, **overrides):
        result = dict(
            source_id=turn["id"],
            speaker="therapist",
            quote=quote,
            entity="daily log",
            attribute="title",
            value=value,
            status="reported",
        )
        result.update(overrides)
        return result

    def consolidate(self, facts, **overrides):
        generate = Mock(return_value=json.dumps({"facts": facts}))
        options = dict(
            patient_id=self.patient,
            therapist_id=self.therapist,
            session_id=self.session,
            generate=generate,
        )
        options.update(overrides)
        self.memory.consolidate_session(**options)
        return generate

    def current(self, include_history=False):
        return self.memory.current_facts(
            self.patient, self.therapist, include_history=include_history,
        )

    def retrieve(self, query, **overrides):
        options = dict(
            patient_id=self.patient,
            therapist_id=self.therapist,
            query=query,
            limit=8,
            token_budget=1800,
        )
        options.update(overrides)
        return self.memory.retrieve(**options)

    def test_facts_and_sources_survive_a_new_memory_instance(self):
        quote = "The daily log is called Coral Notebook (blue cover)."
        turn = self.record(quote)
        self.consolidate([self.fact(turn, quote, "Coral Notebook")])

        reopened = EvidenceMemory(JsonlMemoryStore(Path(self.directory.name)))
        facts = reopened.current_facts(self.patient, self.therapist)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["value"], "Coral Notebook")
        self.assertEqual(facts[0]["quote"], quote)
        self.assertEqual(facts[0]["source_id"], turn["id"])
        self.assertEqual(facts[0]["session_id"], self.session)
        self.assertEqual(facts[0]["turn_index"], 1)
        records = reopened.retrieve(
            patient_id=self.patient,
            therapist_id=self.therapist,
            query="What is the daily log called?",
        )
        self.assertIn(quote, render_evidence(records))

    def test_patient_and_therapist_namespaces_are_isolated(self):
        scopes = [
            (self.patient, self.therapist, "Coral Notebook"),
            ("synthetic-patient-b", self.therapist, "Copper Journal"),
            (self.patient, "therapist-b", "Winter Pages"),
        ]
        for patient, therapist, title in scopes:
            quote = f"The daily log is called {title}."
            turn = self.record(quote, patient_id=patient, therapist_id=therapist)
            self.consolidate(
                [self.fact(turn, quote, title)],
                patient_id=patient,
                therapist_id=therapist,
            )

        for patient, therapist, title in scopes:
            with self.subTest(patient=patient, therapist=therapist):
                facts = self.memory.current_facts(patient, therapist)
                self.assertEqual([fact["value"] for fact in facts], [title])
                evidence = render_evidence(self.memory.retrieve(
                    patient_id=patient,
                    therapist_id=therapist,
                    query="What is the daily log called?",
                ))
                self.assertIn(title, evidence)
                for _, _, other_title in scopes:
                    if other_title != title:
                        self.assertNotIn(other_title, evidence)

    def test_rename_and_time_change_preserve_previous_versions(self):
        first_quote = "The daily log is Coral Notebook; our meeting is at 08:15."
        first = self.record(first_quote, turn_index=7)
        self.consolidate([
            self.fact(first, first_quote, "Coral Notebook"),
            self.fact(first, first_quote, "08:15", entity="meeting", attribute="time"),
        ])
        next_session = "session-b"
        latest_quote = "The daily log is now Orchard Journal; our meeting moves to 17:45."
        latest = self.record(latest_quote, session_id=next_session, turn_index=1)
        self.consolidate([
            self.fact(latest, latest_quote, "Orchard Journal"),
            self.fact(latest, latest_quote, "17:45", entity="meeting", attribute="time"),
        ], session_id=next_session)

        current = {(f["entity"], f["attribute"]): f for f in self.current()}
        self.assertEqual(current[("daily log", "title")]["value"], "Orchard Journal")
        self.assertEqual(current[("daily log", "title")]["previous_value"], "Coral Notebook")
        self.assertEqual(current[("meeting", "time")]["value"], "17:45")
        self.assertEqual(current[("meeting", "time")]["previous_value"], "08:15")
        history = self.current(include_history=True)
        self.assertEqual(len(history), 4)
        self.assertEqual({f["value"] for f in history}, {
            "Coral Notebook", "Orchard Journal", "08:15", "17:45",
        })
        self.assertEqual({f["value"] for f in history if f["current"]}, {
            "Orchard Journal", "17:45",
        })

    def test_source_turn_order_controls_updates_not_extractor_array_order(self):
        first_quote = "The meeting is at 08:15."
        latest_quote = "The meeting has moved to 17:45."
        first = self.record(first_quote, turn_index=1)
        latest = self.record(latest_quote, turn_index=7)
        # A model can output its facts in a different order than the dialogue.
        self.consolidate([
            self.fact(latest, latest_quote, "17:45", entity="meeting", attribute="time"),
            self.fact(first, first_quote, "08:15", entity="meeting", attribute="time"),
        ])
        current = self.current()
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]["value"], "17:45")
        self.assertEqual(current[0]["turn_index"], 7)
        self.assertEqual(current[0]["previous_value"], "08:15")

    def test_query_using_an_old_name_also_retrieves_the_current_version(self):
        for index, title in enumerate(("Coral Notebook", "Orchard Journal"), 1):
            session = f"rename-session-{index}"
            quote = f"The daily log is called {title}."
            turn = self.record(quote, session_id=session)
            self.consolidate([self.fact(turn, quote, title)], session_id=session)

        # A query can identify an entity by an obsolete label. Searching that
        # label must follow its update, including when semantic search is down.
        failed_encoder = Mock(side_effect=RuntimeError("Synthetic encoder outage"))
        evidence = render_evidence(self.retrieve(
            "What is the current name of Coral Notebook?", embed=failed_encoder,
        ))
        self.assertIn("Orchard Journal", evidence)
        self.assertIn("current", evidence)

    def test_explicit_unambiguous_rename_reuses_the_previous_attribute(self):
        old_quote = "The journal's name is Coral Notebook."
        first = self.record(old_quote, turn_index=1)
        self.consolidate([self.fact(
            first, old_quote, "Coral Notebook", entity="journal", attribute="name",
        )])

        new_text = "I am renaming the same journal. Its title is now Orchard Journal."
        new_quote = "Its title is now Orchard Journal."
        latest = self.record(new_text, turn_index=2)
        self.consolidate([self.fact(
            latest, new_quote, "Orchard Journal", entity="journal", attribute="title",
        )])

        facts = self.current()
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["attribute"], "name")
        self.assertEqual(facts[0]["value"], "Orchard Journal")
        self.assertEqual(facts[0]["previous_value"], "Coral Notebook")
        self.assertEqual(facts[0]["source_id"], latest["id"])
        self.assertEqual(facts[0]["quote"], new_quote)
        history = self.current(include_history=True)
        self.assertEqual(len(history), 2)
        self.assertEqual({f["value"] for f in history if f["current"]}, {"Orchard Journal"})

    def test_name_and_title_stay_distinct_without_explicit_rename(self):
        first_quote = "The journal's name is Coral Notebook."
        first = self.record(first_quote, turn_index=1)
        self.consolidate([self.fact(
            first, first_quote, "Coral Notebook", entity="journal", attribute="name",
        )])
        second_quote = "The journal's title is Winter Pages."
        second = self.record(second_quote, turn_index=2)
        self.consolidate([self.fact(
            second, second_quote, "Winter Pages", entity="journal", attribute="title",
        )])

        self.assertEqual({(f["attribute"], f["value"]) for f in self.current()}, {
            ("name", "Coral Notebook"), ("title", "Winter Pages"),
        })
        self.assertTrue(all(f["current"] for f in self.current(include_history=True)))

    def test_ambiguous_journal_reference_does_not_merge_existing_entities(self):
        old_facts = []
        for index, (entity, title) in enumerate((
            ("daily journal", "Coral Notebook"),
            ("work journal", "Winter Pages"),
        ), 1):
            quote = f"The {entity}'s name is {title}."
            turn = self.record(quote, turn_index=index)
            old_facts.append(self.fact(turn, quote, title, entity=entity, attribute="name"))
        self.consolidate(old_facts)

        text = "I am renaming the same journal. Its title is now Orchard Journal."
        new_turn = self.record(text, turn_index=3)
        self.consolidate([self.fact(
            new_turn, "Its title is now Orchard Journal.", "Orchard Journal",
            entity="journal", attribute="title",
        )])

        # The source and extracted generic entity do not identify which of the
        # two journals changed. Preserve the ambiguity instead of inventing it.
        self.assertEqual({(f["entity"], f["value"]) for f in self.current()}, {
            ("daily journal", "Coral Notebook"),
            ("work journal", "Winter Pages"),
            ("journal", "Orchard Journal"),
        })

    def test_compatible_values_from_one_source_are_current_as_a_group(self):
        quote = "The breathing exercise is optional and unconfirmed."
        turn = self.record(quote, turn_index=1)
        self.consolidate([
            self.fact(turn, quote, value, entity="breathing exercise", attribute="commitment")
            for value in ("optional", "unconfirmed")
        ])

        facts = self.current()
        self.assertEqual({f["value"] for f in facts}, {"optional", "unconfirmed"})
        self.assertEqual(len(facts), 2)
        self.assertTrue(all(f["current"] for f in facts))
        self.assertTrue(all("supersedes" not in f for f in facts))

        # A later assertion updates the earlier source group, not whichever
        # member happens to have the larger hash.
        updated_quote = "The breathing exercise is confirmed."
        latest = self.record(updated_quote, turn_index=2)
        self.consolidate([self.fact(
            latest, updated_quote, "confirmed", entity="breathing exercise",
            attribute="commitment",
        )])
        self.assertEqual([f["value"] for f in self.current()], ["confirmed"])
        history = self.current(include_history=True)
        self.assertEqual(len(history), 3)
        self.assertEqual({f["value"] for f in history if not f["current"]}, {
            "optional", "unconfirmed",
        })

    def test_patient_report_does_not_overwrite_therapist_statement(self):
        therapist_quote = "I remember the meeting time as 08:15."
        patient_quote = "I remember the meeting time as 17:45."
        turn = self.record(therapist_quote, patient_quote)
        self.consolidate([
            self.fact(turn, therapist_quote, "08:15", entity="meeting",
                      attribute="time", status="reported"),
            self.fact(turn, patient_quote, "17:45", entity="meeting",
                      attribute="time", status="reported", speaker="patient"),
        ])
        facts = self.current()
        self.assertEqual(len(facts), 2)
        self.assertEqual({(f["speaker"], f["value"]) for f in facts}, {
            ("therapist", "08:15"), ("patient", "17:45"),
        })

    def test_proposed_activity_remains_proposed(self):
        quote = "I suggest a ten-minute walk on Wednesday."
        turn = self.record(quote, "I will think about it.")
        self.consolidate([self.fact(
            turn, quote, "ten-minute walk", entity="homework",
            attribute="activity", status="proposed",
        )])
        facts = self.current()
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["status"], "proposed")
        self.assertFalse(any(f["status"] in {"agreed", "completed"} for f in facts))
        self.assertIn("proposed", render_evidence(self.retrieve("What homework was proposed?")))

    def test_later_proposal_does_not_replace_an_existing_agreement(self):
        agreed_quote = "We have agreed to a ten-minute walk on Wednesday."
        proposed_quote = "I suggest a breathing exercise instead."
        agreed = self.record(agreed_quote, turn_index=1)
        proposed = self.record(proposed_quote, "I have not decided yet.", turn_index=2)
        self.consolidate([
            self.fact(agreed, agreed_quote, "ten-minute walk", entity="homework",
                      attribute="activity", status="agreed"),
            self.fact(proposed, proposed_quote, "breathing exercise", entity="homework",
                      attribute="activity", status="proposed"),
        ])
        self.assertEqual({(f["status"], f["value"]) for f in self.current()}, {
            ("agreed", "ten-minute walk"), ("proposed", "breathing exercise"),
        })

    def test_proposal_promoted_to_agreement_or_completion_rejects_batch(self):
        assertion = "The daily log is called Coral Notebook."
        good_turn = self.record(assertion, turn_index=1)
        proposal = "I propose an optional breathing exercise on Tuesday."
        proposal_turn = self.record(proposal, "I have not decided.", turn_index=2)
        good_fact = self.fact(good_turn, assertion, "Coral Notebook")
        proposal_fact = self.fact(
            proposal_turn, proposal, "breathing exercise", entity="homework",
            attribute="activity", status="proposed",
        )
        before = self.store.file_path(self.patient, self.therapist).read_bytes()

        for invented_status in ("agreed", "completed"):
            with self.subTest(status=invented_status):
                with self.assertRaises(MemoryExtractionError):
                    self.consolidate([
                        good_fact, {**proposal_fact, "status": invented_status},
                    ])
                self.assertEqual(self.current(include_history=True), [])
                self.assertEqual(
                    self.store.file_path(self.patient, self.therapist).read_bytes(),
                    before,
                )

        retry = self.consolidate([good_fact, proposal_fact])
        retry.assert_called_once()
        self.assertEqual({(f["status"], f["value"]) for f in self.current()}, {
            ("reported", "Coral Notebook"), ("proposed", "breathing exercise"),
        })

    def test_short_quote_cannot_hide_proposal_context_and_invent_completion(self):
        assertion = "The daily log is called Coral Notebook."
        good_turn = self.record(assertion, turn_index=1)
        proposal = "I suggest an optional breathing exercise on Tuesday."
        proposal_turn = self.record(proposal, "I have not decided.", turn_index=2)
        good_fact = self.fact(good_turn, assertion, "Coral Notebook")
        unsupported_fact = self.fact(
            proposal_turn, "breathing exercise", "breathing exercise",
            entity="homework", attribute="activity", status="completed",
        )

        # The short quote is literal, but its containing assertion is only a
        # suggestion. Literal substring validation cannot establish completion.
        with self.assertRaises(MemoryExtractionError):
            self.consolidate([good_fact, unsupported_fact])
        self.assertEqual(self.current(include_history=True), [])
        retry = self.consolidate([
            good_fact, {**unsupported_fact, "quote": proposal, "status": "proposed"},
        ])
        retry.assert_called_once()

    def test_completed_optional_activity_is_not_misclassified_as_a_proposal(self):
        answer = "I completed the optional breathing exercise yesterday."
        turn = self.record("How did the exercise go?", answer)
        self.consolidate([self.fact(
            turn, answer, "breathing exercise", entity="homework", attribute="activity",
            speaker="patient", status="completed",
        )])
        facts = self.current()
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["status"], "completed")
        self.assertEqual(facts[0]["quote"], answer)

    def test_short_quote_inside_a_question_cannot_establish_a_fact(self):
        turn = self.record("Did you complete the breathing exercise?", "No.")
        for status in ("reported", "agreed", "completed"):
            with self.subTest(status=status), self.assertRaises(MemoryExtractionError):
                self.consolidate([self.fact(
                    turn, "breathing exercise", "breathing exercise",
                    entity="homework", attribute="activity", status=status,
                )])
            self.assertEqual(self.current(include_history=True), [])

    def test_assertion_before_a_question_is_eligible_for_extraction(self):
        turn = self.record("The journal is Coral Notebook. How do you feel about it?")
        self.consolidate([self.fact(turn, "The journal is Coral Notebook.", "Coral Notebook")])
        self.assertEqual(self.current()[0]["value"], "Coral Notebook")

    def test_unsupported_quote_or_value_rejects_the_entire_batch(self):
        quote = "The daily log is called Coral Notebook."
        turn = self.record(quote, "I understand the instructions.")
        good = self.fact(turn, quote, "Coral Notebook")
        invalid_facts = [
            {**good, "quote": "The daily log is called Invented Pages.",
             "value": "Invented Pages"},
            {**good, "value": "Invented Pages"},
            {**good, "speaker": "patient"},
        ]
        for invalid in invalid_facts:
            with self.subTest(invalid=invalid):
                with self.assertRaises(MemoryExtractionError):
                    self.consolidate([good, invalid])
                self.assertEqual(self.current(include_history=True), [])

        # The failed extraction must not prevent a later valid extraction.
        retry = self.consolidate([good])
        retry.assert_called_once()
        self.assertEqual([f["value"] for f in self.current()], ["Coral Notebook"])

    def test_foreign_patient_source_is_rejected(self):
        own_quote = "The daily log is called Coral Notebook."
        self.record(own_quote)
        foreign_quote = "The daily log is called Copper Journal."
        foreign = self.record(foreign_quote, patient_id="synthetic-patient-b")
        with self.assertRaises(MemoryExtractionError):
            self.consolidate([self.fact(foreign, foreign_quote, "Copper Journal")])
        self.assertEqual(self.current(include_history=True), [])

    def test_malformed_json_does_not_mark_extraction_complete(self):
        quote = "The daily log is called Coral Notebook."
        turn = self.record(quote)
        generate = Mock(return_value='{"facts": [')
        with self.assertRaises(MemoryExtractionError):
            self.memory.consolidate_session(
                patient_id=self.patient,
                therapist_id=self.therapist,
                session_id=self.session,
                generate=generate,
            )
        self.assertEqual(self.current(include_history=True), [])

        self.memory = EvidenceMemory(JsonlMemoryStore(Path(self.directory.name)))
        retry = self.consolidate([self.fact(turn, quote, "Coral Notebook")])
        retry.assert_called_once()
        self.assertEqual(len(self.current()), 1)

    def test_successful_consolidation_is_idempotent_after_restart(self):
        quote = "The daily log is called Coral Notebook."
        turn = self.record(quote)
        facts = [self.fact(turn, quote, "Coral Notebook")]
        first = self.consolidate(facts)
        first.assert_called_once()
        records_before = list(self.store.iter_records(self.patient, self.therapist))

        self.memory = EvidenceMemory(JsonlMemoryStore(Path(self.directory.name)))
        second = self.consolidate(facts)
        second.assert_not_called()
        self.assertEqual(len(self.current(include_history=True)), 1)
        self.assertEqual(
            list(self.store.iter_records(self.patient, self.therapist)),
            records_before,
        )

    def test_archived_raw_dialogue_is_retrievable_without_extracted_facts(self):
        quote = "The worksheet title is Copper Finch (revised edition)."
        self.record(quote, topic="planning")
        self.consolidate([])
        self.record("How has your sleep been?", "Much better.", session_id="session-b")
        self.memory = EvidenceMemory(JsonlMemoryStore(Path(self.directory.name)))

        self.assertEqual(self.current(), [])
        evidence = render_evidence(self.retrieve("What was the worksheet title?"))
        self.assertIn(quote, evidence)
        self.assertIn("(revised edition)", evidence)

    def test_assertion_followed_by_question_outscores_a_pure_question(self):
        mixed_text = "The worksheet is Copper Finch. How does that feel?"
        mixed = self.record(mixed_text, turn_index=1)
        self.record("What is the worksheet called?", turn_index=2)

        # Both turns mention the queried noun once. The later pure question
        # must not outrank the earlier assertion because both end in '?'.
        evidence = self.retrieve("worksheet", limit=1)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["source_id"], mixed["id"])
        self.assertEqual(evidence[0]["quote"], mixed_text)
        self.assertIn(mixed_text, render_evidence(evidence))

    def test_generic_autobiographical_question_preserves_patient_answer(self):
        question = "Do you remember what happened last weekend?"
        answer = "Last weekend I visited my aunt at Lakeside Park."
        turn = self.record(question, answer)

        self.assertIn(answer, render_evidence(self.retrieve("What happened last weekend?")))
        self.consolidate([self.fact(
            turn, answer, "Lakeside Park", entity="weekend visit", attribute="place",
            speaker="patient", status="reported",
        )])
        self.memory = EvidenceMemory(JsonlMemoryStore(Path(self.directory.name)))
        facts = self.current()
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["speaker"], "patient")
        self.assertEqual(facts[0]["value"], "Lakeside Park")
        self.assertIn(answer, render_evidence(self.retrieve("Where was the weekend visit?")))

    def test_extractor_failure_keeps_the_original_turn_available(self):
        quote = "The worksheet title is Copper Finch (revised edition)."
        self.record(quote)
        generate = Mock(side_effect=RuntimeError("Synthetic extractor outage"))
        with self.assertRaises((RuntimeError, MemoryExtractionError)):
            self.memory.consolidate_session(
                patient_id=self.patient,
                therapist_id=self.therapist,
                session_id=self.session,
                generate=generate,
            )
        self.memory = EvidenceMemory(JsonlMemoryStore(Path(self.directory.name)))
        self.assertEqual(self.current(), [])
        self.assertIn(quote, render_evidence(self.retrieve("What was the worksheet title?")))

    def test_evidence_budget_does_not_cut_through_a_quote(self):
        quote = "QUOTE_START " + ("complete evidence " * 80) + "(QUOTE_END)."
        turn = self.record(quote)
        self.consolidate([self.fact(
            turn, quote, "complete evidence", entity="evidence", attribute="text",
        )])
        records = self.retrieve("complete evidence", token_budget=12000)
        self.assertIn(quote, render_evidence(records, token_budget=12000))

        constrained = render_evidence(records, token_budget=50)
        # Omit an oversized source as a whole; a clipped quotation is misleading.
        self.assertNotIn("QUOTE_START", constrained)
        self.assertNotIn("QUOTE_END", constrained)
        self.assertNotIn("complete evidence", constrained)


if __name__ == "__main__":
    unittest.main()
