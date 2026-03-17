import unittest
from pathlib import Path
from types import SimpleNamespace

from agent.core.patient_profile import PatientProfile
from agent.core.prompt_builder import build_prompt


ROOT_DIR = Path(__file__).resolve().parents[1]


class TestPromptBuilderIdentity(unittest.TestCase):
    def test_stable_identity_facts_include_partner_for_unknown_topic(self):
        profile = PatientProfile.from_file(
            ROOT_DIR / "data" / "patients" / "daniel_isherwood_001_erika.yaml"
        )
        state = SimpleNamespace(
            patient_profile=profile,
            intent_topic={"top": "unknown", "sub": "unknown"},
            emotion_state={},
            classified_emotion=None,
            summary="",
            session_reflection="",
            history=[],
            episodic_context=[],
            safety_flags=[],
            emotion_intensity=0.6,
            last_topic=None,
            emotion_event=None,
            safe_user_input="come si chiama tuo marito?",
            user_input="come si chiama tuo marito?",
        )

        prompt = build_prompt(state)["prompt"]

        self.assertIn("Stable Identity Facts", prompt)
        self.assertIn("Spoken language: english", prompt)
        self.assertIn("Primary relationship: Married to a supportive husband, Erik", prompt)


if __name__ == "__main__":
    unittest.main()
