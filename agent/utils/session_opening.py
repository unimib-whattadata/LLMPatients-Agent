"""Helpers for resuming therapy sessions with a contextual greeting."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from agent.core.patient_profile import PatientProfile

ROOT_DIR = Path(__file__).resolve().parents[2]
PATIENTS_DIR = ROOT_DIR / "data" / "patients"


def load_patient_profile(patient_id: str) -> PatientProfile:
    """Load the requested patient JSON file and convert it into a profile instance."""
    patient_path = PATIENTS_DIR / f"{patient_id}.json"
    if not patient_path.exists():
        raise FileNotFoundError(f"Patient file not found: {patient_path}")
    return PatientProfile.from_file(str(patient_path))


def default_welcome(profile: PatientProfile) -> str:
    """Return the static welcome message or fall back to a simple greeting."""
    metadata = getattr(profile, "Metadata", None)
    if metadata and getattr(metadata, "welcomeMessage", None):
        return metadata.welcomeMessage
    return f"Hi, I'm {profile.name}. It's good to meet again."


def build_session_opening(
    profile: PatientProfile,
    restored_state: Optional[dict],
    llm_runner,
) -> Optional[str]:
    """Create a session-opening greeting that acknowledges prior context."""
    if not restored_state:
        return None

    summary = restored_state.get("summary", "").strip()
    last_topic = restored_state.get("last_topic") or {}
    last_topic_label = ""
    if last_topic:
        last_topic_label = f"{last_topic.get('top', 'unknown')} → {last_topic.get('sub', 'unknown')}"
    previous_patient_reply = ""
    history = restored_state.get("history") or []
    if history:
        previous_patient_reply = history[-1].get("patient", "")

    if not summary and not previous_patient_reply:
        return None

    prompt = f"""
You are {profile.to_text_summary()}.
You're greeting your therapist at the start of a new session after some time has passed.

Previous context:
- Summary: {summary or 'No summary recorded.'}
- Last thing you said: {previous_patient_reply or '[not available]'}
- Last discussed topic: {last_topic_label or 'unknown'}

Write the first thing you would say now. Requirements:
- It should sound like a natural hello/check-in after time apart, not a continuation mid-sentence.
- Briefly hint at how you've been feeling since last session (overall mood, energy, etc.) without diving into details.
- Tone must match the patient’s personality; be genuine, 1-2 sentences max.
""".strip()

    try:
        opening = llm_runner.generate(prompt=prompt).strip()
    except Exception:
        return None
    return opening or None
