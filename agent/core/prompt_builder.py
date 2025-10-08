import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TOPICS_PATH = Path("../data/topics_tree.json")
with open(TOPICS_PATH, "r", encoding="utf-8") as f:
    TOPICS_JSON = json.load(f)


def build_prompt(state):
    profile = state.patient_profile
    intent_topic = state.intent_topic or {}
    intent = intent_topic.get("intent", "unknown")
    top_topic = intent_topic.get("top", "unknown")
    sub_topic = intent_topic.get("sub", "unknown")

    # === Core patient data ===
    psych = profile.PsychologicalProfile
    demographic = profile.DemographicInfo

    always_sections = [
        ("🧠 Psychological Profile", psych.dict() if hasattr(psych, "dict") else psych),
        ("🧍 Demographic Information", demographic.dict() if hasattr(demographic, "dict") else demographic),
    ]

    always_text = "\n".join(
        f"---\n{title}\n{json.dumps(data, indent=2)}"
        for title, data in always_sections
    )

    # === Dynamically include extra fields based on topic ===
    dynamic_sections = []
    used_fields = []
    if top_topic in TOPICS_JSON:
        metadata = TOPICS_JSON[top_topic].get("metadata", {})
        profile_fields = metadata.get("profile_fields", [])
        for section_name in profile_fields:
            if section_name in {"DemographicInfo", "PsychologicalProfile"}:
                continue
            section = getattr(profile, section_name, None)
            if section:
                section_data = section.dict() if hasattr(section, "dict") else section
                dynamic_sections.append(
                    f"\n---\n📂 {section_name}\n{json.dumps(section_data, indent=2)}"
                )
                used_fields.append(section_name)

    logger.info(f"🧩 Building prompt for topic: {top_topic} → {sub_topic}")
    if used_fields:
        logger.info(f"   → Included patient fields: {', '.join(used_fields)}")
    else:
        logger.info("   → No dynamic patient fields added for this topic.")

    # === Build conversation memory ===
    history_text = ""
    if state.summary.strip():
        history_text += f"\n🧾 Summary of previous sessions:\n{state.summary.strip()}\n"
    if state.history:
        last_turns = "\n".join(
            [f"👩‍⚕️ Therapist: {h['therapist']}\n🧍 Patient: {h['patient']}" for h in state.history[-5:]]
        )
        history_text += f"\n💬 Recent conversation (last {len(state.history[-5:])} turns):\n{last_turns}\n"

    # === Emotional continuity (if tracked) ===
    emotional_tone = getattr(profile, "current_emotional_state", "not specified")
    last_topic = (
        f"{state.last_topic['top']} → {state.last_topic['sub']}"
        if state.last_topic else "unknown"
    )

    # === Build final prompt ===
    summary = f"""
You are impersonating a therapy patient described below. You must respond naturally and consistently across turns, preserving emotional tone, personality traits, and prior conversational themes.

{always_text}

{''.join(dynamic_sections)}

{history_text}

---
🧩 Context for This Turn
• Last discussed topic: {last_topic}
• Current detected topic: {top_topic} → {sub_topic}
• Current emotional tone: {emotional_tone}
• Therapist's latest message: "{state.user_input}"

---
✳️ Instruction
Generate a **emotionally authentic reply** (1–3 sentences) as this patient would respond *in the middle of a real session*. 
Your reply must:
- Be consistent with their personality and emotional patterns.
- Reflect continuity with the ongoing dialogue and prior mood.
- Avoid narration or analysis—speak as the patient, not about them.
""".strip()

    return {"prompt": summary}