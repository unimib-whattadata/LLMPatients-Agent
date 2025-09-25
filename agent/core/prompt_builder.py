import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TOPICS_PATH = Path("../data/topics_tree.json")  # adjust path if needed
with open(TOPICS_PATH, "r", encoding="utf-8") as f:
    TOPICS_JSON = json.load(f)

def build_prompt(state):
    profile = state.patient_profile
    intent_topic = state.intent_topic or {}
    intent = intent_topic.get("intent", "unknown") 
    top_topic = intent_topic.get("top", "unknown")
    sub_topic = intent_topic.get("sub", "unknown")

    # Always include psychological profile and demographics
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

    # Dynamic inclusion based on metadata
    dynamic_sections = []
    used_fields = []  # for logging
    if top_topic in TOPICS_JSON:
        metadata = TOPICS_JSON[top_topic].get("metadata", {})
        profile_fields = metadata.get("profile_fields", [])
        for section_name in profile_fields:
            # Skip DemographicInfo & PsychologicalProfile since already included
            if section_name in {"DemographicInfo", "PsychologicalProfile"}:
                continue
            section = getattr(profile, section_name, None)
            if section:
                # If section is a pydantic model, convert to dict
                if hasattr(section, "dict"):
                    section_data = section.dict()
                else:
                    section_data = section
                dynamic_sections.append(
                    f"\n---\n📂 {section_name}\n{json.dumps(section_data, indent=2)}"
                )
                used_fields.append(section_name)

    # Logging what we’re including
    logger.info("🧩 Building prompt for topic:")
    logger.info(f"   → Top: {top_topic}, Sub: {sub_topic}")
    if used_fields:
        logger.info(f"   → Included patient fields: {', '.join(used_fields)}")
    else:
        logger.info("   → No dynamic patient fields added for this topic.")

    # === Conversation history handling ===
    history_text = ""
    if state.summary:
        history_text += f"\n📝 Conversation Summary (earlier):\n{state.summary.strip()}\n"
    if state.history:
        last_turns = "\n".join(
            [f"👩‍⚕️ Therapist: {h['therapist']}\n🧍 Patient: {h['patient']}" for h in state.history]
        )
        history_text += f"\n💬 Recent Conversation (last {len(state.history)} turns):\n{last_turns}\n"

    # Build final prompt
    summary = f"""
You are impersonating a therapy patient described below. Respond naturally and concisely, as this individual would during a live therapy session. Use their emotional tone, beliefs, and conversational tendencies.

{always_text}

{''.join(dynamic_sections)}

{history_text}

---  
🧩 Therapist Context  
Intent: {intent}  
Topic: {top_topic} → {sub_topic}  
Therapist input:  
"{state.user_input}"

---  
✳️ Instruction  
Based on the above, generate **a short, emotionally authentic response** that reflects how this patient would react in context. Consider their tone, defenses, and symptoms. The response should sound like a real utterance in session, not an explanation or narration.
""".strip()

    return {"prompt": summary}