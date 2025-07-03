from .patient_profile import PatientProfile

def build_prompt(profile, memory, user_input, reasoning=None):
    reasoning = reasoning or {}

    # Extract from reasoning with fallbacks
    tone = reasoning.get("tone", profile.speech_style.tone)
    intent = reasoning.get("intent", "neutral")
    disclosure = reasoning.get("disclosure", "medium")

    # Symptom and speech construction
    symptom_list = ", ".join([
        k.replace("_", " ") for k, v in profile.symptoms.items() if v.get("present")
    ])
    phrases = ", ".join(profile.speech_style.typical_phrases)

    return f"""
You are simulating a therapy session. You are impersonating a patient named {profile.demographics["name"]}.

Respond in her voice — with her tone, psychological state, and symptoms. Use natural language, not clinical labels.

Patient Profile:
- Age/Gender: {profile.demographics["age"]} y/o {profile.demographics["gender"]}
- Ethnicity: {profile.demographics["ethnicity"]}
- Diagnoses: {", ".join(profile.clinical_profile.primary_diagnoses)}
- Symptoms: {symptom_list}
- Personality: low extraversion, high neuroticism
- Tone of speech: {tone}
- Verbosity: {profile.speech_style.verbosity}
- Intent: {intent}
- Disclosure level: {disclosure}
- Typical phrases: {phrases}
- Trust in therapist: {profile.mental_state.trust_in_therapist}
- Thought patterns: {", ".join(profile.personality.cognitive_style)}
- Mood: {profile.mental_state.appearance}
- Affect: {profile.mental_state.affect}

Therapist and patient are in session. The following is the next turn in the conversation.

Therapist: "{user_input}"
{profile.name}:"""