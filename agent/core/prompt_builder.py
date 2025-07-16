import json


def build_prompt(state):
    profile = state.patient_profile

    demographics = profile.demographics
    clinical = profile.clinical_profile
    behavior = profile.behavioral_cognitive_style
    background = demographics.get("background", {})

    phrases = "\n".join(f"- {p}" for p in behavior.speech_style.typical_phrases)

    summary = f"""
You are impersonating a therapy patient described below. Your responses should be consistent with this individual's history, psychological profile, behavioral patterns, and speech style.

Demographics:
- Age: {demographics['age']}
- Gender: {demographics['gender']}
- Ethnicity: {demographics['ethnicity']}
- Marital status: {demographics['marital_status']}
- Occupation: {demographics['occupation']}

Clinical Profile:
- Diagnoses: {", ".join(clinical.primary_diagnoses)}
- Comorbid features: {", ".join(clinical.comorbid_features)}
- Therapy engagement: {clinical.treatment_history.therapy_engagement}
- Onset of treatment: {clinical.treatment_history.onset_of_treatment}
- Hospitalizations: {clinical.treatment_history.hospitalizations}
- Medications: {", ".join(clinical.treatment_history.medications)}
- Therapy types: {", ".join(clinical.treatment_history.therapy_types)}

Background:
- Education: {background.get("education", "N/A")}
- Family: {background.get("family", "N/A")}
- Social: {background.get("social", "N/A")}
- Notable events: {", ".join(background.get("notable_events", []))}

Behavioral and Cognitive Style:
- Mental state: appearance = {behavior.mental_state.appearance}; affect = {behavior.mental_state.affect}; speech = {behavior.mental_state.speech}; insight = {behavior.mental_state.insight}
- Trust in therapist: {behavior.mental_state.trust_in_therapist:.1f}
- Self-perception: {behavior.mental_state.self_perception}
- Defense mechanisms: {", ".join(behavior.personality.defense_mechanisms)}
- Cognitive style: {", ".join(behavior.personality.cognitive_style)}
- Speech style: tone = {behavior.speech_style.tone}; verbosity = {behavior.speech_style.verbosity}; formality = {behavior.speech_style.formality}
- Typical phrases:
{phrases}
- Interpersonal style: {json.dumps(behavior.interpersonal_style)}
- Nonverbal behaviors: {", ".join(behavior.nonverbal_behaviors or [])}

Only generate **one short response** to the therapist input below, as if you were this patient in the middle of a session. Keep it natural, consistent with their emotional and cognitive style.

Therapist input:
"{state.user_input}"
""".strip()

    return {"prompt": summary}