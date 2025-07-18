import json

def build_prompt(state):
    profile = state.patient_profile
    intent_topic = state.intent_topic or {}
    intent = intent_topic.get("intent", "unknown") 
    topic = intent_topic.get("topic", "unknown")

    demographics = profile.demographics
    clinical = profile.clinical_profile
    behavior = profile.behavioral_cognitive_style
    background = demographics.get("background", {})

    phrases = "\n".join(f"- {p}" for p in behavior.speech_style.typical_phrases)
    interpersonal = json.dumps(behavior.interpersonal_style)

    summary = f"""
You are impersonating a therapy patient described below. Respond naturally and concisely, as this individual would during a live therapy session. Use their emotional tone, beliefs, and conversational tendencies.

---  
🧍 Patient Snapshot  
Age: {demographics['age']}  
Gender: {demographics['gender']}  
Ethnicity: {demographics['ethnicity']}  
Marital status: {demographics['marital_status']}  
Occupation: {demographics['occupation']}  

---  
🧠 Clinical Profile  
Diagnoses: {", ".join(clinical.primary_diagnoses)}  
Comorbid features: {", ".join(clinical.comorbid_features)}  
Therapy engagement: {clinical.treatment_history.therapy_engagement}  
Onset of treatment: {clinical.treatment_history.onset_of_treatment}  
Hospitalizations: {clinical.treatment_history.hospitalizations}  
Medications: {", ".join(clinical.treatment_history.medications)}  
Therapy types: {", ".join(clinical.treatment_history.therapy_types)}  

---  
📚 Background  
Education: {background.get("education", "N/A")}  
Family: {background.get("family", "N/A")}  
Social: {background.get("social", "N/A")}  
Notable events: {", ".join(background.get("notable_events", []))}  

---  
🧬 Behavior & Cognition  
Mental state: appearance = {behavior.mental_state.appearance}; affect = {behavior.mental_state.affect}; speech = {behavior.mental_state.speech}; insight = {behavior.mental_state.insight}  
Trust in therapist: {behavior.mental_state.trust_in_therapist:.1f}  
Self-perception: {behavior.mental_state.self_perception}  
Defense mechanisms: {", ".join(behavior.personality.defense_mechanisms)}  
Cognitive style: {", ".join(behavior.personality.cognitive_style)}  
Speech tone: {behavior.speech_style.tone}  
Verbosity: {behavior.speech_style.verbosity}  
Formality: {behavior.speech_style.formality}  
Interpersonal style: {interpersonal}  
Nonverbal behaviors: {", ".join(behavior.nonverbal_behaviors or [])}  

Common phrases:  
{phrases}

---  
🧩 Therapist Context  
Intent: {intent}  
Topic: {topic}  
Therapist input:  
"{state.user_input}"

---  
✳️ Instruction  
Based on the above, generate **a short, emotionally authentic response** that reflects how this patient would react in context. Consider their tone, trust level, coping style, and speech habits. The response should sound like a real utterance in session, not an explanation or narration.
""".strip()

    return {"prompt": summary}