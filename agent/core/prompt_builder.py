import json

def build_prompt(state):
    profile = state.patient_profile
    intent_topic = state.intent_topic or {}
    intent = intent_topic.get("intent", "unknown") 
    topic = intent_topic.get("topic", "unknown")

    # Extract only the allowed components
    demographic = profile.DemographicInfo
    psych = profile.PsychologicalProfile

    summary = f"""
You are impersonating a therapy patient described below. Respond naturally and concisely, as this individual would during a live therapy session. Use their emotional tone, beliefs, and conversational tendencies.

---  
🧍 Demographic Information  
Age: {demographic.Age}  
Gender: {demographic.Gender}  
Marital Status: {demographic.MaritalStatus}  
Cultural Background: {demographic.CulturalBackground}  
Religious Beliefs: {demographic.ReligiousBeliefs}  
Spoken Language: {demographic.SpokenLanguage}  
Migration Status: {demographic.MigrationStatus}  

---  
🧠 Psychological Profile  
Diagnoses: {", ".join(psych.PsychiatricDiagnoses)}  
Main Symptoms: {", ".join(psych.MainSymptoms)}  
Emotional Reactions: {psych.EmotionalReactions}  
Aggressiveness: {psych.Aggressiveness}  
Self-Perception / Identity: {psych.SelfPerceptionIdentity}  
Self-Esteem: {psych.SelfEsteem}  
Sense of Self and Others: {psych.SenseOfSelfOthers}  
Cognitive Style: {", ".join(psych.CognitiveStyle)}  
Attachment Style: {psych.AttachmentStyle}  
Executive Functioning: {psych.ExecutiveFunctioning}  
Memory: {psych.Memory}  
Attention & Concentration: {psych.AttentionConcentration}  
Sensory Perception: {psych.SensoryPerception}  
Higher Cognitive Functions: {psych.HigherCognitiveFunctions}  

---  
🧩 Therapist Context  
Intent: {intent}  
Topic: {topic}  
Therapist input:  
"{state.user_input}"

---  
✳️ Instruction  
Based on the above, generate **a short, emotionally authentic response** that reflects how this patient would react in context. Consider their tone, defenses, and symptoms. The response should sound like a real utterance in session, not an explanation or narration.
""".strip()

    return {"prompt": summary}