from adk import Agent, DeferredTopic, LLMFunctionCallAction, Session, tools
from typing import Any
from app.core.patient_profile import PatientProfile  # This includes your classes

patient = PatientProfile.from_file("path/to/profile.json")

def build_system_prompt(profile: PatientProfile) -> str:
    return f"""
You are impersonating a patient named {profile.name} in a therapeutic roleplay.
Base your answers on the following attributes:

Demographics: {profile.demographics}
Primary Diagnoses: {profile.clinical_profile.primary_diagnoses}
Comorbid Features: {profile.clinical_profile.comorbid_features}
Mental State: {profile.mental_state.dict()}
Personality Traits: {profile.personality.traits}
Speech Style: {profile.speech_style.dict()}
Nonverbal Behaviors: {profile.nonverbal_behaviors}
Interpersonal Style: {profile.interpersonal_style}

You respond with a tone and verbosity consistent with the above profile.
Avoid breaking character. If asked questions beyond the patient’s knowledge, defer or say “I’m not sure.”
"""

trust_topic = DeferredTopic(
    name="trust_in_therapist",
    description="Disclose thoughts about trusting the therapist",
    open_condition=lambda history, state: state.get("trust_in_therapist", 0.0) > 0.6,
    message="I'm starting to feel like I can open up more about that."
)

identity_topic = DeferredTopic(
    name="identity_disturbance",
    description="Discuss identity confusion",
    open_condition=lambda history, state: state.get("identity_disturbance") is True,
    message="Lately, I’ve been struggling to understand who I really am."
)

class PatientAgent(Agent):
    def __init__(self, profile: PatientProfile):
        super().__init__(
            name=profile.name,
            system_prompt=build_system_prompt(profile),
            actions=[
                LLMFunctionCallAction(
                    name="impersonate_patient",
                    description="Respond to the therapist while staying in character.",
                    parameters={"input": "str"},
                    func=self.generate_response
                )
            ],
            deferred_topics=[trust_topic, identity_topic]
        )
        self.profile = profile
        self.state = {
            "trust_in_therapist": profile.mental_state.trust_in_therapist,
            "identity_disturbance": profile.mental_state.identity_disturbance
        }

    def generate_response(self, input: str, **kwargs) -> str:
        # Use self.profile and self.state to guide response generation
        return f"As {self.profile.name}, here's how I’d respond: (model output based on traits)"

def main():  
    agent = PatientAgent(patient)
    session = Session(agent=agent)

    # Simulate interaction
    while True:
        user_input = input("Therapist: ")
        response = session.step(user_input)
        print(f"Patient: {response}")

if __name__ == "__main__":
    main()