from typing import List, Dict, Optional, Any
from pydantic import BaseModel
import json

class TreatmentHistory(BaseModel):
    onset_of_treatment: int
    hospitalizations: int
    medications: List[str]
    therapy_types: List[str]
    therapy_engagement: Optional[str] = None

class ClinicalProfile(BaseModel):
    primary_diagnoses: List[str]
    comorbid_features: List[str]
    treatment_history: TreatmentHistory
    icd11_reference: Optional[str] = None
    active_symptoms: Optional[Dict[str, str]] = {}

class MentalState(BaseModel):
    appearance: str
    affect: str
    speech: str
    insight: str
    trust_in_therapist: float
    self_perception: str

class Personality(BaseModel):
    traits: Dict[str, float]
    defense_mechanisms: List[str]
    cognitive_style: List[str]

class SpeechStyle(BaseModel):
    verbosity: str
    tone: str
    formality: str
    typical_phrases: List[str]

class BehavioralCognitiveStyle(BaseModel):
    mental_state: MentalState
    personality: Personality
    speech_style: SpeechStyle
    nonverbal_behaviors: Optional[List[str]] = []
    interpersonal_style: Optional[Dict[str, str]] = {}

class PatientProfile(BaseModel):
    id: str
    demographics: Dict[str, Any]
    clinical_profile: ClinicalProfile
    behavioral_cognitive_style: BehavioralCognitiveStyle

    @property
    def name(self) -> str:
        return self.demographics.get("name", self.id.replace("_", " ").title())

    @classmethod
    def from_file(cls, path: str) -> "PatientProfile":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)