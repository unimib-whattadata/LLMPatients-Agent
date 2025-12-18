"""Structured patient profile models aligned with the JSON schema in data/patients."""

import json

from pathlib import Path
from pydantic import BaseModel, Field
from agent.core.emotion_model import EMOTIONS
from typing import Dict, List, Optional, Union
from agent.core.safety import NOT_REPORTED_MARKERS

# === Emotion Traits ===
class EmotionTraits(BaseModel):
    """Baseline affective systems and volatility taken from patient JSON."""

    trait_baseline: Dict[str, float] = Field(default_factory=dict)
    volatility_level: str = "medium"

    def normalized_baseline(self) -> Dict[str, float]:
        """Return a full trait vector with missing values filled and clamped."""
        normalized = {}
        for key in EMOTIONS:
            raw_value = (
                self.trait_baseline.get(key)
                or self.trait_baseline.get(key.lower())
                or self.trait_baseline.get(key.capitalize())
                or 0.5
            )
            normalized[key] = _clamp_emotion_value(raw_value)
        return normalized


# === Detail sub-components ===
class DemographicAndSocioculturalInformation(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    maritalStatus: Optional[str] = None
    culturalBackground: Optional[str] = None
    religiousBeliefs: Optional[str] = None
    spokenLanguage: Optional[str] = None
    migrationStatus: Optional[str] = None

    def to_prompt(self) -> str:
        """Concise demographic identity string."""
        parts = []
        if self.name:
            parts.append(self.name)
        if self.age:
            parts.append(f"{self.age}-year-old")
        if self.culturalBackground:
            parts.append(self.culturalBackground)
        if self.gender:
            gender_map = {"female": "woman", "male": "man", "non-binary": "non-binary person"}
            parts.append(gender_map.get(self.gender.lower(), self.gender))

        sentence = " ".join(parts).strip()
        if self.maritalStatus and self.maritalStatus.lower() not in NOT_REPORTED_MARKERS:
            sentence = (sentence + f", {self.maritalStatus.lower()}").strip()

        return f"You are {sentence}." if sentence else ""


class CurrentRelationshipWithParents(BaseModel):
    mother: Optional[str] = None
    father: Optional[str] = None

    def summary(self) -> str:
        parts = []
        if self.mother:
            parts.append(f"Mother: {self.mother}")
        if self.father:
            parts.append(f"Father: {self.father}")
        return "; ".join(parts)


class FamilyHistory(BaseModel):
    familyDynamicsDuringDevelopment: Optional[str] = None
    familyPsychiatricIllnesses: Optional[str] = None
    currentRelationshipWithParents: Optional[CurrentRelationshipWithParents] = None
    childhoodExperiences: Optional[str] = None
    significantDevelopmentalExperiences: Optional[str] = None

    def to_prompt(self) -> Optional[str]:
        parts = []
        if self.familyDynamicsDuringDevelopment:
            parts.append(f"Family environment: {self.familyDynamicsDuringDevelopment}")
        if self.currentRelationshipWithParents:
            relation = self.currentRelationshipWithParents.summary()
            if relation:
                parts.append(f"Current parents relationship: {relation}")
        if self.childhoodExperiences:
            parts.append(f"Childhood experiences: {self.childhoodExperiences}")
        if self.significantDevelopmentalExperiences:
            parts.append(f"Key developmental events: {self.significantDevelopmentalExperiences}")
        if not parts:
            return None
        return " ".join(parts)


class EducationAndEmployment(BaseModel):
    educationLevel: Optional[str] = None
    workHistory: Optional[str] = None
    housingStability: Optional[str] = None
    financialSituation: Optional[str] = None
    hobbiesAndInterests: Optional[str] = None

    def to_prompt(self) -> Optional[str]:
        pieces = []
        if self.educationLevel:
            pieces.append(f"Education: {self.educationLevel}")
        if self.workHistory:
            pieces.append(f"Work: {self.workHistory}")
        if self.housingStability:
            pieces.append(f"Housing: {self.housingStability}")
        if self.financialSituation:
            pieces.append(f"Finances: {self.financialSituation}")
        if self.hobbiesAndInterests:
            pieces.append(f"Hobbies: {self.hobbiesAndInterests}")
        return " ".join(pieces) if pieces else None


class SocialRelationshipsAndInteractions(BaseModel):
    friendships: Optional[str] = None
    romanticRelationships: Optional[str] = None
    sexualRelationships: Optional[str] = None
    familyInteractions: Optional[str] = None
    relationshipsWithPeersAndColleagues: Optional[str] = None
    socialMediaUseAndImpact: Optional[str] = None

    def to_prompt(self) -> Optional[str]:
        parts = []
        if self.friendships:
            parts.append(f"Friendships: {self.friendships}")
        if self.romanticRelationships:
            parts.append(f"Romantic relationships: {self.romanticRelationships}")
        if self.sexualRelationships:
            parts.append(f"Sexual relationships: {self.sexualRelationships}")
        if self.familyInteractions:
            parts.append(f"Family interactions: {self.familyInteractions}")
        if self.relationshipsWithPeersAndColleagues:
            parts.append(f"Peers/colleagues: {self.relationshipsWithPeersAndColleagues}")
        if not parts:
            return None
        return " ".join(parts)


class TreatmentsAndInterventions(BaseModel):
    previousTherapeuticExperiences: Optional[Union[str, List[str]]] = None
    treatmentResistance: Optional[str] = None
    medicationHistory: List[str] = Field(default_factory=list)
    responseToMedications: Optional[str] = None
    previousHospitalizations: Optional[Union[str, int]] = None
    emergencyDepartmentVisits: Optional[str] = None
    previousDropouts: Optional[str] = None
    previousPsychiatricDiagnoses: List[str] = Field(default_factory=list)

    def to_prompt(self) -> Optional[str]:
        items = []
        if self.previousTherapeuticExperiences:
            prev = (
                "; ".join(self.previousTherapeuticExperiences)
                if isinstance(self.previousTherapeuticExperiences, list)
                else self.previousTherapeuticExperiences
            )
            items.append(f"Therapy history: {prev}")
        if self.treatmentResistance:
            items.append(f"Engagement/resistance: {self.treatmentResistance}")
        if self.medicationHistory:
            items.append(f"Medications tried: {', '.join(self.medicationHistory)}")
        if self.responseToMedications:
            items.append(f"Medication response: {self.responseToMedications}")
        if self.previousHospitalizations:
            items.append(f"Hospitalizations: {self.previousHospitalizations}")
        if self.emergencyDepartmentVisits:
            items.append(f"ER visits: {self.emergencyDepartmentVisits}")
        if self.previousPsychiatricDiagnoses:
            items.append(f"Previous diagnoses: {', '.join(self.previousPsychiatricDiagnoses)}")
        return " ".join(items) if items else None


class MedicalAndPhysicalHistory(BaseModel):
    preExistingMedicalConditions: Optional[str] = None
    pharmacologicalTreatments: Optional[Union[str, List[str]]] = None
    lifestyle: Optional[str] = None
    generalPhysicalHealth: Optional[str] = None
    sleepPatterns: Optional[str] = None
    eatingHabits: Optional[str] = None

    def to_prompt(self) -> Optional[str]:
        parts = []
        if self.preExistingMedicalConditions:
            parts.append(f"Medical conditions: {self.preExistingMedicalConditions}")
        if self.lifestyle:
            parts.append(f"Lifestyle: {self.lifestyle}")
        if self.generalPhysicalHealth:
            parts.append(f"Health: {self.generalPhysicalHealth}")
        if self.sleepPatterns:
            parts.append(f"Sleep: {self.sleepPatterns}")
        if self.eatingHabits:
            parts.append(f"Eating: {self.eatingHabits}")
        return " ".join(parts) if parts else None


class BehaviorDuringTestAdministration(BaseModel):
    recurringDynamicsTransferenceCountertransference: Optional[str] = None
    expressedEmotionsAndCongruence: Optional[Union[str, List[str]]] = None
    speechCharacteristics: Optional[str] = None
    nonVerbalBehavior: Optional[str] = None
    appearanceSelfCareOrientation: Optional[str] = None

    def to_prompt(self) -> Optional[str]:
        parts = []
        if self.recurringDynamicsTransferenceCountertransference:
            parts.append(self.recurringDynamicsTransferenceCountertransference)
        if self.expressedEmotionsAndCongruence:
            emotions = (
                ", ".join(self.expressedEmotionsAndCongruence)
                if isinstance(self.expressedEmotionsAndCongruence, list)
                else self.expressedEmotionsAndCongruence
            )
            parts.append(f"Observed affect: {emotions}")
        if self.speechCharacteristics:
            parts.append(f"Speech: {self.speechCharacteristics}")
        if self.nonVerbalBehavior:
            parts.append(f"Non-verbal: {self.nonVerbalBehavior}")
        if self.appearanceSelfCareOrientation:
            parts.append(f"Appearance/self-care: {self.appearanceSelfCareOrientation}")
        return " ".join(parts) if parts else None


class ImpairmentDetail(BaseModel):
    impairment: Optional[str] = None
    impairmentLevel: Optional[str] = Field(default=None, alias="level")
    description: Optional[str] = None

    class Config:
        allow_population_by_field_name = True
        extra = "ignore"
    
    def brief(self) -> Optional[str]:
        if self.description and self.impairmentLevel:
            return f"{self.description} (severity: {self.impairmentLevel})"
        if self.description:
            return self.description
        if self.impairmentLevel:
            return f"Severity: {self.impairmentLevel}"
        return None


class OverallPersonalityOrganization(BaseModel):
    organization: Optional[str] = None
    severityRange: Optional[str] = None
    description: Optional[str] = None

    def brief(self) -> Optional[str]:
        parts = []
        if self.organization:
            parts.append(self.organization)
        if self.severityRange:
            parts.append(f"severity {self.severityRange}")
        if self.description:
            parts.append(self.description)

        return ": ".join(parts) if parts else None


class PersonalityAndSymptomAxis(BaseModel):
    identity: ImpairmentDetail = Field(default_factory=ImpairmentDetail)
    objectRelations: ImpairmentDetail = Field(default_factory=ImpairmentDetail)
    defensiveLevel: ImpairmentDetail = Field(default_factory=ImpairmentDetail)
    realityTesting: ImpairmentDetail = Field(default_factory=ImpairmentDetail)
    overallPersonalityOrganization: OverallPersonalityOrganization = Field(default_factory=OverallPersonalityOrganization)
    personalitySyndrome: Optional[str] = None
    symptomPatterns: Dict[str, str] = Field(default_factory=dict)
    comorbidity: Optional[str] = None

    def to_prompt(self) -> Optional[str]:
        chunks = []
        if self.identity.brief():
            chunks.append(f"Identity: {self.identity.brief()}")
        if self.objectRelations.brief():
            chunks.append(f"Relationships: {self.objectRelations.brief()}")
        if self.defensiveLevel.brief():
            chunks.append(f"Defenses: {self.defensiveLevel.brief()}")
        if self.realityTesting.brief():
            chunks.append(f"Reality testing: {self.realityTesting.brief()}")
        if self.overallPersonalityOrganization.brief():
            chunks.append(f"Personality organization: {self.overallPersonalityOrganization.brief()}")
        return " ".join(chunks) if chunks else None


class MentalFunctioningAxis(BaseModel):
    affectExperienceAndRegulation: ImpairmentDetail = Field(default_factory=ImpairmentDetail)
    identityIntegration: ImpairmentDetail = Field(default_factory=ImpairmentDetail)
    selfEsteemRegulation: ImpairmentDetail = Field(default_factory=ImpairmentDetail)
    attentionAndLearning: ImpairmentDetail = Field(default_factory=ImpairmentDetail)
    defensiveFunctioning: ImpairmentDetail = Field(default_factory=ImpairmentDetail)
    impulseControl: ImpairmentDetail = Field(default_factory=ImpairmentDetail)
    moralStandardsAndIdeals: ImpairmentDetail = Field(default_factory=ImpairmentDetail)
    relationshipsAndIntimacy: ImpairmentDetail = Field(default_factory=ImpairmentDetail)
    mentalization: ImpairmentDetail = Field(default_factory=ImpairmentDetail)
    selfObservation: ImpairmentDetail = Field(default_factory=ImpairmentDetail)
    adaptationAndResilience: ImpairmentDetail = Field(default_factory=ImpairmentDetail)
    meaningAndDirectionality: ImpairmentDetail = Field(default_factory=ImpairmentDetail)

    def to_prompt(self) -> Optional[str]:
        parts = []
        if self.affectExperienceAndRegulation.brief():
            parts.append(f"Affect regulation: {self.affectExperienceAndRegulation.brief()}")
        if self.identityIntegration.brief():
            parts.append(f"Identity integration: {self.identityIntegration.brief()}")
        if self.selfEsteemRegulation.brief():
            parts.append(f"Self-esteem: {self.selfEsteemRegulation.brief()}")
        if self.mentalization.brief():
            parts.append(f"Mentalization: {self.mentalization.brief()}")
        if self.impulseControl.brief():
            parts.append(f"Impulse control: {self.impulseControl.brief()}")
        if self.meaningAndDirectionality.brief():
            parts.append(f"Meaning/direction: {self.meaningAndDirectionality.brief()}")
        return " ".join(parts) if parts else None


class ClinicalFunctioning(BaseModel):
    personalityAndSymptomAxis: PersonalityAndSymptomAxis = Field(default_factory=PersonalityAndSymptomAxis)
    mentalFunctioningAxis: MentalFunctioningAxis = Field(default_factory=MentalFunctioningAxis)

    def to_prompt(self) -> Optional[str]:
        pieces = []
        psa = self.personalityAndSymptomAxis.to_prompt()
        if psa:
            pieces.append(psa)
        mfa = self.mentalFunctioningAxis.to_prompt()
        if mfa:
            pieces.append(mfa)
        return " ".join(pieces) if pieces else None


class PatientDetails(BaseModel):
    demographicAndSocioculturalInformation: DemographicAndSocioculturalInformation = Field(
        default_factory=DemographicAndSocioculturalInformation
    )
    familyHistory: FamilyHistory = Field(default_factory=FamilyHistory)
    educationAndEmployment: EducationAndEmployment = Field(default_factory=EducationAndEmployment)
    socialRelationshipsAndInteractions: SocialRelationshipsAndInteractions = Field(default_factory=SocialRelationshipsAndInteractions)
    treatmentsAndInterventions: TreatmentsAndInterventions = Field(default_factory=TreatmentsAndInterventions)
    medicalAndPhysicalHistory: MedicalAndPhysicalHistory = Field(default_factory=MedicalAndPhysicalHistory)
    behaviorDuringTestAdministration: BehaviorDuringTestAdministration = Field(default_factory=BehaviorDuringTestAdministration)
    clinicalFunctioning: ClinicalFunctioning = Field(default_factory=ClinicalFunctioning)
    disorder: Optional[dict] = None


# === Main Patient Profile ===
class PatientProfile(BaseModel):
    patient_id: str = Field(..., alias="patientId")
    name: str
    brief_description: Optional[str] = Field(None, alias="briefDescription")
    avatar_url: Optional[str] = Field(None, alias="avatarUrl")
    voiceId: Optional[str] = None
    welcomeMessage: Optional[str] = None
    objectives: List[str] = Field(default_factory=list)
    difficulty: Optional[int] = None
    estimated_duration: Optional[int] = Field(None, alias="estimatedDuration")
    emotionTraits: EmotionTraits = Field(default_factory=EmotionTraits)
    details: PatientDetails = Field(default_factory=PatientDetails)
    clinicalCase: Optional[str] = None

    # Runtime fields
    current_emotional_state: str = "base"
    emotion_state: Dict[str, float] = Field(default_factory=dict)
    session_notes: Optional[str] = None
    emotion_intensity: float = 0.6

    @classmethod
    def from_file(cls, path: str) -> "PatientProfile":
        """Load a patient JSON file that follows the new schema."""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        prepared = _prepare_payload(raw, path)
        return cls(**prepared)

    def to_text_summary(self) -> str:
        """Compact description used for LLM greetings or summaries."""
        demo = self.details.demographicAndSocioculturalInformation
        name = self.name or self.patient_id.replace("_", " ").title()
        age_gender = []
        if demo.age:
            age_gender.append(f"{demo.age}-year-old")
        if demo.gender:
            age_gender.append(demo.gender.lower())
        background = demo.culturalBackground or ""
        descriptor = " ".join(age_gender).strip() or "patient"
        overview = self.brief_description or self.clinicalCase or ""
        background_suffix = f" ({background})" if background else ""
        return f"{name}, {descriptor}{background_suffix}. {overview}".strip()

    class Config:
        allow_population_by_field_name = True
        extra = "ignore"


def _prepare_payload(raw: dict, path: str) -> dict:
    """Normalize raw JSON into the structure expected by PatientProfile."""
    payload = dict(raw)

    if "patientId" not in payload:
        payload["patientId"] = payload.get("patient_id") or Path(path).stem

    if "name" not in payload or not payload["name"]:
        payload["name"] = payload["patientId"].replace("_", " ").title()

    payload["briefDescription"] = (
        payload.get("briefDescription")
        or payload.get("brief_description")
        or ""
    )

    payload["avatarUrl"] = payload.get("avatarUrl") or payload.get("avatar_url")
    payload["estimatedDuration"] = payload.get("estimatedDuration") or payload.get("estimated_duration")
    payload["emotionTraits"] = _build_emotion_traits(payload)

    # ---- DETAILS NORMALIZATION ----
    payload.setdefault("details", {})
    payload["details"].setdefault("demographicAndSocioculturalInformation", {})

    # 👇 THIS IS THE IMPORTANT LINE
    payload["details"]["demographicAndSocioculturalInformation"].setdefault(
        "name", payload["name"]
    )

    return payload


def _build_emotion_traits(raw: dict) -> dict:
    """Normalize any provided emotion trait metadata into the expected structure."""
    container = (
        raw.get("emotionTraits")
        or raw.get("EmotionDynamics")
        or raw.get("emotion_traits")
        or {}
    )
    baseline = container.get("trait_baseline") or raw.get("trait_baseline") or {}
    volatility = (
        container.get("volatility_level")
        or container.get("volatilityLevel")
        or raw.get("volatility_level")
        or raw.get("volatility")
        or "medium"
    )

    normalized = {}
    for key in EMOTIONS:
        raw_value = (
            baseline.get(key)
            or baseline.get(key.lower())
            or baseline.get(key.capitalize())
            or 0.5
        )
        normalized[key] = _clamp_emotion_value(raw_value)
    return {"trait_baseline": normalized, "volatility_level": str(volatility).lower()}


def _clamp_emotion_value(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, numeric))
