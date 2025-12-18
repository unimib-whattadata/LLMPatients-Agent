"""Structured patient profile models for the updated patient JSON schema."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class EmotionDynamics(BaseModel):
    """Traits controlling affective baseline and volatility."""

    trait_baseline: Dict[str, float] = Field(default_factory=dict)
    volatility_level: str = "medium"


class PatientMetadata(BaseModel):
    """Fields primarily consumed by the UI/simulation layers (voice, avatar, etc.)."""

    background: Optional[str] = None
    therapy_goals: List[str] = Field(default_factory=list)
    difficulty: Optional[int] = None
    estimatedDuration: Optional[int] = None
    avatarUrl: Optional[str] = None
    voiceId: Optional[str] = None
    welcomeMessage: Optional[str] = None


class Demographics(BaseModel):
    """Basic demographic snapshot used for prompts and UI."""

    age: Optional[int] = None
    gender: Optional[str] = None
    marital_status: Optional[str] = None
    cultural_background: Optional[str] = None
    religious_beliefs: Optional[str] = None
    spoken_language: Optional[str] = None
    migration_status: Optional[str] = None


class FamilyHistory(BaseModel):
    """Developmental and relational history of the patient."""

    family_dynamics_during_development: Optional[str] = None
    family_psychiatric_illnesses: Optional[str] = None
    current_relationship_with_parents: Optional[str] = None
    childhood_experiences: Optional[str] = None
    significant_developmental_experiences: Optional[str] = None


class EducationEmployment(BaseModel):
    """Educational background plus work/life stability indicators."""

    education_level: Optional[str] = None
    work_history: Optional[str] = None
    housing_stability: Optional[str] = None
    financial_situation: Optional[str] = None
    hobbies_and_interests: Optional[str] = None


class TreatmentsInterventions(BaseModel):
    """Medication/therapy history plus adherence and treatment goals."""

    previous_therapeutic_experiences: Optional[str] = None
    treatment_resistance: Optional[str] = None
    therapeutic_goals: List[str] = Field(default_factory=list)
    medication_history: List[str] = Field(default_factory=list)
    response_to_medications: Optional[str] = None
    previous_hospitalizations: Optional[str] = None
    emergency_department_visits: Optional[str] = None
    previous_dropouts: Optional[str] = None
    previous_psychiatric_diagnoses: List[str] = Field(default_factory=list)


class MedicalPhysicalHistory(BaseModel):
    """Physical health context relevant to mental health treatment."""

    pre_existing_medical_conditions: Optional[str] = None
    pharmacological_treatments: Optional[str] = None
    lifestyle: Optional[str] = None
    general_physical_health: Optional[str] = None
    eating_habits: Optional[str] = None


class BehaviorObservations(BaseModel):
    """Observations captured during assessments (speech, affect, posture)."""

    recurring_dynamics: Optional[str] = None
    expressed_emotions_and_congruence: Optional[str] = None
    speech_characteristics: Optional[str] = None
    non_verbal_behavior: Optional[str] = None
    appearance_self_care_orientation: Optional[str] = None


class PersonalityAndSymptomAxis(BaseModel):
    """Personality organization, defenses, and symptom patterns."""

    identity: Dict[str, str] = Field(default_factory=dict)
    object_relations: Dict[str, str] = Field(default_factory=dict)
    defensive_level: Dict[str, str] = Field(default_factory=dict)
    reality_testing: Dict[str, str] = Field(default_factory=dict)
    overall_personality_organization: Dict[str, str] = Field(default_factory=dict)
    personality_syndrome: Optional[str] = None
    symptom_patterns: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    comorbidity: Optional[str] = None


class MentalFunctioningAxis(BaseModel):
    """Mental functioning capacities (identity integration, regulation, etc.)."""

    affect_experience_and_regulation: Dict[str, str] = Field(default_factory=dict)
    identity_integration: Dict[str, str] = Field(default_factory=dict)
    self_esteem_regulation: Dict[str, str] = Field(default_factory=dict)
    attention_and_learning: Dict[str, str] = Field(default_factory=dict)
    defensive_functioning: Dict[str, str] = Field(default_factory=dict)
    impulse_control: Dict[str, str] = Field(default_factory=dict)
    moral_standards_and_ideals: Dict[str, str] = Field(default_factory=dict)
    relationships_and_intimacy: Dict[str, str] = Field(default_factory=dict)
    mentalization: Dict[str, str] = Field(default_factory=dict)
    self_observation: Dict[str, str] = Field(default_factory=dict)
    adaptation_and_resilience: Dict[str, str] = Field(default_factory=dict)
    meaning_and_directionality: Dict[str, str] = Field(default_factory=dict)


class ClinicalFunctioning(BaseModel):
    """Aggregated clinical functioning axes."""

    personality_and_symptom_axis: PersonalityAndSymptomAxis = Field(
        default_factory=PersonalityAndSymptomAxis
    )
    mental_functioning_axis: MentalFunctioningAxis = Field(
        default_factory=MentalFunctioningAxis
    )


class PatientProfile(BaseModel):
    """Aggregated patient record used to condition the simulated agent."""

    patient_id: str
    name: str
    disorder: str
    Metadata: Optional[PatientMetadata] = None

    Demographics: Demographics
    FamilyHistory: FamilyHistory
    EducationEmployment: EducationEmployment
    TreatmentsInterventions: TreatmentsInterventions
    MedicalHistory: MedicalPhysicalHistory
    BehaviorObservations: BehaviorObservations
    ClinicalFunctioning: ClinicalFunctioning
    EmotionDynamics: EmotionDynamics

    # === Clinical and dynamic fields ===
    ClinicalSummary: Optional[str] = None
    brief_description: Optional[str] = None
    current_emotional_state: Optional[str] = "base"
    session_notes: Optional[str] = None
    emotion_state: Dict[str, float] = Field(default_factory=dict)

    @property
    def short_name(self) -> str:
        """Return formatted patient name."""
        if self.name:
            return self.name
        return self.patient_id.replace("_", " ").title()

    @classmethod
    def from_file(cls, path: str) -> "PatientProfile":
        """Load the updated patient schema from disk and normalize it."""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        converted = cls._convert_from_new_schema(raw, path)
        return cls(**converted)

    @staticmethod
    def _convert_from_new_schema(raw: dict, path: str) -> dict:
        """Map the current schema (details + clinicalFunctioning) into models."""
        details = raw.get("details", {})
        patient_id = raw.get("patientId") or raw.get("patient_id") or Path(path).stem
        name = raw.get("name") or patient_id.replace("_", " ").title()

        demographics = details.get("demographicAndSocioculturalInformation", {})
        family = details.get("familyHistory", {})
        education = details.get("educationAndEmployment", {})
        treatments = details.get("treatmentsAndInterventions", {})
        medical = details.get("medicalAndPhysicalHistory", {})
        behavior = details.get("behaviorDuringTestAdministration", {})

        metadata = PatientMetadata(
            background=_clean_text(raw.get("briefDescription")),
            therapy_goals=_ensure_list(raw.get("objectives")),
            difficulty=raw.get("difficulty"),
            estimatedDuration=raw.get("estimatedDuration"),
            avatarUrl=raw.get("avatarUrl"),
            voiceId=raw.get("voiceId"),
            welcomeMessage=raw.get("welcomeMessage"),
        )

        profile = {
            "patient_id": patient_id,
            "name": name,
            "disorder": _clean_text(details.get("disorder", {}).get("disorderName"), default="Unspecified disorder"),
            "Metadata": metadata,
            "Demographics": {
                "age": _safe_int(demographics.get("age")),
                "gender": _clean_text(demographics.get("gender")),
                "marital_status": _clean_text(demographics.get("maritalStatus")),
                "cultural_background": _clean_text(demographics.get("culturalBackground")),
                "religious_beliefs": _clean_text(demographics.get("religiousBeliefs")),
                "spoken_language": _clean_text(demographics.get("spokenLanguage")),
                "migration_status": _clean_text(demographics.get("migrationStatus")),
            },
            "FamilyHistory": {
                "family_dynamics_during_development": _clean_text(family.get("familyDynamicsDuringDevelopment")),
                "family_psychiatric_illnesses": _clean_text(family.get("familyPsychiatricIllnesses")),
                "current_relationship_with_parents": _clean_text(family.get("currentRelationshipWithParents")),
                "childhood_experiences": _clean_text(family.get("childhoodExperiences")),
                "significant_developmental_experiences": _clean_text(family.get("significantDevelopmentalExperiences")),
            },
            "EducationEmployment": {
                "education_level": _clean_text(education.get("educationLevel")),
                "work_history": _clean_text(education.get("workHistory")),
                "housing_stability": _clean_text(education.get("housingStability")),
                "financial_situation": _clean_text(education.get("financialSituation")),
                "hobbies_and_interests": _clean_text(education.get("hobbiesAndInterests")),
            },
            "TreatmentsInterventions": {
                "previous_therapeutic_experiences": _clean_text(treatments.get("previousTherapeuticExperiences")),
                "treatment_resistance": _clean_text(treatments.get("treatmentResistance")),
                "therapeutic_goals": _ensure_list(treatments.get("therapeuticGoals")),
                "medication_history": _ensure_list(treatments.get("medicationHistory")),
                "response_to_medications": _clean_text(treatments.get("responseToMedications")),
                "previous_hospitalizations": _clean_text(treatments.get("previousHospitalizations")),
                "emergency_department_visits": _clean_text(treatments.get("emergencyDepartmentVisits")),
                "previous_dropouts": _clean_text(treatments.get("previousDropouts")),
                "previous_psychiatric_diagnoses": _ensure_list(treatments.get("previousPsychiatricDiagnoses")),
            },
            "MedicalHistory": {
                "pre_existing_medical_conditions": _clean_text(medical.get("preExistingMedicalConditions")),
                "pharmacological_treatments": _clean_text(medical.get("pharmacologicalTreatments")),
                "lifestyle": _clean_text(medical.get("lifestyle")),
                "general_physical_health": _clean_text(medical.get("generalPhysicalHealth")),
                "eating_habits": _clean_text(medical.get("eatingHabits")),
            },
            "BehaviorObservations": {
                "recurring_dynamics": _clean_text(behavior.get("recurringDynamicsTransferenceCountertransference")),
                "expressed_emotions_and_congruence": _clean_text(behavior.get("expressedEmotionsAndCongruence")),
                "speech_characteristics": _clean_text(behavior.get("speechCharacteristics")),
                "non_verbal_behavior": _clean_text(behavior.get("nonVerbalBehavior")),
                "appearance_self_care_orientation": _clean_text(behavior.get("appearanceSelfCareOrientation")),
            },
            "ClinicalFunctioning": _normalize_clinical_functioning(details.get("clinicalFunctioning", {})),
            "EmotionDynamics": _build_emotion_traits(raw),
            "ClinicalSummary": raw.get("clinicalCase", ""),
            "brief_description": _clean_text(raw.get("briefDescription"), default=""),
            "current_emotional_state": "base",
            "session_notes": None,
        }
        return profile

    def to_text_summary(self) -> str:
        """Compact summary for LLM or prompt context."""
        demo = self.Demographics
        age = demo.age or "unknown age"
        gender = demo.gender.lower() if demo.gender else "person"
        description = self.brief_description or (self.Metadata.background if self.Metadata else "")
        descriptor = description or "No brief description provided."
        return (
            f"{self.short_name}, {age}-year-old {gender}. "
            f"Disorder: {self.disorder}. {descriptor}"
        )

    class Config:
        extra = "ignore"  # Ignore unexpected fields when loading from JSON


EMOTION_KEYS = ["SEEKING", "RAGE", "FEAR", "CARE", "LUST", "SADNESS", "PLAY"]


def _build_emotion_traits(raw: dict) -> dict:
    """Normalize any provided emotion trait metadata into the expected structure."""
    container = raw.get("emotionTraits") or raw.get("emotion_traits") or {}
    baseline = container.get("trait_baseline") or raw.get("trait_baseline") or {}
    volatility = container.get("volatility_level") or raw.get("volatility_level") or "medium"

    normalized = {}
    for key in EMOTION_KEYS:
        raw_value = (
            baseline.get(key)
            or baseline.get(key.lower())
            or baseline.get(key.capitalize())
            or 0.5
        )
        normalized[key] = _clamp_emotion_value(raw_value)
    return {"trait_baseline": normalized, "volatility_level": volatility}


def _normalize_clinical_functioning(raw: dict) -> dict:
    """Normalize the clinicalFunctioning block into our models."""
    personality = raw.get("personalityAndSymptomAxis", {}) if isinstance(raw, dict) else {}
    mental = raw.get("mentalFunctioningAxis", {}) if isinstance(raw, dict) else {}

    def _level_block(data: Optional[dict]) -> Dict[str, str]:
        if not isinstance(data, dict):
            return {}
        return _clean_dict({
            "level": _clean_text(data.get("impairmentLevel") or data.get("level")),
            "description": _clean_text(data.get("description")),
        })

    def _impairment_block(data: Optional[dict]) -> Dict[str, str]:
        if not isinstance(data, dict):
            return {}
        return _clean_dict({
            "impairment": _clean_text(data.get("impairment")),
            "description": _clean_text(data.get("description")),
        })

    def _symptom_patterns(data: Optional[dict]) -> Dict[str, Dict[str, str]]:
        if not isinstance(data, dict):
            return {}
        cleaned = {}
        for key, value in data.items():
            if isinstance(value, dict):
                cleaned[key] = _clean_dict({k: _clean_text(v) for k, v in value.items()})
        return cleaned

    return {
        "personality_and_symptom_axis": {
            "identity": _level_block(personality.get("identity")),
            "object_relations": _level_block(personality.get("objectRelations")),
            "defensive_level": _level_block(personality.get("defensiveLevel")),
            "reality_testing": _level_block(personality.get("realityTesting")),
            "overall_personality_organization": _clean_dict({
                "organization": _clean_text(personality.get("overallPersonalityOrganization", {}).get("organization")),
                "severity_range": _clean_text(personality.get("overallPersonalityOrganization", {}).get("severityRange")),
                "description": _clean_text(personality.get("overallPersonalityOrganization", {}).get("description")),
            }),
            "personality_syndrome": _clean_text(personality.get("personalitySyndrome")),
            "symptom_patterns": _symptom_patterns(personality.get("symptomPatterns")),
            "comorbidity": _clean_text(personality.get("comorbidity")),
        },
        "mental_functioning_axis": {
            "affect_experience_and_regulation": _impairment_block(mental.get("affectExperienceAndRegulation")),
            "identity_integration": _impairment_block(mental.get("identityIntegration")),
            "self_esteem_regulation": _impairment_block(mental.get("selfEsteemRegulation")),
            "attention_and_learning": _impairment_block(mental.get("attentionAndLearning")),
            "defensive_functioning": _impairment_block(mental.get("defensiveFunctioning")),
            "impulse_control": _impairment_block(mental.get("impulseControl")),
            "moral_standards_and_ideals": _impairment_block(mental.get("moralStandardsAndIdeals")),
            "relationships_and_intimacy": _impairment_block(mental.get("relationshipsAndIntimacy")),
            "mentalization": _impairment_block(mental.get("mentalization")),
            "self_observation": _impairment_block(mental.get("selfObservation")),
            "adaptation_and_resilience": _impairment_block(mental.get("adaptationAndResilience")),
            "meaning_and_directionality": _impairment_block(mental.get("meaningAndDirectionality")),
        },
    }


def _clamp_emotion_value(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, numeric))


def _clean_text(value, default: Optional[str] = None):
    """Normalize various json fields into trimmed strings with sensible fallbacks."""
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        return text
    return str(value)


def _clean_dict(payload: Dict[str, Optional[str]]) -> Dict[str, str]:
    return {k: v for k, v in payload.items() if v is not None and str(v).strip()}


def _ensure_list(value) -> List[str]:
    """Coerce string/list inputs into a cleaned list that omits empty markers."""
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return [item for item in cleaned if item.lower() not in {"none", "not reported", "n/a"}]
    if isinstance(value, str):
        cleaned = [
            segment.strip()
            for segment in re.split(r"[;,]", value)
            if segment.strip()
        ]
        return [item for item in cleaned if item.lower() not in {"none", "not reported", "n/a"}]
    if value is None:
        return []
    return [str(value)]


def _safe_int(value, default: Optional[int] = None) -> Optional[int]:
    """Extract an integer from loosely formatted sources (words, strings, etc.)."""
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        digits = re.findall(r"-?\d+", value)
        if digits:
            try:
                return int(digits[0])
            except ValueError:
                return default
        lowered = value.strip().lower()
        if lowered in {"none", "no", "not reported", "n/a", "zero"}:
            return 0
        word_numbers = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
        }
        for word, number in word_numbers.items():
            if word in lowered:
                return number
    return default
