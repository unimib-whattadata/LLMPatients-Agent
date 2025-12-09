"""Structured patient profile models plus helpers for loading legacy JSON schemas."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


# === Subcomponents ===
class DemographicInfo(BaseModel):
    """Basic demographic snapshot used for prompts and UI."""
    Name: str
    Surname: str
    Age: int
    Gender: str
    MaritalStatus: str
    CulturalBackground: str
    ReligiousBeliefs: str
    SpokenLanguage: str
    MigrationStatus: str

class FamilySocialHistory(BaseModel):
    """Developmental and relational history of the patient."""
    ChildhoodFamilyDynamics: str
    CurrentParentRelations: str
    ChildhoodExperiences: str
    SignificantDevelopmentalExperiences: Optional[str] = None
    FamilyPsychiatricHistory: Optional[str] = None
    AbuseHistory: Optional[str] = None
    SocialSupport: Optional[str] = None
    SupportNetwork: Optional[str] = None

class EducationOccupation(BaseModel):
    """Educational background plus work/life stability indicators."""
    EducationLevel: str
    WorkHistory: str
    HousingStability: str
    FinancialSituation: str
    HobbiesInterests: str

class PsychologicalProfile(BaseModel):
    """Clinical symptoms, diagnoses, and cognitive functioning markers."""
    PsychiatricDiagnoses: List[str]
    MainSymptoms: List[str]
    AffectiveEmotionalFunctioning: Optional[str] = None
    PsychiatricComorbidities: Optional[str] = None
    SenseOfSelfOthers: Optional[str] = None
    ThoughtCognitiveStyle: Optional[str] = None
    EmotionalReactions: Optional[str] = None
    Aggressiveness: Optional[str] = None
    SelfPerceptionIdentity: Optional[str] = None
    SelfEsteem: Optional[str] = None
    CognitiveStyle: Optional[List[str]] = None
    AttachmentStyle: Optional[str] = None
    ExecutiveFunctioning: Optional[str] = None
    Memory: Optional[str] = None
    AttentionConcentration: Optional[str] = None
    SensoryPerception: Optional[str] = None
    HigherCognitiveFunctions: Optional[str] = None

class CopingDefenses(BaseModel):
    """Summaries of coping strategies, defenses, and risk behaviors."""
    CopingStrategies: str
    DefenseMechanisms: List[str]
    SelfHarmSuicidality: str
    SubstanceAbuse: str
    Avoidance: Optional[str] = None
    ImpulsiveRiskBehaviors: Optional[str] = None
    Morality: str

class SocialRelations(BaseModel):
    """Descriptions of interpersonal dynamics across key relationship categories."""
    Friendships: str
    RomanticRelationships: str
    SexualRelationships: str
    FamilyInteractions: str
    PeerColleagueRelations: str
    SocialMediaBehavior: str

class TreatmentsInterventions(BaseModel):
    """Medication/therapy history plus adherence and treatment goals."""
    PastTherapies: List[str]
    ProgressResistance: str
    TherapeuticGoals: Union[str, List[str]]
    MedicationHistory: List[str]
    MedicationResponse: str
    Hospitalizations: int
    EmergencyRoomVisits: str
    PastPsychiatricDiagnoses: List[str]
    PreviousDropouts: Optional[str] = None

class ClinicalJudgment(BaseModel):
    """Clinician-rated insight, judgment, and impulse control."""
    JudgmentCapacity: str
    Insight: str
    ImpulseControl: str

class ResilienceWellbeing(BaseModel):
    """Protective factors and measures of psychological wellbeing."""
    PsychologicalResilience: str
    SelfCompassion: str
    LifeSatisfaction: str
    PersonalGrowth: str
    SenseOfCoherence: str
    Empowerment: str
    HopeForFuture: str
    LongTermLifeGoals: str

class MedicalHistory(BaseModel):
    """Physical health context relevant to mental health treatment."""
    PreexistingConditions: str
    CurrentMedications: List[str]
    PharmacologicalTreatments: Optional[List[str]] = None
    Allergies: Optional[str] = None
    Lifestyle: str
    GeneralHealth: str
    SleepPatterns: str
    EatingHabits: str

class SocialEnvironment(BaseModel):
    """Environmental determinants such as housing and support networks."""
    SocialSupport: str
    HousingConditions: str
    SocialSecurity: str
    SocialCohesion: str
    SocialDeterminantsMentalHealth: str

class TestBehavior(BaseModel):
    """Observations captured during assessments (speech, affect, posture)."""
    RecurringDynamics: str
    PredominantEmotions: List[str]
    Speech: str
    EyeContactPostureGestures: str
    AvoidantAttitudesMoodChange: str


class EmotionDynamics(BaseModel):
    """Traits controlling affective baseline and volatility."""
    trait_baseline: Dict[str, float] = Field(default_factory=dict)
    volatility_level: str = "medium"

# === Metadata for UI/Simulation Layer ===
class PatientMetadata(BaseModel):
    """Fields primarily consumed by the UI/simulation layers (voice, avatar, etc.)."""
    background: str
    therapy_goals: List[str]
    difficulty: int
    estimatedDuration: int
    avatarUrl: str
    voiceId: str
    welcomeMessage: str

# === Main Patient Profile ===
class PatientProfile(BaseModel):
    """Aggregated patient record used to condition the simulated agent."""
    patient_id: str
    disorder_id: Optional[str] = None
    Metadata: Optional[PatientMetadata] = None

    DemographicInfo: DemographicInfo
    FamilySocialHistory: FamilySocialHistory
    EducationOccupation: EducationOccupation
    PsychologicalProfile: PsychologicalProfile
    CopingDefenses: CopingDefenses
    SocialRelations: SocialRelations
    TreatmentsInterventions: TreatmentsInterventions
    ClinicalJudgment: ClinicalJudgment
    ResilienceWellbeing: ResilienceWellbeing
    MedicalHistory: MedicalHistory
    SocialEnvironment: SocialEnvironment
    TestBehavior: TestBehavior
    EmotionDynamics: EmotionDynamics

    # === Clinical and dynamic fields ===
    ClinicalSummary: Optional[str] = None
    current_emotional_state: Optional[str] = "base"
    session_notes: Optional[str] = None
    emotion_state: Dict[str, float] = Field(default_factory=dict)
    

    @property
    def name(self) -> str:
        """Return formatted patient name."""
        demo = getattr(self, "DemographicInfo", None)
        if demo:
            return f"{demo.Name} {demo.Surname}"
        return self.patient_id.replace("_", " ").title()

    @classmethod
    def from_file(cls, path: str) -> "PatientProfile":
        """Load legacy or new attribute-style patient files."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "details" in data:
            converted = cls._convert_from_attribute_schema(data, path)
            return cls(**converted)

        if "EmotionDynamics" not in data:
            data["EmotionDynamics"] = _build_emotion_traits(data)
        return cls(**data)

    @staticmethod
    def _convert_from_attribute_schema(raw: dict, path: str) -> dict:
        """Map the new attribute schema into the legacy PatientProfile structure."""
        details = raw.get("details", {})
        patient_id = (
            raw.get("patientId")
            or raw.get("patient_id")
            or Path(path).stem
        )
        name = raw.get("name", patient_id.replace("_", " ").title())
        first_name, last_name = _split_name(name)

        demo = details.get("demographicAndSocioculturalInformation", {})
        family = details.get("familyHistory", {})
        edu = details.get("educationAndEmployment", {})
        psych = details.get("psychologicalProfileAndCognitiveFunctioning", {})
        coping = details.get("copingMechanismsAndDefenses", {})
        dysfunction = details.get("dysfunctionalBehaviorsAndRiskyConduct", {})
        morality = details.get("morality", {})
        social = details.get("socialRelationshipsAndInteractions", {})
        tx = details.get("treatmentsAndInterventions", {})
        resources = details.get("psychologicalResources", {})
        medical = details.get("medicalAndPhysicalHistory", {})
        behavior = details.get("behaviorDuringTestAdministration", {})

        metadata = {
            "background": raw.get("briefDescription", ""),
            "therapy_goals": raw.get("objectives", []) or [],
            "difficulty": raw.get("difficulty"),
            "estimatedDuration": raw.get("estimatedDuration"),
            "avatarUrl": raw.get("avatarUrl"),
            "voiceId": raw.get("voiceId"),
            "welcomeMessage": raw.get("welcomeMessage"),
        }

        disorder_name = details.get("disorder", {}).get("disorderName", "Unspecified disorder")
        diagnoses = _ensure_list(disorder_name)
        main_symptoms = [
            psych.get("affectiveEmotionalFunctioningAndMoodRegulation"),
            psych.get("psychiatricComorbidities"),
            psych.get("senseOfSelfAndOthers"),
            psych.get("thoughtFunctioningAndCognitiveStyle"),
            dysfunction.get("selfHarmAndSuicidality"),
            dysfunction.get("impulsiveAndRiskyBehaviors"),
        ]
        main_symptoms = [
            s for s in (_clean_text(item, default="") for item in main_symptoms) if s
        ]

        therapeutic_goals = tx.get("therapeuticGoals")
        if isinstance(therapeutic_goals, list):
            goals_value = therapeutic_goals
        elif therapeutic_goals:
            goals_value = [therapeutic_goals]
        else:
            goals_value = []

        profile = {
            "patient_id": patient_id,
            "disorder_id": _slugify(disorder_name),
            "Metadata": metadata,
             "EmotionDynamics": _build_emotion_traits(raw),
            "DemographicInfo": {
                "Name": first_name,
                "Surname": last_name,
                "Age": _safe_int(demo.get("age"), default=0),
                "Gender": _clean_text(demo.get("gender"), default="Not reported"),
                "MaritalStatus": _clean_text(demo.get("maritalStatus"), default="Not reported"),
                "CulturalBackground": _clean_text(demo.get("culturalBackground")),
                "ReligiousBeliefs": _clean_text(demo.get("religiousBeliefs")),
                "SpokenLanguage": _clean_text(demo.get("spokenLanguage")),
                "MigrationStatus": _clean_text(demo.get("migrationStatus")),
            },
            "FamilySocialHistory": {
                "ChildhoodFamilyDynamics": _clean_text(family.get("familyDynamicsDuringDevelopment")),
                "CurrentParentRelations": _clean_text(family.get("currentRelationshipWithParents")),
                "ChildhoodExperiences": _clean_text(family.get("childhoodExperiences")),
                "SignificantDevelopmentalExperiences": _clean_text(family.get("significantDevelopmentalExperiences")),
                "FamilyPsychiatricHistory": _clean_text(family.get("familyPsychiatricIllnesses")),
                "AbuseHistory": None,
                "SocialSupport": None,
                "SupportNetwork": None,
            },
            "EducationOccupation": {
                "EducationLevel": _clean_text(edu.get("educationLevel")),
                "WorkHistory": _clean_text(edu.get("workHistory")),
                "HousingStability": _clean_text(edu.get("housingStability")),
                "FinancialSituation": _clean_text(edu.get("financialSituation")),
                "HobbiesInterests": _clean_text(edu.get("hobbiesAndInterests")),
            },
            "PsychologicalProfile": {
                "PsychiatricDiagnoses": diagnoses,
                "MainSymptoms": main_symptoms or ["Not specified"],
                "AffectiveEmotionalFunctioning": _clean_text(psych.get("affectiveEmotionalFunctioningAndMoodRegulation")),
                "PsychiatricComorbidities": _clean_text(psych.get("psychiatricComorbidities")),
                "SenseOfSelfOthers": _clean_text(psych.get("senseOfSelfAndOthers")),
                "ThoughtCognitiveStyle": _clean_text(psych.get("thoughtFunctioningAndCognitiveStyle")),
                "EmotionalReactions": None,
                "Aggressiveness": None,
                "SelfPerceptionIdentity": None,
                "SelfEsteem": None,
                "CognitiveStyle": None,
                "AttachmentStyle": None,
                "ExecutiveFunctioning": None,
                "Memory": _clean_text(psych.get("memory")),
                "AttentionConcentration": _clean_text(psych.get("attentionAndConcentration")),
                "SensoryPerception": _clean_text(psych.get("sensoryPerception")),
                "HigherCognitiveFunctions": _clean_text(psych.get("higherCognitiveFunctions")),
            },
            "CopingDefenses": {
                "CopingStrategies": _clean_text(coping.get("copingStrategies")),
                "DefenseMechanisms": _ensure_list(coping.get("defenseMechanisms")),
                "SelfHarmSuicidality": _clean_text(dysfunction.get("selfHarmAndSuicidality")),
                "SubstanceAbuse": _clean_text(dysfunction.get("substanceAbuse")),
                "Avoidance": _clean_text(coping.get("copingStrategies")),
                "ImpulsiveRiskBehaviors": _clean_text(dysfunction.get("impulsiveAndRiskyBehaviors")),
                "Morality": _clean_text(morality.get("moralValues")),
            },
            "SocialRelations": {
                "Friendships": _clean_text(social.get("friendships")),
                "RomanticRelationships": _clean_text(social.get("romanticRelationships")),
                "SexualRelationships": _clean_text(social.get("sexualRelationships")),
                "FamilyInteractions": _clean_text(social.get("familyInteractions")),
                "PeerColleagueRelations": _clean_text(social.get("relationshipsWithPeersAndColleagues")),
                "SocialMediaBehavior": _clean_text(social.get("socialMediaUseAndImpact")),
            },
            "TreatmentsInterventions": {
                "PastTherapies": _ensure_list(tx.get("previousTherapeuticExperiences")),
                "ProgressResistance": _clean_text(tx.get("treatmentResistance")),
                "TherapeuticGoals": goals_value,
                "MedicationHistory": _ensure_list(tx.get("medicationHistory")),
                "MedicationResponse": _clean_text(tx.get("responseToMedications")),
                "Hospitalizations": _safe_int(tx.get("previousHospitalizations"), default=0),
                "EmergencyRoomVisits": _clean_text(tx.get("emergencyDepartmentVisits")),
                "PastPsychiatricDiagnoses": _ensure_list(tx.get("previousPsychiatricDiagnoses")),
                "PreviousDropouts": _clean_text(tx.get("previousDropouts")),
            },
            "ClinicalJudgment": {
                "JudgmentCapacity": _clean_text(psych.get("higherCognitiveFunctions")),
                "Insight": _clean_text(psych.get("higherCognitiveFunctions")),
                "ImpulseControl": _clean_text(dysfunction.get("impulsiveAndRiskyBehaviors")),
            },
            "ResilienceWellbeing": {
                "PsychologicalResilience": _clean_text(resources.get("psychologicalResilience")),
                "SelfCompassion": "Working on self-kindness while managing shame and self-criticism.",
                "LifeSatisfaction": _clean_text(resources.get("lifeSatisfaction")),
                "PersonalGrowth": "Committed to therapy to regain stability and purpose.",
                "SenseOfCoherence": "Seeks to make sense of stressors and how they affect functioning.",
                "Empowerment": "Therapeutic support is used to build agency and healthier routines.",
                "HopeForFuture": _clean_text(resources.get("hopeForTheFuture")),
                "LongTermLifeGoals": _clean_text(resources.get("longTermLifeGoals")),
            },
            "MedicalHistory": {
                "PreexistingConditions": _clean_text(medical.get("preExistingMedicalConditions")),
                "CurrentMedications": [],
                "PharmacologicalTreatments": _ensure_list(medical.get("pharmacologicalTreatments")),
                "Allergies": None,
                "Lifestyle": _clean_text(medical.get("lifestyle")),
                "GeneralHealth": _clean_text(medical.get("generalPhysicalHealth")),
                "SleepPatterns": _clean_text(medical.get("sleepPatterns")),
                "EatingHabits": _clean_text(medical.get("eatingHabits")),
            },
            "SocialEnvironment": {
                "SocialSupport": _infer_social_support(family, social),
                "HousingConditions": _clean_text(edu.get("housingStability")),
                "SocialSecurity": "Not reported.",
                "SocialCohesion": "Navigates academic, occupational, and cultural environments with varying belonging.",
                "SocialDeterminantsMentalHealth": "Financial pressure, discrimination, and relational stress influence mental health.",
            },
            "TestBehavior": {
                "RecurringDynamics": _clean_text(behavior.get("recurringDynamicsTransferenceCountertransference")),
                "PredominantEmotions": _ensure_list(behavior.get("expressedEmotionsAndCongruence")) or ["Not reported."],
                "Speech": _clean_text(behavior.get("speechCharacteristics")),
                "EyeContactPostureGestures": _clean_text(behavior.get("nonVerbalBehavior")),
                "AvoidantAttitudesMoodChange": _clean_text(behavior.get("appearanceSelfCareOrientation")),
            },
            "ClinicalSummary": raw.get("clinicalCase", ""),
            "current_emotional_state": "base",
            "session_notes": None,
        }
        return profile

    def to_text_summary(self) -> str:
        """Compact summary for LLM or prompt context."""
        demo = self.DemographicInfo
        profile = self.PsychologicalProfile
        symptoms = ", ".join(profile.MainSymptoms[:3]) if profile.MainSymptoms else "no major symptoms"
        return (
            f"{demo.Name} {demo.Surname}, {demo.Age}-year-old {demo.Gender.lower()} "
            f"with background: {demo.CulturalBackground.lower()}. "
            f"Known for {symptoms}. Typical tone: {self.current_emotional_state}."
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


def _clamp_emotion_value(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, numeric))


def _split_name(full_name: str) -> tuple[str, str]:
    """Best-effort split of arbitrary name strings into first/last components."""
    if not full_name:
        return "Unknown", "Unknown"
    parts = full_name.strip().split()
    if not parts:
        return "Unknown", "Unknown"
    if len(parts) == 1:
        return parts[0], parts[0]
    return parts[0], " ".join(parts[1:])


def _clean_text(value, default: str = "Not reported."):
    """Normalize various json fields into trimmed strings with sensible fallbacks."""
    if value is None:
        return default if default != "" else ""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default if default != "" else ""
        return text
    return str(value)


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


def _safe_int(value, default: int = 0) -> int:
    """Extract an integer from loosely formatted sources (words, strings, etc.)."""
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        digits = re.findall(r"-?\\d+", value)
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


def _slugify(text: str) -> str:
    """Machine-friendly slug for disorders/patient IDs."""
    if not text:
        return "unspecified"
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return cleaned.strip("_") or "unspecified"


def _infer_social_support(family: dict, social: dict) -> str:
    """Synthesize a concise statement about the patient's practical support system."""
    parts = []
    fam = family.get("currentRelationshipWithParents")
    if fam:
        parts.append(fam)
    cousin = social.get("friendships")
    if cousin:
        parts.append(cousin)
    peers = social.get("relationshipsWithPeersAndColleagues")
    if peers:
        parts.append(peers)
    return " ".join(parts).strip() or "Not reported."
