from typing import List, Optional, Union
from pydantic import BaseModel
import json


# === Subcomponents ===
class DemographicInfo(BaseModel):
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
    ChildhoodFamilyDynamics: str
    CurrentParentRelations: str
    ChildhoodExperiences: str
    SignificantDevelopmentalExperiences: Optional[str] = None
    FamilyPsychiatricHistory: Optional[str] = None
    AbuseHistory: Optional[str] = None
    SocialSupport: Optional[str] = None
    SupportNetwork: Optional[str] = None

class EducationOccupation(BaseModel):
    EducationLevel: str
    WorkHistory: str
    HousingStability: str
    FinancialSituation: str
    HobbiesInterests: str

class PsychologicalProfile(BaseModel):
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
    CopingStrategies: str
    DefenseMechanisms: List[str]
    SelfHarmSuicidality: str
    SubstanceAbuse: str
    Avoidance: Optional[str] = None
    ImpulsiveRiskBehaviors: Optional[str] = None
    Morality: str

class SocialRelations(BaseModel):
    Friendships: str
    RomanticRelationships: str
    SexualRelationships: str
    FamilyInteractions: str
    PeerColleagueRelations: str
    SocialMediaBehavior: str

class TreatmentsInterventions(BaseModel):
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
    JudgmentCapacity: str
    Insight: str
    ImpulseControl: str

class ResilienceWellbeing(BaseModel):
    PsychologicalResilience: str
    SelfCompassion: str
    LifeSatisfaction: str
    PersonalGrowth: str
    SenseOfCoherence: str
    Empowerment: str
    HopeForFuture: str
    LongTermLifeGoals: str

class MedicalHistory(BaseModel):
    PreexistingConditions: str
    CurrentMedications: List[str]
    PharmacologicalTreatments: Optional[List[str]] = None
    Allergies: Optional[str] = None
    Lifestyle: str
    GeneralHealth: str
    SleepPatterns: str
    EatingHabits: str

class SocialEnvironment(BaseModel):
    SocialSupport: str
    HousingConditions: str
    SocialSecurity: str
    SocialCohesion: str
    SocialDeterminantsMentalHealth: str

class TestBehavior(BaseModel):
    RecurringDynamics: str
    PredominantEmotions: List[str]
    Speech: str
    EyeContactPostureGestures: str
    AvoidantAttitudesMoodChange: str

# === Metadata for UI/Simulation Layer ===
class PatientMetadata(BaseModel):
    background: str
    therapy_goals: List[str]
    difficulty: int
    estimatedDuration: int
    avatarUrl: str
    voiceId: str
    welcomeMessage: str

# === Main Patient Profile ===
class PatientProfile(BaseModel):
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

    # === Clinical and dynamic fields ===
    ClinicalSummary: Optional[str] = None
    current_emotional_state: Optional[str] = "base"
    session_notes: Optional[str] = None
    

    @property
    def name(self) -> str:
        """Return formatted patient name."""
        demo = getattr(self, "DemographicInfo", None)
        if demo:
            return f"{demo.Name} {demo.Surname}"
        return self.patient_id.replace("_", " ").title()

    @classmethod
    def from_file(cls, path: str) -> "PatientProfile":
        """Safely load JSON and ignore missing dynamic fields."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

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