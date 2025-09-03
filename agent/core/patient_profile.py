from typing import List, Dict, Optional
from pydantic import BaseModel
import json

# === Subcomponents ===
class DemographicInfo(BaseModel):
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
    AbuseHistory: str
    SocialSupport: str
    SupportNetwork: str

class EducationOccupation(BaseModel):
    EducationLevel: str
    WorkHistory: str
    HousingStability: str
    FinancialSituation: str
    HobbiesInterests: str

class PsychologicalProfile(BaseModel):
    PsychiatricDiagnoses: List[str]
    MainSymptoms: List[str]
    EmotionalReactions: str
    Aggressiveness: str
    SelfPerceptionIdentity: str
    SelfEsteem: str
    SenseOfSelfOthers: str
    CognitiveStyle: List[str]
    AttachmentStyle: str
    ExecutiveFunctioning: str
    Memory: str
    AttentionConcentration: str
    SensoryPerception: str
    HigherCognitiveFunctions: str

class CopingDefenses(BaseModel):
    CopingStrategies: str
    DefenseMechanisms: List[str]
    SelfHarmSuicidality: str
    SubstanceAbuse: str
    Avoidance: str
    ImpulsiveRiskBehaviors: str
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
    TherapeuticGoals: str
    MedicationHistory: List[str]
    MedicationResponse: str
    Hospitalizations: int
    EmergencyRoomVisits: str
    PastPsychiatricDiagnoses: List[str]

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
    Allergies: str
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

# === Main Patient Profile ===
class PatientProfile(BaseModel):
    patient_id: str
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

    @property
    def name(self) -> str:
        return getattr(self.DemographicInfo, "Name", self.patient_id.replace("_", " ").title())

    @classmethod
    def from_file(cls, path: str) -> "PatientProfile":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)