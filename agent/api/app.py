"""FastAPI entrypoint that exposes the simulated patient via /api/message."""

import re
import json
import time

from pathlib import Path
from typing import Literal
from datetime import datetime
from pydantic import BaseModel, Field
from agent.utils.run_logger import RunLogger
from fastapi import FastAPI, HTTPException, status
from agent.core.langgraph_builder import build_graph

ROOT_DIR = Path(__file__).resolve().parents[2]
PATIENTS_DIR = ROOT_DIR / "data" / "patients"

app = FastAPI(title="PsyLLM Patient Agent API")

graph = build_graph()
session_loggers: dict[tuple[str, str], dict] = {}

# === Request Schema ===
class MessageRequest(BaseModel):
    """Payload describing a single therapist-to-patient turn."""
    external_patient_id: str
    user_message: str
    session_id: str
    step_id: int
    therapist_id: str | None = "therapist0"

# === Response Schema ===
class MessageResponse(BaseModel):
    """Normalized response sent back to the caller/UI."""
    message: str
    reasoning_time: float
    emotion: str
    topic: str
    timestamp: str


class PatientInitRequest(BaseModel):
    """Request body for creating or initializing a patient record."""

    id: str
    name: str
    age: int
    gender: str
    diagnosis: str
    difficulty_level: int
    psychological_profile: str
    background: str
    current_medications: list[str] = Field(default_factory=list)
    therapy_goals: list[str] = Field(default_factory=list)
    previous_sessions: int = 0
    session_id: str


class PatientInitResponse(BaseModel):
    """Standardized acknowledgement for patient initialization."""

    status: Literal["success", "exists"]
    code: Literal["PATIENT_CREATED", "PATIENT_EXISTS"]
    external_patient_id: str
    message: str
    timestamp: str


def _sanitize_patient_id(raw_id: str) -> str:
    """Normalize and validate patient IDs to safe filenames."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw_id.strip()).strip("_").lower()
    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Patient id must contain alphanumeric characters",
        )
    return cleaned


def _patient_file_path(patient_id: str) -> Path:
    """Return the target path for a patient's JSON profile."""
    return PATIENTS_DIR / f"{patient_id}.json"


def _difficulty_to_volatility(level: int) -> str:
    """Map difficulty level into a coarse volatility bucket."""
    if level >= 4:
        return "high"
    if level <= 1:
        return "low"
    return "medium"


def _serialize_patient(req: PatientInitRequest, patient_id: str) -> dict:
    """Convert the external payload into the internal attribute-style schema."""
    volatility = _difficulty_to_volatility(req.difficulty_level)
    welcome = f"Hi, I'm {req.name.split()[0] if req.name else 'the patient'}. Thanks for meeting with me."
    previous_sessions = (
        f"{req.previous_sessions} previous sessions; latest session id: {req.session_id}"
        if req.previous_sessions
        else "No previous sessions; intake session."
    )

    return {
        "patientId": patient_id,
        "name": req.name,
        "briefDescription": req.background,
        "welcomeMessage": welcome,
        "difficulty": req.difficulty_level,
        "objectives": req.therapy_goals or [],
        "emotionTraits": {
            "volatility_level": volatility,
            "trait_baseline": {
                "SEEKING": 0.45,
                "RAGE": 0.3,
                "FEAR": 0.35,
                "CARE": 0.5,
                "LUST": 0.25,
                "SADNESS": 0.35,
                "PLAY": 0.35,
            },
        },
        "details": {
            "demographicAndSocioculturalInformation": {
                "age": req.age,
                "gender": req.gender,
            },
            "disorder": {"disorderName": req.diagnosis},
            "educationAndEmployment": {
                "workHistory": req.background,
            },
            "psychologicalProfileAndCognitiveFunctioning": {
                "affectiveEmotionalFunctioningAndMoodRegulation": req.psychological_profile,
            },
            "treatmentsAndInterventions": {
                "therapeuticGoals": req.therapy_goals or [],
                "medicationHistory": req.current_medications or [],
                "previousTherapeuticExperiences": previous_sessions,
            },
            "medicalAndPhysicalHistory": {
                "pharmacologicalTreatments": req.current_medications or [],
            },
        },
        "clinicalCase": req.psychological_profile,
    }


@app.post("/chat-response", response_model=MessageResponse)
async def send_message(req: MessageRequest):
    """Main conversational endpoint."""

    patient_id = req.external_patient_id
    therapist_id = req.therapist_id or "therapist0"
    session_key = (therapist_id, req.session_id)

    config = {"configurable": {"thread_id": req.session_id}}

    # === Track reasoning time ===
    start_time = time.time()

    # === Ensure run logger for this therapist/session ===
    entry = session_loggers.get(session_key)
    if not entry:
        run_logger = RunLogger(therapist_id)
        try:
            base_state = run_logger.restore_state(patient_id)
        except Exception as e:
            base_state = None
            print(f"[WARN] restore_state failed: {e}")
        run_logger.start_run(
            patient_id=patient_id,
            session_id=req.session_id,
            source="api",
            mode="live",
            metadata={"initial_step_id": req.step_id},
        )
        entry = {"logger": run_logger, "base_state": base_state}
        session_loggers[session_key] = entry
    run_logger = entry["logger"]
    base_state = entry.get("base_state")
    if base_state:
        entry["base_state"] = None

    # === Run the agent ===
    payload = {"user_input": req.user_message, "patient_id": patient_id}
    if base_state:
        payload.update(base_state)
    result = graph.invoke(payload, config=config)

    reasoning_time = round(time.time() - start_time, 3)

    # === Extract relevant info ===
    message = result.get("response", "...")
    patient_profile = result.get("patient_profile", {})
    
    emotion = (
        patient_profile.get("current_emotional_state", "base")
        if isinstance(patient_profile, dict)
        else "base"
    )
    topic_info = result.get("last_topic", {})
    topic = topic_info.get("sub", "general") if isinstance(topic_info, dict) else "general"

    # === Persist run info ===
    run_logger.log_turn(result, req.user_message)
    entry["latest_state"] = result

    # === Return unified JSON ===
    return MessageResponse(
        message=message,
        reasoning_time=reasoning_time,
        emotion=emotion,
        topic=topic,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/patients", response_model=PatientInitResponse)
async def create_patient(req: PatientInitRequest):
    """Create a patient file if it does not already exist."""

    patient_id = _sanitize_patient_id(req.id)
    patient_path = _patient_file_path(patient_id)

    if patient_path.exists():
        return PatientInitResponse(
            status="exists",
            code="PATIENT_EXISTS",
            external_patient_id=patient_id,
            message="Paziente già presente nel sistema esterno",
            timestamp=datetime.utcnow().isoformat(),
        )

    PATIENTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = _serialize_patient(req, patient_id)
    with open(patient_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")

    return PatientInitResponse(
        status="success",
        code="PATIENT_CREATED",
        external_patient_id=patient_id,
        message="Paziente inizializzato correttamente nel sistema esterno",
        timestamp=datetime.utcnow().isoformat(),
    )
