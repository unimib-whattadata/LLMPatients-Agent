"""FastAPI entrypoint that exposes the simulated patient via /api/message."""

import re
import time
import yaml

from pathlib import Path
from typing import Literal
from datetime import datetime
from pydantic import BaseModel, Field
from agent.core.emotion_model import EMOTION_LABELS
from agent.core.patient_profile import resolve_patient_profile_path
from agent.utils.run_logger import RunLogger
from fastapi import FastAPI, HTTPException, status
from agent.core.langgraph_builder import build_graph, finalize_session_memory

ROOT_DIR = Path(__file__).resolve().parents[2]
PATIENTS_DIR = ROOT_DIR / "data" / "patients"

app = FastAPI(title="LLMPatients-Agent API")

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
class EmotionPoint(BaseModel):
    """Single point for line-chart rendering."""
    turn_index: int
    timestamp: str
    emotion: str
    intensity: float


class EmotionSnapshot(BaseModel):
    """Current emotional state payload for chart + short text."""
    dominant: str
    intensity: float
    vector: dict[str, float] = Field(default_factory=dict)
    event: str | None = None
    salience: float | None = None
    description: str


class MessageResponse(BaseModel):
    """Normalized response sent back to the caller/UI."""
    message: str
    reasoning_time: float
    emotion: str
    topic: str
    timestamp: str
    patient_name: str | None = None
    avatar_url: str | None = None
    emotion_snapshot: EmotionSnapshot | None = None
    emotion_timeline: list[EmotionPoint] = Field(default_factory=list)


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


class SessionEndRequest(BaseModel):
    """Payload describing a session end event."""
    external_patient_id: str
    session_id: str
    therapist_id: str | None = "therapist0"


class SessionEndResponse(BaseModel):
    """Ack for session finalization."""
    status: Literal["finalized", "not_found"]
    message: str
    timestamp: str


def _profile_value(profile, *keys, default=None):
    """Safely read attributes from either dict profiles or model instances."""
    if isinstance(profile, dict):
        for key in keys:
            if key in profile and profile.get(key) is not None:
                return profile.get(key)
        return default
    for key in keys:
        value = getattr(profile, key, None)
        if value is not None:
            return value
    return default


def _normalize_emotion_vector(raw_vector) -> dict[str, float]:
    """Clamp and normalize raw emotion vectors into chart-safe floats."""
    if not isinstance(raw_vector, dict):
        return {}
    normalized: dict[str, float] = {}
    for key, value in raw_vector.items():
        if not isinstance(key, str):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        normalized[key.upper()] = max(0.0, min(1.0, numeric))
    return normalized


def _emotion_label(label: str) -> str:
    normalized = (label or "").strip().upper()
    if not normalized:
        return "Unknown"
    return EMOTION_LABELS.get(normalized, normalized.replace("_", " ").title())


def _build_emotion_description(
    *,
    dominant: str,
    intensity: float,
    vector: dict[str, float],
) -> str:
    """Small textual summary suitable for rendering below the chart."""
    if not vector:
        return f"Dominant emotion: {_emotion_label(dominant)}. Overall intensity: {intensity:.2f}."

    ranked = sorted(vector.items(), key=lambda item: item[1], reverse=True)
    primary_label, primary_value = ranked[0]
    secondary_label, secondary_value = ranked[1] if len(ranked) > 1 else ranked[0]
    return (
        f"Dominant emotion: {_emotion_label(primary_label)} ({primary_value:.2f}). "
        f"Secondary tone: {_emotion_label(secondary_label)} ({secondary_value:.2f}). "
        f"Overall intensity: {intensity:.2f}."
    )


def _build_emotion_timeline(run_logger: RunLogger, *, max_points: int = 60) -> list[dict]:
    """Extract a compact turn-by-turn timeline from the active run logger."""
    idx = run_logger.current_session_index
    if idx is None:
        return []

    sessions = run_logger.data.get("sessions", [])
    if idx < 0 or idx >= len(sessions):
        return []

    turns = sessions[idx].get("turns", [])
    if not isinstance(turns, list):
        return []

    points = []
    for turn in turns[-max_points:]:
        if not isinstance(turn, dict):
            continue
        try:
            turn_index = int(turn.get("turn_index", len(points) + 1))
        except (TypeError, ValueError):
            turn_index = len(points) + 1
        timestamp = str(turn.get("timestamp") or datetime.utcnow().isoformat())
        emotion = str(turn.get("current_emotion") or "unknown")
        try:
            intensity = float(turn.get("emotion_intensity", 0.0))
        except (TypeError, ValueError):
            intensity = 0.0
        points.append({
            "turn_index": turn_index,
            "timestamp": timestamp,
            "emotion": emotion,
            "intensity": max(0.0, min(1.0, intensity)),
        })
    return points


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
    """Return the target path for a patient's YAML profile."""
    return PATIENTS_DIR / f"{patient_id}.yaml"


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
                "PANIC_GRIEF": 0.35,
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
    payload = {
        "user_input": req.user_message,
        "patient_id": patient_id,
        "therapist_id": therapist_id,
        "session_id": req.session_id,
    }
    if base_state:
        payload.update(base_state)
    result = graph.invoke(payload, config=config)

    reasoning_time = round(time.time() - start_time, 3)

    # === Extract relevant info ===
    message = result.get("response", "...")
    patient_profile = result.get("patient_profile", {})

    emotion = str(
        _profile_value(patient_profile, "current_emotional_state", default=None)
        or result.get("core_emotion")
        or "seeking"
    ).lower()
    topic_info = result.get("last_topic", {})
    topic = topic_info.get("sub", "general") if isinstance(topic_info, dict) else "general"

    emotion_vector = _normalize_emotion_vector(
        result.get("emotion_state")
        or _profile_value(patient_profile, "emotion_state", default={})
    )
    try:
        intensity = float(
            result.get("emotion_intensity")
            or _profile_value(patient_profile, "emotion_intensity", default=0.0)
            or 0.0
        )
    except (TypeError, ValueError):
        intensity = 0.0
    intensity = max(0.0, min(1.0, intensity))

    salience = result.get("emotion_salience")
    try:
        salience_value = max(0.0, min(1.0, float(salience))) if salience is not None else None
    except (TypeError, ValueError):
        salience_value = None

    emotion_snapshot = EmotionSnapshot(
        dominant=emotion,
        intensity=intensity,
        vector=emotion_vector,
        event=result.get("emotion_event"),
        salience=salience_value,
        description=_build_emotion_description(
            dominant=emotion,
            intensity=intensity,
            vector=emotion_vector,
        ),
    )

    patient_name = _profile_value(patient_profile, "name", default=None)
    avatar_url = _profile_value(patient_profile, "avatar_url", "avatarUrl", default=None)

    # === Persist run info ===
    run_logger.log_turn(result, req.user_message)
    entry["latest_state"] = result
    emotion_timeline = _build_emotion_timeline(run_logger)

    # === Return unified JSON ===
    return MessageResponse(
        message=message,
        reasoning_time=reasoning_time,
        emotion=emotion,
        topic=topic,
        timestamp=datetime.utcnow().isoformat(),
        patient_name=patient_name,
        avatar_url=avatar_url,
        emotion_snapshot=emotion_snapshot,
        emotion_timeline=emotion_timeline,
    )


@app.post("/session-end", response_model=SessionEndResponse)
async def end_session(req: SessionEndRequest):
    """Finalize memory and logs for a therapist/patient session."""
    therapist_id = req.therapist_id or "therapist0"
    session_key = (therapist_id, req.session_id)
    entry = session_loggers.get(session_key)
    if not entry:
        return SessionEndResponse(
            status="not_found",
            message="No active session found for this therapist/session id.",
            timestamp=datetime.utcnow().isoformat(),
        )

    run_logger = entry.get("logger")
    state = entry.get("latest_state") or entry.get("base_state") or {}
    state.setdefault("patient_id", req.external_patient_id)
    state.setdefault("therapist_id", therapist_id)
    state.setdefault("session_id", req.session_id)
    state = finalize_session_memory(state)
    if run_logger:
        run_logger.finalize(state or {})
    session_loggers.pop(session_key, None)

    return SessionEndResponse(
        status="finalized",
        message="Session memory finalized.",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/patients", response_model=PatientInitResponse)
async def create_patient(req: PatientInitRequest):
    """Create a patient file if it does not already exist."""

    patient_id = _sanitize_patient_id(req.id)
    patient_path = _patient_file_path(patient_id)
    existing_path = None
    try:
        existing_path = resolve_patient_profile_path(patient_id, PATIENTS_DIR)
    except FileNotFoundError:
        existing_path = None

    if existing_path:
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
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)

    return PatientInitResponse(
        status="success",
        code="PATIENT_CREATED",
        external_patient_id=patient_id,
        message="Paziente inizializzato correttamente nel sistema esterno",
        timestamp=datetime.utcnow().isoformat(),
    )
