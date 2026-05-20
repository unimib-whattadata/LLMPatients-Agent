"""FastAPI entrypoint that exposes the simulated patient via /api/message."""

import json
import logging
import os
import re
import tarfile
import tempfile
import time
import yaml

from pathlib import Path
from typing import Literal
from datetime import datetime
from pydantic import BaseModel, Field
from agent.core.emotion_model import EMOTION_LABELS
from agent.core.patient_profile import resolve_patient_profile_path
from agent.utils.run_logger import RunLogger
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from agent.core.langgraph_builder import build_graph, finalize_session_memory

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
PATIENTS_DIR = ROOT_DIR / "data" / "patients"
RUNS_DIR = ROOT_DIR / "tests" / "runs"
MEMORY_DIR = ROOT_DIR / "data" / "memory"
EXPORT_TOKEN = os.getenv("PSYLLM_EXPORT_TOKEN")
DEFAULT_THERAPIST_ID = "therapist0"
DEFAULT_TOPIC = "general"
DEFAULT_EMOTION = "seeking"
MAX_EMOTION_TIMELINE_POINTS = 60

app = FastAPI(title="LLMPatients-Agent API")

graph = build_graph()
session_loggers: dict[tuple[str, str, str], dict] = {}


def _runtime_session_key(therapist_id: str, patient_id: str, session_id: str) -> tuple[str, str, str]:
    """Isolate in-memory sessions by therapist, patient, and client session id."""
    return (therapist_id or DEFAULT_THERAPIST_ID, patient_id, session_id)


def _runtime_thread_id(therapist_id: str, patient_id: str, session_id: str) -> str:
    """Build a checkpointer thread id that cannot collide across patients."""
    return f"{therapist_id or DEFAULT_THERAPIST_ID}::{patient_id}::{session_id}"

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


def _clamp01(value: float) -> float:
    """Clamp chart-facing numeric values into the valid [0, 1] range."""
    return max(0.0, min(1.0, value))


def _safe_float(value, default: float = 0.0) -> float:
    """Best-effort float conversion for values returned by graph state."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _utc_now() -> str:
    return datetime.utcnow().isoformat()


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
        normalized[key.upper()] = _clamp01(numeric)
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


def _build_emotion_timeline(run_logger: RunLogger, *, max_points: int = MAX_EMOTION_TIMELINE_POINTS) -> list[dict]:
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
        timestamp = str(turn.get("timestamp") or _utc_now())
        emotion = str(turn.get("current_emotion") or "unknown")
        points.append({
            "turn_index": turn_index,
            "timestamp": timestamp,
            "emotion": emotion,
            "intensity": _clamp01(_safe_float(turn.get("emotion_intensity", 0.0))),
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


def _export_archive_path() -> Path:
    """Create a tar.gz with run logs and therapist/patient memory pairs."""
    run_files = sorted(path for path in RUNS_DIR.glob("*.json") if path.is_file())
    memory_files = set()
    for run_file in run_files:
        try:
            run_payload = json.loads(run_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        therapist_id = run_payload.get("therapist_id")
        sessions = run_payload.get("sessions", []) if isinstance(run_payload, dict) else []
        patient_ids = {session.get("patient_id") for session in sessions if session.get("patient_id")}
        for patient_id in patient_ids:
            if therapist_id and patient_id:
                memory_path = MEMORY_DIR / f"{therapist_id}__{patient_id}.jsonl"
                if memory_path.exists():
                    memory_files.add(memory_path)

    files = run_files + sorted(memory_files)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz")
    tmp_path = Path(tmp.name)
    tmp.close()
    with tarfile.open(tmp_path, mode="w:gz") as tar:
        for path in files:
            arcname = path.relative_to(ROOT_DIR)
            tar.add(path, arcname=str(arcname))
    return tmp_path


def _cleanup_path(path: Path) -> None:
    if path.exists():
        try:
            path.unlink()
        except OSError as exc:
            logger.warning("Could not delete temporary export archive %s: %s", path, exc)


def _restore_base_state(run_logger: RunLogger, patient_id: str) -> dict | None:
    """Load resumable graph state without letting corrupt logs break a new session."""
    try:
        return run_logger.restore_state(patient_id)
    except Exception as exc:
        logger.warning("Could not restore state for patient %s: %s", patient_id, exc)
        return None


def _get_or_start_session(
    *,
    req: MessageRequest,
    patient_id: str,
    therapist_id: str,
    session_key: tuple[str, str, str],
) -> dict:
    """Return the active session logger entry, creating it on first turn."""
    entry = session_loggers.get(session_key)
    if entry:
        return entry

    run_logger = RunLogger(therapist_id)
    base_state = _restore_base_state(run_logger, patient_id)
    run_logger.start_run(
        patient_id=patient_id,
        session_id=req.session_id,
        source="api",
        mode="live",
        metadata={"initial_step_id": req.step_id},
    )
    entry = {"logger": run_logger, "base_state": base_state}
    session_loggers[session_key] = entry
    return entry


def _consume_base_state(entry: dict) -> dict | None:
    """Use restored state exactly once at the start of a live API session."""
    base_state = entry.get("base_state")
    if base_state:
        entry["base_state"] = None
    return base_state


def _build_graph_payload(
    *,
    req: MessageRequest,
    patient_id: str,
    therapist_id: str,
    base_state: dict | None,
) -> dict:
    """Merge restored state with the current request while preserving live identifiers."""
    payload = {
        "user_input": req.user_message,
        "patient_id": patient_id,
        "therapist_id": therapist_id,
        "session_id": req.session_id,
    }
    if base_state:
        payload = {**base_state, **payload}
    return payload


def _topic_from_result(result: dict) -> str:
    topic_info = result.get("last_topic", {})
    if isinstance(topic_info, dict):
        return topic_info.get("sub") or DEFAULT_TOPIC
    return DEFAULT_TOPIC


def _emotion_snapshot_from_result(result: dict, patient_profile) -> tuple[str, EmotionSnapshot]:
    """Build both the public emotion label and the structured chart snapshot."""
    emotion = str(
        _profile_value(patient_profile, "current_emotional_state", default=None)
        or result.get("core_emotion")
        or DEFAULT_EMOTION
    ).lower()
    emotion_vector = _normalize_emotion_vector(
        result.get("emotion_state")
        or _profile_value(patient_profile, "emotion_state", default={})
    )
    intensity = _clamp01(
        _safe_float(
            result.get("emotion_intensity")
            or _profile_value(patient_profile, "emotion_intensity", default=0.0)
        )
    )
    raw_salience = result.get("emotion_salience")
    salience = _clamp01(_safe_float(raw_salience)) if raw_salience is not None else None
    snapshot = EmotionSnapshot(
        dominant=emotion,
        intensity=intensity,
        vector=emotion_vector,
        event=result.get("emotion_event"),
        salience=salience,
        description=_build_emotion_description(
            dominant=emotion,
            intensity=intensity,
            vector=emotion_vector,
        ),
    )
    return emotion, snapshot


def _message_response(
    *,
    result: dict,
    run_logger: RunLogger,
    reasoning_time: float,
) -> MessageResponse:
    """Translate internal graph state into the stable API response model."""
    patient_profile = result.get("patient_profile", {})
    emotion, emotion_snapshot = _emotion_snapshot_from_result(result, patient_profile)
    return MessageResponse(
        message=result.get("response", "..."),
        reasoning_time=reasoning_time,
        emotion=emotion,
        topic=_topic_from_result(result),
        timestamp=_utc_now(),
        patient_name=_profile_value(patient_profile, "name", default=None),
        avatar_url=_profile_value(patient_profile, "avatar_url", "avatarUrl", default=None),
        emotion_snapshot=emotion_snapshot,
        emotion_timeline=_build_emotion_timeline(run_logger),
    )


@app.post("/chat-response", response_model=MessageResponse)
async def send_message(req: MessageRequest):
    """Main conversational endpoint."""

    patient_id = req.external_patient_id
    therapist_id = req.therapist_id or DEFAULT_THERAPIST_ID
    session_key = _runtime_session_key(therapist_id, patient_id, req.session_id)

    config = {"configurable": {"thread_id": _runtime_thread_id(therapist_id, patient_id, req.session_id)}}

    start_time = time.time()
    entry = _get_or_start_session(
        req=req,
        patient_id=patient_id,
        therapist_id=therapist_id,
        session_key=session_key,
    )
    run_logger = entry["logger"]

    payload = _build_graph_payload(
        req=req,
        patient_id=patient_id,
        therapist_id=therapist_id,
        base_state=_consume_base_state(entry),
    )
    result = graph.invoke(payload, config=config)
    reasoning_time = round(time.time() - start_time, 3)

    run_logger.log_turn(result, req.user_message)
    entry["latest_state"] = result
    return _message_response(result=result, run_logger=run_logger, reasoning_time=reasoning_time)


@app.post("/session-end", response_model=SessionEndResponse)
async def end_session(req: SessionEndRequest):
    """Finalize memory and logs for a therapist/patient session."""
    therapist_id = req.therapist_id or DEFAULT_THERAPIST_ID
    session_key = _runtime_session_key(therapist_id, req.external_patient_id, req.session_id)
    entry = session_loggers.get(session_key)
    if not entry:
        return SessionEndResponse(
            status="not_found",
            message="No active session found for this therapist/session id.",
            timestamp=_utc_now(),
        )

    run_logger = entry.get("logger")
    state = entry.get("latest_state") or entry.get("base_state") or {}
    state["patient_id"] = req.external_patient_id
    state["therapist_id"] = therapist_id
    state["session_id"] = req.session_id
    state = finalize_session_memory(state)
    if run_logger:
        run_logger.finalize(state or {})
    session_loggers.pop(session_key, None)

    return SessionEndResponse(
        status="finalized",
        message="Session memory finalized.",
        timestamp=_utc_now(),
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
            timestamp=_utc_now(),
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
        timestamp=_utc_now(),
    )


@app.get("/export-logs")
async def export_logs(
    background_tasks: BackgroundTasks,
    export_token: str | None = Header(default=None, alias="X-Export-Token"),
):
    """Download run logs and therapist/patient memory pairs."""
    if EXPORT_TOKEN and export_token != EXPORT_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid export token.")
    archive_path = _export_archive_path()
    background_tasks.add_task(_cleanup_path, archive_path)
    filename = f"llmpatients_export_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.tar.gz"
    return StreamingResponse(
        archive_path.open("rb"),
        media_type="application/gzip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
