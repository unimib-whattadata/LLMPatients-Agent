"""FastAPI entrypoint that exposes the simulated patient via /api/message."""

from fastapi import FastAPI
from pydantic import BaseModel
from agent.core.langgraph_builder import build_graph
from agent.utils.run_logger import RunLogger
from datetime import datetime
import time

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


@app.post("/api/message", response_model=MessageResponse)
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
        base_state = run_logger.restore_state(patient_id)
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
    emotion = getattr(result.get("patient_profile", None), "current_emotional_state", "base")
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
