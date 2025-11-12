"""FastAPI entrypoint that exposes the simulated patient via /api/message."""

from fastapi import FastAPI
from pydantic import BaseModel
from agent.core.langgraph_builder import build_graph
from agent.utils.run_logger import RunLogger
from datetime import datetime
import time

app = FastAPI(title="PsyLLM Patient Agent API")

graph = build_graph()
session_loggers = {}

# === Request Schema ===
class MessageRequest(BaseModel):
    """Payload describing a single therapist-to-patient turn."""
    external_patient_id: str
    user_message: str
    session_id: str
    step_id: int

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

    config = {"configurable": {"thread_id": req.session_id}}

    # === Track reasoning time ===
    start_time = time.time()

    # === Ensure run logger for this session ===
    run_logger = session_loggers.get(req.session_id)
    if not run_logger:
        run_logger = RunLogger()
        run_logger.start_run(
            patient_id=patient_id,
            session_id=req.session_id,
            source="api",
            mode="live",
            metadata={"initial_step_id": req.step_id},
        )
        session_loggers[req.session_id] = run_logger

    # === Run the agent ===
    result = graph.invoke(
        {
            "user_input": req.user_message,
            "patient_id": patient_id,
        },
        config=config,
    )

    reasoning_time = round(time.time() - start_time, 3)

    # === Extract relevant info ===
    message = result.get("response", "...")
    emotion = getattr(result.get("patient_profile", None), "current_emotional_state", "base")
    topic_info = result.get("last_topic", {})
    topic = topic_info.get("sub", "general") if isinstance(topic_info, dict) else "general"

    # === Persist run info ===
    run_logger.log_turn(result, req.user_message)

    # === Return unified JSON ===
    return MessageResponse(
        message=message,
        reasoning_time=reasoning_time,
        emotion=emotion,
        topic=topic,
        timestamp=datetime.utcnow().isoformat()
    )
