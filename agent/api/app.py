from fastapi import FastAPI
from pydantic import BaseModel
from agent.core.langgraph_builder import build_graph
from datetime import datetime
import time

app = FastAPI(title="PsyLLM Patient Agent API")

graph = build_graph()

# === Request Schema ===
class MessageRequest(BaseModel):
    external_patient_id: str
    user_message: str
    session_id: str
    step_id: int

# === Response Schema ===
class MessageResponse(BaseModel):
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

    # === Return unified JSON ===
    return MessageResponse(
        message=message,
        reasoning_time=reasoning_time,
        emotion=emotion,
        topic=topic,
        timestamp=datetime.utcnow().isoformat()
    )
