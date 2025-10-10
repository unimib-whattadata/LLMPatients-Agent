from fastapi import FastAPI
from pydantic import BaseModel
from agent.core.langgraph_builder import build_graph
from datetime import datetime
import time

app = FastAPI(title="PsyLLM Patient Agent API")

# Keep agent state in memory (for prototype)
sessions = {}

# === Request Schema ===
class MessageRequest(BaseModel):
    session_id: str
    user_input: str

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

    # Retrieve or initialize session
    state = sessions.get(req.session_id)
    if not state:
        graph = build_graph(initial=True)
    else:
        graph = build_graph(initial=False)

    # === Track reasoning time ===
    start_time = time.time()

    # Run the agent
    if not state:
        result = graph.invoke({"user_input": req.user_input})
    else:
        result = graph.invoke({
            **state,
            "user_input": req.user_input
        })

    reasoning_time = round(time.time() - start_time, 3)

    # === Extract relevant info ===
    message = result.get("response", "...")  # ← corrected
    emotion = getattr(result.get("patient_profile", None), "current_emotional_state", "base")
    topic_info = result.get("last_topic", {})
    topic = topic_info["sub"] if isinstance(topic_info, dict) and "sub" in topic_info else "general"

    # Save state for continuity
    sessions[req.session_id] = result

    # === Return unified JSON ===
    return MessageResponse(
        message=message,
        reasoning_time=reasoning_time,
        emotion=emotion,
        topic=topic,
        timestamp=datetime.utcnow().isoformat()
    )