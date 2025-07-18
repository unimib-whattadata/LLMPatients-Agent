import os
import re
import json
import logging

from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda

from dotenv import load_dotenv
from core.patient_profile import PatientProfile
from core.prompt_builder import build_prompt
from core.llm_runner import create_llm_runner

# === Configure Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Load Environment ===
env_path = Path(__file__).resolve().parent.parent / "config" / ".env"
load_dotenv(dotenv_path=env_path)

# === Load Persona Path ===
PATIENT_PATH = Path("../data/patients/juanita_delgado.json")
with open(PATIENT_PATH, "r") as f:
    PATIENT = json.load(f)

# === Initialize LLM Runner ===
llm_runner = create_llm_runner()

# === LangGraph State ===
class State(BaseModel):
    user_input: Optional[str] = None
    patient_profile: Optional[PatientProfile] = None
    intent_topic: Optional[dict] = None
    prompt: Optional[str] = None
    response: Optional[str] = None

# === Build Nodes ===
def load_profile(state):
    logger.info("🔄 Loading patient profile...")
    profile = PatientProfile.from_file(str(PATIENT_PATH))
    logger.info("✅ Patient profile loaded.")
    return {"patient_profile": profile}

def load_prompt_template(path: str) -> str:
    logger.info(f"📄 Loading prompt template from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def detect_intent_topic(state):
    logger.info("🔍 Detecting intent and topic...")

    # Get path from .env
    prompt_rel_path = os.getenv("INTENT_TOPIC_PROMPT_PATH", "prompts/intent_topic.txt")
    
    # Compute absolute path relative to this file
    prompt_abs_path = (Path(__file__).resolve().parent.parent / prompt_rel_path).resolve()
    logger.info(f"📄 Loading prompt template from: {prompt_abs_path}")

    with open(prompt_abs_path, "r", encoding="utf-8") as f:
        template = f.read()

    formatted_prompt = template.format(therapist_input=state.user_input.strip())
    output = llm_runner.generate(prompt=formatted_prompt)
    logger.info(f"🧾 Raw model output:\n{output}")

    # Try to extract JSON safely
    match = re.search(r'{.*?}', output, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            logger.info(f"🧠 Detected intent and topic: {parsed}")
            return {"intent_topic": parsed}
        except json.JSONDecodeError as e:
            logger.warning(f"[WARNING] JSON parse error: {e}\nMatched:\n{match.group(0)}")
    else:
        logger.warning(f"[WARNING] No JSON object found in output.")

    return {"intent_topic": {"intent": "unknown", "topic": "unknown"}}

def generate_response(state):
    logger.info("💬 Generating response to therapist input...")
    result = llm_runner.generate(prompt=state.prompt)
    logger.info("✅ Response generated.")
    return {"response": result}

def display_response(state):
    logger.info("🖨️ Displaying response:")
    print(f"\n🧠 Juanita: {state.response.strip()}\n")
    return state

# === Build LangGraph ===
def build_graph():
    builder = StateGraph(State)

    builder.add_node("load_profile", RunnableLambda(load_profile))
    builder.add_node("detect_intent_topic", RunnableLambda(detect_intent_topic))
    builder.add_node("build_prompt", RunnableLambda(build_prompt))
    builder.add_node("generate", RunnableLambda(generate_response))
    builder.add_node("display", RunnableLambda(display_response))

    builder.set_entry_point("load_profile")
    builder.add_edge("load_profile", "detect_intent_topic")
    builder.add_edge("detect_intent_topic", "build_prompt")
    builder.add_edge("build_prompt", "generate")
    builder.add_edge("generate", "display")

    logger.info("✅ LangGraph pipeline built and compiled.")
    return builder.compile()