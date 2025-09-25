import os
import re
import json
import torch
import logging

from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from core.prompt_builder import build_prompt
from core.llm_runner import create_llm_runner
from core.patient_profile import PatientProfile
from langchain_core.runnables import RunnableLambda
from sentence_transformers import SentenceTransformer, util

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

# === Device for embeddings (MPS if available, else CPU) ===
device = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"⚙️ Using device for embeddings: {device}")

# === Load SentenceTransformer ===
st_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# === Load Topic Tree JSON ===
TOPIC_PATH = Path("../data/topics_tree.json")
with open(TOPIC_PATH, "r") as f:
    TOPIC_TREE = json.load(f)

# === Flatten enriched topics JSON into a list of dicts ===
def flatten_topics(topics_json):
    flat = []
    for top_topic, content in topics_json.items():
        for sub_topic, desc in content.items():
            if sub_topic == "metadata":  # skip metadata
                continue
            flat.append({
                "top": top_topic,
                "sub": sub_topic,
                "desc": desc
            })
    return flat

# === Build embeddings ===
TOPIC_EMBEDDINGS = {
    f"{t['top']} → {t['sub']}": st_model.encode([t["desc"]], convert_to_tensor=True)[0]
    for t in flatten_topics(TOPIC_TREE)
}

# === LangGraph State ===
class State(BaseModel):
    user_input: Optional[str] = None
    patient_profile: Optional[PatientProfile] = None
    intent_topic: Optional[dict] = None
    prompt: Optional[str] = None
    response: Optional[str] = None
    last_topic: Optional[dict] = None
    history: list = []              # list of full turns
    summary: str = ""               # rolling summary of older turns

# === Build Nodes ===
def load_profile(state):
    logger.info("🔄 Loading patient profile...")
    profile = PatientProfile.from_file(str(PATIENT_PATH))
    logger.info("✅ Patient profile loaded.")
    return {"patient_profile": profile}

def detect_intent_topic(state, threshold: float = 0.3):
    logger.info("🔍 Detecting topic with SentenceTransformer...")

    if not state.user_input:
        return {
            "intent_topic": {
                "intent": "unknown",
                "top": "unknown",
                "sub": "unknown",
                "score": 0.0
            },
            "last_topic": state.last_topic
        }

    # Encode therapist input
    text_emb = st_model.encode(state.user_input.strip(), convert_to_tensor=True)

    # Compute similarity to each subtopic
    scores = {
        key: util.cos_sim(text_emb, emb).item()
        for key, emb in TOPIC_EMBEDDINGS.items()
    }

    # Pick best match
    best_key, best_score = max(scores.items(), key=lambda x: x[1])
    top, sub = best_key.split(" → ")

    # === TODO: Add explicit check for "generic utterances"
    # e.g., if state.user_input.lower() in {"how?", "and then?", "what do you mean?"}
    # then force continuation with state.last_topic

    # Decide whether to reuse previous topic
    if best_score < threshold and state.last_topic:
        logger.info(
            f"↪️ Low similarity ({best_score:.3f} < {threshold}). "
            f"Continuing previous topic: {state.last_topic['top']} → {state.last_topic['sub']}"
        )
        topic = state.last_topic
    else:
        topic = {
            "intent": "topic_detection",
            "top": top,
            "sub": sub if best_score >= threshold else "general",
            "score": best_score
        }
        logger.info(f"🧠 Detected topic: {topic['top']} → {topic['sub']} (score={topic['score']:.3f})")

    logger.info(f"📌 State update → intent_topic={topic}, last_topic={topic}")
    return {"intent_topic": topic, "last_topic": topic}

def generate_response(state):
    logger.info("💬 Generating response to therapist input...")
    result = llm_runner.generate(prompt=state.prompt)
    logger.info(f"✅ Response generated: {result}")
    return {"response": result}

def update_memory(state):
    """
    Append latest turn to history.
    If >5 turns, fold oldest 5 into summary and keep last 5 verbatim.
    """
    state.history.append({
        "therapist": state.user_input,
        "patient": state.response,
        "topic": state.intent_topic
    })

    logger.info(f"🧾 Memory before folding: {len(state.history)} turns, summary length={len(state.summary)} chars")

    if len(state.history) > 5:
        old_turns = state.history[:-5]
        old_text = "\n".join(
            [f"T: {h['therapist']} | P: {h['patient']}" for h in old_turns]
        )
        logger.info("📝 Summarizing older conversation turns into memory...")
        summary_update = llm_runner.generate(
            prompt=f"Summarize the following therapy dialogue into a concise memory that preserves meaning, tone, and key topics:\n\n{old_text}"
        )
        state.summary += "\n" + summary_update.strip()
        state.history = state.history[-5:]

        logger.info("✅ Memory updated (older turns folded into summary).")

    logger.info(f"📌 State update → history_len={len(state.history)}, summary_len={len(state.summary)}")
    if state.history:
        last = state.history[-1]
        logger.info(f"   → Last turn: T='{last['therapist']}' | P='{last['patient']}' | Topic={last['topic']}")

    return state

def display_response(state):
    logger.info("Displaying response:")
    print(f"\n Juanita: {state.response}\n")
    return state

# === Build LangGraph ===
def build_graph(initial=True):
    builder = StateGraph(State)

    # Nodes
    builder.add_node("load_profile", RunnableLambda(load_profile))
    builder.add_node("detect_intent_topic", RunnableLambda(detect_intent_topic))
    builder.add_node("build_prompt", RunnableLambda(build_prompt))
    builder.add_node("generate", RunnableLambda(generate_response))
    builder.add_node("update_memory", RunnableLambda(update_memory))
    builder.add_node("display", RunnableLambda(display_response))

    # First run: load profile from disk
    if initial:
        builder.set_entry_point("load_profile")
        builder.add_edge("load_profile", "detect_intent_topic")
    else:
        builder.set_entry_point("detect_intent_topic")

    # Common edges
    builder.add_edge("detect_intent_topic", "build_prompt")
    builder.add_edge("build_prompt", "generate")
    builder.add_edge("generate", "update_memory")
    builder.add_edge("update_memory", "display")

    logger.info("✅ LangGraph pipeline built and compiled.")
    return builder.compile()