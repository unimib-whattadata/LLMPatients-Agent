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
from agent.core.prompt_builder import build_prompt
from agent.core.llm_runner import create_llm_runner
from agent.core.patient_profile import PatientProfile
from langchain_core.runnables import RunnableLambda
from sentence_transformers import SentenceTransformer, util

# === Configure Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]

# === Load Environment ===
env_path = ROOT_DIR / "config" / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# === Load Persona Path ===
PATIENT_PATH = ROOT_DIR / "data" / "patients" / "john_wayne.json"
with open(PATIENT_PATH, "r") as f:
    PATIENT = json.load(f)

# === Initialize LLM Runner ===
llm_runner = create_llm_runner()

# === Load SentenceTransformer ===
st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# === Load Topic Tree JSON ===
TOPIC_PATH = ROOT_DIR / "data" / "topics_tree.json"
with open(TOPIC_PATH, "r") as f:
    TOPIC_TREE = json.load(f)


# === Flatten enriched topics JSON into a list of dicts ===
def flatten_topics(topics_json):
    flat = []
    for top_topic, content in topics_json.items():
        for sub_topic, desc in content.items():
            if sub_topic == "metadata":  # skip metadata
                continue
            flat.append({"top": top_topic, "sub": sub_topic, "desc": desc})
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
    history: list = []  # list of full turns
    summary: str = ""  # rolling summary of older turns


# === Build Nodes ===
def load_profile(state):
    logger.info("🔄 Loading patient profile...")
    profile = PatientProfile.from_file(str(PATIENT_PATH))
    if not hasattr(profile, "current_emotional_state"):
        profile.current_emotional_state = "neutral, guarded tone"
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
                "score": 0.0,
            },
            "last_topic": state.last_topic,
        }

    # Encode therapist input
    text_emb = st_model.encode(state.user_input.strip(), convert_to_tensor=True)

    # Compute similarity to each subtopic
    scores = {
        key: util.cos_sim(text_emb, emb).item() for key, emb in TOPIC_EMBEDDINGS.items()
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
            "score": best_score,
        }
        logger.info(
            f"🧠 Detected topic: {topic['top']} → {topic['sub']} (score={topic['score']:.3f})"
        )

    logger.info(f"📌 State update → intent_topic={topic}, last_topic={topic}")
    return {"intent_topic": topic, "last_topic": topic}


def generate_response(state):
    logger.info("💬 Generating response to therapist input...")
    result = llm_runner.generate(prompt=state.prompt)
    logger.info(f"✅ Response generated: {result}")
    return {"response": result}


def update_memory(state):
    """
    Append the latest therapist–patient exchange to memory.
    If >5 turns, fold older ones into the long-term summary.
    Also updates the patient's emotional tone.
    """
    logger.info("🧠 Entering update_memory()")

    # === 1. Append new turn ===
    new_turn = {
        "therapist": state.user_input,
        "patient": state.response,
        "topic": state.intent_topic,
    }
    state.history.append(new_turn)

    logger.info(f"🧾 Added new turn. Total turns: {len(state.history)}")
    logger.debug(f"🧩 New turn content: {json.dumps(new_turn, indent=2)}")

    # === 2. Summarize older turns ===
    if len(state.history) > 5:
        old_turns = state.history[:-5]
        old_text = "\n".join(
            [f"Therapist: {h['therapist']}\nPatient: {h['patient']}" for h in old_turns]
        )
        logger.info("📝 Summarizing older conversation turns into long-term memory...")
        summary_update = llm_runner.generate(
            prompt=f"Summarize the following therapy dialogue into a concise memory that preserves meaning, tone, and themes:\n\n{old_text}"
        )
        summary_update = summary_update.strip()
        logger.info(
            f"🧾 Summary update (chars={len(summary_update)}): {summary_update[:120]}..."
        )

        state.summary += "\n" + summary_update
        state.history = state.history[-5:]
        logger.info(
            f"✅ Folded old turns. New history len={len(state.history)} | Summary len={len(state.summary)}"
        )

    # === 3. Extract emotional tone ===
    try:
        tone_prompt = (
            f"Based on the patient's latest reply below, describe their current emotional tone "
            f"in one short, clinician-style phrase (e.g., 'anxious and defensive', 'sad but receptive', 'flat affect and withdrawn').\n\n"
            f"Patient reply:\n{state.response}"
        )
        tone_summary = llm_runner.generate(prompt=tone_prompt).strip()
        prev_tone = getattr(state.patient_profile, "current_emotional_state", "unknown")
        state.patient_profile.current_emotional_state = tone_summary
        logger.info(f"🫀 Emotional tone updated: '{prev_tone}' → '{tone_summary}'")

    except Exception as e:
        logger.warning(f"⚠️ Could not extract emotional tone: {e}")
        state.patient_profile.current_emotional_state = "unspecified"

    # === 4. Inspect and return ===
    logger.info(f"📊 Summary length: {len(state.summary)} chars")
    logger.info(f"📈 History length: {len(state.history)} turns")
    for i, h in enumerate(state.history, 1):
        logger.debug(
            f"   🗣️ Turn {i}: Therapist='{h['therapist'][:40]}...' | Patient='{h['patient'][:40]}...'"
        )

    return {
        "history": state.history,
        "summary": state.summary,
        "patient_profile": state.patient_profile,
    }


def display_response(state):
    logger.info("Displaying response:")
    logger.info(f"\n Patient: {state.response}\n")
    logger.info(
        f"📜 Current emotional tone: {state.patient_profile.current_emotional_state}"
    )
    logger.info(
        f"🕓 Turns so far: {len(state.history)} | Summary length: {len(state.summary)} chars\n"
    )

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
