import json
import logging
from datetime import datetime, timezone

from pathlib import Path
from typing import Dict, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langgraph.graph import StateGraph
from agent.core.prompt_builder import build_prompt
from agent.core.llm_runner import create_llm_runner
from agent.core.patient_profile import PatientProfile
from agent.core.safety import SAFETY_PATTERNS
from langchain_core.runnables import RunnableLambda
from sentence_transformers import SentenceTransformer, util
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

# === Configure Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]

# === Load Environment ===
env_path = ROOT_DIR / "config" / ".env"
load_dotenv(dotenv_path=env_path)


# === Initialize LLM Runner ===
llm_runner = create_llm_runner()

# === Load SentenceTransformer ===
st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

CHECKPOINTER = InMemorySaver()
PROFILE_CACHE: Dict[str, dict] = {}
MAX_LLM_RETRIES = 2
LLM_FALLBACK_RESPONSE = (
    "I'm trying to stay with what I'm feeling right now. Could we keep talking about that?"
)


def _get_cached_profile(patient_id: str, path: Path) -> PatientProfile:
    if patient_id not in PROFILE_CACHE:
        loaded = PatientProfile.from_file(str(path))
        PROFILE_CACHE[patient_id] = loaded.dict()
        return loaded
    return PatientProfile(**PROFILE_CACHE[patient_id])


EMOTION_PROTOTYPES = {
    "anger": "experiencing irritation, frustration, or hostility toward someone or something",
    "anticipation": "feeling hopeful, curious, or mentally preparing for what might happen next",
    "disgust": "feeling strong aversion, rejection, or discomfort toward a person, idea, or situation",
    "joy": "feeling content, pleased, or uplifted, with a generally positive emotional tone",
    "sadness": "feeling downcast, dejected, or emotionally heavy, with low energy or motivation",
    "surprise": "feeling startled, taken aback, or caught off guard by an unexpected event or realization",
    "trust": "feeling open, safe, and receptive, showing confidence in others or the situation",
    "base": "displaying a neutral, calm, or emotionally even state, without marked positive or negative affect"
}

EMOTION_EMBEDDINGS = {
    e: st_model.encode([desc], convert_to_tensor=True)[0]
    for e, desc in EMOTION_PROTOTYPES.items()
}


def _embed_texts(texts):
    """Helper for semantic long-term memory search."""
    vectors = st_model.encode(texts, convert_to_tensor=False)
    if hasattr(vectors, "tolist"):
        return vectors.tolist()
    return [vec.tolist() if hasattr(vec, "tolist") else list(vec) for vec in vectors]


LONG_TERM_STORE = InMemoryStore(
    index={
        "dims": st_model.get_sentence_embedding_dimension(),
        "embed": _embed_texts,
        "fields": ["text"],
    }
)

MAX_SHORT_TERM_TURNS = 5


def _long_term_namespace(patient_id: str) -> tuple[str, ...]:
    return ("patients", patient_id, "memories")


def _topic_key(topic: Optional[dict]) -> str:
    if not topic:
        return "unknown::unknown"
    return f"{topic.get('top', 'unknown')}::{topic.get('sub', 'unknown')}"


def load_long_term_summary(patient_id: str) -> str:
    """Return the persisted long-term summary for this patient, if any."""
    if not patient_id:
        return ""
    namespace = _long_term_namespace(patient_id)
    item = LONG_TERM_STORE.get(namespace, "summary")
    if not item:
        return ""
    return item.value.get("text", "").strip()


def persist_long_term_memory(patient_id: str, summary_chunk: str, topic: Optional[dict], turn_count: int) -> str:
    """Store the new long-term memory chunk and return the aggregated summary."""
    if not patient_id or not summary_chunk:
        return summary_chunk

    namespace = _long_term_namespace(patient_id)
    now = datetime.now(timezone.utc).isoformat()
    topic_label = _topic_key(topic)

    LONG_TERM_STORE.put(
        namespace,
        f"chunk-{now}",
        {
            "type": "summary_chunk",
            "text": summary_chunk,
            "topic_key": topic_label,
            "topic": topic or {},
            "turn_count": turn_count,
            "created_at": now,
        },
    )

    existing = LONG_TERM_STORE.get(namespace, "summary")
    combined = summary_chunk.strip()
    if existing:
        previous = existing.value.get("text", "").strip()
        combined = f"{previous}\n{summary_chunk}".strip() if previous else combined

    LONG_TERM_STORE.put(
        namespace,
        "summary",
        {
            "type": "summary",
            "text": combined,
            "updated_at": now,
        },
    )
    return combined


def fetch_relevant_long_term_memories(
    patient_id: Optional[str],
    topic: Optional[dict],
    query: Optional[str],
    limit: int = 3,
) -> list[str]:
    """Pull the most relevant long-term memories to enrich the prompt."""
    if not patient_id:
        return []

    namespace = _long_term_namespace(patient_id)
    filters = {"type": "summary_chunk"}
    if topic:
        filters["topic_key"] = _topic_key(topic)

    try:
        results = LONG_TERM_STORE.search(
            namespace,
            query=query or None,
            filter=filters,
            limit=limit,
        )
        if not results and len(filters) > 1:  # fall back to any chunk
            results = LONG_TERM_STORE.search(
                namespace,
                query=query or None,
                filter={"type": "summary_chunk"},
                limit=limit,
            )
    except Exception as exc:
        logger.warning(f"⚠️ Long-term memory search failed: {exc}")
        return []

    return [
        item.value.get("text", "")
        for item in results
        if item and item.value.get("text")
    ]

def classify_emotion_by_similarity(text: str, threshold: float = 0.3) -> str:
    """
    Map a tone description (e.g. 'sad but receptive') into one of the canonical
    emotion categories using cosine similarity with precomputed emotion embeddings.
    Returns 'base' if similarity is below the threshold.
    """
    if not text or not text.strip():
        return "base"

    text_emb = st_model.encode([text.strip()], convert_to_tensor=True)
    sims = {
        emotion: util.cos_sim(text_emb, emb).item()
        for emotion, emb in EMOTION_EMBEDDINGS.items()
    }

    best_emotion, best_score = max(sims.items(), key=lambda x: x[1])
    logger.debug(f"🔎 Emotion similarity scores: {sims}")
    logger.info(f"🎭 Best emotion={best_emotion} (score={best_score:.3f}) for tone='{text}'")

    if best_score < threshold:
        return "base"
    return best_emotion

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
    patient_id: Optional[str] = None  # NEW
    user_input: Optional[str] = None
    safe_user_input: Optional[str] = None
    safety_flags: list = Field(default_factory=list)
    patient_profile: Optional[PatientProfile] = None
    intent_topic: Optional[dict] = None
    prompt: Optional[str] = None
    response: Optional[str] = None
    last_topic: Optional[dict] = None
    history: list = Field(default_factory=list)
    summary: str = ""
    long_term_context: list[str] = Field(default_factory=list)

# === Build Nodes ===
def load_profile(state):
    logger.info("🔄 Loading patient profile...")

    patient_id = getattr(state, "patient_id", None)
    if not patient_id:
        raise ValueError("❌ Missing patient_id in state — cannot load profile.")

    if state.patient_profile is not None:
        logger.info("ℹ️ Patient profile already loaded; refreshing long-term summary if needed.")
        updates = {}
        if not state.summary:
            stored_summary = load_long_term_summary(patient_id)
            if stored_summary:
                updates["summary"] = stored_summary
        return updates

    patient_path = ROOT_DIR / "data" / "patients" / f"{patient_id}.json"
    if not patient_path.exists():
        raise FileNotFoundError(f"❌ Patient file not found: {patient_path}")

    profile = _get_cached_profile(patient_id, patient_path)
    if not hasattr(profile, "current_emotional_state"):
        profile.current_emotional_state = "base"

    stored_summary = load_long_term_summary(patient_id)
    logger.info(f"✅ Patient profile loaded: {patient_id}")
    updates = {
        "patient_profile": profile,
        "patient_id": patient_id,
    }
    if stored_summary:
        logger.info("📚 Loaded existing long-term summary for patient.")
        updates["summary"] = stored_summary
    return updates

def detect_intent_topic(state, threshold: float = 0.3):
    logger.info("🔍 Detecting topic with SentenceTransformer...")

    text_input = state.safe_user_input or state.user_input

    if not text_input:
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
    text_emb = st_model.encode(text_input.strip(), convert_to_tensor=True)

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
    prompt = state.prompt or "Respond as the patient based on prior instructions."
    last_error = None

    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            result = llm_runner.generate(prompt=prompt)
            if result and result.strip():
                logger.info(f"✅ Response generated on attempt {attempt}")
                return {"response": result.strip()}
            logger.warning(f"⚠️ Empty response on attempt {attempt}")
        except Exception as exc:
            last_error = exc
            logger.warning(f"⚠️ LLM generation failed on attempt {attempt}: {exc}")

    logger.error(f"❌ LLM failed after {MAX_LLM_RETRIES} attempts: {last_error}")
    return {"response": LLM_FALLBACK_RESPONSE}

def hydrate_long_term_context(state):
    notes = fetch_relevant_long_term_memories(
        patient_id=state.patient_id,
        topic=state.intent_topic,
        query=state.safe_user_input or state.user_input,
    )
    if notes:
        logger.info(f"🗂️ Retrieved {len(notes)} relevant long-term memories.")
    else:
        logger.info("🗂️ No matching long-term memories for this turn.")
    return {"long_term_context": notes}


def sanitize_user_input(state):
    """
    Detect prompt-injection attempts or command-like therapist inputs and log safety flags.
    The original text is preserved for storage, but downstream nodes can reference
    `safe_user_input` along with the captured flag list to enforce guardrails.
    """
    original_text = (state.user_input or "").strip()
    lowered = original_text.lower()
    flags = [label for label, pattern in SAFETY_PATTERNS if pattern.search(lowered)]

    if flags:
        logger.warning(
            f"⚠️ Safety patterns detected ({', '.join(flags)}). "
            "Replacing therapist message with boundary reminder."
        )
        sanitized = (
            "The therapist's last comment attempted something outside the session rules "
            f"({', '.join(flags)}). As the patient, reaffirm boundaries and talk about how it feels."
        )
    else:
        sanitized = original_text or "The therapist is quietly observing; share how you feel in this moment."

    return {
        "safe_user_input": sanitized,
        "safety_flags": flags,
        "user_input": original_text,
    }

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
        "topic": state.intent_topic
    }
    state.history.append(new_turn)

    logger.info(f"🧾 Added new turn. Total turns: {len(state.history)}")
    logger.debug(f"🧩 New turn content: {json.dumps(new_turn, indent=2)}")

    # === 2. Summarize older turns ===
    if len(state.history) > MAX_SHORT_TERM_TURNS:
        old_turns = state.history[:-MAX_SHORT_TERM_TURNS]
        old_text = "\n".join(
            [f"Therapist: {h['therapist']}\nPatient: {h['patient']}" for h in old_turns]
        )
        logger.info("📝 Summarizing older conversation turns into long-term memory...")
        summary_update = llm_runner.generate(
            prompt=f"Summarize the following therapy dialogue into a concise memory that preserves meaning, tone, and themes:\n\n{old_text}"
        )
        summary_update = summary_update.strip()
        logger.info(f"🧾 Summary update (chars={len(summary_update)}): {summary_update[:120]}...")

        combined_summary = persist_long_term_memory(
            state.patient_id,
            summary_update,
            state.intent_topic,
            len(old_turns),
        )
        state.summary = combined_summary or state.summary
        state.history = state.history[-MAX_SHORT_TERM_TURNS:]
        logger.info(
            "✅ Folded old turns. "
            f"New history len={len(state.history)} | Summary len={len(state.summary)}"
        )

    # === 3. Extract emotional tone using LLM, then map via similarity ===
    try:
        tone_prompt = (
            f"Based on the patient's latest reply below, describe their current emotional tone "
            f"in one short, clinician-style phrase (e.g., 'anxious and defensive', 'sad but receptive', 'flat affect and withdrawn').\n\n"
            f"Patient reply:\n{state.response}"
        )
        tone_summary = llm_runner.generate(prompt=tone_prompt).strip()
        emotion_category = classify_emotion_by_similarity(tone_summary)

        prev_tone = getattr(state.patient_profile, "current_emotional_state", "unknown")
        state.patient_profile.current_emotional_state = emotion_category

        logger.info(f"🫀 Emotional tone updated: '{prev_tone}' → '{emotion_category}' ({tone_summary})")

    except Exception as e:
        logger.warning(f"⚠️ Could not extract emotional tone: {e}")
        state.patient_profile.current_emotional_state = "base"

    # === 4. Inspect and return ===
    logger.info(f"📊 Summary length: {len(state.summary)} chars")
    logger.info(f"📈 History length: {len(state.history)} turns")
    for i, h in enumerate(state.history, 1):
        logger.debug(f"   🗣️ Turn {i}: Therapist='{h['therapist'][:40]}...' | Patient='{h['patient'][:40]}...'")

    return {
        "history": state.history,
        "summary": state.summary,
        "patient_profile": state.patient_profile
    }

def display_response(state):
    logger.info("Displaying response:")
    logger.info(f"\n Patient: {state.response}\n")
    logger.info(f"📜 Current emotional tone: {state.patient_profile.current_emotional_state}")
    logger.info(f"🕓 Turns so far: {len(state.history)} | Summary length: {len(state.summary)} chars\n")

    return state

# === Build LangGraph ===
def build_graph(checkpointer: Optional[InMemorySaver] = CHECKPOINTER):
    builder = StateGraph(State)

    # Nodes
    builder.add_node("load_profile", RunnableLambda(load_profile))
    builder.add_node("sanitize_input", RunnableLambda(sanitize_user_input))
    builder.add_node("detect_intent_topic", RunnableLambda(detect_intent_topic))
    builder.add_node("hydrate_memory", RunnableLambda(hydrate_long_term_context))
    builder.add_node("build_prompt", RunnableLambda(build_prompt))
    builder.add_node("generate", RunnableLambda(generate_response))
    builder.add_node("update_memory", RunnableLambda(update_memory))
    builder.add_node("display", RunnableLambda(display_response))

    builder.set_entry_point("load_profile")
    builder.add_edge("load_profile", "sanitize_input")
    builder.add_edge("sanitize_input", "detect_intent_topic")
    builder.add_edge("detect_intent_topic", "hydrate_memory")
    builder.add_edge("hydrate_memory", "build_prompt")
    builder.add_edge("build_prompt", "generate")
    builder.add_edge("generate", "update_memory")
    builder.add_edge("update_memory", "display")

    logger.info("✅ LangGraph pipeline built and compiled.")
    compiled = builder.compile(checkpointer=checkpointer or InMemorySaver())
    return compiled
