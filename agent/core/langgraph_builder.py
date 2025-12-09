import json
import logging
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, wait, ALL_COMPLETED
from datetime import datetime, timezone

from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import atexit

from langgraph.graph import StateGraph
from agent.core.prompt_builder import build_prompt
from agent.core.llm_runner import create_llm_runner
from agent.core.emotion_model import EMOTIONS, compute_emotional_state
from agent.core.patient_profile import PatientProfile
from agent.core.safety import SAFETY_PATTERNS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from sentence_transformers import SentenceTransformer, util
from langgraph.checkpoint.memory import MemorySaver
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

CHECKPOINTER = MemorySaver()
PROFILE_CACHE: Dict[str, dict] = {}
MAX_LLM_RETRIES = 2
LLM_FALLBACK_RESPONSE = (
    "I'm trying to stay with what I'm feeling right now. Could we keep talking about that?"
)
SUMMARY_BATCH_SIZE = 3
SUMMARY_EXECUTOR = ThreadPoolExecutor(max_workers=2)
SUMMARY_TASKS: Dict[str, List[Future]] = defaultdict(list)
SUMMARY_TIMEOUT_SECONDS = 10
MAX_SHORT_TERM_TURNS = 5
MAX_MESSAGE_WINDOW = 10


def _shutdown_summary_executor():
    """Drain pending summary futures and close the executor on interpreter shutdown."""
    for patient_id, futures in SUMMARY_TASKS.items():
        done, not_done = wait(futures, timeout=SUMMARY_TIMEOUT_SECONDS, return_when=ALL_COMPLETED)
        for fut in done:
            try:
                fut.result()
            except Exception as exc:
                logger.warning(f"⚠️ Summary future error during shutdown ({patient_id}): {exc}")
        for fut in not_done:
            logger.warning(f"⚠️ Summary future still running for {patient_id}; cancelling.")
            fut.cancel()
    SUMMARY_EXECUTOR.shutdown(wait=False)


atexit.register(_shutdown_summary_executor)


def _get_cached_profile(patient_id: str, path: Path) -> PatientProfile:
    """Load a patient profile from disk and memoize it for subsequent requests."""
    if patient_id not in PROFILE_CACHE:
        loaded = PatientProfile.from_file(str(path))
        PROFILE_CACHE[patient_id] = loaded.dict()
        return loaded
    return PatientProfile(**PROFILE_CACHE[patient_id])


def _trait_baseline_from_profile(profile: PatientProfile) -> Dict[str, float]:
    """Return a clamped baseline vector for the patient's affective systems."""
    dynamics = getattr(profile, "EmotionDynamics", None)
    baseline = getattr(dynamics, "trait_baseline", None) if dynamics else None
    if not baseline:
        return dict(DEFAULT_TRAIT_BASELINE)
    normalized = {}
    for emotion in EMOTIONS:
        value = baseline.get(emotion, baseline.get(emotion.lower(), 0.5))
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.5
        normalized[emotion] = max(0.0, min(1.0, numeric))
    return normalized


def _volatility_from_profile(profile: PatientProfile) -> str:
    dynamics = getattr(profile, "EmotionDynamics", None)
    if dynamics and getattr(dynamics, "volatility_level", None):
        level = dynamics.volatility_level.lower()
        if level in ("low", "medium", "high"):
            return level
    fallback = getattr(profile, "volatility_level", None)
    if fallback in {"low", "medium", "high"}:
        return fallback
    return "medium"


def _detect_context_event(text: str, safety_flags: List[str]) -> str:
    """Map therapist actions to deterministic context modifiers."""
    if safety_flags:
        return "boundary"
    lowered = text.lower().strip()
    if not lowered:
        return "neutral"
    for event, keywords in CONTEXT_EVENT_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return event
    return "neutral"


def _infer_core_emotion(profile: PatientProfile) -> str:
    metadata = getattr(profile, "Metadata", None)
    if metadata:
        core = getattr(metadata, "coreEmotion", None)
        if core in EMOTION_PROTOTYPES:
            return core
    diagnoses = getattr(profile, "PsychiatricDiagnoses", []) or []
    joined = " ".join(diagnoses).lower()
    if "depress" in joined:
        return "sadness"
    if "anx" in joined:
        return "anticipation"
    if "ptsd" in joined or "trauma" in joined:
        return "sadness"
    return "base"


def _blend_scores_with_baseline(scores: dict, core: str, intensity: float) -> dict:
    blended = {}
    for emotion, score in scores.items():
        boost = 0.0
        if emotion == core:
            boost += 0.25 * intensity
        else:
            boost -= 0.08 * intensity
        blended[emotion] = max(0.0, score + boost)
    return blended


def _adjust_intensity(current: float, emotion: str) -> float:
    if emotion in POSITIVE_EMOTIONS:
        return max(0.0, current - 0.03)
    if emotion in NEGATIVE_EMOTIONS:
        return min(1.0, current + 0.02)
    # drift slowly toward midpoint
    if current > 0.5:
        return current - 0.01
    elif current < 0.5:
        return current + 0.01
    return current


def _collect_completed_summaries(patient_id: str, state):
    """Merge finished summary futures into the running state summary buffer."""
    futures = SUMMARY_TASKS.get(patient_id, [])
    if not futures:
        return

    remaining = []
    new_chunks = []
    for fut in futures:
        if fut.done():
            try:
                summary_text = fut.result()
                if summary_text:
                    new_chunks.append(summary_text)
            except Exception as exc:
                logger.warning(f"⚠️ Summary future failed for {patient_id}: {exc}")
        else:
            remaining.append(fut)

    SUMMARY_TASKS[patient_id] = remaining
    if new_chunks:
        addition = "\n".join(new_chunks)
        state.summary = (state.summary + "\n" + addition).strip() if state.summary else addition


def _schedule_summary_job(patient_id: str, chunk: list, topic: Optional[dict]):
    """Fire-and-forget a background task that summarizes a chunk of conversation turns."""
    chunk_text = _format_chunk_text(chunk)
    turn_count = len(chunk)
    future = SUMMARY_EXECUTOR.submit(
        _summarize_chunk,
        patient_id,
        chunk_text,
        topic,
        turn_count,
    )
    SUMMARY_TASKS[patient_id].append(future)
    logger.info(f"📨 Scheduled async summary for {patient_id} (turns={turn_count}).")


def _format_chunk_text(chunk: list) -> str:
    """Format a batch of turns into the alternating Therapist/Patient text block expected by the LLM."""
    return "\n".join(
        f"Therapist: {h['therapist']}\nPatient: {h['patient']}"
        for h in chunk
    )


def _summarize_chunk(patient_id: str, chunk_text: str, topic: Optional[dict], turn_count: int) -> str:
    """Call the LLM to summarize a chunk and persist the result as long-term memory."""
    prompt = (
        "You are maintaining a patient's long-term therapy memory. Summarize the dialogue below in 2-3 natural sentences that capture:\n"
        "- Concrete events or stressors mentioned\n"
        "- Emotional tone shifts and trust toward the therapist\n"
        "- Any unresolved questions or worries to revisit\n"
        "Keep the summary expressive yet concise.\n\n"
        f"{chunk_text}"
    )
    try:
        summary_update = llm_runner.generate(prompt=prompt).strip()
    except Exception as exc:
        logger.warning(f"⚠️ Async summary generation failed: {exc}")
        return ""

    if summary_update:
        persist_long_term_memory(patient_id, summary_update, topic, turn_count)
    return summary_update


def _messages_to_turns(messages: List[BaseMessage]) -> list:
    """Convert alternating Human/AI messages to turn dicts."""
    turns = []
    last_human = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            last_human = msg.content
        elif isinstance(msg, AIMessage) and last_human is not None:
            turns.append({
                "therapist": last_human,
                "patient": msg.content,
                "topic": None,
            })
            last_human = None
    return turns


def _turns_to_messages(turns: list) -> List[BaseMessage]:
    """Flatten structured turns back into a LangChain message list."""
    msgs: List[BaseMessage] = []
    for turn in turns:
        msgs.append(HumanMessage(content=turn.get("therapist", "")))
        msgs.append(AIMessage(content=turn.get("patient", "")))
    return msgs


def _chunk_turns(turns: list, size: int) -> List[list]:
    """Split turn history into equal-sized chunks for asynchronous summarization."""
    return [turns[i:i + size] for i in range(0, len(turns), size)]


FOLLOW_UP_CUES = {
    "what do you mean",
    "can you say more",
    "tell me more",
    "go on",
    "and then",
    "how so",
    "why",
    "uh huh",
    "i see",
    "okay",
    "ok",
    "mmh",
    "hmm",
    "right",
    "continue",
    "please continue",
}

DEFAULT_TRAIT_BASELINE = {emotion: 0.5 for emotion in EMOTIONS}

CONTEXT_EVENT_KEYWORDS = {
    "empathy": [
        "i'm here",
        "here for you",
        "understand",
        "hear you",
        "holding space",
        "take your time",
    ],
    "boundary": [
        "not appropriate",
        "can't do that",
        "won't do that",
        "we should stay focused",
        "stay in role",
        "remember our roles",
        "boundary",
    ],
    "abandonment_cue": [
        "wrap up",
        "time is up",
        "see you next week",
        "end here",
        "goodbye",
        "leave it there",
        "stop for today",
        "ending soon",
    ],
    "success_discussion": [
        "progress",
        "proud of you",
        "improvement",
        "doing better",
        "win",
        "success",
        "better lately",
        "great job",
    ],
}


def _is_follow_up(user_input: Optional[str]) -> bool:
    """Heuristically decide if the therapist merely nudged the patient to continue."""
    if not user_input:
        return False
    text = user_input.strip().lower()
    if len(text) <= 20:
        return True
    return any(cue in text for cue in FOLLOW_UP_CUES)


def _is_small_topic_shift(previous_score: float, new_score: float, epsilon: float = 0.05) -> bool:
    """Return True when cosine scores suggest the conversation is still on the same topic."""
    return previous_score and abs(previous_score - new_score) <= epsilon


def _build_topic_text(state) -> str:
    """Assemble the text snippet used for topic detection, preferring sanitized input and prior state."""
    pieces = []
    if state.safe_user_input:
        pieces.append(state.safe_user_input)
    elif state.user_input:
        pieces.append(state.user_input)
    if state.last_topic:
        pieces.append(f"(previous topic: {state.last_topic.get('top')} → {state.last_topic.get('sub')})")
    if state.history:
        last_patient = state.history[-1].get("patient")
        if last_patient:
            pieces.append(f"(patient previously said: {last_patient})")
    return " ".join(pieces).strip()


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

EMOTION_ORDER = ["anger", "disgust", "sadness", "base", "trust", "anticipation", "joy", "surprise"]
EMOTION_TRANSITION_SIM = {}
for a, emb_a in EMOTION_EMBEDDINGS.items():
    EMOTION_TRANSITION_SIM[a] = {}
    for b, emb_b in EMOTION_EMBEDDINGS.items():
        sim = util.cos_sim(emb_a, emb_b).item()
        if a == b:
            sim = 1.0
        EMOTION_TRANSITION_SIM[a][b] = sim

POSITIVE_EMOTIONS = {"trust", "anticipation", "joy", "surprise"}
NEGATIVE_EMOTIONS = {"anger", "disgust", "sadness"}


def _embed_texts(texts):
    """Vectorize arbitrary strings for use inside the in-memory similarity index."""
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
    """Return the namespace tuple under which a patient's memories are stored."""
    return ("patients", patient_id, "memories")


def _topic_key(topic: Optional[dict]) -> str:
    """Represent a topic dictionary as a consistent lookup key."""
    if not topic:
        return "unknown::unknown"
    return f"{topic.get('top', 'unknown')}::{topic.get('sub', 'unknown')}"


def load_long_term_summary(patient_id: str) -> str:
    """Return the persisted long-term summary for this patient, if any."""
    if not patient_id:
        return ""
    namespace = ("patients", patient_id, "memories")
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

def classify_emotion_by_similarity(text: str, threshold: float = 0.3):
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
        best_emotion = "base"
    return best_emotion, best_score, sims


def smooth_emotion_transition(previous: Optional[str], proposed: str, scores: dict, intensity: float, core: str) -> str:
    if not previous or previous in {"unknown", ""}:
        return proposed
    if previous == proposed:
        return proposed
    prev_score = scores.get(previous, -1.0)
    proposed_score = scores.get(proposed, -1.0)
    transition_sim = EMOTION_TRANSITION_SIM.get(previous, {}).get(proposed, 0.0)

    threshold_sim = 0.5 + 0.2 * (1 - intensity)  # higher intensity -> higher threshold
    threshold_delta = 0.12 + 0.1 * (1 - intensity)

    if proposed_score - prev_score >= threshold_delta or transition_sim >= threshold_sim:
        return proposed

    # Try to find intermediate emotion with high similarity to both
    candidates = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )
    for candidate, score in candidates:
        if candidate == previous:
            continue
        sim = EMOTION_TRANSITION_SIM.get(previous, {}).get(candidate, 0.0)
        if score >= proposed_score - 0.05 and sim >= threshold_sim:
            return candidate

    # Fall back to whichever emotion has highest blend of prev similarity and evidence
    blend_best = previous
    blend_score = prev_score
    for candidate, score in scores.items():
        sim = EMOTION_TRANSITION_SIM.get(previous, {}).get(candidate, 0.0)
        core_bonus = 0.2 * intensity if candidate == core else 0.0
        blended = 0.5 * sim + 0.4 * score + core_bonus
        if blended > blend_score + 0.05:
            blend_score = blended
            blend_best = candidate
    return blend_best

# === Load Topic Tree JSON ===
TOPIC_PATH = ROOT_DIR / "data" / "topics_tree.json"
with open(TOPIC_PATH, "r") as f:
    TOPIC_TREE = json.load(f)

# === Flatten enriched topics JSON into a list of dicts ===
def flatten_topics(topics_json):
    """Flatten the nested topics JSON into {top, sub, desc} records for embedding."""
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
    """Central LangGraph state container passed between nodes."""
    patient_id: Optional[str] = None  # NEW
    user_input: Optional[str] = None
    safe_user_input: Optional[str] = None
    safety_flags: list = Field(default_factory=list)
    patient_profile: Optional[PatientProfile] = None
    intent_topic: Optional[dict] = None
    prompt: Optional[str] = None
    response: Optional[str] = None
    last_topic: Optional[dict] = None
    topic_similarity: float = 0.0
    history: list = Field(default_factory=list)
    summary: str = ""
    long_term_context: list[str] = Field(default_factory=list)
    messages: List[BaseMessage] = Field(default_factory=list)
    total_turns: int = 0
    core_emotion: Optional[str] = None
    emotion_intensity: float = 0.7
    emotion_state: Dict[str, float] = Field(default_factory=dict)
    emotion_event: str = "neutral"

    class Config:
        arbitrary_types_allowed = True

# === Build Nodes ===
def load_profile(state):
    """Ensure the patient profile and long-term summary are attached to the state."""
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
        if state.core_emotion is None and hasattr(state.patient_profile, "core_emotion"):
            updates["core_emotion"] = state.patient_profile.core_emotion
        if state.emotion_intensity is None and hasattr(state.patient_profile, "emotion_intensity"):
            updates["emotion_intensity"] = state.patient_profile.emotion_intensity
        if not state.emotion_state and getattr(state.patient_profile, "emotion_state", None):
            updates["emotion_state"] = state.patient_profile.emotion_state
        return updates

    patient_path = ROOT_DIR / "data" / "patients" / f"{patient_id}.json"
    if not patient_path.exists():
        raise FileNotFoundError(f"❌ Patient file not found: {patient_path}")

    profile = _get_cached_profile(patient_id, patient_path)
    core = getattr(state, "core_emotion", None) or getattr(profile, "core_emotion", None)
    if not core:
        core = _infer_core_emotion(profile)
    intensity = getattr(state, "emotion_intensity", None)
    if intensity is None:
        intensity = getattr(profile, "emotion_intensity", 0.7)

    profile.__dict__["core_emotion"] = core
    profile.__dict__["emotion_intensity"] = float(intensity)
    if not hasattr(profile, "current_emotional_state"):
        profile.current_emotional_state = core
    if not getattr(profile, "emotion_state", None):
        profile.emotion_state = dict(DEFAULT_TRAIT_BASELINE)

    stored_summary = load_long_term_summary(patient_id)
    logger.info(f"✅ Patient profile loaded: {patient_id}")
    updates = {
        "patient_profile": profile,
        "patient_id": patient_id,
        "core_emotion": core,
        "emotion_intensity": float(intensity),
    }
    if stored_summary:
        logger.info("📚 Loaded existing long-term summary for patient.")
        updates["summary"] = stored_summary
    return updates


def update_emotional_state(state):
    """Synthesize momentary emotion vector from baseline, volatility, and context event."""
    profile = state.patient_profile
    if not profile:
        return {}

    therapist_text = state.safe_user_input or state.user_input or ""
    event = _detect_context_event(therapist_text, state.safety_flags)
    baseline = _trait_baseline_from_profile(profile)
    volatility = _volatility_from_profile(profile)
    previous = state.emotion_state or getattr(profile, "emotion_state", None)
    snapshot = compute_emotional_state(
        baseline,
        volatility_level=volatility,
        event=event,
        previous_state=previous,
    )

    dominant = sorted(snapshot.items(), key=lambda item: item[1], reverse=True)
    primary_emotion, peak_value = dominant[0]
    profile.current_emotional_state = primary_emotion.lower()
    profile.emotion_state = snapshot
    state.core_emotion = primary_emotion.lower()
    state.emotion_state = snapshot
    state.emotion_event = event
    state.emotion_intensity = peak_value
    profile.__dict__["emotion_intensity"] = peak_value

    logger.info(
        f"🎚️ Emotional systems updated → {primary_emotion}={peak_value:.2f} "
        f"(event={event}, volatility={volatility})"
    )
    return {
        "emotion_state": snapshot,
        "emotion_event": event,
        "emotion_intensity": peak_value,
        "core_emotion": state.core_emotion,
        "patient_profile": profile,
    }

def detect_intent_topic(state, threshold: float = 0.3):
    """Infer the most likely topic for the current turn via semantic similarity."""
    logger.info("🔍 Detecting topic with SentenceTransformer...")

    text_input = _build_topic_text(state)

    if not text_input:
        return {
            "intent_topic": {
                "intent": "unknown",
                "top": "unknown",
                "sub": "unknown",
                "score": 0.0
            },
            "last_topic": state.last_topic,
            "topic_similarity": 0.0,
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

    topic = {
        "intent": "topic_detection",
        "top": top,
        "sub": sub if best_score >= threshold else "general",
        "score": best_score
    }

    reuse_previous = (
        state.last_topic
        and (
            best_score < threshold
            or _is_follow_up(state.safe_user_input or state.user_input)
            or _is_small_topic_shift(state.topic_similarity, best_score)
        )
    )

    if reuse_previous:
        logger.info(
            f"↪️ Continuing previous topic: {state.last_topic['top']} → {state.last_topic['sub']}"
        )
        topic = state.last_topic
    else:
        logger.info(f"🧠 Detected topic: {topic['top']} → {topic['sub']} (score={topic['score']:.3f})")

    logger.info(f"📌 State update → intent_topic={topic}, last_topic={topic}")
    return {"intent_topic": topic, "last_topic": topic, "topic_similarity": topic["score"]}

def generate_response(state):
    """Call the configured LLM runner with retry/fallback logic."""
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


def append_messages(state):
    """Record the most recent therapist/patient exchange in structured message form."""
    human_text = (state.safe_user_input or state.user_input or "").strip()
    patient_text = (state.response or "").strip() or "[no reply]"

    messages = list(state.messages)
    if human_text:
        messages.append(HumanMessage(content=human_text))
    else:
        messages.append(HumanMessage(content="[Therapist silently observes]"))
    messages.append(AIMessage(content=patient_text))
    return {"messages": messages}


def trim_messages(state):
    """Keep a bounded recency window and summarize overflow batches."""
    messages = state.messages
    if len(messages) <= MAX_MESSAGE_WINDOW:
        return {}

    overflow_msgs = messages[:-MAX_MESSAGE_WINDOW]
    trimmed = messages[-MAX_MESSAGE_WINDOW:]
    turns = _messages_to_turns(overflow_msgs)
    leftover_turns = []

    if state.patient_id and turns:
        for chunk in _chunk_turns(turns, SUMMARY_BATCH_SIZE):
            if len(chunk) == SUMMARY_BATCH_SIZE:
                _schedule_summary_job(state.patient_id, chunk, state.intent_topic)
            else:
                leftover_turns.extend(chunk)
    else:
        leftover_turns = turns

    if leftover_turns:
        trimmed = _turns_to_messages(leftover_turns) + trimmed
        trimmed = trimmed[-MAX_MESSAGE_WINDOW:]

    return {"messages": trimmed}


def hydrate_long_term_context(state):
    """Retrieve long-term memories relevant to the therapist input/topic for grounding."""
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
        sanitized = original_text or "The therapist is quietly observing. Just wait for their next comment."

    state.safe_user_input = sanitized
    state.safety_flags = flags
    return {
        "safe_user_input": sanitized,
        "safety_flags": flags,
    }

def update_memory(state):
    """
    Append the latest therapist–patient exchange to memory.
    If the backlog exceeds thresholds, asynchronously fold older batches into long-term memory.
    Also updates the patient's emotional tone.
    """
    logger.info("🧠 Entering update_memory()")

    if state.patient_id:
        _collect_completed_summaries(state.patient_id, state)

    # === 1. Append new turn ===
    new_turn = {
        "therapist": state.user_input,
        "patient": state.response,
        "topic": state.intent_topic
    }
    state.history.append(new_turn)
    state.total_turns = (state.total_turns or 0) + 1

    logger.info(f"🧾 Added new turn. Total turns overall: {state.total_turns}")
    logger.debug(f"🧩 New turn content: {json.dumps(new_turn, indent=2)}")

    # === 2. Keep bounded short-term window ===
    if len(state.history) > MAX_SHORT_TERM_TURNS:
        state.history = state.history[-MAX_SHORT_TERM_TURNS:]

    # === 3. Update emotional tone (prefer synthesized vector; fallback to LLM heuristic) ===
    if state.emotion_state:
        dominant_emotion, dominant_value = max(state.emotion_state.items(), key=lambda item: item[1])
        prev_tone = getattr(state.patient_profile, "current_emotional_state", "unknown")
        state.patient_profile.current_emotional_state = dominant_emotion.lower()
        state.patient_profile.emotion_state = state.emotion_state
        state.patient_profile.__dict__["emotion_intensity"] = dominant_value
        state.core_emotion = dominant_emotion.lower()
        state.emotion_intensity = dominant_value
        logger.info(
            f"🫀 Emotional tone (model-driven): '{prev_tone}' → '{dominant_emotion.lower()}' "
            f"(intensity={dominant_value:.2f}, event={state.emotion_event})"
        )
    else:
        try:
            tone_prompt = (
                f"Based on the patient's latest reply below, describe their current emotional tone "
                f"in one short, clinician-style phrase (e.g., 'anxious and defensive', 'sad but receptive', 'flat affect and withdrawn').\n\n"
                f"Patient reply:\n{state.response}"
            )
            tone_summary = llm_runner.generate(prompt=tone_prompt).strip()
            proposed_emotion, _, emotion_scores = classify_emotion_by_similarity(tone_summary)
            core_emotion = getattr(state.patient_profile, "core_emotion", None) or state.core_emotion or _infer_core_emotion(state.patient_profile)
            intensity = getattr(state.patient_profile, "emotion_intensity", None)
            if intensity is None:
                intensity = state.emotion_intensity
            if intensity is None:
                intensity = 0.7
            blended_scores = _blend_scores_with_baseline(emotion_scores, core_emotion, intensity)
            proposed = max(blended_scores.items(), key=lambda item: item[1])[0]

            prev_tone = getattr(state.patient_profile, "current_emotional_state", "unknown")
            smoothed = smooth_emotion_transition(prev_tone, proposed, blended_scores, intensity, core_emotion)
            state.patient_profile.current_emotional_state = smoothed
            new_intensity = _adjust_intensity(float(intensity), smoothed)
            state.patient_profile.__dict__["emotion_intensity"] = new_intensity
            state.core_emotion = core_emotion
            state.emotion_intensity = new_intensity

            logger.info(
                f"🫀 Emotional tone updated: '{prev_tone}' → '{smoothed}' "
                f"(raw='{proposed_emotion}', intensity={new_intensity:.2f}, desc='{tone_summary}')"
            )

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
        "patient_profile": state.patient_profile,
        "total_turns": state.total_turns,
        "emotion_state": state.emotion_state,
    }

def display_response(state):
    """Log the agent's response and lightweight telemetry for observability."""
    logger.info("Displaying response:")
    logger.info(f"\n Patient: {state.response}\n")
    logger.info(f"📜 Current emotional tone: {state.patient_profile.current_emotional_state}")
    logger.info(f"🕓 Turns so far: {len(state.history)} | Summary length: {len(state.summary)} chars\n")

    return state

# === Build LangGraph ===
def build_graph(checkpointer: Optional[MemorySaver] = CHECKPOINTER):
    """Assemble and compile the LangGraph pipeline that powers the agent."""
    builder = StateGraph(State)

    # Nodes
    builder.add_node("load_profile", RunnableLambda(load_profile))
    builder.add_node("sanitize_input", RunnableLambda(sanitize_user_input))
    builder.add_node("detect_intent_topic", RunnableLambda(detect_intent_topic))
    builder.add_node("hydrate_memory", RunnableLambda(hydrate_long_term_context))
    builder.add_node("update_emotions", RunnableLambda(update_emotional_state))
    builder.add_node("build_prompt", RunnableLambda(build_prompt))
    builder.add_node("generate", RunnableLambda(generate_response))
    builder.add_node("append_messages", RunnableLambda(append_messages))
    builder.add_node("trim_messages", RunnableLambda(trim_messages))
    builder.add_node("update_memory", RunnableLambda(update_memory))
    builder.add_node("display", RunnableLambda(display_response))

    builder.set_entry_point("load_profile")
    builder.add_edge("load_profile", "sanitize_input")
    builder.add_edge("sanitize_input", "detect_intent_topic")
    builder.add_edge("detect_intent_topic", "hydrate_memory")
    builder.add_edge("hydrate_memory", "update_emotions")
    builder.add_edge("update_emotions", "build_prompt")
    builder.add_edge("build_prompt", "generate")
    builder.add_edge("generate", "append_messages")
    builder.add_edge("append_messages", "trim_messages")
    builder.add_edge("trim_messages", "update_memory")
    builder.add_edge("update_memory", "display")

    logger.info("✅ LangGraph pipeline built and compiled.")
    compiled = builder.compile(checkpointer=checkpointer or InMemorySaver())
    return compiled
