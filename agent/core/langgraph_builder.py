import hashlib
import json
import logging
import re
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, wait, ALL_COMPLETED
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import atexit

from langgraph.graph import StateGraph
from agent.core.prompt_builder import build_prompt
from agent.core.memory_store import JsonlMemoryStore
from agent.core.llm_runner import create_llm_runner
from agent.core.emotion_model import EMOTIONS, EVENT_SALIENCE, compute_emotional_state
from agent.core.patient_profile import (
    PatientProfile,
    PatientDetails,
    resolve_patient_profile_path,
)
from agent.core.safety import SAFETY_PATTERNS, FOLLOW_UP_CUES, CONTEXT_EVENT_KEYWORDS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from sentence_transformers import SentenceTransformer
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
SUMMARY_EXECUTOR = ThreadPoolExecutor(max_workers=2)
EPISODE_TASKS: Dict[tuple[str, str], List[Future]] = defaultdict(list)
SUMMARY_TIMEOUT_SECONDS = 10
MAX_SHORT_TERM_TURNS = 5
MAX_MESSAGE_WINDOW = 10
EPISODE_BATCH_SIZE = 5

MEMORY_DIR = ROOT_DIR / "data" / "memory"
MEMORY_STORE = JsonlMemoryStore(MEMORY_DIR)
MEMORY_CACHE_LOADED: Dict[tuple[str, str], Optional[float]] = {}
LATEST_SUMMARY_CACHE: Dict[tuple[str, str], str] = {}
LATEST_REFLECTION_CACHE: Dict[tuple[str, str], str] = {}

DEFAULT_TRAIT_BASELINE = {emotion: 0.5 for emotion in EMOTIONS}



def _shutdown_summary_executor():
    """Drain pending summary futures and close the executor on interpreter shutdown."""
    for key, futures in EPISODE_TASKS.items():
        done, not_done = wait(futures, timeout=SUMMARY_TIMEOUT_SECONDS, return_when=ALL_COMPLETED)
        for fut in done:
            try:
                fut.result()
            except Exception as exc:
                logger.warning(f"⚠️ Summary future error during shutdown ({key}): {exc}")
        for fut in not_done:
            logger.warning(f"⚠️ Summary future still running for {key}; cancelling.")
            fut.cancel()
    SUMMARY_EXECUTOR.shutdown(wait=False)


atexit.register(_shutdown_summary_executor)


def _get_cached_profile(patient_id: str, path: Path) -> PatientProfile:
    """Load a patient profile from disk and memoize it for subsequent requests."""
    if patient_id not in PROFILE_CACHE:
        loaded = PatientProfile.from_file(str(path))
        PROFILE_CACHE[patient_id] = loaded
        return loaded
    cached = PROFILE_CACHE[patient_id]
    if isinstance(cached, dict):
        cached = PatientProfile(**cached)
        PROFILE_CACHE[patient_id] = cached
    return cached


def _trait_baseline_from_profile(profile: PatientProfile) -> Dict[str, float]:
    """Return a clamped baseline vector for the patient's affective systems."""
    dynamics = getattr(profile, "emotionTraits", None) or getattr(profile, "EmotionDynamics", None)
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
    dynamics = getattr(profile, "emotionTraits", None) or getattr(profile, "EmotionDynamics", None)
    if dynamics and getattr(dynamics, "volatility_level", None):
        level = dynamics.volatility_level.lower()
        if level in ("low", "medium", "high"):
            return level
    fallback = getattr(profile, "volatility_level", None)
    if fallback in {"low", "medium", "high"}:
        return fallback
    return "medium"


def _detect_context_event(text: str, safety_flags: List[str], topic_changed: bool) -> tuple[str, float]:
    """Map therapist actions to deterministic context modifiers and salience."""
    if safety_flags:
        return "boundary", 0.85
    lowered = text.lower().strip()
    if not lowered:
        return "neutral", EVENT_SALIENCE.get("neutral", 0.2)

    event = "neutral"
    for candidate, keywords in CONTEXT_EVENT_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            event = candidate
            break

    salience = EVENT_SALIENCE.get(event, EVENT_SALIENCE["neutral"])
    # Length bump for disclosures
    if len(lowered) > 140:
        salience += 0.05
    # Topic change implies novelty
    if topic_changed:
        salience += 0.1
    salience = min(1.0, max(salience, 0.0))
    return event, salience


def _infer_core_emotion(profile: PatientProfile) -> str:
    details = getattr(profile, "details", None)
    tx = getattr(details, "treatmentsAndInterventions", None) if details else None
    diagnoses = tx.previousPsychiatricDiagnoses if tx else []
    clinical = getattr(details, "clinicalFunctioning", None) if details else None
    syndrome = None
    if clinical and getattr(clinical, "personalityAndSymptomAxis", None):
        syndrome = clinical.personalityAndSymptomAxis.personalitySyndrome

    joined = " ".join(diagnoses or [])
    if syndrome:
        joined += f" {syndrome}"
    if profile.brief_description:
        joined += f" {profile.brief_description}"
    joined = joined.lower()
    if "depress" in joined or "grief" in joined or "loss" in joined:
        return "PANIC_GRIEF"
    if "anx" in joined:
        return "FEAR"
    if "ptsd" in joined or "trauma" in joined:
        return "FEAR"
    return "SEEKING"


def _normalize_emotion_key(emotion: Optional[str]) -> str:
    if not emotion:
        return ""
    return (
        emotion.strip()
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .upper()
    )


def _memory_key(patient_id: Optional[str], therapist_id: Optional[str]) -> tuple[str, str]:
    """Return a stable key for per-therapist/per-patient memory caches."""
    return (therapist_id or "therapist0", patient_id or "unknown")


def _memory_namespace(patient_id: str, therapist_id: str) -> tuple[str, ...]:
    """Namespace for all memory artifacts tied to a therapist/patient pair."""
    return ("therapists", therapist_id, "patients", patient_id, "memories")


def _append_memory_record(record: dict) -> None:
    MEMORY_STORE.append(record)


def _index_memory_record(record: dict) -> None:
    patient_id = record.get("patient_id")
    therapist_id = record.get("therapist_id")
    if not patient_id or not therapist_id:
        return
    namespace = _memory_namespace(patient_id, therapist_id)
    record_id = record.get("id") or _record_fingerprint(record)
    LONG_TERM_STORE.put(namespace, record_id, record)


def _load_persisted_memories(patient_id: str, therapist_id: str) -> None:
    memory_key = _memory_key(patient_id, therapist_id)
    current_mtime = MEMORY_STORE.file_mtime(patient_id, therapist_id)
    cached_mtime = MEMORY_CACHE_LOADED.get(memory_key)
    if cached_mtime is not None and current_mtime is not None and current_mtime <= cached_mtime:
        return
    LATEST_SUMMARY_CACHE.pop(memory_key, None)
    LATEST_REFLECTION_CACHE.pop(memory_key, None)
    latest_summary = None
    latest_summary_time = None
    latest_reflection = None
    latest_reflection_time = None
    for record in MEMORY_STORE.iter_records(patient_id, therapist_id):
        _index_memory_record(record)
        record_type = record.get("type")
        if record_type == "long_term_summary":
            updated_at = record.get("updated_at")
            if updated_at and (latest_summary_time is None or updated_at > latest_summary_time):
                latest_summary_time = updated_at
                latest_summary = record.get("text", "")
        elif record_type == "session_reflection":
            created_at = record.get("created_at")
            if created_at and (latest_reflection_time is None or created_at > latest_reflection_time):
                latest_reflection_time = created_at
                latest_reflection = record.get("text", "")
    if latest_summary:
        LATEST_SUMMARY_CACHE[memory_key] = latest_summary
        namespace = _memory_namespace(patient_id, therapist_id)
        LONG_TERM_STORE.put(
            namespace,
            "summary",
            {"type": "long_term_summary", "text": latest_summary, "updated_at": latest_summary_time},
        )
    if latest_reflection:
        LATEST_REFLECTION_CACHE[memory_key] = latest_reflection
    MEMORY_CACHE_LOADED[memory_key] = current_mtime


def _record_fingerprint(record: dict) -> str:
    serialized = json.dumps(record, sort_keys=True, ensure_ascii=True)
    return f"{record.get('type', 'memory')}-{hashlib.sha1(serialized.encode('utf-8')).hexdigest()}"

def _collect_completed_episodes(memory_key: tuple[str, str]) -> None:
    """Drain completed episode futures to prevent unbounded task buildup."""
    futures = EPISODE_TASKS.get(memory_key, [])
    if not futures:
        return

    remaining = []
    for fut in futures:
        if fut.done():
            try:
                fut.result()
            except Exception as exc:
                logger.warning(f"⚠️ Episode future failed for {memory_key}: {exc}")
        else:
            remaining.append(fut)

    EPISODE_TASKS[memory_key] = remaining


def _schedule_episode_job(
    patient_id: str,
    therapist_id: str,
    session_id: str,
    chunk: list,
    topic: Optional[dict],
    turn_range: tuple[int, int],
    emotion_label: str,
    emotion_intensity: float,
    salience: float,
):
    """Fire-and-forget a background task that summarizes a chunk into an episode."""
    chunk_text = _format_chunk_text(chunk)
    turn_count = len(chunk)
    memory_key = _memory_key(patient_id, therapist_id)
    future = SUMMARY_EXECUTOR.submit(
        _summarize_episode,
        patient_id,
        therapist_id,
        session_id,
        chunk_text,
        topic,
        turn_range,
        turn_count,
        emotion_label,
        emotion_intensity,
        salience,
    )
    EPISODE_TASKS[memory_key].append(future)
    logger.info(
        f"📨 Scheduled async episode summary for {patient_id}/{therapist_id} "
        f"(turns={turn_count}, range={turn_range})."
    )


def _format_chunk_text(chunk: list) -> str:
    """Format a batch of turns into the alternating Therapist/Patient text block expected by the LLM."""
    return "\n".join(
        f"Therapist: {h['therapist']}\nPatient: {h['patient']}"
        for h in chunk
    )


def _summarize_episode(
    patient_id: str,
    therapist_id: str,
    session_id: str,
    chunk_text: str,
    topic: Optional[dict],
    turn_range: tuple[int, int],
    turn_count: int,
    emotion_label: str,
    emotion_intensity: float,
    salience: float,
) -> str:
    """Call the LLM to summarize a chunk into an episodic memory item."""
    prompt = (
        "You are maintaining episodic therapy memory. Summarize the dialogue below in 2-3 natural sentences that capture:\n"
        "- Concrete events or stressors mentioned\n"
        "- Emotional tone shifts, trust, or relational dynamics\n"
        "- Any unresolved questions or worries to revisit\n"
        "Write as a compact clinical note, but in plain language.\n\n"
        f"{chunk_text}"
    )
    try:
        summary_update = llm_runner.generate(prompt=prompt).strip()
    except Exception as exc:
        logger.warning(f"⚠️ Async episode generation failed: {exc}")
        return ""

    if summary_update:
        persist_episode_summary(
            patient_id=patient_id,
            therapist_id=therapist_id,
            session_id=session_id,
            summary_text=summary_update,
            topic=topic,
            turn_range=turn_range,
            turn_count=turn_count,
            emotion_label=emotion_label,
            emotion_intensity=emotion_intensity,
            salience=salience,
        )
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


def _is_follow_up(user_input: Optional[str]) -> bool:
    """Heuristically decide if the therapist merely nudged the patient to continue."""
    if not user_input:
        return False
    text = user_input.strip().lower()
    if len(text) <= 8:
        return True
    return any(cue in text for cue in FOLLOW_UP_CUES)


def _build_topic_text(state) -> str:
    """
    Assemble the text snippet used for topic detection.
    Combines recent therapist inputs with the latest patient reply for more signal.
    """
    def _clean_topic_text(text: str) -> str:
        if not text:
            return ""
        cleaned = text.lower().strip()
        cleaned = re.sub(r"[^\w\s'-]+", " ", cleaned)
        filler = [
            "hi", "hey", "hello", "good morning", "good afternoon", "good evening",
            "how are you", "how are things", "checking in", "how have you been",
            "how's it going", "you ok", "you okay", "thanks", "thank you"
        ]
        for phrase in filler:
            cleaned = cleaned.replace(phrase, " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    pieces = []

    # Last 2 therapist inputs (including current)
    therapist_texts = []
    current = (state.safe_user_input or state.user_input or "").strip()
    if current:
        therapist_texts.append(_clean_topic_text(current))
    if state.history:
        prev = [turn.get("therapist", "") for turn in state.history[-2:]]
        therapist_texts.extend([_clean_topic_text(t) for t in prev if t])
    if therapist_texts:
        pieces.append(" | ".join(t for t in therapist_texts if t))

    # Latest patient reply for context
    if state.history:
        last_patient = state.history[-1].get("patient")
        if last_patient:
            pieces.append(_clean_topic_text(last_patient))

    return " ".join(piece for piece in pieces if piece).strip()


PANKSEPP_LABELS = ["SEEKING", "FEAR", "RAGE", "LUST", "CARE", "PANIC_GRIEF", "PLAY"]


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


def _topic_key(topic: Optional[dict]) -> str:
    """Represent a topic dictionary as a consistent lookup key."""
    if not topic:
        return "unknown::unknown"
    return f"{topic.get('top', 'unknown')}::{topic.get('sub', 'unknown')}"


def load_long_term_summary(patient_id: str, therapist_id: str) -> str:
    """Return the persisted long-term summary for this therapist/patient pair."""
    if not patient_id or not therapist_id:
        return ""
    _load_persisted_memories(patient_id, therapist_id)
    memory_key = _memory_key(patient_id, therapist_id)
    cached = LATEST_SUMMARY_CACHE.get(memory_key)
    if cached:
        return cached.strip()
    namespace = _memory_namespace(patient_id, therapist_id)
    item = LONG_TERM_STORE.get(namespace, "summary")
    if not item:
        return ""
    return item.value.get("text", "").strip()


def load_latest_session_reflection(patient_id: str, therapist_id: str) -> str:
    """Return the most recent session reflection, if any."""
    if not patient_id or not therapist_id:
        return ""
    _load_persisted_memories(patient_id, therapist_id)
    memory_key = _memory_key(patient_id, therapist_id)
    cached = LATEST_REFLECTION_CACHE.get(memory_key)
    return cached.strip() if cached else ""


def persist_episode_summary(
    *,
    patient_id: str,
    therapist_id: str,
    session_id: str,
    summary_text: str,
    topic: Optional[dict],
    turn_range: tuple[int, int],
    turn_count: int,
    emotion_label: str,
    emotion_intensity: float,
    salience: float,
) -> dict:
    """Store a new episodic memory chunk and return its record."""
    if not patient_id or not therapist_id or not summary_text:
        return {}
    now = datetime.now(timezone.utc).isoformat()
    topic_label = _topic_key(topic)
    record = {
        "id": uuid4().hex,
        "type": "episode_summary",
        "patient_id": patient_id,
        "therapist_id": therapist_id,
        "session_id": session_id or "unknown",
        "text": summary_text.strip(),
        "topic_key": topic_label,
        "topic": topic or {},
        "turn_range": turn_range,
        "turn_count": turn_count,
        "emotion_label": emotion_label,
        "emotion_intensity": float(emotion_intensity),
        "salience": float(salience),
        "created_at": now,
    }
    _append_memory_record(record)
    _index_memory_record(record)
    return record


def persist_session_reflection(
    *,
    patient_id: str,
    therapist_id: str,
    session_id: str,
    reflection_text: str,
) -> dict:
    """Persist a session reflection and update caches."""
    if not patient_id or not therapist_id or not reflection_text:
        return {}
    now = datetime.now(timezone.utc).isoformat()
    record = {
        "id": uuid4().hex,
        "type": "session_reflection",
        "patient_id": patient_id,
        "therapist_id": therapist_id,
        "session_id": session_id or "unknown",
        "text": reflection_text.strip(),
        "created_at": now,
    }
    _append_memory_record(record)
    _index_memory_record(record)
    LATEST_REFLECTION_CACHE[_memory_key(patient_id, therapist_id)] = reflection_text.strip()
    return record


def persist_long_term_summary(
    *,
    patient_id: str,
    therapist_id: str,
    summary_text: str,
) -> dict:
    """Persist the rolling long-term summary."""
    if not patient_id or not therapist_id or not summary_text:
        return {}
    now = datetime.now(timezone.utc).isoformat()
    record = {
        "id": uuid4().hex,
        "type": "long_term_summary",
        "patient_id": patient_id,
        "therapist_id": therapist_id,
        "text": summary_text.strip(),
        "updated_at": now,
    }
    _append_memory_record(record)
    _index_memory_record(record)
    namespace = _memory_namespace(patient_id, therapist_id)
    LONG_TERM_STORE.put(
        namespace,
        "summary",
        {
            "type": "long_term_summary",
            "text": summary_text.strip(),
            "updated_at": now,
        },
    )
    LATEST_SUMMARY_CACHE[_memory_key(patient_id, therapist_id)] = summary_text.strip()
    return record


def fetch_relevant_episodic_memories(
    *,
    patient_id: Optional[str],
    therapist_id: Optional[str],
    topic: Optional[dict],
    query: Optional[str],
    limit: int = 3,
) -> list[str]:
    """Pull the most relevant episodic memories to enrich the prompt."""
    if not patient_id or not therapist_id:
        return []

    namespace = _memory_namespace(patient_id, therapist_id)
    filters = {"type": "episode_summary"}
    if topic:
        filters["topic_key"] = _topic_key(topic)

    try:
        results = LONG_TERM_STORE.search(
            namespace,
            query=query or None,
            filter=filters,
            limit=limit,
        )
        if not results and len(filters) > 1:
            results = LONG_TERM_STORE.search(
                namespace,
                query=query or None,
                filter={"type": "episode_summary"},
                limit=limit,
            )
    except Exception as exc:
        logger.warning(f"⚠️ Episodic memory search failed: {exc}")
        return []

    return [
        item.value.get("text", "")
        for item in results
        if item and item.value.get("text")
    ]

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
TOPIC_RECORDS = flatten_topics(TOPIC_TREE)
TOPIC_LABELS = [f"{t['top']} → {t['sub']}" for t in TOPIC_RECORDS]


def _coerce_topic_label(label: str) -> str:
    if not label:
        return "unknown"
    cleaned = label.strip()
    if cleaned.lower() == "unknown":
        return "unknown"
    if cleaned in TOPIC_LABELS:
        return cleaned
    normalized = " → ".join(p.strip() for p in re.split(r"[→>]", cleaned) if p.strip())
    return normalized if normalized in TOPIC_LABELS else "unknown"


def _label_to_topic(label: str, fallback: Optional[dict]) -> dict:
    if not label or label.lower() == "unknown":
        return fallback or {"intent": "topic_detection", "top": "unknown", "sub": "unknown", "score": 0.0}
    if "→" in label:
        parts = [p.strip() for p in label.split("→")]
    elif ">" in label:
        parts = [p.strip() for p in label.split(">")]
    else:
        parts = [label.strip(), "general"]
    if len(parts) == 1:
        parts.append("general")
    return {
        "intent": "topic_detection",
        "top": parts[0],
        "sub": parts[1],
        "score": 1.0,
    }


def classify_topic_and_emotion(
    therapist_text: str,
    patient_text: str,
    context_text: str = "",
) -> tuple[str, str]:
    """Single LLM call to classify topic label + patient emotion label."""
    if not therapist_text and not patient_text:
        return "unknown", "SEEKING"
    labels_text = "\n".join(f"- {label}" for label in TOPIC_LABELS)
    emotion_labels = ", ".join(PANKSEPP_LABELS)
    prompt = (
        "You are a classifier. Given the therapist message and context, "
        "return the best topic label from the list or 'unknown', and the patient's "
        f"likely emotion label from: {emotion_labels}. Respond ONLY with JSON in the form "
        '{"topic_label": "...", "emotion_label": "..."}.\n\n'
        f"Therapist message:\n{therapist_text.strip() or '[none]'}\n\n"
        f"Recent patient context:\n{patient_text.strip() or '[none]'}\n\n"
        f"Topic context:\n{context_text.strip() or '[none]'}\n\n"
        "Topic labels:\n"
        f"{labels_text}\n"
    )
    try:
        raw = llm_runner.generate(prompt=prompt).strip()
    except Exception as exc:
        logger.warning(f"⚠️ Joint classification failed: {exc}")
        return "unknown", "SEEKING"

    payload = {}
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            payload = json.loads(raw[start : end + 1])
    except Exception:
        payload = {}

    topic_label = payload.get("topic_label") if isinstance(payload, dict) else None
    emotion_label = payload.get("emotion_label") if isinstance(payload, dict) else None

    if not topic_label:
        match = re.search(r"topic_label\\s*[:=]\\s*([\\w\\s→>-]+)", raw, flags=re.IGNORECASE)
        if match:
            topic_label = match.group(1).strip()
    if not emotion_label:
        match = re.search(r"emotion_label\\s*[:=]\\s*([A-Za-z_/-]+)", raw, flags=re.IGNORECASE)
        if match:
            emotion_label = match.group(1).strip()

    topic_label = _coerce_topic_label(topic_label or "unknown")
    normalized = _normalize_emotion_key(emotion_label or "")
    emotion_label = normalized if normalized in PANKSEPP_LABELS else "SEEKING"
    return topic_label, emotion_label

# === LangGraph State ===
class State(BaseModel):
    """Central LangGraph state container passed between nodes."""
    patient_id: Optional[str] = None  # NEW
    therapist_id: Optional[str] = None
    session_id: Optional[str] = None
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
    session_reflection: str = ""
    long_term_context: list[str] = Field(default_factory=list)
    episodic_context: list[str] = Field(default_factory=list)
    messages: List[BaseMessage] = Field(default_factory=list)
    total_turns: int = 0
    last_episode_turn: int = 0
    core_emotion: Optional[str] = None
    emotion_intensity: float = 0.7
    emotion_state: Dict[str, float] = Field(default_factory=dict)
    emotion_event: str = "neutral"
    emotion_salience: float = 0.2
    low_salience_streak: int = 0
    classified_emotion: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

# === Build Nodes ===
def load_profile(state):
    """Ensure the patient profile and long-term summary are attached to the state."""
    logger.info("🔄 Loading patient profile...")

    patient_id = getattr(state, "patient_id", None)
    therapist_id = getattr(state, "therapist_id", None) or "therapist0"
    if not patient_id:
        raise ValueError("❌ Missing patient_id in state — cannot load profile.")

    if state.patient_profile is not None:
        logger.info("ℹ️ Patient profile already loaded; refreshing long-term summary if needed.")
        updates = {}
        if isinstance(state.patient_profile.details, dict):
            try:
                state.patient_profile.details = PatientDetails(**state.patient_profile.details)
            except Exception:
                state.patient_profile.details = PatientDetails()
        if not state.summary:
            stored_summary = load_long_term_summary(patient_id, therapist_id)
            if stored_summary:
                updates["summary"] = stored_summary
        if not state.session_reflection:
            stored_reflection = load_latest_session_reflection(patient_id, therapist_id)
            if stored_reflection:
                updates["session_reflection"] = stored_reflection
        if state.core_emotion is None and hasattr(state.patient_profile, "core_emotion"):
            updates["core_emotion"] = _normalize_emotion_key(state.patient_profile.core_emotion).lower()
        if state.emotion_intensity is None and hasattr(state.patient_profile, "emotion_intensity"):
            updates["emotion_intensity"] = state.patient_profile.emotion_intensity
        if not state.emotion_state and getattr(state.patient_profile, "emotion_state", None):
            updates["emotion_state"] = state.patient_profile.emotion_state
        if not state.therapist_id:
            updates["therapist_id"] = therapist_id
        return updates

    patients_dir = ROOT_DIR / "data" / "patients"
    patient_path = resolve_patient_profile_path(patient_id, patients_dir)

    profile = _get_cached_profile(patient_id, patient_path)
    if isinstance(profile.details, dict):
        profile.details = PatientDetails(**profile.details)
    core = getattr(state, "core_emotion", None) or getattr(profile, "core_emotion", None)
    if not core:
        core = _infer_core_emotion(profile)
    core = _normalize_emotion_key(core) or "SEEKING"
    intensity = getattr(state, "emotion_intensity", None)
    if intensity is None:
        intensity = getattr(profile, "emotion_intensity", 0.7)

    profile.__dict__["core_emotion"] = core
    profile.__dict__["emotion_intensity"] = float(intensity)
    if not hasattr(profile, "current_emotional_state"):
        profile.current_emotional_state = core.lower()
    if not getattr(profile, "emotion_state", None):
        profile.emotion_state = dict(DEFAULT_TRAIT_BASELINE)

    stored_summary = load_long_term_summary(patient_id, therapist_id)
    stored_reflection = load_latest_session_reflection(patient_id, therapist_id)
    logger.info(f"✅ Patient profile loaded: {patient_id}")
    updates = {
        "patient_profile": profile,
        "patient_id": patient_id,
        "therapist_id": therapist_id,
        "core_emotion": core,
        "emotion_intensity": float(intensity),
    }
    if stored_summary:
        logger.info("📚 Loaded existing long-term summary for patient.")
        updates["summary"] = stored_summary
    if stored_reflection:
        updates["session_reflection"] = stored_reflection
    return updates


def update_emotional_state(state):
    """Synthesize momentary emotion vector from baseline, volatility, and context event."""
    profile = state.patient_profile
    if not profile:
        return {}

    therapist_text = state.safe_user_input or state.user_input or ""
    topic_changed = bool(state.last_topic and state.intent_topic and state.last_topic != state.intent_topic)
    event, salience = _detect_context_event(therapist_text, state.safety_flags, topic_changed)
    baseline = _trait_baseline_from_profile(profile)
    volatility = _volatility_from_profile(profile)
    def _prior_bias_vector(label: Optional[str], baseline_vec: Dict[str, float]) -> Optional[Dict[str, float]]:
        if not label:
            return None
        key = _normalize_emotion_key(label)
        if key not in EMOTIONS:
            return None
        biased = dict(baseline_vec)
        biased[key] = max(baseline_vec.get(key, 0.5), 0.7)
        return biased

    previous = state.emotion_state or getattr(profile, "emotion_state", None)
    prior_bias = _prior_bias_vector(getattr(state, "classified_emotion", None), baseline)
    previous_state = prior_bias or previous
    snapshot = compute_emotional_state(
        baseline,
        volatility_level=volatility,
        event=event,
        previous_state=previous_state,
    )
    if prior_bias:
        blend_weight = 0.15
        for emotion in snapshot:
            blended = (1 - blend_weight) * snapshot[emotion] + blend_weight * prior_bias.get(emotion, baseline.get(emotion, 0.5))
            snapshot[emotion] = max(0.0, min(1.0, blended))

    # Gentle decay toward baseline when multiple low-salience turns occur.
    if salience < 0.3:
        state.low_salience_streak = (state.low_salience_streak or 0) + 1
    else:
        state.low_salience_streak = 0
    if state.low_salience_streak >= 2:
        for emotion in snapshot:
            delta = snapshot[emotion] - baseline.get(emotion, 0.5)
            snapshot[emotion] = max(0.0, min(1.0, snapshot[emotion] - 0.05 * delta))

    dominant = sorted(snapshot.items(), key=lambda item: item[1], reverse=True)
    primary_emotion, peak_value = dominant[0]
    secondary = dominant[1][1] if len(dominant) > 1 else primary_emotion
    intensity = 0.6 * peak_value + 0.4 * (secondary if isinstance(secondary, (int, float)) else peak_value)

    profile.emotion_state = snapshot
    state.core_emotion = primary_emotion.lower()
    state.emotion_state = snapshot
    state.emotion_event = event
    state.emotion_salience = salience
    state.emotion_intensity = intensity
    profile.__dict__["emotion_intensity"] = intensity

    logger.info(
        f"🎚️ Emotional systems updated → {primary_emotion}={peak_value:.2f} "
        f"(event={event}, salience={salience:.2f}, volatility={volatility})"
    )
    return {
        "emotion_state": snapshot,
        "emotion_event": event,
        "emotion_intensity": intensity,
        "emotion_salience": salience,
        "core_emotion": state.core_emotion,
        "patient_profile": profile,
        "low_salience_streak": state.low_salience_streak,
    }

def _is_greeting_or_checkin(text: str) -> bool:
    """Quick check for greeting/check-in phrases that should not force a new topic."""
    lowered = text.lower()
    simple = {"hi", "hey", "hello", "good morning", "good afternoon", "good evening"}
    if any(lowered.startswith(g) for g in simple):
        return True
    cues = ["how are you", "how are things", "checking in", "how have you been", "how's it going", "you ok", "you okay"]
    return any(cue in lowered for cue in cues)




def classify_topic_and_emotion_pre(state):
    """Use a single LLM call to classify topic + likely patient emotion before generation."""
    text_input = _build_topic_text(state)
    if not text_input:
        return {
            "intent_topic": {"intent": "topic_detection", "top": "unknown", "sub": "unknown", "score": 0.0},
            "last_topic": state.last_topic,
            "topic_similarity": 0.0,
            "classified_emotion": None,
        }

    therapist_text = (state.safe_user_input or state.user_input or "").strip()
    last_patient = ""
    if state.history:
        last_patient = state.history[-1].get("patient") or ""

    try:
        topic_label, emotion_label = classify_topic_and_emotion(
            therapist_text=therapist_text,
            patient_text=last_patient,
            context_text=text_input,
        )
    except Exception as exc:
        logger.warning(f"⚠️ Joint classification failed: {exc}")
        topic_label, emotion_label = "unknown", "SEEKING"

    greeting = _is_greeting_or_checkin(state.user_input or "")
    follow_up = _is_follow_up(state.safe_user_input or state.user_input)
    short_turn = len((therapist_text or "").split()) <= 3
    low_signal = greeting or follow_up or short_turn
    prev_topic = state.last_topic

    unknown_label = topic_label.lower() == "unknown"
    if unknown_label:
        if low_signal and prev_topic:
            topic = prev_topic
        else:
            topic = {"intent": "topic_detection", "top": "unknown", "sub": "unknown", "score": 0.0}
    else:
        topic = _label_to_topic(topic_label, None)

    if state.patient_profile and emotion_label:
        state.patient_profile.current_emotional_state = emotion_label.lower()

    return {
        "intent_topic": topic,
        "last_topic": topic,
        "topic_similarity": topic.get("score", 0.0),
        "classified_emotion": emotion_label,
    }

def generate_response(state):
    """Call the configured LLM runner with retry/fallback logic."""
    logger.info("💬 Generating response to therapist input...")
    prompt = state.prompt or "Respond as the patient based on prior instructions."
    last_error = None

    def _enforce_response_format(text: str) -> str:
        """Ensure non-spoken content is in (), spoken text otherwise."""
        if not text:
            return text
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return text.strip()

        thought_re = re.compile(r"^(thoughts?|thinking|internal|inner)\s*:\s*(.+)", re.IGNORECASE)
        desc_re = re.compile(r"^(description|narration|scene|action|setting)\s*:\s*(.+)", re.IGNORECASE)
        formatted = []
        for line in lines:
            line = re.sub(r"\*([^*]+)\*", r"(\1)", line)
            if re.match(r"^\*.*\*$", line):
                formatted.append(f"({line.strip('*').strip()})")
                continue
            if re.match(r"^\(.*\)$", line):
                formatted.append(line)
                continue
            match = thought_re.match(line)
            if match:
                formatted.append(f"({match.group(2).strip()})")
                continue
            match = desc_re.match(line)
            if match:
                formatted.append(f"({match.group(2).strip()})")
                continue
            formatted.append(line)
        return " ".join(formatted).strip()

    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            result = llm_runner.generate(prompt=prompt)
            if result and result.strip():
                logger.info(f"✅ Response generated on attempt {attempt}")
                return {"response": _enforce_response_format(result.strip())}
            logger.warning(f"⚠️ Empty response on attempt {attempt}")
        except Exception as exc:
            last_error = exc
            logger.warning(f"⚠️ LLM generation failed on attempt {attempt}: {exc}")

    logger.error(f"❌ LLM failed after {MAX_LLM_RETRIES} attempts: {last_error}")
    return {"response":  "I'm trying to stay with what I'm feeling right now. Could we keep talking about that?"}


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
    """Keep a bounded recency window for messages."""
    messages = state.messages
    if len(messages) <= MAX_MESSAGE_WINDOW:
        return {}

    overflow_msgs = messages[:-MAX_MESSAGE_WINDOW]
    trimmed = messages[-MAX_MESSAGE_WINDOW:]
    turns = _messages_to_turns(overflow_msgs)
    leftover_turns = []
    leftover_turns = turns

    if leftover_turns:
        trimmed = _turns_to_messages(leftover_turns) + trimmed
        trimmed = trimmed[-MAX_MESSAGE_WINDOW:]

    return {"messages": trimmed}


def hydrate_long_term_context(state):
    """Retrieve long-term memories relevant to the therapist input/topic for grounding."""
    notes = fetch_relevant_episodic_memories(
        patient_id=state.patient_id,
        therapist_id=state.therapist_id,
        topic=state.intent_topic,
        query=state.safe_user_input or state.user_input,
    )
    if notes:
        logger.info(f"🗂️ Retrieved {len(notes)} relevant episodic memories.")
    else:
        logger.info("🗂️ No matching episodic memories for this turn.")
    return {"long_term_context": notes, "episodic_context": notes}


def sanitize_user_input(state):
    """Detect prompt-injection attempts and log safety flags."""
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

    if state.patient_id and state.therapist_id:
        _collect_completed_episodes(_memory_key(state.patient_id, state.therapist_id))

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

    # === 2b. Summarize episodic batches ===
    if state.patient_id and state.therapist_id:
        last_episode_turn = state.last_episode_turn or 0
        turns_since = state.total_turns - last_episode_turn
        if turns_since >= EPISODE_BATCH_SIZE and len(state.history) >= EPISODE_BATCH_SIZE:
            chunk = state.history[-EPISODE_BATCH_SIZE:]
            turn_range = (state.total_turns - EPISODE_BATCH_SIZE + 1, state.total_turns)
            _schedule_episode_job(
                patient_id=state.patient_id,
                therapist_id=state.therapist_id,
                session_id=state.session_id or "unknown",
                chunk=chunk,
                topic=state.intent_topic,
                turn_range=turn_range,
                emotion_label=_normalize_emotion_key(state.core_emotion) or "SEEKING",
                emotion_intensity=state.emotion_intensity or 0.5,
                salience=state.emotion_salience or 0.2,
            )
            state.last_episode_turn = state.total_turns

    # === 3. No LLM calls here; topic/tone handled before generation. ===

    # === 4. Inspect and return ===
    logger.debug(f"📊 Summary length: {len(state.summary)} chars")
    logger.debug(f"📈 History length: {len(state.history)} turns")
    for i, h in enumerate(state.history, 1):
        logger.debug(f"   🗣️ Turn {i}: Therapist='{h['therapist'][:40]}...' | Patient='{h['patient'][:40]}...'")

    return {
        "history": state.history,
        "summary": state.summary,
        "patient_profile": state.patient_profile,
        "total_turns": state.total_turns,
        "emotion_state": state.emotion_state,
        "last_episode_turn": state.last_episode_turn,
    }

def display_response(state):
    """Log the agent's response and lightweight telemetry for observability."""
    logger.info("Displaying response:")
    logger.info(f"\n Patient: {state.response}\n")
    logger.info(f"📜 Current emotional tone: {state.patient_profile.current_emotional_state}")
    logger.info(f"🕓 Turns so far: {len(state.history)} | Summary length: {len(state.summary)} chars\n")

    return state


def _drain_episode_futures(patient_id: str, therapist_id: str) -> None:
    """Wait briefly for any pending episode summaries to finish."""
    memory_key = _memory_key(patient_id, therapist_id)
    futures = EPISODE_TASKS.get(memory_key, [])
    if not futures:
        return
    done, not_done = wait(futures, timeout=SUMMARY_TIMEOUT_SECONDS, return_when=ALL_COMPLETED)
    for fut in done:
        try:
            fut.result()
        except Exception as exc:
            logger.warning(f"⚠️ Episode future error during finalize ({memory_key}): {exc}")
    for fut in not_done:
        logger.warning(f"⚠️ Episode future still running for {memory_key}; cancelling.")
        fut.cancel()
    EPISODE_TASKS[memory_key] = []


def _load_session_episode_texts(patient_id: str, therapist_id: str, session_id: str) -> list[str]:
    texts = []
    for record in MEMORY_STORE.iter_records(patient_id, therapist_id):
        if record.get("type") != "episode_summary":
            continue
        if record.get("session_id") != session_id:
            continue
        text = record.get("text", "")
        if text:
            texts.append(text)
    return texts


def _generate_session_reflection(episode_texts: list[str], fallback_history: list[dict]) -> str:
    if episode_texts:
        bullets = "\n".join(f"- {text}" for text in episode_texts)
        context = f"Episodes:\n{bullets}"
    else:
        turns = "\n".join(
            f"Therapist: {h.get('therapist')}\nPatient: {h.get('patient')}"
            for h in fallback_history[-EPISODE_BATCH_SIZE:]
        )
        context = f"Recent turns:\n{turns}"

    prompt = (
        "You are producing a session reflection for a therapy patient. "
        "Write 4-6 sentences that the patient could have said about the session. "
        "Use first person (I/me), highlight key themes, emotional shifts, "
        "relational dynamics with the therapist, and any open questions. "
        "Keep it concise, plain, and faithful to what was said.\n\n"
        f"{context}"
    )
    try:
        return llm_runner.generate(prompt=prompt).strip()
    except Exception as exc:
        logger.warning(f"⚠️ Session reflection generation failed: {exc}")
        return ""


def _update_long_term_summary_from_reflection(
    patient_id: str,
    therapist_id: str,
    reflection_text: str,
) -> str:
    existing = load_long_term_summary(patient_id, therapist_id)
    prompt = (
        "You maintain a long-term therapy memory written in the patient's voice. "
        "Update the memory using the new session reflection below. "
        "Write 5-7 sentences in first person (I/me). Preserve stable facts and "
        "incorporate new developments or shifts, staying close to what was said. "
        "Keep it plain and informative for grounding future sessions.\n\n"
        f"Existing summary:\n{existing or '[none]'}\n\n"
        f"New session reflection:\n{reflection_text}"
    )
    try:
        updated = llm_runner.generate(prompt=prompt).strip()
    except Exception as exc:
        logger.warning(f"⚠️ Long-term summary update failed: {exc}")
        return existing
    if updated:
        persist_long_term_summary(
            patient_id=patient_id,
            therapist_id=therapist_id,
            summary_text=updated,
        )
        return updated
    return existing


def finalize_session_memory(state: dict) -> dict:
    """Finalize session-level reflection and long-term summary updates."""
    patient_id = state.get("patient_id")
    therapist_id = state.get("therapist_id") or "therapist0"
    session_id = state.get("session_id") or "unknown"
    if not patient_id:
        return state

    _drain_episode_futures(patient_id, therapist_id)
    episode_texts = _load_session_episode_texts(patient_id, therapist_id, session_id)
    reflection = _generate_session_reflection(episode_texts, state.get("history", []))
    if reflection:
        persist_session_reflection(
            patient_id=patient_id,
            therapist_id=therapist_id,
            session_id=session_id,
            reflection_text=reflection,
        )
        updated_summary = _update_long_term_summary_from_reflection(
            patient_id=patient_id,
            therapist_id=therapist_id,
            reflection_text=reflection,
        )
        state["session_reflection"] = reflection
        state["summary"] = updated_summary
    return state

# === Build LangGraph ===
def build_graph(checkpointer: Optional[MemorySaver] = CHECKPOINTER):
    """Assemble and compile the LangGraph pipeline that powers the agent."""
    builder = StateGraph(State)

    # Nodes
    builder.add_node("load_profile", RunnableLambda(load_profile))
    builder.add_node("sanitize_input", RunnableLambda(sanitize_user_input))
    builder.add_node("classify_topic_and_emotion", RunnableLambda(classify_topic_and_emotion_pre))
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
    builder.add_edge("sanitize_input", "classify_topic_and_emotion")
    builder.add_edge("classify_topic_and_emotion", "hydrate_memory")
    builder.add_edge("hydrate_memory", "update_emotions")
    builder.add_edge("update_emotions", "build_prompt")
    builder.add_edge("build_prompt", "generate")
    builder.add_edge("generate", "append_messages")
    builder.add_edge("append_messages", "trim_messages")
    builder.add_edge("trim_messages", "update_memory")
    builder.add_edge("update_memory", "display")

    logger.info("✅ LangGraph pipeline built and compiled.")
    compiled = builder.compile(checkpointer=checkpointer)
    return compiled
