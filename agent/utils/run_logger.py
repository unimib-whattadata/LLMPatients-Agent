"""Session logger that snapshots LangGraph state between therapist turns."""

import copy
import json
import os
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

RUNS_BASE_DIR = Path("tests") / "runs"
RUNS_BASE_DIR.mkdir(parents=True, exist_ok=True)


def _serialize_message(msg: BaseMessage) -> Dict[str, str]:
    """Convert a LangChain message into a JSON-friendly payload."""
    if isinstance(msg, HumanMessage):
        msg_type = "human"
    elif isinstance(msg, AIMessage):
        msg_type = "ai"
    else:
        msg_type = msg.__class__.__name__.lower()
    return {"type": msg_type, "content": msg.content}


def _deserialize_message(payload: Dict[str, str]) -> BaseMessage:
    """Reconstruct LangChain message objects from stored payloads."""
    msg_type = payload.get("type", "human")
    content = payload.get("content", "")
    if msg_type == "ai":
        return AIMessage(content=content)
    return HumanMessage(content=content)


def _serialize_messages(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    """Vectorized helper for message serialization."""
    return [_serialize_message(msg) for msg in messages]


def _deserialize_messages(payload: List[Dict[str, str]]) -> List[BaseMessage]:
    """Vectorized helper for message hydration."""
    return [_deserialize_message(item) for item in payload]


def _state_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    """Persist only the state fields required to resume a conversation."""
    snapshot = {
        "therapist_id": state.get("therapist_id"),
        "session_id": state.get("session_id"),
        "summary": state.get("summary", ""),
        "session_reflection": state.get("session_reflection", ""),
        "history": state.get("history", []),
        "long_term_context": state.get("long_term_context", []),
        "episodic_context": state.get("episodic_context", []),
        "last_topic": state.get("last_topic"),
        "topic_similarity": state.get("topic_similarity"),
        "total_turns": state.get("total_turns"),
        "last_episode_turn": state.get("last_episode_turn", 0),
        "messages": _serialize_messages(state.get("messages", [])),
        "core_emotion": state.get("core_emotion"),
        "emotion_intensity": state.get("emotion_intensity"),
        "emotion_state": state.get("emotion_state", {}),
        "emotion_event": state.get("emotion_event"),
        "emotion_salience": state.get("emotion_salience"),
        "low_salience_streak": state.get("low_salience_streak", 0),
    }
    return snapshot


def _hydrate_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild a usable state dict from a stored snapshot."""
    if not snapshot:
        return {}
    hydrated = copy.deepcopy(snapshot)
    if "messages" in hydrated:
        hydrated["messages"] = _deserialize_messages(hydrated.get("messages", []))
    return hydrated


def _env_flag(name: str) -> bool:
    """Return True if the env var looks truthy."""
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _sanitize_identifier(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def migrate_legacy_runs(base_dir: Path) -> None:
    """Coalesce older single-run files into therapist-scoped session logs."""
    for legacy_path in list(base_dir.glob("*.json")):
        try:
            data = json.loads(legacy_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "sessions" in data and "therapist_id" in data:
            continue
        if "run_id" not in data:
            continue
        therapist_id = data.get("therapist_id", "therapist0")
        session = {
            "session_id": data.get("session_id") or data["run_id"],
            "run_id": data.get("run_id"),
            "patient_id": data.get("patient_id"),
            "source": data.get("source"),
            "mode": data.get("mode"),
            "started_at": data.get("started_at"),
            "ended_at": data.get("ended_at"),
            "metadata": data.get("metadata", {}),
            "turns": data.get("turns", []),
            "final_state": data.get("final_state") or {},
        }
        dest_path = base_dir / f"{therapist_id}.json"
        if dest_path.exists():
            try:
                dest_data = json.loads(dest_path.read_text(encoding="utf-8"))
            except Exception:
                dest_data = {"therapist_id": therapist_id, "sessions": []}
        else:
            dest_data = {"therapist_id": therapist_id, "sessions": []}
        dest_data["sessions"].append(session)
        dest_path.write_text(json.dumps(dest_data, indent=2, ensure_ascii=False), encoding="utf-8")
        legacy_path.unlink(missing_ok=True)


migrate_legacy_runs(RUNS_BASE_DIR)


class RunLogger:
    """JSON-backed logger that tracks every turn for a therapist's sessions."""
    _counter = count(1)

    def __init__(self, therapist_id: str, base_dir: Optional[Path] = None):
        self.therapist_id = therapist_id or "therapist0"
        self.base_dir = Path(base_dir or RUNS_BASE_DIR)
        self.file_path = self.base_dir / f"{self.therapist_id}.json"
        self.transcript_enabled = True
        self._transcript_dir = self.base_dir / "transcripts"
        self._transcript_path: Optional[Path] = None
        self._current_session_id: Optional[str] = None
        self.data = self._load()
        self.current_session_index: Optional[int] = None

    def _load(self) -> Dict[str, Any]:
        """Load or initialize the therapist's consolidated session file."""
        if self.file_path.exists():
            try:
                return json.loads(self.file_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"therapist_id": self.therapist_id, "sessions": []}

    def start_run(
        self,
        *,
        patient_id: str,
        session_id: str,
        source: str,
        mode: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Open a new session record and return its run identifier."""
        run_id = self._generate_run_id()
        safe_session = _sanitize_identifier(session_id)
        self._current_session_id = session_id
        if self.transcript_enabled:
            self._transcript_path = self._transcript_dir / f"{self.therapist_id}__{safe_session}.jsonl"
        session = {
            "session_id": session_id,
            "run_id": run_id,
            "patient_id": patient_id,
            "source": source,
            "mode": mode,
            "started_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
            "turns": [],
            "final_state": {},
        }
        self.data["sessions"].append(session)
        self.current_session_index = len(self.data["sessions"]) - 1
        self._persist()
        return run_id

    def log_turn(self, state: Dict[str, Any], therapist_input: str) -> None:
        """Append a turn with sanitized/raw inputs plus summary state."""
        if self.current_session_index is None:
            raise RuntimeError("RunLogger.start_run must be called before logging turns.")
        session = self.data["sessions"][self.current_session_index]
        turn_entry = {
            "turn_index": len(session["turns"]) + 1,
            "timestamp": datetime.utcnow().isoformat(),
            "therapist_input_raw": therapist_input,
            "therapist_input_safe": state.get("safe_user_input"),
            "patient_response": state.get("response"),
            "detected_topic": state.get("last_topic"),
            "intent_topic": state.get("intent_topic"),
            "current_emotion": getattr(
                state.get("patient_profile"), "current_emotional_state", "unknown"
            ),
            "emotion_intensity": state.get("emotion_intensity"),
            "safety_flags": state.get("safety_flags", []),
            "long_term_context": state.get("long_term_context", []),
            "summary_so_far": state.get("summary", ""),
            "history_length": len(state.get("history", [])),
            "total_turns": state.get("total_turns"),
        }
        session["turns"].append(turn_entry)
        session["last_updated_at"] = datetime.utcnow().isoformat()
        session["final_state"] = _state_snapshot(state)
        self._persist()
        if self.transcript_enabled and self._transcript_path:
            transcript_entry = {
                "turn_index": turn_entry["turn_index"],
                "timestamp": turn_entry["timestamp"],
                "session_id": session.get("session_id"),
                "patient_id": session.get("patient_id"),
                "therapist_id": self.therapist_id,
                "therapist_input_raw": therapist_input,
                "therapist_input_safe": state.get("safe_user_input"),
                "patient_response": state.get("response"),
            }
            _append_jsonl(self._transcript_path, transcript_entry)

    def finalize(self, state: Optional[Dict[str, Any]] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        """Seal the current session and optionally attach final metadata."""
        if self.current_session_index is None:
            return
        session = self.data["sessions"][self.current_session_index]
        session["ended_at"] = datetime.utcnow().isoformat()
        if state:
            session["final_state"] = _state_snapshot(state)
            session["final_summary"] = state.get("summary", "")
        if extra:
            session.setdefault("extra", {}).update(extra)
        self._persist()
        self.current_session_index = None
        self._current_session_id = None
        self._transcript_path = None

    def restore_state(self, patient_id: str) -> Dict[str, Any]:
        """Return the last saved snapshot for the specified patient, if any."""
        for session in reversed(self.data.get("sessions", [])):
            if session.get("patient_id") == patient_id and session.get("final_state"):
                return _hydrate_snapshot(session["final_state"])
        return {}

    def _generate_run_id(self) -> str:
        """Produce a sortable run identifier (UTC timestamp + counter)."""
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        idx = next(self._counter)
        return f"{timestamp}_{idx:04d}"

    def _persist(self) -> None:
        """Write the therapist's session ledger back to disk."""
        self.file_path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
