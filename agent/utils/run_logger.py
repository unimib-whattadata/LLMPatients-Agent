"""Lightweight JSON logger that records each simulated therapy session."""

import json
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any, Dict, Optional


class RunLogger:
    """Accumulates per-turn metadata and persists it to disk for audits/tests."""
    _counter = count(1)

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir or Path("tests") / "runs")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.run_id: Optional[str] = None
        self.file_path: Optional[Path] = None
        self.data: Dict[str, Any] = {}

    def start_run(
        self,
        *,
        patient_id: str,
        session_id: str,
        source: str,
        mode: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Initialize a new log file and return its run identifier."""
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        run_index = next(self._counter)
        self.run_id = f"{timestamp}_{run_index:04d}"
        self.file_path = self.base_dir / f"{self.run_id}.json"
        self.data = {
            "run_id": self.run_id,
            "patient_id": patient_id,
            "session_id": session_id,
            "source": source,
            "mode": mode,
            "started_at": datetime.utcnow().isoformat(),
            "turns": [],
        }
        if metadata:
            self.data["metadata"] = metadata
        self._persist()
        return self.run_id

    def log_turn(self, state: Dict[str, Any], therapist_input: str):
        """Append a turn entry pulled from the LangGraph state."""
        if not self.run_id:
            raise RuntimeError("RunLogger.start_run must be called before logging turns.")

        turn_entry = {
            "turn_index": len(self.data["turns"]) + 1,
            "timestamp": datetime.utcnow().isoformat(),
            "therapist_input_raw": therapist_input,
            "therapist_input_safe": state.get("safe_user_input"),
            "patient_response": state.get("response"),
            "detected_topic": state.get("last_topic"),
            "intent_topic": state.get("intent_topic"),
            "current_emotion": getattr(
                state.get("patient_profile"), "current_emotional_state", "unknown"
            ),
            "safety_flags": state.get("safety_flags", []),
            "long_term_context": state.get("long_term_context", []),
            "summary_so_far": state.get("summary", ""),
            "history_length": len(state.get("history", [])),
            "total_turns": state.get("total_turns"),
        }
        self.data["turns"].append(turn_entry)
        self.data["last_updated_at"] = datetime.utcnow().isoformat()
        self._persist()

    def finalize(self, extra: Optional[Dict[str, Any]] = None):
        """Mark the run as ended and persist optional summary metadata."""
        if not self.run_id:
            return
        self.data["ended_at"] = datetime.utcnow().isoformat()
        if extra:
            self.data.update(extra)
        self._persist()

    def _persist(self):
        """Write the current log payload to disk."""
        if not self.file_path:
            raise RuntimeError("RunLogger file path not initialized.")
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
