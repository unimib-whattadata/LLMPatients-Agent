"""JSONL-backed memory store for therapist/patient session artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional

MEMORY_SCHEMA_VERSION = 1

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None


class JsonlMemoryStore:
    """Append-only JSONL store with optional file locking."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def file_path(self, patient_id: str, therapist_id: str) -> Path:
        safe_patient = self._sanitize_identifier(patient_id)
        safe_therapist = self._sanitize_identifier(therapist_id)
        return self.base_dir / f"{safe_therapist}__{safe_patient}.jsonl"

    def append(self, record: dict) -> None:
        payload = dict(record)
        payload.setdefault("schema_version", MEMORY_SCHEMA_VERSION)
        path = self.file_path(payload["patient_id"], payload["therapist_id"])
        line = json.dumps(payload, ensure_ascii=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            self._lock_file(f)
            try:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                self._unlock_file(f)

    def iter_records(
        self,
        patient_id: str,
        therapist_id: str,
        *,
        max_records: Optional[int] = None,
    ) -> Iterable[dict]:
        path = self.file_path(patient_id, therapist_id)
        if not path.exists():
            return
        yielded = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record
                    yielded += 1
                    if max_records is not None and yielded >= max_records:
                        return

    def file_mtime(self, patient_id: str, therapist_id: str) -> Optional[float]:
        path = self.file_path(patient_id, therapist_id)
        if not path.exists():
            return None
        return path.stat().st_mtime

    def _lock_file(self, f) -> None:
        if fcntl is None:
            return
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

    def _unlock_file(self, f) -> None:
        if fcntl is None:
            return
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _sanitize_identifier(value: str) -> str:
        return value.replace("/", "_").replace("\\", "_")
