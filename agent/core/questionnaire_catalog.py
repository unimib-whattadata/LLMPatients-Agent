"""Helpers for loading questionnaire definitions from data/questionnaires."""

from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
QUESTIONNAIRES_DIR = ROOT_DIR / "data" / "questionnaires"


def _read_questionnaire_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Questionnaire definition must be a mapping: {path}")
    payload.setdefault("id", path.stem)
    payload.setdefault("name", path.stem)
    payload["_path"] = str(path)
    return payload


def load_questionnaire_definition(questionnaire_id: str, questionnaires_dir: Path = QUESTIONNAIRES_DIR) -> dict[str, Any]:
    """Load a questionnaire YAML definition by stem ID."""
    q_path = questionnaires_dir / f"{questionnaire_id}.yaml"
    if not q_path.exists():
        available = [p.stem for p in questionnaires_dir.glob("*.yaml")]
        raise FileNotFoundError(
            f"Questionnaire '{questionnaire_id}' not found in {questionnaires_dir}. "
            f"Available: {', '.join(sorted(available)) or 'none'}"
        )
    return _read_questionnaire_yaml(q_path)


def iter_questionnaire_definitions(questionnaires_dir: Path = QUESTIONNAIRES_DIR) -> list[dict[str, Any]]:
    """Return all questionnaire definitions sorted by filename."""
    return [_read_questionnaire_yaml(path) for path in sorted(questionnaires_dir.glob("*.yaml"))]


def questionnaire_is_runnable(questionnaire_def: dict[str, Any]) -> bool:
    """Questionnaires are runnable unless they are explicitly marked otherwise."""
    return bool(questionnaire_def.get("runnable", True))


def questionnaire_non_runnable_reason(questionnaire_def: dict[str, Any]) -> str:
    """Human-readable reason used when a definition is intentionally excluded."""
    return questionnaire_def.get("runnable_reason") or "Questionnaire definition is marked as non-runnable."
