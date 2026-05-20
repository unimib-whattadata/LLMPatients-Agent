"""
Automated questionnaire runner for virtual patients.

Loads a questionnaire YAML definition and a patient profile, then drives the LLM
to answer each item in character. Results are stored once per (patient, questionnaire)
pair in data/questionnaire_results/<patient_id>/<questionnaire_id>.json.

Design notes:
- Uses PatientProfile.from_file() and the same to_prompt() methods as build_prompt()
  in the interactive pipeline — ensuring the patient is represented identically.
- Uses emotionTraits.normalized_baseline() (stable trait values) rather than the
  per-turn stochastic emotion update, which is appropriate for self-report instruments
  that ask about general or recent-weeks patterns.
- Does NOT import langgraph_builder (it initialises an LLM runner at module level).
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.core.questionnaire_catalog import (
    load_questionnaire_definition,
    questionnaire_is_runnable,
    questionnaire_non_runnable_reason,
)
from agent.core.emotion_model import EMOTION_LABELS, EMOTION_SYSTEM_HINTS
from agent.core.llm_runner import create_llm_runner
from agent.core.patient_profile import PatientDetails, PatientProfile, resolve_patient_profile_path

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "data" / "questionnaire_results"

CLINICAL_CASE_MAX_CHARS = 400
MAX_RETRIES = 3
ORDINAL_CHOICE_MAX_TOKENS = 256
SINGLE_CHOICE_MAX_TOKENS = 256
CHOICE_BATCH_MIN_MAX_TOKENS = 1024
PROGRESS_SEPARATOR_WIDTH = 60


def is_questionnaire_eligible_patient_id(patient_id: str) -> bool:
    """Questionnaires only run on canonical patient IDs ending exactly with _001."""
    return bool(re.fullmatch(r".+_001", patient_id or ""))


def questionnaire_inter_batch_delay_seconds() -> float:
    """Throttle questionnaire runs a bit on hosted providers to reduce 429 bursts."""
    provider = os.getenv("model_provider", "local").lower()
    default_delay = 3.0 if provider == "vertex_ai" else 0.0
    raw_value = os.getenv("QUESTIONNAIRE_INTER_BATCH_DELAY_SECONDS")
    if raw_value is None:
        return default_delay
    try:
        delay_seconds = float(raw_value)
        if delay_seconds < 0:
            raise ValueError
        return delay_seconds
    except ValueError:
        logger.warning(
            "Ignoring invalid QUESTIONNAIRE_INTER_BATCH_DELAY_SECONDS=%r. Using %.1fs.",
            raw_value,
            default_delay,
        )
        return default_delay


def choice_batch_max_tokens(item_count: int) -> int:
    """Give T/F batches extra output headroom on hosted models with hidden reasoning."""
    safe_item_count = max(int(item_count), 1)
    return max(CHOICE_BATCH_MIN_MAX_TOKENS, safe_item_count * 32 + 256)


def _env_optional_float(name: str) -> Optional[float]:
    raw_value = os.getenv(name)
    if raw_value is None:
        return None
    try:
        return float(raw_value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r.", name, raw_value)
        return None


def _env_optional_positive_int(name: str) -> Optional[int]:
    raw_value = os.getenv(name)
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        logger.warning("Ignoring invalid %s=%r.", name, raw_value)
        return None


def questionnaire_llm_overrides() -> Dict[str, Any]:
    """Allow questionnaires to run on a lighter or cheaper model than chat."""
    overrides: Dict[str, Any] = {}

    provider = os.getenv("QUESTIONNAIRE_MODEL_PROVIDER")
    if provider:
        overrides["provider"] = provider.strip().lower()

    model_id = os.getenv("QUESTIONNAIRE_MODEL_ID")
    if model_id:
        overrides["model_id"] = model_id.strip()

    temperature = _env_optional_float("QUESTIONNAIRE_TEMPERATURE")
    if temperature is not None:
        overrides["temperature"] = temperature

    max_tokens = _env_optional_positive_int("QUESTIONNAIRE_MAX_TOKENS")
    if max_tokens is not None:
        overrides["max_tokens"] = max_tokens

    return overrides


class QuestionnaireRunner:
    """Run one questionnaire for one canonical patient and persist resumable results."""

    def __init__(self, questionnaire_id: str, patient_id: str, force: bool = False):
        if not is_questionnaire_eligible_patient_id(patient_id):
            raise ValueError(
                "Questionnaires can only be run for canonical patients whose ID ends exactly with '_001'. "
                f"Received: {patient_id!r}"
            )

        self.questionnaire_id = questionnaire_id
        self.patient_id = patient_id
        self.force = force

        # Load questionnaire definition
        self.q_def = load_questionnaire_definition(questionnaire_id)
        if not questionnaire_is_runnable(self.q_def):
            raise ValueError(
                f"Questionnaire '{questionnaire_id}' is marked as non-runnable. "
                f"{questionnaire_non_runnable_reason(self.q_def)}"
            )

        # Load patient profile — same path as interactive mode
        patients_dir = ROOT_DIR / "data" / "patients"
        patient_path = resolve_patient_profile_path(patient_id, patients_dir)
        self.profile = PatientProfile.from_file(str(patient_path))

        # Ensure details are fully deserialised (mirrors load_profile node)
        if isinstance(self.profile.details, dict):
            self.profile.details = PatientDetails(**(self.profile.details or {}))

        # Questionnaires can target a lighter model than interactive chat.
        llm_overrides = questionnaire_llm_overrides()
        self.llm = create_llm_runner(**llm_overrides)
        logger.info(
            "Questionnaire runner using provider=%s model=%s",
            llm_overrides.get("provider", os.getenv("model_provider", "local")).strip().lower(),
            llm_overrides.get("model_id", os.getenv("model_id")),
        )
        self.inter_batch_delay_seconds = questionnaire_inter_batch_delay_seconds()

        # Resolve storage paths
        result_dir = RESULTS_DIR / patient_id
        result_dir.mkdir(parents=True, exist_ok=True)
        self.result_path = result_dir / f"{questionnaire_id}.json"
        self.partial_path = result_dir / f"{questionnaire_id}.partial.json"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Execute the questionnaire and return the result dict."""
        if self.result_path.exists() and not self.force:
            result = json.loads(self.result_path.read_text(encoding="utf-8"))
            self._print_existing_result(result)
            return result

        answers = self._load_partial_answers()
        patient_context = self._build_patient_context()
        patient_name = self.profile.name
        items: List[dict] = self.q_def.get("items", [])
        batch_size: int = self.q_def.get("batch_size", 1)
        remaining = [it for it in items if str(it["id"]) not in answers]
        chunks = [remaining[i : i + batch_size] for i in range(0, len(remaining), batch_size)]
        total_items = len(items)
        done_count = len(answers)

        self._print_run_header(
            patient_name=patient_name,
            total_items=total_items,
            done_count=done_count,
            remaining_count=len(remaining),
            batch_count=len(chunks),
            batch_size=batch_size,
        )

        for i, chunk in enumerate(chunks):
            batch_answers = self._prompt_and_validate(chunk, patient_context, patient_name)
            answers.update(batch_answers)
            self._write_partial_answers(answers, last_item=chunk[-1]["id"])

            completed = done_count + sum(len(c) for c in chunks[: i + 1])
            print(f"  [{min(completed, total_items):>4}/{total_items}]  batch {i + 1}/{len(chunks)}", flush=True)

            if self.inter_batch_delay_seconds > 0 and i < len(chunks) - 1:
                time.sleep(self.inter_batch_delay_seconds)

        return self._finalize_result(answers)

    def _load_partial_answers(self) -> Dict[str, Any]:
        """Resume answers from a partial run when possible."""
        if not self.partial_path.exists():
            return {}
        try:
            partial = json.loads(self.partial_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Partial file corrupt; starting fresh: %s", self.partial_path)
            return {}
        answers = partial.get("answers", {})
        print(f"[Resuming] Loaded {len(answers)} answers from partial save.")
        return answers

    def _write_partial_answers(self, answers: Dict[str, Any], *, last_item: str) -> None:
        """Persist progress after every successful batch."""
        self.partial_path.write_text(
            json.dumps({"answers": answers, "last_item": last_item}, indent=2),
            encoding="utf-8",
        )

    def _finalize_result(self, answers: Dict[str, Any]) -> dict:
        """Compute scores, write final JSON and remove resumable partial state."""
        scores = self._compute_scores(answers)
        result = {
            "questionnaire_id": self.questionnaire_id,
            "questionnaire_name": self.q_def["name"],
            "patient_id": self.patient_id,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "answers": {str(k): v for k, v in answers.items()},
            "scores": scores,
        }

        self.result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        if self.partial_path.exists():
            self.partial_path.unlink()

        self._print_final_result(scores)
        return result

    def _print_existing_result(self, result: dict) -> None:
        print(f"[Already complete] {self.result_path}")
        print(f"Scores: {json.dumps(result.get('scores', {}), indent=2)}")

    def _print_run_header(
        self,
        *,
        patient_name: str,
        total_items: int,
        done_count: int,
        remaining_count: int,
        batch_count: int,
        batch_size: int,
    ) -> None:
        separator = "=" * PROGRESS_SEPARATOR_WIDTH
        print(f"\n{separator}")
        print(f"Questionnaire : {self.q_def['name']}")
        print(f"Patient       : {patient_name} ({self.patient_id})")
        print(f"Items         : {total_items} total | {done_count} already done | {remaining_count} remaining")
        print(f"Batches       : {batch_count} (batch_size={batch_size})")
        print(f"{separator}\n")

    def _print_final_result(self, scores: dict) -> None:
        print(f"\n[Done] Saved to {self.result_path}")
        print(f"Scores:\n{json.dumps(scores, indent=2)}")

    # ------------------------------------------------------------------
    # Patient context builder
    # ------------------------------------------------------------------

    def _build_patient_context(self) -> str:
        """
        Build a focused patient description using the same profile fields and
        to_prompt() methods as build_prompt() in the interactive pipeline:
          - demographicAndSocioculturalInformation.to_prompt()  → identity
          - profile.cognitive_style_prompt()                    → cognitive style
          - behaviorDuringTestAdministration.to_prompt()        → interaction style
          - profile.clinicalCase (truncated)                    → clinical background
          - emotionTraits.normalized_baseline()                 → stable emotion traits
        """
        profile = self.profile
        details = profile.details
        if isinstance(details, dict):
            try:
                details = PatientDetails(**details)
            except Exception:
                details = None

        parts: List[str] = []

        # Identity — mirrors build_prompt's "🧍 Identity" section
        if details and details.demographicAndSocioculturalInformation:
            identity = details.demographicAndSocioculturalInformation.to_prompt()
            if identity:
                parts.append(identity)

        # Clinical background
        if profile.clinicalCase:
            case_text = profile.clinicalCase.strip()
            if len(case_text) > CLINICAL_CASE_MAX_CHARS:
                case_text = case_text[:CLINICAL_CASE_MAX_CHARS].rsplit(" ", 1)[0].strip() + "..."
            parts.append(f"Clinical background: {case_text}")

        # Cognitive style — mirrors build_prompt's "Cognitive Style" section
        if hasattr(profile, "cognitive_style_prompt"):
            cog = profile.cognitive_style_prompt(max_bullets=4)
            if cog:
                parts.append(f"Psychological profile:\n{cog}")

        # Interaction style — mirrors build_prompt's "🎭 Observed Interaction Style"
        if details and details.behaviorDuringTestAdministration:
            style = details.behaviorDuringTestAdministration.to_prompt()
            if style:
                parts.append(f"Interaction style: {style}")

        # Stable emotion baseline (not the per-turn stochastic state)
        baseline = profile.emotionTraits.normalized_baseline()
        if baseline:
            dominant = sorted(baseline.items(), key=lambda x: x[1], reverse=True)[:3]
            emotion_lines = [
                f"  - {EMOTION_LABELS.get(k, k)}: {EMOTION_SYSTEM_HINTS.get(k, '')} ({v:.2f})"
                for k, v in dominant
            ]
            parts.append("Characteristic emotional tendencies:\n" + "\n".join(emotion_lines))

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Prompt dispatch and validation
    # ------------------------------------------------------------------

    def _prompt_and_validate(
        self,
        items: List[dict],
        patient_context: str,
        patient_name: str,
        retries: int = MAX_RETRIES,
    ) -> Dict[str, Any]:
        """Generate and validate answers for a chunk of items, with retry logic."""
        scale = self.q_def["scale"]

        last_error: Optional[Exception] = None
        for attempt in range(retries):
            try:
                if scale["type"] == "choice":
                    prompt = self._build_batch_prompt(items, patient_context, patient_name, scale)
                    raw = self.llm.generate(
                        prompt,
                        temperature=0.0,
                        max_tokens=choice_batch_max_tokens(len(items)),
                    )
                    return self._parse_batch_tf(raw, items)
                elif scale["type"] == "ordinal_choice":
                    assert len(items) == 1, "ordinal_choice questionnaires must use batch_size=1"
                    item = items[0]
                    prompt = self._build_choice_prompt(item, patient_context, patient_name)
                    raw = self.llm.generate(
                        prompt,
                        temperature=0.0,
                        max_tokens=max(ORDINAL_CHOICE_MAX_TOKENS, len(item.get("choices", [])) * 64),
                    )
                    return self._parse_ordinal_choice(raw.strip(), item)
                else:
                    assert len(items) == 1, "Integer-scale questionnaires must use batch_size=1"
                    item = items[0]
                    item_scale = item.get("scale_override") or scale
                    prompt = self._build_single_prompt(item, patient_context, patient_name, item_scale)
                    raw = self.llm.generate(prompt, temperature=0.1, max_tokens=220)
                    validated = self._parse_integer(raw.strip(), item_scale)
                    return {str(item["id"]): validated}
            except ValueError as exc:
                last_error = exc
                if attempt < retries - 1:
                    logger.warning(f"Attempt {attempt + 1}/{retries} failed: {exc}")

        if scale["type"] == "choice":
            logger.warning(
                "Retry budget exhausted for item IDs %s; answering remaining items one by one.",
                [it["id"] for it in items],
            )
            answers: Dict[str, str] = {}
            for item in items:
                answers[str(item["id"])] = self._answer_single_choice_item(
                    item, patient_context, patient_name, retries=retries
                )
            return answers

        raise ValueError(
            f"Failed after {retries} attempts for items "
            f"{[it['id'] for it in items]}. Last error: {last_error}"
        )

    def _build_single_prompt(
        self,
        item: dict,
        patient_context: str,
        patient_name: str,
        scale: dict,
    ) -> str:
        labels: dict = scale.get("labels", {})
        scale_desc = ", ".join(f"{k}={v}" for k, v in sorted(labels.items(), key=lambda x: int(x[0])))
        time_frame = self.q_def.get("time_frame", "")
        time_instruction = f" Think about {time_frame}." if time_frame else ""

        return (
            f"You are {patient_name}. {patient_context}\n\n"
            f"You are completing a self-report questionnaire.{time_instruction} "
            f"Answer the statement below using ONLY a single integer — no other text.\n\n"
            f"Valid responses: {scale_desc}\n\n"
            f'Statement: "{item["text"]}"\n\n'
            f"Your answer (integer only):"
        )

    def _build_batch_prompt(
        self,
        items: List[dict],
        patient_context: str,
        patient_name: str,
        scale: dict,
    ) -> str:
        time_frame = self.q_def.get("time_frame", "")
        time_instruction = f" Think about {time_frame}." if time_frame else ""
        item_lines = "\n".join(f"{it['id']}. {it['text']}" for it in items)
        required_ids = ", ".join(str(it["id"]) for it in items)

        return (
            f"You are {patient_name}. {patient_context}\n\n"
            f"You are completing a self-report questionnaire.{time_instruction}\n"
            f"For each statement, answer T (True or Mostly True) or F (False or Mostly False) "
            f"AS YOURSELF.\n\n"
            f"Reply ONLY with one line per statement in the format '<id>. <T/F>'.\n"
            f"Return exactly {len(items)} lines and answer every ID exactly once.\n"
            f"Required IDs: {required_ids}\n\n"
            f"Do not include any explanation or extra text.\n\n"
            f"Statements:\n{item_lines}\n\n"
            f"Your answers:"
        )

    def _build_choice_prompt(self, item: dict, patient_context: str, patient_name: str) -> str:
        choices = item.get("choices", [])
        time_frame = self.q_def.get("time_frame", "")
        time_instruction = f" Think about {time_frame}." if time_frame else ""
        options_text = "\n".join(f"{i + 1}. {c['text']}" for i, c in enumerate(choices))
        n = len(choices)

        return (
            f"You are {patient_name}. {patient_context}\n\n"
            f"You are completing a self-report questionnaire.{time_instruction} "
            f"Read the options below and choose the ONE that best describes you. "
            f"If multiple options partly fit, choose the single best match. "
            f"Reply with EXACTLY one digit from 1 to {n} — no words, no punctuation, no explanation.\n\n"
            f"Options:\n{options_text}\n\n"
            f"Your answer (number only):"
        )

    def _build_single_choice_prompt(
        self,
        item: dict,
        patient_context: str,
        patient_name: str,
    ) -> str:
        """Focused prompt for one T/F item when batch parsing repeatedly truncates."""
        time_frame = self.q_def.get("time_frame", "")
        time_instruction = f" Think about {time_frame}." if time_frame else ""

        return (
            f"You are {patient_name}. {patient_context}\n\n"
            f"You are completing a self-report questionnaire.{time_instruction}\n"
            f"Answer the following statement AS YOURSELF.\n"
            f"Reply with ONLY one character: T or F.\n\n"
            f'Statement {item["id"]}: "{item["text"]}"\n\n'
            f"Your answer (T/F only):"
        )

    def _answer_single_choice_item(
        self,
        item: dict,
        patient_context: str,
        patient_name: str,
        retries: int = MAX_RETRIES,
    ) -> str:
        """Recover a missing T/F answer with a constrained single-item prompt."""
        last_error: Optional[Exception] = None
        max_attempts = max(retries * 2, 6)
        scale = self.q_def["scale"]

        for attempt in range(max_attempts):
            try:
                # Alternate between a numbered single-line format and a one-character
                # format; some model/provider combinations are flaky with one style only.
                if attempt % 2 == 0:
                    prompt = self._build_single_choice_prompt(item, patient_context, patient_name)
                else:
                    prompt = self._build_batch_prompt([item], patient_context, patient_name, scale)

                raw = self.llm.generate(prompt, temperature=0.0, max_tokens=SINGLE_CHOICE_MAX_TOKENS)

                parsed_numbered = self._parse_batch_tf_partial(raw)
                if item["id"] in parsed_numbered:
                    return parsed_numbered[item["id"]]

                return self._parse_single_tf(raw)
            except ValueError as exc:
                last_error = exc
                if attempt < max_attempts - 1:
                    logger.warning(
                        "Single-item fallback attempt %d/%d failed for item %s: %s",
                        attempt + 1,
                        max_attempts,
                        item["id"],
                        exc,
                    )

        raise ValueError(f"Single-item fallback failed for item {item['id']}. Last error: {last_error}")

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _parse_integer(self, raw: str, scale: dict) -> int:
        match = re.search(r"\b(\d+)\b", raw)
        if not match:
            raise ValueError(f"No integer found in response: '{raw}'")
        value = int(match.group(1))
        min_val = int(scale.get("min", 0))
        max_val = int(scale.get("max", 3))
        if not (min_val <= value <= max_val):
            raise ValueError(f"Value {value} out of valid range [{min_val}, {max_val}]")
        return value

    def _parse_ordinal_choice(self, raw: str, item: dict) -> Dict[str, int]:
        choices = item.get("choices", [])
        match = re.search(r"\b(\d+)\b", raw)
        if not match:
            raise ValueError(f"No option number found in response: '{raw}'")
        idx = int(match.group(1)) - 1
        if not (0 <= idx < len(choices)):
            raise ValueError(f"Option {idx + 1} out of range [1, {len(choices)}] for item {item['id']}")
        return {str(item["id"]): choices[idx]["score"]}

    def _parse_batch_tf(self, raw: str, items: List[dict]) -> Dict[str, str]:
        """Parse a numbered True/False list response like '1. T\\n2. F\\n...'"""
        matches = self._parse_batch_tf_partial(raw)

        missing = [it["id"] for it in items if it["id"] not in matches]
        if missing:
            raise ValueError(f"Missing answers for item IDs: {missing}. Raw: '{raw[:200]}'")

        return {str(it["id"]): matches[it["id"]] for it in items}

    def _parse_batch_tf_partial(self, raw: str) -> Dict[int, str]:
        """Extract any valid 'N. T/F' pairs from model output."""
        pattern = re.compile(r"(\d+)\s*[.:)]\s*([TF])", re.IGNORECASE)
        return {int(m.group(1)): m.group(2).upper() for m in pattern.finditer(raw)}

    def _parse_single_tf(self, raw: str) -> str:
        """Parse a constrained single T/F response."""
        cleaned = raw.strip()
        if cleaned in {"T", "F"}:
            return cleaned

        single_letter = re.search(r"\b([TF])\b", cleaned, re.IGNORECASE)
        if single_letter:
            return single_letter.group(1).upper()

        tf_word = re.search(r"\b(TRUE|FALSE)\b", cleaned, re.IGNORECASE)
        if tf_word:
            return "T" if tf_word.group(1).upper() == "TRUE" else "F"

        raise ValueError(f"No T/F answer found in response: '{cleaned[:120]}'")

    # ------------------------------------------------------------------
    # Scorer
    # ------------------------------------------------------------------

    def _compute_scores(self, answers: Dict[str, Any]) -> dict:
        """Apply the scoring config defined in the questionnaire YAML."""
        scoring = self.q_def.get("scoring", {})
        result: dict = {}

        # -- Subscales --
        subscale_scores: Dict[str, float] = {}
        for name, cfg in scoring.get("subscales", {}).items():
            item_ids = cfg.get("items", [])
            method = cfg.get("method", "sum")
            values = [
                answers[str(i)]
                for i in item_ids
                if str(i) in answers and isinstance(answers[str(i)], (int, float))
            ]
            if not values:
                continue
            subscale_scores[name] = sum(values) if method == "sum" else round(sum(values) / len(values), 3)

        if subscale_scores:
            result["subscales"] = subscale_scores

        # -- Domain scores (average of subscales) --
        domain_scores: Dict[str, float] = {}
        for name, cfg in scoring.get("domains", {}).items():
            sub_names = cfg.get("subscales", [])
            method = cfg.get("method", "mean")
            values = [subscale_scores[s] for s in sub_names if s in subscale_scores]
            if not values:
                continue
            domain_scores[name] = (
                sum(values) if method == "sum" else round(sum(values) / len(values), 3)
            )

        if domain_scores:
            result["domains"] = domain_scores

        # -- Total --
        total_cfg = scoring.get("total", {})
        if total_cfg:
            if "subscales" in total_cfg:
                sub_vals = [subscale_scores[s] for s in total_cfg["subscales"] if s in subscale_scores]
                method = total_cfg.get("method", "sum")
                if sub_vals:
                    result["total"] = (
                        sum(sub_vals) if method == "sum" else round(sum(sub_vals) / len(sub_vals), 3)
                    )
            elif "items" in total_cfg:
                item_ids = total_cfg["items"]
                vals = [
                    answers[str(i)]
                    for i in item_ids
                    if str(i) in answers and isinstance(answers[str(i)], (int, float))
                ]
                method = total_cfg.get("method", "sum")
                if vals:
                    result["total"] = sum(vals) if method == "sum" else round(sum(vals) / len(vals), 3)

        # -- Severity label --
        if "total" in result:
            for threshold in scoring.get("severity_thresholds", []):
                if result["total"] <= threshold["max"]:
                    result["severity"] = threshold["label"]
                    break

        # -- Domain flags (DSM-5-TR style: highest item score per domain) --
        domain_flags: dict = {}
        for domain_name, dcfg in scoring.get("domain_flags", {}).items():
            item_ids = dcfg.get("items", [])
            flag_threshold = dcfg.get("flag_threshold", 2)
            vals = [
                answers[str(i)]
                for i in item_ids
                if str(i) in answers and isinstance(answers[str(i)], (int, float))
            ]
            highest = max(vals) if vals else 0
            domain_flags[domain_name] = {"highest": highest, "flag": highest >= flag_threshold}

        if domain_flags:
            result["domain_flags"] = domain_flags

        # -- Functional item (PHQ-9 item 10, stored separately) --
        functional_item = scoring.get("functional_item")
        if functional_item is not None and str(functional_item) in answers:
            val = answers[str(functional_item)]
            item_def = next(
                (it for it in self.q_def.get("items", []) if it["id"] == functional_item), None
            )
            func_labels = {}
            if item_def:
                raw_labels = (item_def.get("scale_override") or {}).get("labels") or {}
                func_labels = {int(k): v for k, v in raw_labels.items()}
            result["functional_impairment"] = {
                "score": val,
                "label": func_labels.get(val, str(val)),
            }

        # -- Notes (e.g. SNAP-2 scoring key notice) --
        if scoring.get("notes"):
            result["notes"] = scoring["notes"]

        return result
