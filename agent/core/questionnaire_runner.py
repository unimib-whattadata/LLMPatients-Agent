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
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from agent.core.emotion_model import EMOTION_LABELS, EMOTION_SYSTEM_HINTS
from agent.core.llm_runner import create_llm_runner
from agent.core.patient_profile import PatientDetails, PatientProfile, resolve_patient_profile_path

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
QUESTIONNAIRES_DIR = ROOT_DIR / "data" / "questionnaires"
RESULTS_DIR = ROOT_DIR / "data" / "questionnaire_results"

CLINICAL_CASE_MAX_CHARS = 400
MAX_RETRIES = 3


class QuestionnaireRunner:
    def __init__(self, questionnaire_id: str, patient_id: str, force: bool = False):
        self.questionnaire_id = questionnaire_id
        self.patient_id = patient_id
        self.force = force

        # Load questionnaire definition
        q_path = QUESTIONNAIRES_DIR / f"{questionnaire_id}.yaml"
        if not q_path.exists():
            available = [p.stem for p in QUESTIONNAIRES_DIR.glob("*.yaml")]
            raise FileNotFoundError(
                f"Questionnaire '{questionnaire_id}' not found in {QUESTIONNAIRES_DIR}. "
                f"Available: {', '.join(sorted(available)) or 'none'}"
            )
        with open(q_path, "r", encoding="utf-8") as f:
            self.q_def = yaml.safe_load(f)

        # Load patient profile — same path as interactive mode
        patients_dir = ROOT_DIR / "data" / "patients"
        patient_path = resolve_patient_profile_path(patient_id, patients_dir)
        self.profile = PatientProfile.from_file(str(patient_path))

        # Ensure details are fully deserialised (mirrors load_profile node)
        if isinstance(self.profile.details, dict):
            self.profile.details = PatientDetails(**(self.profile.details or {}))

        # Create LLM runner (same factory as the interactive pipeline)
        self.llm = create_llm_runner()

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
        # Idempotency guard
        if self.result_path.exists() and not self.force:
            result = json.loads(self.result_path.read_text(encoding="utf-8"))
            print(f"[Already complete] {self.result_path}")
            print(f"Scores: {json.dumps(result.get('scores', {}), indent=2)}")
            return result

        # Resume from partial progress if available
        answers: Dict[str, Any] = {}
        if self.partial_path.exists():
            try:
                partial = json.loads(self.partial_path.read_text(encoding="utf-8"))
                answers = partial.get("answers", {})
                print(f"[Resuming] Loaded {len(answers)} answers from partial save.")
            except json.JSONDecodeError:
                logger.warning("Partial file corrupt; starting fresh.")
                answers = {}

        # Build patient context once (reuses the same profile fields as build_prompt)
        patient_context = self._build_patient_context()
        patient_name = self.profile.name

        # Prepare item chunks
        items: List[dict] = self.q_def.get("items", [])
        batch_size: int = self.q_def.get("batch_size", 1)
        remaining = [it for it in items if str(it["id"]) not in answers]
        chunks = [remaining[i : i + batch_size] for i in range(0, len(remaining), batch_size)]

        total_items = len(items)
        done_count = len(answers)

        print(f"\n{'=' * 60}")
        print(f"Questionnaire : {self.q_def['name']}")
        print(f"Patient       : {patient_name} ({self.patient_id})")
        print(f"Items         : {total_items} total | {done_count} already done | {len(remaining)} remaining")
        print(f"Batches       : {len(chunks)} (batch_size={batch_size})")
        print(f"{'=' * 60}\n")

        for i, chunk in enumerate(chunks):
            batch_answers = self._prompt_and_validate(chunk, patient_context, patient_name)
            answers.update(batch_answers)

            # Persist partial progress after every batch
            self.partial_path.write_text(
                json.dumps({"answers": answers, "last_item": chunk[-1]["id"]}, indent=2),
                encoding="utf-8",
            )

            completed = done_count + sum(len(c) for c in chunks[: i + 1])
            print(f"  [{min(completed, total_items):>4}/{total_items}]  batch {i + 1}/{len(chunks)}", flush=True)

        # Score and finalise
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

        print(f"\n[Done] Saved to {self.result_path}")
        print(f"Scores:\n{json.dumps(scores, indent=2)}")
        return result

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
        item_ids = [it["id"] for it in items]

        last_error: Optional[Exception] = None
        collected_choice_answers: Dict[str, str] = {}
        pending_items = list(items)

        for attempt in range(retries):
            try:
                if scale["type"] == "choice":
                    prompt = self._build_batch_prompt(pending_items, patient_context, patient_name, scale)
                    raw = self.llm.generate(
                        prompt,
                        temperature=0.1,
                        max_tokens=max(len(pending_items) * 15 + 200, 300),
                    )
                    parsed = self._parse_batch_tf_partial(raw)

                    for it in pending_items:
                        item_id = it["id"]
                        if item_id in parsed:
                            collected_choice_answers[str(item_id)] = parsed[item_id]

                    missing = [it["id"] for it in items if str(it["id"]) not in collected_choice_answers]
                    if not missing:
                        return {str(it["id"]): collected_choice_answers[str(it["id"])] for it in items}

                    pending_items = [it for it in items if str(it["id"]) not in collected_choice_answers]
                    raise ValueError(f"Missing answers for item IDs: {missing}. Raw: '{raw[:200]}'")
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
            # If a large batch consistently returns partial output, recursively split
            # into smaller chunks to recover missing answers instead of failing hard.
            missing_items = [it for it in items if str(it["id"]) not in collected_choice_answers]
            if len(missing_items) > 1 and collected_choice_answers:
                midpoint = len(missing_items) // 2
                logger.warning(
                    "Retry budget exhausted for item IDs %s; splitting into %d and %d items.",
                    [it["id"] for it in missing_items],
                    midpoint,
                    len(missing_items) - midpoint,
                )
                left = self._prompt_and_validate(
                    missing_items[:midpoint], patient_context, patient_name, retries=retries
                )
                right = self._prompt_and_validate(
                    missing_items[midpoint:], patient_context, patient_name, retries=retries
                )
                collected_choice_answers.update(left)
                collected_choice_answers.update(right)
                return {str(item_id): collected_choice_answers[str(item_id)] for item_id in item_ids}

        raise ValueError(
            f"Failed after {retries} attempts for items "
            f"{item_ids}. Last error: {last_error}"
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
