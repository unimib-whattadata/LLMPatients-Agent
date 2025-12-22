from __future__ import annotations

import json
import statistics
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from agent.core.langgraph_builder import build_graph, finalize_session_memory
from agent.utils.run_logger import RunLogger


def load_scenarios(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load scenario definitions from a JSON file and index them by id."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    scenarios = raw.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("Scenario file must contain a top-level 'scenarios' array.")
    indexed = {}
    for item in scenarios:
        scenario_id = item.get("id")
        if not scenario_id:
            raise ValueError("Every scenario must define an 'id'.")
        indexed[scenario_id] = item
    return indexed


@dataclass
class ScenarioResult:
    scenario_id: str
    scenario_type: str
    patient_id: str
    metrics: Dict[str, Any]
    success: bool
    failure_reasons: List[str] = field(default_factory=list)
    turns: List[Dict[str, Any]] = field(default_factory=list)
    output_path: Optional[Path] = None
    duration_seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.output_path:
            payload["output_path"] = str(self.output_path)
        return payload


def run_suite(
    scenarios: Dict[str, Dict[str, Any]],
    *,
    selected_ids: Optional[Iterable[str]] = None,
    output_dir: Path,
    therapist_prefix: str = "eval",
    write_summary: bool = True,
) -> List[ScenarioResult]:
    """Run a batch of scenarios and optionally persist a summary JSON."""
    graph = build_graph()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_logs_dir = output_dir / "session_logs"
    run_logs_dir.mkdir(parents=True, exist_ok=True)

    targets = selected_ids or scenarios.keys()
    results: List[ScenarioResult] = []
    for scenario_id in targets:
        scenario = scenarios.get(scenario_id)
        if not scenario:
            raise KeyError(f"Unknown scenario id '{scenario_id}'.")
        result = run_scenario(
            scenario,
            graph=graph,
            output_dir=output_dir,
            run_logs_dir=run_logs_dir,
            therapist_prefix=therapist_prefix,
        )
        results.append(result)

    if write_summary:
        summary = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "output_dir": str(output_dir),
            "results": [r.as_dict() for r in results],
        }
        summary_path = output_dir / "latest_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return results


def run_scenario(
    scenario: Dict[str, Any],
    *,
    graph,
    output_dir: Path,
    run_logs_dir: Path,
    therapist_prefix: str,
) -> ScenarioResult:
    """Execute a single scripted scenario and compute metrics."""
    scenario_id = scenario["id"]
    patient_id = scenario["patient_id"]
    thread_id = f"{scenario_id}-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    therapist = scenario.get("therapist_id") or f"{therapist_prefix}_{scenario_id}"
    run_logger = RunLogger(therapist, base_dir=run_logs_dir)
    run_logger.start_run(
        patient_id=patient_id,
        session_id=thread_id,
        source="eval-suite",
        mode=scenario.get("type", "unspecified"),
        metadata={"scenario_id": scenario_id},
    )

    turns: List[Dict[str, Any]] = []
    last_state: Dict[str, Any] = {}
    start_time = time.perf_counter()
    for turn_index, turn in enumerate(scenario.get("messages", []), start=1):
        payload = {
            "patient_id": patient_id,
            "user_input": turn["text"],
            "therapist_id": therapist,
            "session_id": thread_id,
        }
        turn_start = time.perf_counter()
        state = graph.invoke(payload, config=config)
        latency_ms = (time.perf_counter() - turn_start) * 1000.0
        run_logger.log_turn(state, turn["text"])

        turn_record = _build_turn_record(state, turn, latency_ms, turn_index)
        turns.append(turn_record)
        last_state = state

    last_state = finalize_session_memory(last_state or {})
    run_logger.finalize(last_state or {})
    duration = time.perf_counter() - start_time

    metrics = _compute_metrics(turns)
    success, reasons = _evaluate_success(scenario.get("success_criteria", {}), metrics)
    output_path = _write_scenario_result(output_dir, scenario, turns, metrics, success, reasons)

    return ScenarioResult(
        scenario_id=scenario_id,
        scenario_type=scenario.get("type", "unspecified"),
        patient_id=patient_id,
        metrics=metrics,
        success=success,
        failure_reasons=reasons,
        turns=turns,
        output_path=output_path,
        duration_seconds=duration,
    )


def _write_scenario_result(
    output_dir: Path,
    scenario: Dict[str, Any],
    turns: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    success: bool,
    reasons: List[str],
) -> Path:
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    path = Path(output_dir) / f"{timestamp}_{scenario['id']}.json"
    payload = {
        "scenario": {
            "id": scenario["id"],
            "type": scenario.get("type"),
            "description": scenario.get("description"),
            "patient_id": scenario.get("patient_id"),
        },
        "turns": turns,
        "metrics": metrics,
        "success": success,
        "failure_reasons": reasons,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _build_turn_record(state: Dict[str, Any], turn: Dict[str, Any], latency_ms: float, index: int) -> Dict[str, Any]:
    """Capture the key artifacts from a single model turn."""
    last_topic = state.get("last_topic") or {}
    intent_topic = state.get("intent_topic") or {}
    profile = state.get("patient_profile")
    emotion = getattr(profile, "current_emotional_state", "unknown") if profile else "unknown"
    response = state.get("response") or ""
    safety_flags = state.get("safety_flags", [])
    long_term_context = state.get("long_term_context", []) or []

    expectations = turn.get("expectations") or {}
    expectation_results = _evaluate_expectations(expectations, last_topic, intent_topic, safety_flags, response, emotion, long_term_context)

    return {
        "index": index,
        "therapist_input": turn["text"],
        "response": response,
        "topic": last_topic,
        "intent_topic": intent_topic,
        "emotion": emotion,
        "safety_flags": safety_flags,
        "long_term_context_count": len(long_term_context),
        "latency_ms": latency_ms,
        "response_tokens": _count_tokens(response),
        "expectations": expectations,
        "expectation_results": expectation_results,
    }


def _evaluate_expectations(
    expectations: Dict[str, Any],
    topic: Dict[str, Any],
    intent_topic: Dict[str, Any],
    safety_flags: List[str],
    response: str,
    emotion: str,
    long_term_context: Sequence[str],
) -> Dict[str, Optional[bool]]:
    """Check per-turn expectations and record pass/fail booleans."""
    results: Dict[str, Optional[bool]] = {}
    if not expectations:
        return results

    if "topic_top" in expectations:
        results["topic_top"] = _compare_text(topic.get("top"), expectations["topic_top"])
    if "topic_sub" in expectations:
        results["topic_sub"] = _compare_text(topic.get("sub"), expectations["topic_sub"])
    if "intent_top" in expectations:
        results["intent_top"] = _compare_text(intent_topic.get("top"), expectations["intent_top"])
    if "intent_sub" in expectations:
        results["intent_sub"] = _compare_text(intent_topic.get("sub"), expectations["intent_sub"])
    if "expected_emotion" in expectations:
        results["expected_emotion"] = _compare_text(emotion, expectations["expected_emotion"])

    required_flags = expectations.get("requires_safety_flags")
    if required_flags:
        results["requires_safety_flags"] = all(flag in safety_flags for flag in required_flags)

    includes = expectations.get("response_includes")
    if includes:
        lowered = response.lower()
        results["response_includes"] = all(fragment.lower() in lowered for fragment in includes)

    excludes = expectations.get("response_excludes")
    if excludes:
        lowered = response.lower()
        results["response_excludes"] = all(fragment.lower() not in lowered for fragment in excludes)

    min_tokens = expectations.get("min_response_tokens")
    if min_tokens is not None:
        results["min_response_tokens"] = _count_tokens(response) >= min_tokens

    max_tokens = expectations.get("max_response_tokens")
    if max_tokens is not None:
        results["max_response_tokens"] = _count_tokens(response) <= max_tokens

    if expectations.get("requires_memory_reference"):
        results["requires_memory_reference"] = len(long_term_context) > 0

    return results


def _compute_metrics(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate scenario-level metrics across all turns."""
    metrics: Dict[str, Any] = {"total_turns": len(turns)}
    if not turns:
        return metrics

    topic_checks = []
    intent_checks = []
    safety_checks = []
    include_checks = []
    exclude_checks = []
    memory_expectation_checks = []

    latencies = []
    response_tokens = []
    emotions = []
    organic_memory_hits = []

    for turn in turns:
        exp_results = turn.get("expectation_results") or {}
        for key in ("topic_top", "topic_sub"):
            value = exp_results.get(key)
            if value is not None:
                topic_checks.append(value)
        for key in ("intent_top", "intent_sub"):
            value = exp_results.get(key)
            if value is not None:
                intent_checks.append(value)
        if "requires_safety_flags" in exp_results:
            safety_checks.append(exp_results["requires_safety_flags"])
        if "response_includes" in exp_results:
            include_checks.append(exp_results["response_includes"])
        if "response_excludes" in exp_results:
            exclude_checks.append(exp_results["response_excludes"])
        if "requires_memory_reference" in exp_results:
            memory_expectation_checks.append(exp_results["requires_memory_reference"])

        latencies.append(turn["latency_ms"])
        response_tokens.append(turn["response_tokens"])
        organic_memory_hits.append(turn.get("long_term_context_count", 0) > 0)
        emotion = turn.get("emotion")
        if emotion and emotion != "unknown":
            emotions.append(emotion)

    metrics["topic_accuracy"] = _bool_ratio(topic_checks)
    metrics["intent_accuracy"] = _bool_ratio(intent_checks)
    metrics["safety_success_rate"] = _bool_ratio(safety_checks)
    metrics["response_include_rate"] = _bool_ratio(include_checks)
    metrics["response_exclude_rate"] = _bool_ratio(exclude_checks)
    metrics["memory_expectation_success_rate"] = _bool_ratio(memory_expectation_checks)
    metrics["organic_memory_turn_ratio"] = _bool_ratio(organic_memory_hits)
    metrics["avg_response_tokens"] = _safe_mean(response_tokens)
    metrics["median_response_tokens"] = statistics.median(response_tokens) if response_tokens else None
    metrics["mean_latency_ms"] = _safe_mean(latencies)
    metrics["p95_latency_ms"] = _percentile(latencies, 0.95)
    metrics["emotion_drift"] = _emotion_drift(emotions)

    return metrics


def _evaluate_success(criteria: Dict[str, Any], metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
    if not criteria:
        return True, []

    success = True
    reasons: List[str] = []
    checks = {
        "min_topic_accuracy": ("topic_accuracy", lambda m, t: m is not None and m >= t),
        "min_intent_accuracy": ("intent_accuracy", lambda m, t: m is not None and m >= t),
        "min_safety_success": ("safety_success_rate", lambda m, t: m is not None and m >= t),
        "min_memory_expectation_success": (
            "memory_expectation_success_rate",
            lambda m, t: m is not None and m >= t,
        ),
        "min_organic_memory_ratio": ("organic_memory_turn_ratio", lambda m, t: m is not None and m >= t),
        "min_response_include_rate": ("response_include_rate", lambda m, t: m is not None and m >= t),
        "min_response_exclude_rate": ("response_exclude_rate", lambda m, t: m is not None and m >= t),
        "max_emotion_drift": ("emotion_drift", lambda m, t: m is not None and m <= t),
        "max_mean_latency_ms": ("mean_latency_ms", lambda m, t: m is not None and m <= t),
        "max_p95_latency_ms": ("p95_latency_ms", lambda m, t: m is not None and m <= t),
        "min_avg_response_tokens": ("avg_response_tokens", lambda m, t: m is not None and m >= t),
        "max_avg_response_tokens": ("avg_response_tokens", lambda m, t: m is not None and m <= t),
    }

    for criterion, (metric_key, comparator) in checks.items():
        if criterion not in criteria:
            continue
        target = criteria[criterion]
        value = metrics.get(metric_key)
        if not comparator(value, target):
            success = False
            reasons.append(f"{metric_key}={value} violated {criterion}={target}")

    return success, reasons


def _compare_text(left: Optional[str], right: Optional[str]) -> Optional[bool]:
    if left is None or right is None:
        return None
    return left.lower() == right.lower()


def _count_tokens(text: str) -> int:
    return len(text.split())


def _bool_ratio(values: Sequence[bool]) -> Optional[float]:
    if not values:
        return None
    return sum(1 for v in values if v) / len(values)


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _percentile(values: Sequence[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    k = int(round((len(ordered) - 1) * quantile))
    return ordered[k]


def _emotion_drift(emotions: Sequence[str]) -> Optional[float]:
    if not emotions or len(emotions) < 2:
        return 0.0
    changes = sum(1 for prev, curr in zip(emotions, emotions[1:]) if prev != curr)
    return changes / (len(emotions) - 1)
