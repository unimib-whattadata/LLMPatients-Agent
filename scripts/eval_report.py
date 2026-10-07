#!/usr/bin/env python
"""Summarize evaluation run outputs and optionally compare against a baseline."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

HIGHER_IS_BETTER = {
    "topic_accuracy",
    "intent_accuracy",
    "safety_success_rate",
    "memory_expectation_success_rate",
    "organic_memory_turn_ratio",
    "response_include_rate",
    "response_exclude_rate",
}

LOWER_IS_BETTER = {"mean_latency_ms", "p95_latency_ms", "emotion_drift"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report on LLMPatients-Agent evaluation outputs.")
    parser.add_argument(
        "--current",
        default="tests/eval_runs/latest_summary.json",
        help="Path to the latest summary JSON emitted by run_eval_suite.py.",
    )
    parser.add_argument(
        "--baseline",
        help="Optional summary JSON to compare against for regressions.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Minimum absolute delta required to count as a regression.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with status 1 if any scenario failed or regressed beyond tolerance.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> Dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("results"), list) or not data["results"]:
        raise ValueError("Evaluation summary must contain a nonempty 'results' array.")
    results = {}
    for item in data["results"]:
        if not isinstance(item, dict):
            raise ValueError("Every evaluation result must be an object.")
        scenario_id = item.get("scenario_id") or item.get("scenario", {}).get("id")
        if not scenario_id:
            raise ValueError("Every evaluation result must identify a scenario.")
        if scenario_id in results:
            raise ValueError(f"Duplicate result for scenario '{scenario_id}'.")
        results[scenario_id] = item
    return results


def collect_regressions(
    current: Dict[str, dict],
    baseline: Dict[str, dict],
    tolerance: float,
) -> List[Tuple[str, str, float]]:
    regressions: List[Tuple[str, str, float]] = []
    for scenario_id, current_result in current.items():
        baseline_result = baseline.get(scenario_id)
        if not baseline_result:
            continue
        curr_metrics = current_result.get("metrics", {})
        base_metrics = baseline_result.get("metrics", {})
        for metric, curr_value in curr_metrics.items():
            base_value = base_metrics.get(metric)
            if curr_value is None or base_value is None:
                continue
            delta = curr_value - base_value
            if metric in HIGHER_IS_BETTER and delta < -tolerance:
                regressions.append((scenario_id, metric, delta))
            elif metric in LOWER_IS_BETTER and delta > tolerance:
                regressions.append((scenario_id, metric, delta))
    return regressions


def main() -> int:
    args = parse_args()
    current = load_summary(Path(args.current))
    baseline = load_summary(Path(args.baseline)) if args.baseline else {}

    print("=== Evaluation Summary ===")
    failures = 0
    for scenario_id, result in current.items():
        metrics = result.get("metrics", {})
        status = "PASS" if result.get("success") else "FAIL"
        line = (
            f"{scenario_id:>24} | {status:4} | "
            f"topic={metrics.get('topic_accuracy')} | "
            f"safety={metrics.get('safety_success_rate')} | "
            f"latency={metrics.get('mean_latency_ms')}"
        )
        print(line)
        if status == "FAIL":
            failures += 1

    regressions = []
    if baseline:
        regressions = collect_regressions(current, baseline, args.tolerance)
        if regressions:
            print("\n⚠️ Potential regressions vs baseline:")
            for scenario_id, metric, delta in regressions:
                print(f"  - {scenario_id}: {metric} delta={delta:+.3f}")

    if args.fail_on_regression and (failures or regressions):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
