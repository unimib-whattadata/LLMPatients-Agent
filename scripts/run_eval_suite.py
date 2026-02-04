#!/usr/bin/env python
"""Run scripted evaluation scenarios and emit scenario-level metrics."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.eval import load_scenarios, run_suite  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute LLMPatients-Agent evaluation scenarios.")
    parser.add_argument(
        "--scenario-file",
        default="data/eval/scenarios.json",
        help="Path to the scenario definition JSON file.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["all"],
        help="Specific scenario ids to run (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/eval_runs",
        help="Directory where evaluation artifacts will be stored.",
    )
    parser.add_argument(
        "--therapist-prefix",
        default="eval",
        help="Prefix for auto-generated therapist ids used in the run logger.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip writing the aggregated summary JSON file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenario_path = Path(args.scenario_file)
    scenarios = load_scenarios(scenario_path)

    selected = None
    if args.scenarios and args.scenarios != ["all"]:
        selected = args.scenarios

    results = run_suite(
        scenarios,
        selected_ids=selected,
        output_dir=Path(args.output_dir),
        therapist_prefix=args.therapist_prefix,
        write_summary=not args.no_summary,
    )

    failures = 0
    for result in results:
        status = "PASS" if result.success else "FAIL"
        metrics = result.metrics
        topic = metrics.get("topic_accuracy")
        safety = metrics.get("safety_success_rate")
        latency = metrics.get("mean_latency_ms")
        print(
            f"[{status}] {result.scenario_id} "
            f"(topic={topic:.2f} | safety={safety:.2f} | latency={latency:.0f}ms)"
            if topic is not None and safety is not None and latency is not None
            else f"[{status}] {result.scenario_id}"
        )
        if not result.success:
            failures += 1
            for reason in result.failure_reasons:
                print(f"  - {reason}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
