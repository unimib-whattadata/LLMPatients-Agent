#!/usr/bin/env python3
"""
CLI for running clinical questionnaires against a virtual patient.

Usage examples:
  # List available questionnaires
  PYTHONPATH=. python3 scripts/run_questionnaire.py --list

  # Run a questionnaire (canonical patient IDs only)
  PYTHONPATH=. python3 scripts/run_questionnaire.py --patient juanita_delgado_001 --questionnaire phq9

  # Show an existing result
  PYTHONPATH=. python3 scripts/run_questionnaire.py --patient juanita_delgado_001 --questionnaire phq9 --show

  # Force re-run (overwrite existing result)
  PYTHONPATH=. python3 scripts/run_questionnaire.py --patient juanita_delgado_001 --questionnaire phq9 --force
"""

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

QUESTIONNAIRES_DIR = ROOT_DIR / "data" / "questionnaires"
RESULTS_DIR = ROOT_DIR / "data" / "questionnaire_results"


def is_questionnaire_eligible_patient_id(patient_id: str) -> bool:
    """Questionnaires only run on canonical patient IDs ending exactly with _001."""
    return bool(patient_id) and patient_id.endswith("_001")


def cmd_list():
    """Print all available questionnaire definitions."""
    if not QUESTIONNAIRES_DIR.exists():
        print(f"Questionnaires directory not found: {QUESTIONNAIRES_DIR}")
        return

    from agent.core.questionnaire_catalog import (
        iter_questionnaire_definitions,
        questionnaire_is_runnable,
    )

    questionnaire_defs = iter_questionnaire_definitions(QUESTIONNAIRES_DIR)
    if not questionnaire_defs:
        print("No questionnaire definitions found in data/questionnaires/")
        return

    print("\nAvailable questionnaires:\n")
    header = f"  {'ID':<22} {'Run':<3} {'Items':>5}  {'Scale':<10}  Name"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for q in questionnaire_defs:
        n_items = len(q.get("items", []))
        scale_type = q.get("scale", {}).get("type", "?")
        scale_range = ""
        if scale_type == "integer":
            mn = q["scale"].get("min", "?")
            mx = q["scale"].get("max", "?")
            scale_range = f"int {mn}-{mx}"
        elif scale_type == "choice":
            opts = q["scale"].get("options", [])
            scale_range = "/".join(opts)
        elif scale_type == "ordinal_choice":
            scale_range = "ordinal"
        runnable = "yes" if questionnaire_is_runnable(q) else "no"
        print(f"  {q['id']:<22} {runnable:<3} {n_items:>5}  {scale_range:<10}  {q.get('name', '')}")
    print()


def cmd_show(patient_id: str, questionnaire_id: str):
    """Print the stored result for a (patient, questionnaire) pair."""
    result_path = RESULTS_DIR / patient_id / f"{questionnaire_id}.json"
    if not result_path.exists():
        print(f"No result found: {result_path}")
        sys.exit(1)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_run(patient_id: str, questionnaire_id: str, force: bool):
    """Execute the questionnaire for the given patient."""
    if not is_questionnaire_eligible_patient_id(patient_id):
        print(
            "Questionnaires can only be run for canonical patients whose ID ends exactly with '_001'. "
            f"Received: {patient_id!r}",
            file=sys.stderr,
        )
        sys.exit(2)

    from agent.core.questionnaire_catalog import (
        load_questionnaire_definition,
        questionnaire_is_runnable,
        questionnaire_non_runnable_reason,
    )

    q_def = load_questionnaire_definition(questionnaire_id, QUESTIONNAIRES_DIR)
    if not questionnaire_is_runnable(q_def):
        print(
            f"Questionnaire '{questionnaire_id}' is marked as non-runnable. "
            f"{questionnaire_non_runnable_reason(q_def)}",
            file=sys.stderr,
        )
        sys.exit(2)

    from agent.core.questionnaire_runner import QuestionnaireRunner

    runner = QuestionnaireRunner(questionnaire_id, patient_id, force=force)
    runner.run()


def main():
    parser = argparse.ArgumentParser(
        description="Run clinical questionnaires for LLMPatients virtual patients.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--patient",
        metavar="PATIENT_ID",
        help="Patient ID ending exactly with _001 (e.g. juanita_delgado_001)",
    )
    parser.add_argument(
        "--questionnaire",
        metavar="QUESTIONNAIRE_ID",
        help="Questionnaire ID (e.g. phq9, pid5bf, lpfs_bf2, dsm5tr_l1, snap2, bes)",
    )
    parser.add_argument("--list", action="store_true", help="List all available questionnaires and exit")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the stored result for --patient / --questionnaire and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if a result already exists (overwrites the existing file)",
    )

    args = parser.parse_args()

    if args.list:
        cmd_list()
        return

    if not args.patient or not args.questionnaire:
        parser.error("--patient and --questionnaire are required (unless using --list)")

    if args.show:
        cmd_show(args.patient, args.questionnaire)
    else:
        cmd_run(args.patient, args.questionnaire, args.force)


if __name__ == "__main__":
    main()
