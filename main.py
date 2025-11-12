"""CLI utility to run the PsyLLM patient agent in interactive or scripted modes."""

import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path

from agent.core.langgraph_builder import build_graph
from agent.core.patient_profile import PatientProfile

logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent
PATIENTS_DIR = ROOT_DIR / "data" / "patients"


def parse_args() -> argparse.Namespace:
    """Configure and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run a PsyLLM patient conversation using LangGraph memory."
    )
    parser.add_argument(
        "--patient",
        default=os.getenv("DEFAULT_PATIENT_ID", "franklin_johnson_001"),
        help="Patient identifier (defaults to env DEFAULT_PATIENT_ID or franklin_johnson_001).",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Thread/session id to reuse LangGraph short-term memory. Random if omitted.",
    )
    parser.add_argument(
        "--messages",
        nargs="*",
        help="Therapist messages to run non-interactively (provide multiple strings).",
    )
    parser.add_argument(
        "--messages-file",
        type=str,
        default=None,
        help="Path to a file containing therapist turns (JSON list or newline-separated).",
    )
    return parser.parse_args()


def load_messages_from_file(path_str: str | None) -> list[str]:
    """Read therapist turns from disk, supporting JSON arrays or plaintext."""
    if not path_str:
        return []
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Messages file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() in {".json", ".jsonl"}:
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(item) for item in data if str(item).strip()]
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    return [line.strip() for line in text.splitlines() if line.strip()]


def log_state(state: dict) -> None:
    """Pretty-print the most important fields after each turn."""
    logger.info("📊 --- STATE UPDATE AFTER TURN ---")
    patient_profile = state.get("patient_profile")
    emotion = getattr(patient_profile, "current_emotional_state", "unknown")
    logger.info(f"🧠 Emotional tone: {emotion}")
    logger.info(f"🕓 Total turns: {state.get('total_turns', len(state.get('history', [])))}")
    logger.info(f"🧾 Summary length: {len(state.get('summary', ''))} chars")
    logger.info(f"📌 Last topic: {state.get('last_topic')}")
    logger.info("------------------------------------------\n")


def run_turn(graph, patient_id: str, message: str, config: dict, run_logger: RunLogger | None = None) -> dict:
    """Drive a single therapist→patient exchange through LangGraph."""
    result = graph.invoke(
        {"user_input": message, "patient_id": patient_id},
        config=config,
    )
    print(f"👩‍⚕️ Therapist: {message}")
    print(f"🧍 Patient: {result.get('response', '...')}")
    print("-")
    log_state(result)
    if run_logger:
        run_logger.log_turn(result, message)
    return result


def run_interactive(graph, patient_id: str, config: dict, run_logger: RunLogger | None = None) -> None:
    """Prompt the user for inputs until they exit, logging each turn."""
    welcome = _load_welcome_message(patient_id)
    if welcome:
        print(f"🧍 Patient: {welcome}\n")
    print(f"🧠 Simulated patient agent is ready for patient '{patient_id}'. Type 'exit' to quit.\n")
    state = {}
    while True:
        try:
            user_input = input("👩‍⚕️ Therapist: ")
            if user_input.strip().lower() in {"exit", "quit"}:
                print("Session ended.")
                break
            state = run_turn(graph, patient_id, user_input, config, run_logger)
        except KeyboardInterrupt:
            print("\nSession interrupted.")
            break
        except Exception as exc:
            logger.exception("❌ Error during interaction:")
            print(f"❌ Error during interaction: {exc}")
            break
    if run_logger:
        run_logger.finalize({"final_summary": (state or {}).get("summary", "")})


def run_scripted(graph, patient_id: str, config: dict, messages: list[str], run_logger: RunLogger | None = None) -> None:
    """Replay a predefined list of therapist messages against the agent."""
    welcome = _load_welcome_message(patient_id)
    if welcome:
        print(f"🧍 Patient: {welcome}\n")
    print(f"🧠 Running scripted session for patient '{patient_id}' with {len(messages)} turns.\n")
    state = {}
    for idx, msg in enumerate(messages, 1):
        try:
            state = run_turn(graph, patient_id, msg, config, run_logger)
        except Exception as exc:
            logger.exception("❌ Error during scripted interaction:")
            print(f"❌ Halting at turn {idx} due to error: {exc}")
            break
    print("✅ Scripted session completed.")
    if run_logger:
        run_logger.finalize({"final_summary": (state or {}).get("summary", "")})


def _load_welcome_message(patient_id: str) -> str:
    """Return the welcome blurb if the patient JSON defines one."""
    patient_path = PATIENTS_DIR / f"{patient_id}.json"
    if not patient_path.exists():
        return ""
    try:
        profile = PatientProfile.from_file(str(patient_path))
        metadata = getattr(profile, "Metadata", None)
        if metadata and getattr(metadata, "welcomeMessage", None):
            return metadata.welcomeMessage
    except Exception:
        logger.debug("Unable to load welcome message for %s", patient_id, exc_info=True)
    return ""


def main() -> int:
    """Entrypoint for the CLI; wires args, LangGraph, and run logging."""
    args = parse_args()
    graph = build_graph()
    patient_id = args.patient
    thread_id = args.session or f"cli-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    run_logger = RunLogger()
    mode = "scripted" if (args.messages or args.messages_file) else "interactive"
    run_logger.start_run(
        patient_id=patient_id,
        session_id=thread_id,
        source="cli",
        mode=mode,
        metadata={"args": vars(args)},
    )
    print(f"🗂️ Logging run to {run_logger.file_path}")

    scripted = []
    try:
        scripted.extend(load_messages_from_file(args.messages_file))
    except Exception as exc:
        print(f"❌ Unable to read messages file: {exc}")
        return 1
    if args.messages:
        scripted.extend([msg for msg in args.messages if msg.strip()])

    if scripted:
        run_scripted(graph, patient_id, config, scripted, run_logger)
    else:
        run_interactive(graph, patient_id, config, run_logger)

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
