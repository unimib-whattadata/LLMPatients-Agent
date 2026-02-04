"""CLI utility to run the LLMPatients-Agent patient agent in interactive or scripted modes."""

import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

from agent.core.langgraph_builder import build_graph, finalize_session_memory, llm_runner
from agent.core.patient_profile import PatientProfile
from agent.utils.run_logger import RunLogger
from agent.utils.session_opening import build_session_opening

logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Configure and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run a LLMPatients-Agent patient conversation using LangGraph memory."
    )
    parser.add_argument(
        "--patient",
        default=os.getenv("DEFAULT_PATIENT_ID", "juanita_delgado_001"),
        help="Patient identifier (defaults to env DEFAULT_PATIENT_ID or juanita_delgado_001).",
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
    parser.add_argument(
        "--therapist",
        default=os.getenv("DEFAULT_THERAPIST_ID", "therapist0"),
        help="Therapist identifier (used to resume previous sessions).",
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


def run_turn(
    graph,
    patient_id: str,
    message: str,
    config: dict,
    therapist_id: str,
    session_id: str,
    run_logger: RunLogger | None = None,
    base_state: Optional[dict] = None,
) -> dict:
    """Drive a single therapist→patient exchange through LangGraph."""
    payload = {
        "user_input": message,
        "patient_id": patient_id,
        "therapist_id": therapist_id,
        "session_id": session_id,
    }
    if base_state:
        payload = {**base_state, **payload}
    result = graph.invoke(payload, config=config)
    print(f"👩‍⚕️ Therapist: {message}")
    print(f"🧍 Patient: {result.get('response', '...')}")
    print("-")
    log_state(result)
    if run_logger:
        run_logger.log_turn(result, message)
    return result


def run_interactive(
    graph,
    patient_id: str,
    profile: PatientProfile,
    config: dict,
    therapist_id: str,
    session_id: str,
    run_logger: RunLogger | None = None,
    base_state: Optional[dict] = None,
) -> None:
    """Prompt the user for inputs until they exit, logging each turn."""
    welcome = build_session_opening(profile, base_state, llm_runner) if base_state else None
    if not welcome:
        welcome = default_welcome(profile)
    if welcome:
        print(f"🧍 Patient: {welcome}\n")
    print(f"🧠 Simulated patient agent is ready for patient '{patient_id}'. Type 'exit' to quit.\n")
    state = {}
    pending_state = dict(base_state or {})
    while True:
        try:
            user_input = input("👩‍⚕️ Therapist: ")
            if user_input.strip().lower() in {"exit", "quit"}:
                print("Session ended.")
                break
            state = run_turn(
                graph,
                patient_id,
                user_input,
                config,
                therapist_id,
                session_id,
                run_logger,
                pending_state or None,
            )
            pending_state = None
        except KeyboardInterrupt:
            print("\nSession interrupted.")
            break
        except Exception as exc:
            logger.exception("❌ Error during interaction:")
            print(f"❌ Error during interaction: {exc}")
            break
    state = finalize_session_memory(state or {})
    if run_logger:
        run_logger.finalize(state or {})


def run_scripted(
    graph,
    patient_id: str,
    profile: PatientProfile,
    config: dict,
    messages: list[str],
    therapist_id: str,
    session_id: str,
    run_logger: RunLogger | None = None,
    base_state: Optional[dict] = None,
) -> None:
    """Replay a predefined list of therapist messages against the agent."""
    welcome = build_session_opening(profile, base_state, llm_runner) if base_state else None
    if not welcome:
        welcome = default_welcome(profile)
    if welcome:
        print(f"🧍 Patient: {welcome}\n")
    print(f"🧠 Running scripted session for patient '{patient_id}' with {len(messages)} turns.\n")
    state = {}
    pending_state = dict(base_state or {})
    for idx, msg in enumerate(messages, 1):
        try:
            state = run_turn(
                graph,
                patient_id,
                msg,
                config,
                therapist_id,
                session_id,
                run_logger,
                pending_state or None,
            )
            pending_state = None
        except Exception as exc:
            logger.exception("❌ Error during scripted interaction:")
            print(f"❌ Halting at turn {idx} due to error: {exc}")
            break
    print("✅ Scripted session completed.")
    state = finalize_session_memory(state or {})
    if run_logger:
        run_logger.finalize(state or {})


def main() -> int:
    """Entrypoint for the CLI; wires args, LangGraph, and run logging."""
    args = parse_args()
    graph = build_graph()
    patient_id = args.patient
    profile = load_patient_profile(patient_id)
    thread_id = args.session or f"cli-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    therapist_id = args.therapist
    run_logger = RunLogger(therapist_id)
    mode = "scripted" if (args.messages or args.messages_file) else "interactive"
    run_logger.start_run(
        patient_id=patient_id,
        session_id=thread_id,
        source="cli",
        mode=mode,
        metadata={"args": vars(args)},
    )
    print(f"🗂️ Logging run to {run_logger.file_path}")
    restored_state = run_logger.restore_state(patient_id)
    if restored_state:
        print("♻️ Previous session detected; resuming context.\n")

    scripted = []
    try:
        scripted.extend(load_messages_from_file(args.messages_file))
    except Exception as exc:
        print(f"❌ Unable to read messages file: {exc}")
        return 1
    if args.messages:
        scripted.extend([msg for msg in args.messages if msg.strip()])

    if scripted:
        run_scripted(
            graph,
            patient_id,
            profile,
            config,
            scripted,
            therapist_id,
            thread_id,
            run_logger,
            restored_state,
        )
    else:
        run_interactive(
            graph,
            patient_id,
            profile,
            config,
            therapist_id,
            thread_id,
            run_logger,
            restored_state,
        )

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
