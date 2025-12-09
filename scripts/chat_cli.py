"""Small REPL for chatting with a PsyLLM patient from the command line."""

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path

from agent.core.langgraph_builder import build_graph, llm_runner
from agent.utils.run_logger import RunLogger
from agent.utils.session_opening import (
    build_session_opening,
    default_welcome,
    load_patient_profile,
)

logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Collect CLI arguments for selecting patients, sessions, and logging."""
    parser = argparse.ArgumentParser(
        description="Interactive shell for chatting with a PsyLLM patient agent."
    )
    parser.add_argument(
        "--patient",
        default=os.getenv("DEFAULT_PATIENT_ID", "franklin_johnson_001"),
        help="Patient identifier (defaults to DEFAULT_PATIENT_ID env var or Franklin).",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Optional thread id to resume a previous conversation.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging verbosity (DEBUG, INFO, WARNING, ...).",
    )
    parser.add_argument(
        "--therapist",
        default=os.getenv("DEFAULT_THERAPIST_ID", "therapist0"),
        help="Therapist identifier used to resume previous sessions.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point for the interactive therapist ↔ patient session."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    graph = build_graph()
    thread_id = args.session or f"cli-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    profile = load_patient_profile(args.patient)
    run_logger = RunLogger(args.therapist)
    run_logger.start_run(
        patient_id=args.patient,
        session_id=thread_id,
        source="cli-chat",
        mode="interactive",
        metadata={"args": vars(args)},
    )
    restored_state = run_logger.restore_state(args.patient)
    # If we have a saved state for this therapist/patient pair, craft a contextual greeting.
    welcome = build_session_opening(profile, restored_state, llm_runner) if restored_state else None
    if welcome:
        print("♻️ Previous session detected; resuming context.\n")
    else:
        welcome = default_welcome(profile)
    if welcome:
        print(f"🧍 Patient: {welcome}\n")

    print(
        f"🧠 Chatting with patient '{args.patient}'. "
        "Type 'exit' to end the session.\n"
        f"Thread id: {thread_id}\n"
        f"Run log: {run_logger.file_path}\n"
    )

    last_state = {}
    pending_state = dict(restored_state or {})  # primed for one-shot hydration on the next prompt
    while True:
        try:
            therapist_msg = input("👩‍⚕️ Therapist: ").strip()
            if therapist_msg.lower() in {"exit", "quit"}:
                print("Session ended.")
                break

            payload = {"user_input": therapist_msg, "patient_id": args.patient}
            if pending_state:
                payload = {**pending_state, **payload}
                pending_state = None  # only hydrate state once at session start
            state = graph.invoke(payload, config=config)

            print(f"🧍 Patient: {state.get('response', '...')}\n")
            run_logger.log_turn(state, therapist_msg)
            last_state = state

            profile = state.get("patient_profile")
            tone = getattr(profile, "current_emotional_state", "unknown")
            topic = state.get("last_topic") or {}
            topic_label = (
                f"{topic.get('top', 'unknown')} → {topic.get('sub', 'unknown')}"
                if isinstance(topic, dict)
                else "unknown"
            )
            total_turns = state.get("total_turns", len(state.get("history", [])))
            print(
                f"📊 tone={tone} | turns={total_turns} | topic={topic_label}\n---"
            )
        except KeyboardInterrupt:
            print("\nSession interrupted.")
            break
        except Exception as exc:
            logger.exception("Conversation error")
            print(f"❌ Error: {exc}")
            break

    run_logger.finalize(last_state or {})
    return 0


if __name__ == "__main__":
    result = main()
    sys.exit(result)
