# PYTHONPATH=. python scripts/chat_cli.py --patient franklin_johnson_001

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path

from agent.core.langgraph_builder import build_graph
from agent.core.patient_profile import PatientProfile
from agent.utils.run_logger import RunLogger

logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parents[1]
PATIENTS_DIR = ROOT_DIR / "data" / "patients"


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def _load_welcome_message(patient_id: str) -> str:
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
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    graph = build_graph()
    thread_id = args.session or f"cli-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    run_logger = RunLogger()
    run_logger.start_run(
        patient_id=args.patient,
        session_id=thread_id,
        source="cli-chat",
        mode="interactive",
        metadata={"args": vars(args)},
    )

    welcome = _load_welcome_message(args.patient)
    if welcome:
        print(f"🧍 Patient: {welcome}\n")

    print(
        f"🧠 Chatting with patient '{args.patient}'. "
        "Type 'exit' to end the session.\n"
        f"Thread id: {thread_id}\n"
        f"Run log: {run_logger.file_path}\n"
    )

    last_state = {}
    while True:
        try:
            therapist_msg = input("👩‍⚕️ Therapist: ").strip()
            if therapist_msg.lower() in {"exit", "quit"}:
                print("Session ended.")
                break

            state = graph.invoke(
                {"user_input": therapist_msg, "patient_id": args.patient},
                config=config,
            )

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

    run_logger.finalize({"final_summary": (last_state or {}).get("summary", "")})
    return 0


if __name__ == "__main__":
    result = main()
    sys.exit(result)
