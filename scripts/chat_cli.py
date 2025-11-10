import argparse
import logging
import os
import sys
import uuid

from agent.core.langgraph_builder import build_graph

logger = logging.getLogger(__name__)


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


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    graph = build_graph()
    thread_id = args.session or f"cli-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    print(
        f"🧠 Chatting with patient '{args.patient}'. "
        "Type 'exit' to end the session.\n"
        f"Thread id: {thread_id}\n"
    )

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

            profile = state.get("patient_profile")
            tone = getattr(profile, "current_emotional_state", "unknown")
            topic = state.get("last_topic") or {}
            topic_label = (
                f"{topic.get('top', 'unknown')} → {topic.get('sub', 'unknown')}"
                if isinstance(topic, dict)
                else "unknown"
            )
            print(
                f"📊 tone={tone} | turns={len(state.get('history', []))} "
                f"| topic={topic_label}\n---"
            )
        except KeyboardInterrupt:
            print("\nSession interrupted.")
            break
        except Exception as exc:
            logger.exception("Conversation error")
            print(f"❌ Error: {exc}")
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
