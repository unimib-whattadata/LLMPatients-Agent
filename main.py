from agent.core.langgraph_builder import build_graph
import logging
from pprint import pprint

logger = logging.getLogger(__name__)

def run_agent():
    # === Initialize once ===
    graph = build_graph(initial=True)
    print("🧠 Simulated patient agent is ready.\n")

    # Keep full conversation state here
    state = {}

    while True:
        try:
            user_input = input("👩‍⚕️ Therapist: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Session ended.")
                break

            # Determine whether this is the first interaction
            if not state:
                # First run triggers profile loading
                result = graph.invoke({"user_input": user_input})
            else:
                # Subsequent runs reuse *full* state and skip reloading
                result = graph.invoke({
                    **state,               # carry forward previous memory, profile, history, summary
                    "user_input": user_input
                })

            # Keep updated state for next turn
            state = result

            # === Debug logging ===
            logger.info("📊 --- STATE UPDATE AFTER TURN ---")
            logger.info(f"🧠 Emotional tone: {state['patient_profile'].current_emotional_state}")
            logger.info(f"🕓 Turns in memory: {len(state['history'])}")
            logger.info(f"🧾 Summary length: {len(state['summary'])} chars")
            logger.info(f"📌 Last topic: {state.get('last_topic')}")
            logger.info("------------------------------------------\n")

        except KeyboardInterrupt:
            print("\nSession interrupted.")
            break
        except Exception as e:
            logger.exception("❌ Error during interaction:")
            print(f"❌ Error during interaction: {e}")
            break

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_agent()