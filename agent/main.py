from core.langgraph_builder import build_graph

def run_agent():
    # First turn → load profile
    graph = build_graph(initial=True)
    print("🧠 Simulated patient agent is ready.\n")

    state = {}
    while True:
        try:
            user_input = input("👩‍⚕️ Therapist: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Session ended.")
                break

            # First turn: uses load_profile
            if not state:
                result = graph.invoke({"user_input": user_input})
            else:
                # Subsequent turns: reuse profile, skip reload
                graph = build_graph(initial=False)
                result = graph.invoke({
                    "user_input": user_input,
                    "patient_profile": state["patient_profile"]
                })

            state = result  # keep state for next turn

        except KeyboardInterrupt:
            print("\nSession interrupted.")
            break
        except Exception as e:
            print(f"❌ Error during interaction: {e}")
            break

if __name__ == "__main__":
    run_agent()