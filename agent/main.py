from core.langgraph_builder import build_graph  # Assuming your graph builder is here

def run_agent():
    graph = build_graph()
    print("🧠 Simulated patient agent is ready.\n")

    while True:
        try:
            user_input = input("👩‍⚕️ Therapist: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Session ended.")
                break

            result = graph.invoke({"user_input": user_input})
        except KeyboardInterrupt:
            print("\nSession interrupted.")
            break
        except Exception as e:
            print(f"❌ Error during interaction: {e}")
            break

if __name__ == "__main__":
    run_agent()