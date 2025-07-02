from core.patient_profile import PatientProfile
from core.llm_runner import LLMRunner
from core.langgraph_builder import build_graph

def main():
    profile = PatientProfile.from_file("data/patients/juanita_delgado.json")
    llm_runner = LLMRunner()
    graph = build_graph(profile, llm_runner)

    print("🧠 LLMPatient Chat\n")
    while True:
        user_input = input("Therapist: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        graph.invoke({"user_input": user_input})

if __name__ == "__main__":
    main()