import os
import json

from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda

from dotenv import load_dotenv
from core.llm_runner import LLMRunner
from core.patient_profile import PatientProfile
from core.prompt_builder import build_prompt

# === Load Environment ===
env_path = Path(__file__).resolve().parent.parent / "config" / ".env"
load_dotenv(dotenv_path=env_path)

# === Load Persona ===
PATIENT_PATH = Path("../data/patients/juanita_delgado.json")
with open(PATIENT_PATH, "r") as f:
    PATIENT = json.load(f)



# === Initialize LLM Runner ===
llm_runner = LLMRunner()

# === LangGraph State ===
class State(BaseModel):
    user_input: Optional[str] = None
    patient_profile: Optional[PatientProfile] = None
    prompt: Optional[str] = None
    response: Optional[str] = None

# === Build Nodes ===
def load_profile(state):
    profile = PatientProfile.from_file(str(PATIENT_PATH))
    return {"patient_profile": profile}

def generate_response(state):
    result = llm_runner.generate(prompt=state.prompt)
    return {"response": result}

def display_response(state):
    print(f"\n🧠 Juanita: {state.response}\n")
    return state

# === Build LangGraph ===
def build_graph():
    builder = StateGraph(State)
    builder.add_node("load_profile", RunnableLambda(load_profile))
    builder.add_node("build_prompt", RunnableLambda(build_prompt))
    builder.add_node("generate", RunnableLambda(generate_response))
    builder.add_node("display", RunnableLambda(display_response))

    builder.set_entry_point("load_profile")
    builder.add_edge("load_profile", "build_prompt")
    builder.add_edge("build_prompt", "generate")
    builder.add_edge("generate", "display")

    return builder.compile()

# === Entry Point ===
def run_agent():
    graph = build_graph()
    print("\n💬 Type 'exit' to quit.")
    while True:
        user_input = input("\n🧑‍⚕️ You: ")
        if user_input.lower().strip() == "exit":
            break
        state = graph.invoke({"user_input": user_input})

if __name__ == "__main__":
    run_agent()