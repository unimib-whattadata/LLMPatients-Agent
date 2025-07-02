import re

from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel
from typing import Optional
from .patient_profile import PatientProfile
from .prompt_builder import build_prompt

# 👇 Stato condiviso tra tutti i nodi del grafo
class State(BaseModel):
    user_input: Optional[str] = None
    profile: Optional[PatientProfile] = None
    memory: Optional[str] = None
    prompt: Optional[str] = None
    response: Optional[str] = None

def build_graph(patient_profile: PatientProfile, llm_runner):
    # Nodo 1: carica il profilo del paziente
    def load_profile(state):
        return {"profile": patient_profile}

    # Nodo 2: recupera memoria recente
    def retrieve_memory(state):
        return {"memory": "You mentioned feeling hopeless last session."}

    # Nodo 3: costruisce il prompt in base a profilo, memoria e input utente
    def build_prompt_node(state):
        prompt = build_prompt(
            profile=state.profile,
            memory=state.memory,
            user_input=state.user_input
        )
        return {"prompt": prompt}

    # Nodo 4: genera la risposta con LLM
    def run_llm_node(state):
        response = llm_runner.generate(state.prompt)
        return {"response": response}

    # Nodo 5: stampa la risposta
    def postprocess(state):
        raw = state.response.strip()
        prompt = state.prompt.strip()

        cleaned = raw  # default fallback

        # First attempt: remove full prompt if echoed
        if raw.startswith(prompt):
            cleaned = raw[len(prompt):].strip()
        else:
            # Try to locate the last "{name}:" and extract only what's after
            pattern = re.escape(state.profile.name) + r":\s*"
            match = re.search(pattern, raw)
            if match:
                cleaned = raw[match.end():].strip()
            else:
                print("\n[WARNING] No matching speaker cue or prompt found — returning full output.\n")

        # Avoid printing nothing
        if not cleaned:
            print(f"\n[WARNING] Empty cleaned response. Showing raw output instead.\n")
            cleaned = raw

        # Print formatted response
        print(f"\n{state.profile.name}: {cleaned}\n")

        state.response = cleaned
        return state

    # Costruzione del grafo
    builder = StateGraph(state_schema=State)

    builder.add_node("load_profile", RunnableLambda(load_profile))
    builder.add_node("retrieve_memory", RunnableLambda(retrieve_memory))
    builder.add_node("build_prompt", RunnableLambda(build_prompt_node))
    builder.add_node("run_llm", RunnableLambda(run_llm_node))
    builder.add_node("postprocess", RunnableLambda(postprocess))

    builder.set_entry_point("load_profile")
    builder.add_edge("load_profile", "retrieve_memory")
    builder.add_edge("retrieve_memory", "build_prompt")
    builder.add_edge("build_prompt", "run_llm")
    builder.add_edge("run_llm", "postprocess")

    return builder.compile()