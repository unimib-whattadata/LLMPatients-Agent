import re

from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel
from typing import Optional
from .patient_profile import PatientProfile
from .prompt_builder import build_prompt

SYMPTOM_BEHAVIOR_MAP = {
    "anhedonia": {"tone": "flat", "disclosure": "low"},
    "paranoid_ideation": {"tone": "suspicious", "intent": "deflect", "trust_delta": -0.2},
    "self_harm": {"tone": "ashamed", "avoid_topics": ["cutting", "relapse"]},
    "anger_outbursts": {"tone": "irritable", "intent": "confront"},
}

def apply_reasoning(state):
    modifiers = {"tone": "neutral", "intent": "neutral", "disclosure": "medium"}
    symptoms = state.profile.symptoms

    for symptom, behavior in SYMPTOM_BEHAVIOR_MAP.items():
        if symptom in symptoms and symptoms[symptom].get("present"):
            for k, v in behavior.items():
                if k == "trust_delta":
                    state.profile.mental_state.trust_in_therapist += v
                else:
                    modifiers[k] = v

    return {"reasoning": modifiers}

# Stato condiviso tra tutti i nodi del grafo
class State(BaseModel):
    user_input: Optional[str] = None
    profile: Optional[PatientProfile] = None
    reasoning: Optional[dict] = None
    memory: Optional[str] = None
    prompt: Optional[str] = None
    response: Optional[str] = None

def build_graph(patient_profile: PatientProfile, llm_runner):
    def load_profile(state):
        return {"profile": patient_profile}

    def retrieve_memory(state):
        return {"memory": "You mentioned feeling hopeless last session."}

    def build_prompt_node(state):
        prompt = build_prompt(
            profile=state.profile,
            memory=state.memory,
            user_input=state.user_input,
            reasoning=state.reasoning
        )
        return {"prompt": prompt}

    def run_llm_node(state):
        response = llm_runner.generate(state.prompt)
        return {"response": response}

    
    def postprocess(state):
        raw = state.response.strip()

        # 1. Remove prompt echo if present
        if state.prompt and raw.startswith(state.prompt.strip()):
            raw = raw[len(state.prompt.strip()):].strip()

        # 2. Remove speaker cue (e.g., "Juanita Delgado:") if present
        raw = re.sub(r"^Juanita Delgado:\s*", "", raw, flags=re.IGNORECASE)

        # 3. Remove any trailing instruction-like lines
        cleaned_lines = []
        for line in raw.splitlines():
            if re.match(r"^\*\*.*\*\*$", line.strip()) or "please provide" in line.lower():
                continue  # skip lines that are not part of response
            cleaned_lines.append(line.strip())

        cleaned = "\n".join(line for line in cleaned_lines if line)

        # fallback to raw if cleaning stripped everything
        if not cleaned:
            print("[WARNING] Response cleaning stripped everything; reverting to raw.")
            cleaned = raw

        # Print result
        print(f"\n{state.profile.name}: {cleaned}\n")

        state.response = cleaned
        return state

    builder = StateGraph(state_schema=State)

    builder.add_node("load_profile", RunnableLambda(load_profile))
    builder.add_node("retrieve_memory", RunnableLambda(retrieve_memory))
    builder.add_node("build_prompt", RunnableLambda(build_prompt_node))
    builder.add_node("reason", RunnableLambda(apply_reasoning))
    builder.add_node("run_llm", RunnableLambda(run_llm_node))
    builder.add_node("postprocess", RunnableLambda(postprocess))
    

    builder.set_entry_point("load_profile")
    builder.add_edge("load_profile", "retrieve_memory")
    builder.add_edge("retrieve_memory", "build_prompt")
    builder.add_edge("build_prompt", "reason")
    builder.add_edge("reason", "run_llm")
    builder.add_edge("run_llm", "postprocess")

    return builder.compile()