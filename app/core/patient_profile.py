from pydantic import BaseModel
from typing import List
import json

class CognitiveModel(BaseModel):
    relevant_history: str
    core_beliefs: List[str]
    intermediate_beliefs: List[str]
    coping_strategies: List[str]
    situation: str
    automatic_thoughts: List[str]
    emotions: List[str]
    behaviors: List[str]

class PatientProfile(BaseModel):
    name: str
    cognitive_model: CognitiveModel

    @classmethod
    def from_file(cls, path: str) -> "PatientProfile":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)