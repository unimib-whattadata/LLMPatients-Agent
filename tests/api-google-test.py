import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../agent")))
from core.llm_runner import create_llm_runner

if __name__ == "__main__":
    runner = create_llm_runner()
    prompt = "Give me 3 quick stress-reducing techniques for daily life."
    
    print("Prompting Gemini...")
    response = runner.generate(prompt)
    print("Response:\n", response)