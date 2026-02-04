# LLMPatients-Agent

> **A modular, stateful simulated patient agent for psychotherapy training grounded in clinical psychology and affective neuroscience.**

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [Roadmap & Status](#roadmap--status)
- [Troubleshooting](#troubleshooting)
- [Security](#security)
- [License](#license)
- [Credits](#credits)
- [Citation](#citation)

## Overview
Psychotherapy training relies heavily on repeated exposure to complex, emotionally charged clinical interactions. Traditional methods like roleplay or standardized patients are costly and difficult to scale. **LLMPatients-Agent** (introduced as **EPatient** in research) addresses this by providing a safe, repeatable, and clinically realistic training ground.

Unlike generic LLM chatbots that suffer from persona drift or emotional flatness, this agent treats patient simulation as a **constrained dynamical system**. It embeds a Large Language Model within explicit models of:
- **Persona**: Grounded in the *Psychodynamic Diagnostic Manual (PDM-2)*.
- **Affect**: A symbolic emotion engine based on *Panksepp’s affective neuroscience*.
- **Memory**: A layered system for longitudinal continuity across sessions.

This allows clinicians to practice dealing with resistance, emotional volatility, and ruptures in therapeutic alliance in a way that remains coherent over time.

## Features
- **Clinically Grounded Personas**: Patients are modeled not just by text prompts but by structured psychological profiles (Identity, Mental Functioning, Symptoms) inspired by PDM-2.
- **Symbolic Emotion Dynamics**: Uses Panksepp’s primary emotional systems (e.g., FEAR, RAGE, SEEKING) to drive behavior and tone, ensuring consistent affective reactivity.
- **Longitudinal Continuity**: Supports multi-session interactions. The agent "remembers" past sessions, maintains a rolling summary, and evolves its relationship with the therapist.
- **Deterministic Orchestration**: Prompt construction is layered and rule-based, minimizing hallucinations and persona drift.
- **Topic-Gated Memory**:Retrieves only clinically relevant episodic memories based on the conversation topic to maintain focus.
- **Safety Guardrails**: Input sanitization prevents prompt injections, and the agent is constrained to remain in character even under adversarial pressure.
- **Modular Design**: specific components (LLM provider, Emotion Model, Memory Store) can be swapped or upgraded independently.

## Requirements
- **Python**: 3.9+
- **Virtual Environment**: Recommended (venv, conda, or uv).
- **API Keys**:
  - `GOOGLE_APPLICATION_CREDENTIALS` (if using Vertex AI).
  - Appropriate GPU resources if running local LLMs (vLLM).
- **Dependencies**: Listed in `agent/requirements.txt`.

## Installation
Clone the repository and set up the environment:

```bash
git clone https://github.com/your-org/LLMPatients-Agent.git
cd LLMPatients-Agent

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r agent/requirements.txt
```

## Quickstart
To start a session with a pre-configured patient (e.g., `juanita_delgado_001`) using the interactive CLI:

```bash
PYTHONPATH=. python main.py --patient juanita_delgado_001
```
*Type `exit` or `quit` to end the session. The agent will generate a session reflection and save memory state.*

## Configuration
Configuration is managed via environment variables (can be set in a `.env` file in `config/` or exported in the shell).

| Variable | Description | Default |
|----------|-------------|---------|
| `model_provider` | `local` | `vertex_ai` or `local` |
| `model_id` | Model name/path | e.g. `gemini-pro`, `meta-llama/Llama-2-70b-chat-hf` |
| `temperature` | Generation randomness | `0.7` |
| `cache_path` | Path for HF models | `~/.cache/huggingface` |
| `GCP_PROJECT` | Google Cloud Project ID | Required for Vertex AI |
| `GCP_LOCATION` | Region | e.g. `us-central1` |

## Usage

### Interactive CLI
The main entry point for full simulation sessions:
```bash
python main.py --patient <patient_id>
```

### Minimal Chat Shell
For testing simple dialogue loops without full session overhead:
```bash
python scripts/chat_cli.py --patient <patient_id>
```

### API Server
To run the agent as a backend service (FastAPI):
```bash
uvicorn agent.api.app:app --reload --port 8000
```
- **Endpoints**:
  - `POST /chat-response`: Send a therapist turn, receive patient response.
  - `POST /session-end`: Finalize session, write reflections, consolidate memory.

## Architecture
The system follows a cyclic, stateful pipeline for every conversational turn:

```
Therapist Input
   │
   ▼
[ Input Sanitization ] ──→ Safety Flags
   │
   ▼
[ Topic Detection ] ──→ Hierarchical Topic Tree
   │
   ▼
[ Memory Retrieval ] ──→ Episodic & Long-term Summaries
   │
   ▼
[ Emotion Update ] ──→ Panksepp Vector (Trait + Context)
   │
   ▼
[ Prompt Construction ] ──→ (Identity + Affect + Memory + Safety)
   │
   ▼
[ LLM Generation ]
   │
   ▼
[ State Persistence ] ──→ JSONL / Session Snapshots
```

**Key Components:**
- **`agent/core/langgraph_builder.py`**: Manages the state machine (LangGraph).
- **`agent/core/emotion_model.py`**: Calculates affective updates based on volatility and events.
- **`agent/core/prompt_builder.py`**: Assembles the deterministic prompt layers.
- **`data/patients/`**: JSON definitions of patient personas.

## Testing
Run the test suite to verify core logic:
```bash
pytest tests/
```
*Note: Some tests may require API keys if they hit live LLM endpoints.*

**Coverage**:
- Unit tests for emotion dynamics and memory utilities.
- Integration/End-to-End tests for full conversation flows.

## Deployment
### Docker
A `Dockerfile` is provided for containerized deployment.
```bash
docker build -t llmpatients-agent .
docker run -p 8000:8000 --env-file config/.env llmpatients-agent
```

**Release Checklist**:
- [ ] Verify `requirements.txt` is frozen and up to date.
- [ ] Run full test suite.
- [ ] Ensure `.env` is not included in the image.

## Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes.
4. Open a Pull Request.

**Bug Reports**: Please use the GitHub Issues tab and describe the steps to reproduce the issue.

## Roadmap & Status
**Status**: Beta / Research Prototype.

**Upcoming**:
- [ ] Migration from JSONL memory to Vector DB (e.g., Chroma/Pinecone).
- [ ] Enhanced evaluation metrics for "Clinical Validity".
- [ ] Multi-party therapy simulation support.

## Troubleshooting
- **LLM Initialization Error**: Check your `GCP_APPLICATION_CREDENTIALS` or ensure your local GPU has enough VRAM.
- **Slow First Run**: The `SentenceTransformer` model (`all-MiniLM-L6-v2`) downloads on the first execution. This is normal.
- **Memory Errors**: If the context window is exceeded, check `MAX_SHORT_TERM_TURNS` in config.

## Security
- **Input Sanitization**: All inputs are scanned for prompt injection attacks before processing.
- **Role Integrity**: The system uses system-prompt reinforcement to prevent the LLM from breaking character or leaking instructions.
- **Vulnerability Reporting**: Please define how to report vulnerabilities here (e.g., email or private issue).

## License
[License Name] - See `LICENSE` file for details.

## Credits
Based on the research paper **"EPatient: A Modular, Stateful Simulated Patient Agent for Psychotherapy Training"**.
Authors & Contributors: [Author List Placeholder]

## Citation
If you use this software in your research, please cite:
```bibtex
@article{llmpatients2025,
  title={EPatient: A Modular, Stateful Simulated Patient Agent for Psychotherapy Training},
  author={Unknown},
  journal={arXiv preprint},
  year={2025}
}
```
