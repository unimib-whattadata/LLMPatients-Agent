# PsyLLM Patient Agent

> A LangGraph-based simulator of psychotherapy patients that clinicians can converse with across multiple sessions while the agent maintains persona, emotional continuity, and long-term memories.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture in Brief](#architecture-in-brief)
3. [Repository Layout](#repository-layout)
4. [Core Components](#core-components)
5. [Patient Data & Knowledge Sources](#patient-data--knowledge-sources)
6. [Configuration & Environment](#configuration--environment)
7. [Running the Agent](#running-the-agent)
8. [Logging & Memory](#logging--memory)
10. [Development Notes](#development-notes)
11. [Troubleshooting & Next Steps](#troubleshooting--next-steps)
12. [Emotion Dynamics Guide](#emotion-dynamics-guide)
13. [Further Reading](#further-reading)

## Project Overview
- **Goal**: provide psychologists with a safe training ground where a local or cloud LLM impersonates a richly described patient.
- **Continuity**: each therapist input flows through LangGraph state, keeps a short-term window, writes episodic memory, and consolidates reflections into long-term memory.
- **Safety**: therapist injections are sanitized before prompting the patient, and the LLM is reminded to ignore role changes or commands.
- **Modularity**: LLM providers, profile schemas, prompt templates, and entrypoints (CLI/API) live in separate modules.

## Architecture in Brief
```
Therapist Input
   │
   ▼
[ sanitize_user_input ] → guards against prompt injection
   │
   ▼
[ detect_intent_topic ] → LLM-based topic classifier
   │
   ▼
[ hydrate_long_term_context ] → episodic memory retrieval
   │
   ▼
[ update_emotions ] → Panksepp emotion vector
   │
   ▼
[ build_prompt ] → persona + memory + guardrails
   │
   ▼
[ generate ] → LLM runner
   │
   ▼
[ append_messages → trim_messages ] → bounded message window
   │
   ▼
[ update_memory ] → write episodic memory, classify tone
   │
   ▼
[ display ] → logging hook (CLI/API decide how to surface response)
```

## Repository Layout
```
agent/
  api/              FastAPI surface (agent/api/app.py).
  core/             Business logic (LangGraph, prompts, patient schema, safety, LLM runners).
  utils/            Shared helpers (RunLogger, session opening helpers).
config/             .env and model credentials.
data/
  patients/         JSON records describing each simulated patient.
  icd11_templates/  ICD-11 templates (optional reference).
  memory/           JSONL episodic/summary storage per therapist/patient.
  topics_tree.json  Hierarchical topic metadata.
notebooks/          Exploratory notebooks.
scripts/            Utility CLIs.
tests/              Run logs and artifacts.
main.py             Full-featured CLI.
Dockerfile          Container recipe.
docs/               Additional markdown docs.
```

## Core Components
### LangGraph Builder (`agent/core/langgraph_builder.py`)
- Defines the `State` (patient id, sanitized input, topic, prompt, response, message history, memory summary, etc.).
- Loads/caches patient profiles and their trait baselines.
- Topic detection is LLM-based (classifier prompt over the topic tree).
- Emotion synthesis uses Panksepp systems with volatility, context modifiers, salience-based smoothing, and a weighted top-two intensity.
- Episodic memory is summarized asynchronously and stored in JSONL per therapist/patient; session reflections and long-term summaries are written on session end.
- A lightweight LLM classifier assigns `current_emotional_state` for telemetry (the emotion vector still drives prompt tone).

### Prompt Builder (`agent/core/prompt_builder.py`)
- Builds a layered prompt:
  - **Primary guidance**: Identity → Cognitive Style → Observed Interaction Style → Dominant Affective Systems.
  - **Memory**: rolling summary, last reflection, last few turns, episodic snippets (topic-gated).
  - **Reference and topical sections**: topic-dependent profile sections (truncated for length stability).
- Uses size limits and summary compression to keep prompt length stable over time.
- Strips parenthetical asides before sending history/memory to the model.

### Patient Profile Schema (`agent/core/patient_profile.py`)
- Pydantic models for demographics, social history, clinical functioning, treatment, and observed behavior.
- Adds `cognitive_style_prompt()` to compress clinical functioning into actionable prompt guidance.

### LLM Runner (`agent/core/llm_runner.py`)
- `LocalLLMRunner` (vLLM) or `VertexLLMRunner` (Vertex AI), selected via env vars.

### Safety Module (`agent/core/safety.py`)
- Defines guardrails and regex patterns for prompt-injection detection.

### Emotion Model (`agent/core/emotion_model.py`)
- Panksepp systems: SEEKING, FEAR, RAGE, LUST, CARE, PANIC_GRIEF, PLAY.
- Combines baseline + noise + context modifiers, clamping and smoothing into `emotion_state`.

### FastAPI Surface (`agent/api/app.py`)
- `POST /chat-response` for turns.
- `POST /session-end` to finalize reflection + long-term summary and close logs.

## Patient Data & Knowledge Sources
- **Patient JSON** (`data/patients/*.json`): structured persona data used by `PatientProfile`.
- **Emotion Traits**: `trait_baseline` uses the Panksepp keys and `volatility_level`.
- **Topics Tree** (`data/topics_tree.json`): topic metadata used by LLM-based topic selection.

## Configuration & Environment
1. Install dependencies:
   ```bash
   pip install -r agent/requirements.txt
   ```
2. Environment variables (set in `config/.env` or shell):
   - `model_provider`: `local` or `vertex_ai`.
   - `model_id`, `temperature`, `max_tokens`, `cache_path`.
   - `GCP_PROJECT`, `GCP_LOCATION`, `GOOGLE_APPLICATION_CREDENTIALS` (Vertex AI).
3. SentenceTransformer: `all-MiniLM-L6-v2` is downloaded on first run (CPU by default).

## Running the Agent
### 1) Interactive CLI (`main.py`)
```bash
PYTHONPATH=. python main.py --patient juanita_delgado_001
```
- Type `exit`/`quit` to end a session (finalizes reflection + long-term summary).

### 2) Minimal Chat Shell (`scripts/chat_cli.py`)
```bash
PYTHONPATH=. python scripts/chat_cli.py --patient juanita_delgado_001
```

### 3) FastAPI Service
```bash
uvicorn agent.api.app:app --reload --port 8000
```

#### Export logs and memory
```bash
curl -o psyllm_export.tar.gz http://localhost:8000/export-logs
```
The archive includes run logs plus therapist/patient memory pair files.
If `PSYLLM_EXPORT_TOKEN` is set on the server, include the header:
```bash
curl -H "X-Export-Token: <token>" -o psyllm_export.tar.gz http://localhost:8000/export-logs
```

## Logging & Memory
- **Short-term**: last `MAX_SHORT_TERM_TURNS` kept in `state.history`.
- **Episodic memory**: every `EPISODE_BATCH_SIZE` turns are summarized asynchronously and stored in `data/memory/<therapist>__<patient>.jsonl`.
- **Session reflection**: generated at session end and persisted.
- **Long-term summary**: updated from reflections at session end.
- **Run logs**: `tests/runs/<therapist>.json` contains all turns and snapshots for resume.
- **Transcripts**: full turn-by-turn transcripts in `tests/runs/transcripts/<therapist>__<session>.jsonl`.


## Development Notes
- Keep new fields inside `PatientProfile` so prompt access is consistent.
- The async episode executor is drained on shutdown; session end explicitly calls `finalize_session_memory()` in CLI and API.

## Troubleshooting & Next Steps
- **LLM initialization failures**: verify `model_id` and GPU availability (vLLM) or GCP credentials (Vertex).
- **First run slow**: SentenceTransformer downloads on first use.
- **Persistence**: JSONL is used for now; swap the `JsonlMemoryStore` implementation to migrate to a DB later.

## Prompt Anatomy
The prompt is layered so the most actionable guidance appears closest to the model’s response instruction:

```
🧍 Identity
🧠 Cognitive Style
🎭 Observed Interaction Style
🎚️ Dominant Affective Systems

Summary of previous sessions
Last session reflection
Recent conversation (last 3 turns)
Relevant episodic memories (topic-gated)

📂 Topic-conditioned sections (truncated)

🛡️ Safety & Character Guardrails
🧩 Context for This Turn (topic, affect, therapist message)
✳️ Instruction (response rules)
```

## Emotion Dynamics Guide
- `docs/emotion_dynamics_walkthrough.md`: detailed walkthrough of emotion synthesis and prompt surfacing.

## Further Reading
- `docs/architecture.md`: detailed LangGraph state and memory flow.
- `docs/api_usage.md`: API usage examples.
- `docs/prompt_building.md`: how prompt layers are assembled and constrained.
