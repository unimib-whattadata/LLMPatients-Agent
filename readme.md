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
9. [Development Notes](#development-notes)
10. [Troubleshooting & Next Steps](#troubleshooting--next-steps)
11. [Further Reading](#further-reading)

## Project Overview
- **Goal**: provide psychologists with a safe training ground where a local or cloud LLM impersonates a richly described patient.
- **Continuity**: every therapist input passes through a LangGraph that tracks short-term turns, summarizes overflow, and persists long-term memories keyed by topic.
- **Safety**: therapist injections are sanitized before prompting the patient, and the LLM is reminded to ignore role changes or commands.
- **Modularity**: LLM providers (local vLLM or Vertex AI), profile schemas, prompt templates, and entrypoints (CLI/API) live in their own modules so the system can evolve.

## Architecture in Brief
```
Therapist Input
   │
   ▼
[ sanitize_user_input ] → strips unsafe cues and records safety flags
   │
   ▼
[ detect_intent_topic ] → embeds text with SentenceTransformer (MiniLM) and selects topic
   │
   ▼
[ hydrate_long_term_context ] → semantic search over stored memories
   │
   ▼
[ build_prompt ] → assembles profile JSON, summaries, guardrails, therapist turn
   │
   ▼
[ generate ] → LLM runner (local/Vertex)
   │
   ▼
[ append_messages → trim_messages ] → maintain bounded LangChain message window + async summaries
   │
   ▼
[ update_memory ] → update short-term history, trigger long-term storage, classify emotion tone
   │
   ▼
[ display ] → logging hook (CLI/API decide how to surface response)
```
- Graph definition lives in `agent/core/langgraph_builder.py`. Nodes are added via `StateGraph` and compiled with a `MemorySaver` checkpoint to support threaded sessions.
- Short-term history (last ~5 turns) remains in state; older turns are chunked and summarized asynchronously, then persisted as long-term memory snippets.

## Repository Layout
```
agent/
  api/              FastAPI surface (`agent/api/app.py`).
  core/             Business logic (LangGraph, prompts, patient schema, safety, LLM runners).
  prompts/          Static prompt resources (if any future templates).
  utils/            Shared helpers (e.g., `RunLogger`, session opening helpers).
config/             `.env` and model credentials.
data/
  patients/         JSON records describing each simulated patient.
  topics_tree.json  Hierarchical topic metadata → embeddings.
notebooks/          Exploratory or analysis notebooks.
scripts/            Utility CLIs (e.g., `scripts/chat_cli.py`).
tests/              Run-logging destination and future automated tests.
main.py             Full-featured CLI for scripted/interactive runs.
Dockerfile          Container recipe for deployment.
docs/               Additional markdown docs (see `docs/architecture.md`).
```

## Core Components
### LangGraph Builder (`agent/core/langgraph_builder.py`)
- Defines the global `State` (patient id, sanitized input, topic, prompt, response, message history, summaries, etc.).
- Implements helper functions for:
  - Loading and caching patient profiles (`_get_cached_profile`).
  - Detecting therapist intent/topic using `SentenceTransformer('all-MiniLM-L6-v2')` and cosine similarity against embedded topic metadata.
  - Scheduling asynchronous summaries via a `ThreadPoolExecutor`; results are saved as long-term memories inside an `InMemoryStore` namespace (`patients/<id>/memories`).
  - Sanity-checking therapist input (`sanitize_user_input`) against regex patterns defined in `agent/core/safety.py`.
  - Generating responses with retry/fallback logic on top of `llm_runner.generate`.
  - Updating patient emotional tone each turn by prompting the LLM for a tone synopsis, mapping it to canonical emotions through cosine similarity against predefined prototypes.
- Builds the LangGraph pipeline by chaining the nodes listed in the architecture diagram and compiling it with an optional checkpoint store.

### Prompt Builder (`agent/core/prompt_builder.py`)
- Transforms `State` into a single prompt string consumed by the LLM.
- Always includes psychological + demographic sections plus current emotional tone.
- Dynamically appends additional patient fields based on the detected `top` topic’s metadata in `data/topics_tree.json`.
- Injects session summary snippets, latest short-term turns, and relevant long-term memories when available.
- Appends guardrails from `SAFETY_GUARDS` so the patient refuses role swaps or instruction leaks.

### Patient Profile Schema (`agent/core/patient_profile.py`)
- Declares nested Pydantic models covering demographics, social history, psychological profile, coping, treatment, resilience, medical history, environment, and assessment behavior.
- Supports two file formats: the current attribute-based schema and a legacy schema (`details` + nested sections). Helper methods normalize, slugify, and clean textual content before instantiating `PatientProfile` objects.

### LLM Runner Abstraction (`agent/core/llm_runner.py`)
- `LocalLLMRunner` loads a HuggingFace model with vLLM, enabling GPU-backed inference and prefix caching. Controlled by env vars `model_provider=local`, `model_id`, `cache_path`, `temperature`, `max_tokens`.
- `VertexLLMRunner` proxies prompts to Google Vertex AI (Gemini) using credentials pointed to by `GOOGLE_APPLICATION_CREDENTIALS`. Safety settings block high-risk categories.
- `create_llm_runner()` inspects environment variables to choose the backing provider.

### Safety Module (`agent/core/safety.py`)
- Provides textual guardrails and the regex patterns used to detect prompt injection attempts such as “ignore previous instructions” or “act as the therapist.”

### FastAPI Surface (`agent/api/app.py`)
- Instantiates the LangGraph once at import time.
- Exposes `POST /api/message` that accepts `MessageRequest` (patient id, therapist turn, session info) and returns `MessageResponse` (agent reply, reasoning time, inferred emotion/topic, timestamp).
- Each session gets a `RunLogger` so turns are persisted for audit.

- `agent/utils/run_logger.py`: writes structured JSON logs under `tests/runs/`, capturing safe/unsafe therapist input, detected topics, emotion, and summary progression for each turn. Sessions are grouped per therapist so conversations can be resumed later.
- `agent/utils/session_opening.py`: loads patient metadata and crafts contextual “welcome back” messages when a therapist resumes a session with saved state.
- `scripts/chat_cli.py`: lightweight REPL for quick experiments (`PYTHONPATH=. python scripts/chat_cli.py --patient franklin_johnson_001`). Automatically restores the last session for the therapist/patient pair when available.
- `main.py`: richer CLI supporting scripted conversations (from args or files) plus interactive mode; both integrate with `RunLogger`.

## Patient Data & Knowledge Sources
- **Patient JSON** (`data/patients/*.json`): contain the structured fields required by `PatientProfile`. Many include `Metadata.welcomeMessage` shown before a session starts.
- **Topics Tree** (`data/topics_tree.json`): nested dictionary where each top-level topic lists subtopics, textual descriptions, and metadata (e.g., which profile sections to surface when that topic is active). Embeddings are generated once at module import.
- **Emotion Prototypes**: defined inside `langgraph_builder` to classify per-turn tone into one of eight canonical emotions.

## Configuration & Environment
1. **Python environment**: `python>=3.9`. Install dependencies:
   ```bash
   pip install -r agent/requirements.txt
   ```
2. **Environment variables** (set in `config/.env` or shell):
   - `model_provider`: `local` (vLLM) or `vertex_ai`.
   - `model_id`: HuggingFace model name or Vertex model (e.g., `gemini-1.5-pro`).
   - `temperature`, `max_tokens`, `cache_path`: optional overrides.
   - `GCP_PROJECT`, `GCP_LOCATION`, `GOOGLE_APPLICATION_CREDENTIALS`: required for Vertex AI.
   - `DEFAULT_PATIENT_ID`, `LOG_LEVEL`, etc. for CLIs.
3. **SentenceTransformer resources**: the project downloads `all-MiniLM-L6-v2` embeddings and caches them (CPU by default).
4. **Persistent stores**: long-term memories currently use `InMemoryStore`; swap in another LangGraph store for production persistence if desired.

## Running the Agent
### 1. Interactive CLI (`main.py`)
```bash
PYTHONPATH=. python main.py --patient franklin_johnson_001
```
- Enter therapist messages until you type `exit`/`quit`.
- Use `--session` to reuse a checkpoint thread id.
- Provide scripted messages via `--messages "How are you?" "Tell me more"` or `--messages-file turns.json` to replay transcripts.

### 2. Minimal Chat Shell (`scripts/chat_cli.py`)
```bash
PYTHONPATH=. python scripts/chat_cli.py --patient franklin_johnson_001 --log-level DEBUG
```
- Prints topic, tone, and run-log path after each reply.
- Automatically detects prior runs for the therapist/patient pair and resumes context (hydrating summary, last topic, last few turns, etc.).

### 3. FastAPI Service
```bash
uvicorn agent.api.app:app --reload --port 8000
```
- Send JSON requests:
  ```bash
  curl -X POST http://localhost:8000/api/message \
       -H 'Content-Type: application/json' \
       -d '{
             "external_patient_id": "franklin_johnson_001",
             "user_message": "How have you been sleeping?",
             "session_id": "demo-session",
             "step_id": 1
           }'
  ```
- Response includes the patient utterance, reasoning time, inferred emotion, and topic label.

## Logging & Memory
- **Short-term memory**: last `MAX_SHORT_TERM_TURNS` (default 5) therapist/patient pairs retained verbatim in `state.history`.
- **Long-term memory**: when history exceeds window size, overflow turns are chunked (`SUMMARY_BATCH_SIZE=3`), summarized asynchronously, and stored per patient/topic inside `LONG_TERM_STORE`. Subsequent prompts pull relevant snippets to preserve continuity.
- **Run logs**: every CLI/API session writes `tests/runs/<timestamp>.json` for traceability, including sanitized therapist inputs, LLM responses, topics, safety flags, and summary snapshots.

## Development Notes
- The codebase now includes docstrings and inline comments for all non-trivial helpers (langgraph builder, patient schema, runners, safety guards, prompt builder, API, CLI, logger, etc.).
- Docstrings are ASCII-only to keep compatibility with tooling.
- When editing profiles or schema, prefer updating `PatientProfile` models so new fields are automatically available to prompts.
- The asynchronous summary executor is registered via `atexit`; ensure your unit tests clean up by importing `agent.core.langgraph_builder` once per process.

## Troubleshooting & Next Steps
- **LLM initialization failures**: verify `model_id` exists and that you have access to GPUs (for vLLM) or configured GCP credentials (for Vertex).
- **SentenceTransformer downloads**: the first run may take time; set `HF_HOME` to control cache location.
- **Python codec error when running `py_compile`**: indicates a broken stdlib installation; reinstall/repair Python.
- **Persisting memories**: swap `InMemoryStore` with a file- or database-backed implementation for durability beyond process lifetime.
- **Extending topics/patients**: add new entries under `data/topics_tree.json` and `data/patients/*.json`, then restart the service so embeddings and caches refresh.

Happy experimenting! Adapt the prompts, safety rules, or memory backends to match your training scenarios and research questions.

## Further Reading
- 📘 `docs/architecture.md`: deep dive into LangGraph state, async summaries, resume logic, and extension points for new channels or storage backends.
