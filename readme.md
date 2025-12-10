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
9. [Evaluation Suite](#evaluation-suite)
10. [Development Notes](#development-notes)
11. [Troubleshooting & Next Steps](#troubleshooting--next-steps)
12. [Emotion Dynamics Guide](#emotion-dynamics-guide)
13. [Further Reading](#further-reading)

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
  - Updating patient affect each turn by blending trait baselines + volatility + therapist-triggered modifiers (from `emotion_model`), applying salience-aware Gaussian noise, smoothing toward the previous state, and applying light counterweights (e.g., CARE/PLAY can soften RAGE). The legacy LLM classifier only runs if the synthesizer has no context.
- Builds the LangGraph pipeline by chaining the nodes listed in the architecture diagram and compiling it with an optional checkpoint store.

### Prompt Builder (`agent/core/prompt_builder.py`)
- Transforms `State` into a single prompt string consumed by the LLM.
- Always includes psychological + demographic sections plus the current dominant affect systems (derived from the emotion model), filtered so any field equal to “Not reported” is omitted. Morality summaries and dysfunctional-behavior fields are injected every turn so the persona keeps its ethical compass and risk profile top-of-mind.
- Dynamically appends additional patient fields based on the detected `top` topic’s metadata in `data/topics_tree.json`.
- Injects session summary snippets, latest short-term turns, and relevant long-term memories when available.
- Appends guardrails from `SAFETY_GUARDS` so the patient refuses role swaps or instruction leaks.
- Encodes affect-driven style directives so language mirrors the dominant systems (e.g., high RAGE yields clipped, defensive answers; dominant FEAR keeps replies vigilant) and references recent life updates only when it feels natural.

### Patient Profile Schema (`agent/core/patient_profile.py`)
- Declares nested Pydantic models covering demographics, social history, psychological profile, coping, treatment, resilience, medical history, environment, and assessment behavior.
- Supports two file formats: the current attribute-based schema and a legacy schema (`details` + nested sections). Helper methods normalize, slugify, and clean textual content before instantiating `PatientProfile` objects.
- Includes an `EmotionDynamics` sub-model which stores `trait_baseline` and `volatility_level`, making the affect synthesizer configurable per patient (e.g., Juanita's high-volatility rage/fear vs. Franklin's blunted playfulness).

### LLM Runner Abstraction (`agent/core/llm_runner.py`)
- `LocalLLMRunner` loads a HuggingFace model with vLLM, enabling GPU-backed inference and prefix caching. Controlled by env vars `model_provider=local`, `model_id`, `cache_path`, `temperature`, `max_tokens`.
- `VertexLLMRunner` proxies prompts to Google Vertex AI (Gemini) using credentials pointed to by `GOOGLE_APPLICATION_CREDENTIALS`. Safety settings block high-risk categories.
- `create_llm_runner()` inspects environment variables to choose the backing provider.

### Safety Module (`agent/core/safety.py`)
- Provides textual guardrails and the regex patterns used to detect prompt injection attempts such as “ignore previous instructions” or “act as the therapist.”

### Emotion Model (`agent/core/emotion_model.py`)
- Uses each patient's `emotionTraits` baseline plus a volatility tier to synthesize momentary Panksepp-style affect vectors. Gaussian noise is damped for low-salience turns and amplified when therapist actions carry bigger emotional consequences (empathy, boundaries, abandonment cues, success check-ins), with light counterweights so supportive systems can temper hot ones.
- Deterministic modifiers adjust only the relevant systems, then the vector is clamped to [0,1], exponentially smoothed with the prior turn (plus decay after multiple low-salience turns), logged, and only the dominant systems (top 1–3, via softmax + floor) are surfaced to the prompt while muted systems remain hidden. If the synthesizer ever fails, the legacy LLM-based classifier still acts as a safety net.

### FastAPI Surface (`agent/api/app.py`)
- Instantiates the LangGraph once at import time.
- Exposes `POST /api/message` that accepts `MessageRequest` (patient id, therapist turn, session info) and returns `MessageResponse` (agent reply, reasoning time, inferred emotion/topic, timestamp).
- Each session gets a `RunLogger` so turns are persisted for audit.

- `agent/utils/run_logger.py`: writes structured JSON logs under `tests/runs/`, capturing safe/unsafe therapist input, detected topics, emotion, intensity, and summary progression for each turn. Each therapist gets a single file with multiple sessions, making it easy to resume conversations with the correct emotional baseline.
- `agent/utils/session_opening.py`: loads patient metadata plus the last saved state and crafts contextual “welcome back” greetings. If no history exists it falls back to the static `welcomeMessage`.
- `scripts/chat_cli.py`: lightweight REPL for quick experiments (`PYTHONPATH=. python scripts/chat_cli.py --patient franklin_johnson_001`). Automatically restores the last session for the therapist/patient pair when available.
- `main.py`: richer CLI supporting scripted conversations (from args or files) plus interactive mode; both integrate with `RunLogger`.

## Patient Data & Knowledge Sources
- **Patient JSON** (`data/patients/*.json`): contain the structured fields required by `PatientProfile`. Many include `Metadata.welcomeMessage` shown before a session starts.
- **Emotion Traits** (`emotionTraits` block inside each patient file): define `trait_baseline` (0–1 intensity per SEEKING/RAGE/FEAR/CARE/LUST/SADNESS/PLAY) plus a `volatility_level`. The LangGraph consumes this block and updates the affect vector every turn so the prompt emphasizes only clinically representative emotions.
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
- Automatically detects prior runs for the therapist/patient pair and resumes context (hydrating summary, last topic, last few turns, emotional baseline/intensity, etc.).

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

## Evaluation Suite
- **Scenarios**: `data/eval/scenarios.json` defines both goal-based probes and red-team injections, each with per-turn expectations (topic alignment, guardrail flags, memory usage) plus success criteria for realism, soundness, and latency.
- **Runner**: `scripts/run_eval_suite.py` loads those scenarios, drives LangGraph end-to-end, and writes detailed artifacts to `tests/eval_runs/` (per-scenario JSON plus `latest_summary.json`). The run logger output lives in `tests/eval_runs/session_logs/`.
- **Reporting**: `scripts/eval_report.py` can be pointed at `latest_summary.json` (and an optional baseline) to surface pass/fail status and regressions; use `--fail-on-regression` inside CI.
- **Notebook**: `notebooks/eval_suite.ipynb` visualizes the summary file in pandas for manual inspection of realism (topic/tone), safety (flag rates), and reliability (latency/continuity).
- **Automation**: `.github/workflows/eval-suite.yml` describes a scheduled workflow that installs dependencies, optionally runs the suite when `RUN_EVAL_SUITE=true`, and always reports status if artifacts already exist.

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

## Emotion Dynamics Guide
- 📘 `docs/emotion_dynamics_walkthrough.md`: deep dive into how trait baselines, volatility, salience-weighted Gaussian noise, smoothing, and prompt wiring create stable yet reactive personas with an end-to-end example.

## Further Reading
- 📘 `docs/architecture.md`: deep dive into LangGraph state, async summaries, resume logic, and extension points for new channels or storage backends.
