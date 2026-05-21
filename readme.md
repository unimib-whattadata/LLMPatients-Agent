# LLMPatients-Agent

Stateful virtual patient agent backend for psychotherapy training, built with FastAPI, LangGraph, structured YAML clinical profiles and JSONL long-term memory.

![Status](https://img.shields.io/badge/status-prototype-orange)
![Python](https://img.shields.io/badge/python-3.10%2B-green)
![API](https://img.shields.io/badge/api-FastAPI-blue)
![Workflow](https://img.shields.io/badge/workflow-LangGraph-blue)

## Contents

- [Overview](#overview)
- [Core Features](#core-features)
- [Tech Stack](#tech-stack)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [Runtime Workflow](#runtime-workflow)
- [Useful Commands](#useful-commands)
- [Testing](#testing)
- [Production](#production)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Security](#security)
- [Citation](#citation)

## Overview

LLMPatients-Agent is the external patient/orchestrator service used by LLMPatients simulations. It loads a structured virtual patient profile, runs each therapist turn through a LangGraph pipeline, generates a patient response through either local vLLM or Vertex AI, and persists session memory for continuity across future sessions.

The service is designed to pair with LLMPatients-App in remote API mode. The app can initialize patients through `/patients`, request patient turns through `/chat-response`, and close a session through `/session-end`.

Patient profiles are YAML files under `data/patients/`. Runtime memory is persisted as JSONL under `data/memory/`, while session run snapshots are stored under `tests/runs/` when generated locally.

## Core Features

- Structured YAML patient profiles with clinical details, objectives, difficulty, voice/avatar metadata and emotional trait baselines.
- Multi-session conversation state with short-term messages, long-term summaries, session reflections and episodic memory.
- LangGraph turn pipeline for profile loading, safety filtering, topic/emotion classification, prompt building, generation and memory updates.
- Symbolic emotion dynamics inspired by Panksepp systems: `SEEKING`, `FEAR`, `RAGE`, `LUST`, `CARE`, `PANIC_GRIEF`, `PLAY`.
- Local LLM provider through vLLM, with optional Intel XPU support.
- Vertex AI provider for Gemini-based generation when Google Cloud credentials are available.
- FastAPI endpoints compatible with external simulation clients.
- Clinical questionnaire runner for YAML questionnaire definitions and persisted JSON results.
- Export endpoint for run logs and therapist/patient memory files.
- Dockerfile and Docker Compose support for API deployment.

## Tech Stack

- **Runtime**: Python 3.10+
- **API**: FastAPI, Uvicorn, Pydantic
- **Agent workflow**: LangGraph, LangChain Core
- **LLM providers**: local vLLM or Google Vertex AI
- **Embeddings**: SentenceTransformers `all-MiniLM-L6-v2`
- **Data formats**: YAML patient/questionnaire definitions, JSON/JSONL runtime artifacts
- **Container support**: Dockerfile plus Docker Compose service for the API

## Requirements

- Python 3.10 or newer. Python 3.11 is recommended for local tests.
- `pip` and a virtual environment.
- A valid `model_id` for `model_provider=local`.
- GPU/XPU resources for local vLLM in realistic runs.
- Optional Google Cloud project and credentials for `model_provider=vertex_ai`.
- Optional Docker and Docker Compose for containerized API deployment.

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/unimib-whattadata/LLMPatients-Agent.git
cd LLMPatients-Agent

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r agent/requirements.txt
```

> [!NOTE]
> For containerized deployments using the optimized Intel XPU Docker base image (`intel/vllm`), the `Dockerfile` automatically utilizes a separate lightweight dependency manifest at `agent/requirements-xpu.txt` to prevent conflict or override with preloaded hardware-specific libraries.


### 2. Create `.env`

```bash
cp .env.example .env
```

Minimum local configuration:

```env
model_provider=local
model_id=meta-llama/Llama-2-7b-chat-hf
temperature=0.7
max_tokens=512
max_model_len=8192
cache_path=/absolute/path/to/hf-cache
```

Vertex AI configuration:

```env
model_provider=vertex_ai
model_id=gemini-2.5-flash
GCP_PROJECT=your-gcp-project
GCP_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=config/vertex-ai-api-key.json
```

### 3. Run the API

```bash
PYTHONPATH=. uvicorn agent.api.app:app --reload --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for FastAPI docs.

### 4. Run a CLI session

```bash
PYTHONPATH=. python3 scripts/chat_cli.py \
  --patient juanita_delgado_001 \
  --therapist therapist0
```

### 5. Send a chat turn through the API

```bash
curl -X POST http://localhost:8000/chat-response \
  -H "Content-Type: application/json" \
  -d '{
    "external_patient_id": "juanita_delgado_001",
    "user_message": "How have you been sleeping this week?",
    "session_id": "demo-session",
    "step_id": 1,
    "therapist_id": "therapist0"
  }'
```

### 6. Close the session

```bash
curl -X POST http://localhost:8000/session-end \
  -H "Content-Type: application/json" \
  -d '{
    "external_patient_id": "juanita_delgado_001",
    "session_id": "demo-session",
    "therapist_id": "therapist0"
  }'
```

Keep `therapist_id`, `external_patient_id` and `session_id` stable during a session. Call `/session-end` when the simulation ends so memory is finalized.

### 7. Pair with LLMPatients-App

When LLMPatients-App runs in remote API mode against this service, set the App `.env` values to the Agent API:

```env
API="remote"
API_BASE_URL="http://localhost:8000"
API_INITIALIZE_PATIENT_ENDPOINT="/patients"
API_CHAT_RESPONSE_ENDPOINT="/chat-response"
API_TIMEOUT_INITIALIZE_PATIENT="120000"
API_TIMEOUT_CHAT_RESPONSE="120000"
```

When LLMPatients-App itself runs in Docker and this Agent is exposed on the host, use `API_BASE_URL="http://host.docker.internal:8000"` in the App container.

## Environment Variables

| Variable | Required | Description |
| --- | --- | --- |
| `model_provider` | Yes | `local` or `vertex_ai`. Defaults to `local`. |
| `model_id` | Yes | Local model id or Vertex/Gemini model id. |
| `temperature` | Optional | Generation temperature. Defaults to `0.7`. |
| `max_tokens` | Optional | Maximum output tokens per generation. Defaults to `512`. |
| `max_model_len` | Local only | Total context length for vLLM. Useful for long questionnaire prompts. |
| `cache_path` | Optional | HuggingFace model cache used by local vLLM. |
| `HF_HOME` | Optional | Fallback HuggingFace cache directory. |
| `GCP_PROJECT` | Vertex only | Google Cloud project for Vertex AI. |
| `GCP_LOCATION` | Vertex only | Vertex AI location. Defaults to `us-central1`. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Vertex only | Credentials path, relative paths resolve from the repository root. |
| `VERTEX_MAX_ATTEMPTS` | Optional | Retry attempts for Vertex generation. |
| `VERTEX_RETRY_BASE_DELAY_SECONDS` | Optional | Base retry delay for Vertex transient errors. |
| `VERTEX_RETRY_MAX_DELAY_SECONDS` | Optional | Maximum retry delay. |
| `VERTEX_RATE_LIMIT_COOLDOWN_SECONDS` | Optional | Shared cooldown after rate limiting. |
| `VERTEX_MIN_REQUEST_INTERVAL_SECONDS` | Optional | Minimum interval between Vertex requests. |
| `QUESTIONNAIRE_MODEL_PROVIDER` | Optional | Provider override for questionnaire runs. |
| `QUESTIONNAIRE_MODEL_ID` | Optional | Model override for questionnaire runs. |
| `QUESTIONNAIRE_TEMPERATURE` | Optional | Temperature override for questionnaire runs. |
| `QUESTIONNAIRE_MAX_TOKENS` | Optional | Max token override for questionnaire runs. |
| `QUESTIONNAIRE_INTER_BATCH_DELAY_SECONDS` | Optional | Delay between questionnaire batches. |
| `DEFAULT_PATIENT_ID` | Optional | CLI default patient id. |
| `DEFAULT_THERAPIST_ID` | Optional | CLI default therapist id. |
| `LOG_LEVEL` | Optional | CLI logging level. Defaults to `INFO`. |
| `PSYLLM_EXPORT_TOKEN` | Optional | If set, `/export-logs` requires `X-Export-Token`. |

### Docker Compose Variables

`docker-compose.yml` follows the same root `.env` convention as LLMPatients-App. Lowercase variables are passed directly to the Python runtime; `LLMPATIENTS_AGENT_*` variables configure Compose itself.

| Variable | Description |
| --- | --- |
| `LLMPATIENTS_AGENT_IMAGE` | Image tag built and run by Compose. Defaults to `llmpatients-agent:latest`. |
| `LLMPATIENTS_AGENT_BASE_IMAGE` | Docker base image. Defaults to `intel/vllm`. |
| `LLMPATIENTS_AGENT_PORT` | Host port mapped to container port `8000`. Defaults to `8000`. |
| `LLMPATIENTS_AGENT_RENDER_GROUP` | Linux render group id for `/dev/dri` when using `docker-compose.xpu.yml`. Defaults to `109`. |

## Runtime Workflow

### Conversation Turn Pipeline

Each `/chat-response` or CLI turn runs through the LangGraph state machine:

1. `load_profile`
2. `sanitize_input`
3. `classify_topic_and_emotion`
4. `hydrate_memory`
5. `update_emotions`
6. `build_prompt`
7. `generate`
8. `append_messages`
9. `trim_messages`
10. `update_memory`
11. `display`

### Memory Artifacts

| Artifact | Location | Purpose |
| --- | --- | --- |
| `episode_summary` | `data/memory/<therapist_id>__<patient_id>.jsonl` | Compact summaries of conversation chunks. |
| `session_reflection` | `data/memory/<therapist_id>__<patient_id>.jsonl` | End-of-session reflection used on future resumes. |
| `long_term_summary` | `data/memory/<therapist_id>__<patient_id>.jsonl` | Rolling summary for continuity across sessions. |
| Run snapshots | `tests/runs/<therapist_id>.json` | Session ledger and final state snapshots used by `RunLogger`. |

### Questionnaire Workflow

Questionnaire definitions live in `data/questionnaires/`. Runnable questionnaires can be listed and executed from the CLI.

```bash
PYTHONPATH=. python3 scripts/run_questionnaire.py --list
PYTHONPATH=. python3 scripts/run_questionnaire.py --patient juanita_delgado_001 --questionnaire phq9
PYTHONPATH=. python3 scripts/run_questionnaire.py --patient juanita_delgado_001 --questionnaire phq9 --show
```

Results are written to:

```txt
data/questionnaire_results/<patient_id>/<questionnaire_id>.json
```

SCID extracts are present as structured YAML data but are marked non-runnable because they are not self-report questionnaires.

## Useful Commands

| Command | Purpose |
| --- | --- |
| `PYTHONPATH=. uvicorn agent.api.app:app --reload --port 8000` | Start the FastAPI server in development. |
| `PYTHONPATH=. python3 scripts/chat_cli.py --patient juanita_delgado_001` | Start the minimal interactive CLI. |
| `PYTHONPATH=. python3 main.py --patient juanita_delgado_001 --therapist therapist0` | Start the full CLI with scripted-message support. |
| `PYTHONPATH=. python3 main.py --messages "Hello" "Tell me more"` | Run a scripted conversation. |
| `PYTHONPATH=. python3 scripts/run_questionnaire.py --list` | List questionnaire definitions. |
| `PYTHONPATH=. python3 scripts/run_questionnaire.py --patient juanita_delgado_001 --questionnaire phq9` | Run one questionnaire. |
| `bash scripts/run_all_questionnaires.sh` | Run all runnable questionnaires for canonical `_001` patients. |
| `curl -H "X-Export-Token: $PSYLLM_EXPORT_TOKEN" http://localhost:8000/export-logs -o logs.tar.gz` | Export run logs and memory files when token protection is enabled. |
| `docker compose up -d` | Start the API container with Docker Compose. |
| `docker compose up -d --build` | Rebuild the image and start the API container after Dockerfile or dependency changes. |
| `docker compose logs -f` | Follow Compose logs. |
| `docker compose down` | Stop the Compose stack. |

## Testing

Recommended local validation before pushing:

```bash
python -m py_compile \
  filter_reqs.py \
  agent/api/app.py \
  agent/core/emotion_model.py \
  agent/core/langgraph_builder.py \
  agent/core/llm_runner.py

python -m unittest discover -s tests -p 'test_*.py'
```

The automated unit tests cover:

- emotion salience and smoothing behavior;
- prompt identity facts;
- questionnaire catalog and questionnaire runner environment helpers.

Notes:

- Use Python 3.10+ for tests because the codebase uses modern type syntax.
- The full local dependency set is in `agent/requirements.txt`, while the optimized dependency set for the Intel XPU Docker container build is maintained in `agent/requirements-xpu.txt` (which excludes pre-installed frameworks like `torch` and `vllm` to avoid overwriting hardware-optimized packages).
- `tests/api-google-test.py` is a manual smoke test for provider initialization and generation, not part of the `unittest discover` pattern.

## Production

### Start without Docker

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r agent/requirements.txt

PYTHONPATH=. uvicorn agent.api.app:app --host 0.0.0.0 --port 8000
```

Set `.env` before starting the service. For production, keep secrets out of the repository and use a process manager such as systemd, launchd, Docker, or your deployment platform.

### Docker image

```bash
docker build -t llmpatients-agent .
```

> [!TIP]
> The `Dockerfile` is pre-configured to use `agent/requirements-xpu.txt` for dependencies when building the image. It uses `filter_reqs.py` to strip out already-installed packages in the base image, ensuring that optimized, pre-installed versions of PyTorch XPU and vLLM are never overwritten.

Run with local data mounted:

```bash
docker run --rm \
  --env-file .env \
  -p 8000:8000 \
  -v "$PWD/data:/app/data" \
  -v "$PWD/tests/runs:/app/tests/runs" \
  llmpatients-agent
```

For Vertex AI, mount the credentials file into `/app/config/` or inject credentials through your platform secret manager.

### Docker Compose API service

The current `docker-compose.yml` manages the API service and mounts local runtime data:
The Compose project is named `llmpatients-agent`, parallel to LLMPatients-App's `llmpatients-app`.

```bash
cp .env.example .env
docker compose up -d
docker compose logs -f
docker compose down
```

Use `docker compose up -d --build` only when you need to force an image rebuild, such as after changing the `Dockerfile`, dependency files, or files copied into the image during build.
The `.env` file is optional for Compose parsing, but the runtime still needs a valid provider configuration such as `model_provider` and `model_id`.

Compose mounts:

```txt
./data       -> /app/data
./tests/runs -> /app/tests/runs
```

It also creates a named `huggingface-cache` volume for downloaded models.

Intel XPU device mounts are kept in the optional override file so default Compose runs on Docker Desktop and non-XPU hosts:

```bash
docker compose -f docker-compose.yml -f docker-compose.xpu.yml up -d
```

### Deployment Notes

- Keep `model_provider`, `model_id`, token limits and credential paths explicit in the runtime environment.
- Call `/session-end` from the client when a simulation ends; otherwise the latest reflection/summary may not be finalized.
- Treat `data/memory/`, `tests/runs/` and exported log archives as sensitive simulation data.
- Use `docker-compose.xpu.yml` only on Linux hosts with Intel XPU `/dev/dri` devices.
- Do not bake cloud credentials, `.env` files, patient data or memory exports into Docker images.

## Project Structure

```txt
agent/api/                 FastAPI app, request/response schemas and endpoints
agent/core/                LangGraph workflow, LLM providers, prompt building, safety and memory logic
agent/utils/               Session logging and contextual opening helpers
data/patients/             YAML virtual patient profiles
data/memory/               JSONL therapist/patient memory records
data/questionnaires/       YAML questionnaire definitions
data/questionnaire_results/ Persisted questionnaire outputs
data/questionnaires pdf/   Source questionnaire PDFs
data/eval/                 Evaluation scenarios
scripts/                   CLI chat, evaluation and questionnaire scripts
tests/                     Unit tests, manual API smoke test and run ledgers
notebooks/                 Experimental data management and inference notebooks
Dockerfile                 API image build
docker-compose.yml         API service with local mounts and HF cache volume
.dockerignore              Docker build-context exclusions for local secrets and generated data
.env.example               Example runtime and Compose environment file
filter_reqs.py             Requirement filter for preloaded vLLM/XPU images
main.py                    Full CLI entrypoint
readme.md                  Project documentation
```

## Troubleshooting

### `Missing model_id for LocalLLMRunner`

Set `model_id` in `.env` or switch to Vertex AI:

```env
model_provider=local
model_id=your-local-model-id
```

### `vllm module is not installed`

Local generation requires vLLM and compatible Torch packages. Use the Docker image based on `intel/vllm`, install vLLM in a compatible Python environment, or switch to:

```env
model_provider=vertex_ai
```

### Vertex AI errors

Check:

```env
GCP_PROJECT=...
GCP_LOCATION=...
GOOGLE_APPLICATION_CREDENTIALS=config/vertex-ai-api-key.json
```

The credentials path is resolved from the repository root when it is relative.

### Patient profile not found

Patient ids resolve to YAML files under `data/patients/`:

```txt
data/patients/juanita_delgado_001.yaml
```

JSON patient profiles are not part of the current runtime profile loader.

### Session continuity is missing

Use the same `therapist_id`, `external_patient_id` and `session_id` for all turns, then call `/session-end` once the session is complete.

### XPU device access is missing

Default Compose does not mount `/dev/dri`. On Linux hosts with Intel XPU devices, start with the optional override:

```bash
docker compose -f docker-compose.yml -f docker-compose.xpu.yml up -d
```

### Docker cannot connect to the daemon

Start Docker Desktop or the Docker service before running build/check/compose commands.

### First response is slow

The first run may download the embedding model and/or local LLM weights into the HuggingFace cache.

### Questionnaire is marked non-runnable

Some YAML files, such as SCID extracts, are reference interview prompts rather than runnable self-report questionnaires. Use:

```bash
PYTHONPATH=. python3 scripts/run_questionnaire.py --list
```

to see which questionnaires can run.

## Security

- Never commit `.env`, `config/.env`, cloud credentials, API keys, model provider secrets or exported log archives.
- Treat `data/memory/`, `tests/runs/` and `data/questionnaire_results/` as sensitive simulation artifacts.
- Rotate any credential that appears in logs, screenshots or shared messages.
- Use `PSYLLM_EXPORT_TOKEN` before exposing `/export-logs` beyond local development.
- Do not mount broad host directories into the container when a narrower `data`, `config` or cache mount is sufficient.

## Citation

If you use LLMPatients for research, cite the reference paper or project record used by your group:

```bibtex
@article{llmpatient2025,
  title={LLMPatient: un sistema esperto ibrido per la simulazione multi-sessione di pazienti virtuali nella formazione psicoterapeutica},
  author={UNIMIB Team},
  journal={TBD},
  year={2025},
  url={https://github.com/unimib-whattadata/LLMPatients-Agent}
}
```
