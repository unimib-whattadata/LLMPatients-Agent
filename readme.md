# LLMPatients-Agent

Optional Python backend for LLMPatients psychotherapy-training simulations.
FastAPI exposes patient initialization and conversation endpoints; LangGraph
combines YAML profiles, emotion state, prompt construction and persistent
conversation memory. Supported generation providers are Ollama, Vertex AI and
local vLLM. This is a research prototype.

## Setup

From the repository root, use Python 3.12 and a fresh environment:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r agent/requirements-runtime.txt
cp -n .env.example .env
```

For an existing local Ollama model, set these values in `.env`:

```env
model_provider=ollama
model_id=your-installed-model
ollama_base_url=http://localhost:11434
temperature=0.7
max_tokens=512
```

The backend also loads the real `all-MiniLM-L6-v2` encoder. Its weights must be
available locally for offline startup. Model weights are not included here.
Vertex AI additionally needs `GCP_PROJECT`, `GCP_LOCATION` and a private
`GOOGLE_APPLICATION_CREDENTIALS` file. Local vLLM requires a separately installed,
compatible model and hardware stack. The runtime manifest covers cloud/Ollama
dependencies; the full configuration reference is [.env.example](.env.example).

Start the API:

```sh
python -m uvicorn agent.api.app:app --host 127.0.0.1 --port 8000
```

Interactive request schemas are at [localhost:8000/docs](http://localhost:8000/docs).
The App's local demonstration and saved-data recomputation can run without this
backend; configure its remote API mode only when connecting to this service.

## API and storage

| Endpoint | Purpose |
|---|---|
| `POST /patients` | Initialize a profile; `/patient` remains an alias. |
| `POST /chat-response` | Submit a therapist message and receive patient text, topic and emotion data. |
| `POST /session-end` | Finalize session memory; reports complete or partial consolidation. |
| `GET /export-logs` | Export session logs and associated memory files. |

Chat requests require `external_patient_id`, `user_message`, `session_id` and
`step_id`; `therapist_id` defaults to `therapist0`. Keep these identifiers stable
throughout a session and call `/session-end` when it finishes. Use one API
process: active sessions and graph checkpoints are process-local.

Profiles are in `data/patients/`, memory in `data/memory/`, questionnaire outputs
in `data/questionnaire_results/` and session logs in `tests/runs/`.
`LLMPATIENTS_MEMORY_DIR` and `LLMPATIENTS_RUNS_DIR` override runtime output paths.
See [factual memory](docs/factual-memory.md) for evidence and failure handling.

The shared local SQLite limiter paces Vertex requests and applies cooldowns.
Capacity errors return HTTP `503`, `Retry-After` and code `vertex_rate_limited`;
retry after the indicated delay. Failed memory generation keeps the session open.

## Tests and sharing

[Offline validation](docs/runtime-validation.md) uses the pinned test lock and
explicit synthetic model/encoder fixtures. Real model availability is a separate
check. Never commit `.env` files or credentials; review generated records before
sharing. Set `PSYLLM_EXPORT_TOKEN` and send `X-Export-Token` to protect log exports.

Source code is [AGPL-3.0-or-later](LICENSE). For research, cite the associated
LLMPatients manuscript and the [software commit](https://github.com/unimib-whattadata/LLMPatients-Agent)
used. Clinical instruments and dependencies retain their own terms.
