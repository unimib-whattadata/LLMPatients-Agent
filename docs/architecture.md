# PsyLLM Architecture & Contributor Guide

This document explains how the PsyLLM patient agent works under the hood so that new contributors can comfortably extend or debug the system.

## 1. High-Level Flow

```
therapist turn ─┐
                ▼
   sanitize_user_input → detect_intent_topic → hydrate_long_term_context → build_prompt
                                                                                │
                                                                                ▼
                                                                     generate_response (LLM)
                                                                                │
                                                                                ▼
                                               append_messages → trim_messages → update_memory → display
                                                                                │
                                                                                └──────► async summarizer + long-term store
```

Each box runs inside a LangGraph `StateGraph`. The graph is compiled once at startup and invoked for every therapist turn. Nodes exchange data through a shared `State` (Pydantic model) hosted in `agent/core/langgraph_builder.py`.

## 2. State Object Cheat Sheet

Field | Purpose | Producer | Consumer(s)
----- | ------- | -------- | -----------
`patient_id` | Key used to load patient JSON and retrieve memories | CLI/API | `load_profile`, `persist_long_term_memory`
`user_input` / `safe_user_input` | Raw vs. sanitized therapist text (unsafe instructions replaced) | Entry payload / `sanitize_user_input` | Prompt builder, topic detection
`safety_flags` | Regex labels for potential prompt injections | `sanitize_user_input` | Prompt builder (for guardrails messaging)
`patient_profile` | Rich persona definition loaded from JSON | `load_profile` | Prompt builder, tone updates
`history` | Recent therapist/patient turns (max 5) | `update_memory` | Prompt builder, summary chunking
`messages` | LangChain message list used for streaming memory windows | `append_messages`/`trim_messages` | Async summarizer, future LangGraph nodes
`summary` | Accumulated multi-turn narrative produced by async jobs | `_collect_completed_summaries` | Prompt builder, long-term context
`long_term_context` | Relevant snippets retrieved from persistent store | `hydrate_long_term_context` | Prompt builder
`intent_topic` / `last_topic` | Semantic topic classification for the turn | `detect_intent_topic` | Prompt builder, summarizer metadata
`topic_similarity` | Cosine score for the selected topic | `detect_intent_topic` | `_is_small_topic_shift`
`response` | Final LLM output | `generate_response` | `update_memory`, CLI/API
`total_turns` | Count of turns within this LangGraph thread | `update_memory` | Logging/telemetry

## 3. Memory & Summaries

1. **Short-Term Window**: `history` holds the last `MAX_SHORT_TERM_TURNS` exchanges to keep prompts grounded and manageable.
2. **Overflow Chunking**: when `messages` exceed `MAX_MESSAGE_WINDOW`, `trim_messages` converts the oldest items into therapist/patient pairs, slices them into `SUMMARY_BATCH_SIZE` chunks, and submits jobs to `SUMMARY_EXECUTOR`.
3. **Async Summaries**: `_summarize_chunk` prompts the LLM to compress each chunk into a concise note. The output is persisted via `persist_long_term_memory`, tagged by topic, and appended to the running `summary` once the future completes.
4. **Retrieval**: before every prompt, `hydrate_long_term_context` semantically searches the in-memory store (`InMemoryStore`) for relevant chunks filtered by patient + topic. These snippets are fed back into the prompt to maintain continuity across sessions or processes.

You can replace `InMemoryStore` with a LangGraph-compatible backend (Redis, Postgres, etc.) by implementing the same interface and wiring it into `LONG_TERM_STORE`.

## 4. Session Resume Logic

- **RunLogger (`agent/utils/run_logger.py`)** stores every therapist session inside `tests/runs/<therapist>.json`. Each session keeps:
  - turns with raw/sanitized inputs, responses, topics, flags, and summary snapshots;
  - a lightweight `final_state` produced by `_state_snapshot` (summary, topic, last messages, etc.).
- **Restoring**: both the CLI (`scripts/chat_cli.py`) and the FastAPI endpoint look up the latest session for the therapist/patient pair via `RunLogger.restore_state`. The snapshot feeds into the next `graph.invoke` call so the agent immediately remembers the prior conversation.
- **Session Opening**: `agent/utils/session_opening.py` uses the restored state to ask the LLM for a warm “welcome back” line that references the previous summary/topic. If no saved state exists, it falls back to the patient’s `Metadata.welcomeMessage`.

## 5. LLM Providers

The system abstracts inference behind `agent/core/llm_runner.py`:

- `LocalLLMRunner` boots a HuggingFace model using vLLM (GPU-friendly, prefix caching). Configure via `.env`: `model_provider=local`, `model_id`, optional `cache_path`, `temperature`, and `max_tokens`.
- `VertexLLMRunner` forwards prompts to Google Vertex AI (Gemini family). Requires `GCP_PROJECT`, `GCP_LOCATION`, and `GOOGLE_APPLICATION_CREDENTIALS`. Safety settings block high-risk harms by default.
- `llm_runner` is instantiated at import; use dependency injection if you need per-request variation.

## 6. Prompts & Safety

`agent/core/prompt_builder.py` is the single place that shapes the model input. It:
1. Serializes the patient profile into human-readable sections.
2. Appends recent history, long-term memories, and the running summary.
3. Includes guardrails listed in `agent/core/safety.py` so the patient refuses role swaps or hidden-instruction disclosures.
4. States the current and previous topic plus the therapist’s sanitized message.

Update `SAFETY_GUARDS` / `SAFETY_PATTERNS` whenever you encounter new attack vectors.

## 7. Entry Points

Mode | File | Notes
---- | ---- | -----
CLI | `main.py` | Supports scripted runs (`--messages`, `--messages-file`) and interactive sessions. Always writes run logs.
Chat Shell | `scripts/chat_cli.py` | Minimal REPL that restores state per therapist/patient and prints telemetry (tone, topic, total turns) after each reply.
API | `agent/api/app.py` | `POST /api/message` expects `external_patient_id`, `user_message`, `session_id`, and optional `therapist_id`. Maintains per-therapist session caches and returns reasoning time, emotion, and topic labels.

## 8. Adding a New Feature

1. **Extend the State**: update the `State` model in `langgraph_builder.py` and decide which node owns the new field.
2. **Update Prompting**: surface new context in `prompt_builder.py` if the LLM should be aware of it.
3. **Persist Data**: if the feature affects memory, add logic in `update_memory`, `_state_snapshot`, and `RunLogger` so it can be restored.
4. **Expose via CLI/API**: surface new outputs or inputs in `main.py`, `scripts/chat_cli.py`, and/or `agent/api/app.py`.
5. **Document It**: summarize the behavior in `readme.md` plus any relevant markdown files inside `docs/`.

## 9. Testing & Troubleshooting Tips

- Use `tests/runs/` artifacts to replay issues. Each file captures the entire conversation and state snapshot.
- If `python -m py_compile ...` fails with missing `encodings`, the local Python install is corrupted—reinstall or rely on the project’s Docker image.
- When changing the topic tree or patient files, restart any long-lived processes to rebuild embeddings and caches.
- Wrap experimental code with feature flags in `.env` so other contributors can reproduce your setup without editing source.

---
