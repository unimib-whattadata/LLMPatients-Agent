# LLMPatients-Agent Architecture & Contributor Guide

This document explains how the LLMPatients-Agent patient agent works under the hood so that contributors can extend or debug the system.

## 1. High-Level Flow

```
therapist turn ─┐
                ▼
   sanitize_user_input → detect_intent_topic → hydrate_long_term_context → update_emotions → build_prompt
                                                                                              │
                                                                                              ▼
                                                                                    generate_response (LLM)
                                                                                              │
                                                                                              ▼
                                           append_messages → trim_messages → update_memory → display
                                                                                              │
                                                                                              └──────► async episodic summaries + JSONL store
```

Each box runs inside a LangGraph `StateGraph` compiled once at startup. Nodes exchange data through a shared `State` in `agent/core/langgraph_builder.py`.

## 2. State Object Cheat Sheet

Field | Purpose | Producer | Consumer(s)
----- | ------- | -------- | -----------
`patient_id` | Patient profile id | CLI/API | `load_profile`, memory store
`therapist_id` | Therapist identifier for per-pair memory | CLI/API | memory store
`session_id` | Session identifier | CLI/API | memory store, logging
`user_input` / `safe_user_input` | Raw vs. sanitized therapist text | Entry payload / `sanitize_user_input` | Prompt builder, topic detection
`safety_flags` | Regex labels for potential prompt injections | `sanitize_user_input` | Prompt builder
`patient_profile` | Persona definition loaded from JSON | `load_profile` | Prompt builder
`core_emotion` | Baseline emotion label | `load_profile` | memory metadata
`emotion_intensity` | 0–1 scalar intensity (top-two average) | `update_emotional_state` | Prompt builder
`emotion_state` | Full affect vector (Panksepp systems) | `update_emotional_state` | Prompt builder
`emotion_event` | Therapist-triggered modifier label | `update_emotional_state` | Prompt builder
`history` | Recent therapist/patient turns | `update_memory` | Prompt builder
`messages` | LangChain message list for bounded window | `append_messages`/`trim_messages` | window maintenance
`summary` | Rolling long-term summary (per therapist/patient) | session finalize | Prompt builder
`session_reflection` | Most recent session reflection | session finalize | Prompt builder
`episodic_context` | Retrieved episodic memory snippets | `hydrate_long_term_context` | Prompt builder
`intent_topic` / `last_topic` | LLM topic classification | `detect_intent_topic` | Prompt builder, memory metadata
`response` | Final LLM output | `generate_response` | `update_memory`, CLI/API
`total_turns` | Count of turns in this session | `update_memory` | logging

## 3. Memory & Summaries

1) **Short-Term Window**: `history` holds the last `MAX_SHORT_TERM_TURNS` turns.
2) **Episodic Summaries**: every `EPISODE_BATCH_SIZE` turns are summarized asynchronously and stored in `data/memory/<therapist>__<patient>.jsonl`.
3) **Session Reflection**: on session end, the agent summarizes session episodes into a reflection.
4) **Long-Term Summary**: reflections are consolidated into a rolling long-term summary.
5) **Retrieval**: `hydrate_long_term_context` searches episodic memories by topic and query and injects top matches into the prompt.

## 4. Session Resume Logic

- **RunLogger (`agent/utils/run_logger.py`)** stores sessions per therapist under `tests/runs/<therapist>.json`, including final state snapshots.
- **Restoring**: CLI and API restore the latest session for a therapist/patient pair to resume state.
- **Session Opening**: `agent/utils/session_opening.py` uses the restored state for a contextual greeting.

## 5. LLM Providers

- `LocalLLMRunner`: vLLM-backed HF model (GPU-friendly).
- `VertexLLMRunner`: Google Vertex AI (Gemini).

## 6. Prompts & Safety

`agent/core/prompt_builder.py` constructs the prompt in layers:
1) Primary guidance: Identity, Cognitive Style, Observed Interaction Style, Dominant Affective Systems.
2) Memory: summary, reflection, recent turns, episodic snippets (topic-gated).
3) Topic-conditioned sections from `data/topics_tree.json` (truncated for size).
4) Safety guardrails.

Parenthetical asides are stripped before prompting; memory/summary lengths are capped.

## 7. Emotion Synthesizer

- Panksepp systems: SEEKING, FEAR, RAGE, LUST, CARE, PANIC_GRIEF, PLAY.
- `update_emotional_state()` combines baseline + noise + context modifiers and smooths by salience.
- Baseline trait intensities + volatility stay fixed and anchor all updates.
- A single LLM classifier produces the prior-turn emotion label (strictly normalized to the Panksepp set) and topic label before response generation.
- The prior emotion label seeds a gentle bias toward continuity, then `update_emotional_state()` reacts to the therapist input (event + salience + volatility).
- Dominant systems (top 1–3) from the updated state are injected into the prompt, while the prior tone is referenced explicitly.
- `current_emotional_state` is updated from the classifier output and used for telemetry (API/logs).

## 8. Entry Points

Mode | File | Notes
---- | ---- | -----
CLI | `main.py` | Scripted/interactive; session end finalizes reflection + long-term summary.
Chat Shell | `scripts/chat_cli.py` | Minimal REPL; resumes prior state.
API | `agent/api/app.py` | `POST /chat-response` for turns; `POST /session-end` to finalize memory.

## 9. Extending the System

1) Update `State` in `langgraph_builder.py`.
2) Surface new data in `prompt_builder.py` if needed.
3) Persist new fields in `RunLogger` if they must survive sessions.
4) Update docs.
