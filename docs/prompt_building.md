# Prompt Building

This document explains how PsyLLM constructs the final prompt shown to the LLM
when generating a patient response. The implementation lives in
`agent/core/prompt_builder.py` and is called from the LangGraph pipeline.

## Overview

The prompt is layered so the most actionable guidance appears closest to the
final instruction. The builder combines:
- Patient identity and stable traits
- Topic-conditioned profile sections
- Conversation memory (summary, reflection, recent turns, episodic memories)
- Safety guardrails and affect continuity
- The latest therapist message as context only

## Inputs (state fields)

The builder reads these fields from the current LangGraph `state`:
- `patient_profile` (includes `details`, `emotion_state`, `emotion_intensity`)
- `intent_topic` and `last_topic`
- `summary`, `session_reflection`, `history`
- `episodic_context`
- `safety_flags`
- `emotion_event`
- `safe_user_input` (fallback: `user_input`)

## Layering order

From top to bottom, the prompt is assembled in this order:
1) Patient identity and cognitive style
2) Observed interaction style
3) Dominant affective systems (top 1-3)
4) Conversation memory (summary, reflection, recent turns, episodic memories)
5) Topic-conditioned profile sections (based on `data/topics_tree.json`)
6) Safety guardrails and safety flags
7) Turn context (topic, affect intensity, therapist input)
8) Final instruction for the LLM response

## Topic-conditioned sections

The file `data/topics_tree.json` maps a detected topic to relevant profile
fields (for example, `familyHistory` or `clinicalFunctioning`). The builder
uses those mappings to inject only the relevant sections for the current topic.

This keeps the prompt focused and avoids flooding the model with unrelated
patient details.

## Memory handling and size limits

To keep prompts stable over time, several limits are applied:
- Summary compression: long summaries are truncated and reduced to the first
  two and last two sentences.
- Reflection, profile sections, and episodic memory items are truncated by
  character count.
- Recent turns are capped to the last 3 exchanges.

Parenthetical asides are stripped before adding memory text to the prompt.

## Affect and safety

Dominant affective systems are selected by a temperature-scaled softmax over
`emotion_state`, then filtered and capped to the top 1-3 systems. The final
prompt includes:
- A short list of dominant systems with their hints
- An affect intensity descriptor (muted, steady, or high tension)
- A concise affect directive used in the final instruction

Safety guardrails come from `agent/core/safety.py`. Any `safety_flags` triggered
by the therapist input are appended with a reminder to reaffirm boundaries.

## Output

The builder returns a single `prompt` string:
```
return {"prompt": prompt}
```

This is passed to the configured LLM runner (local vLLM or Vertex AI).

## Extending the prompt

If you add new fields to the patient profile:
1) Extend `agent/core/patient_profile.py` to expose them via `to_prompt()`.
2) Add the field name to relevant topic metadata in `data/topics_tree.json`.
3) Update `agent/core/prompt_builder.py` if the field needs special handling.
