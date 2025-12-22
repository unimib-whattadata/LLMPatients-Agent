# Emotion Dynamics Walkthrough

This document explains how PsyLLM models a patient's momentary emotional state and how that affects the prompt.

## 1. Inputs from the Patient JSON

Each profile carries `emotionTraits`:

```json
"emotionTraits": {
  "volatility_level": "high",
  "trait_baseline": {
    "SEEKING": 0.70,
    "RAGE": 0.85,
    "FEAR": 0.80,
    "CARE": 0.65,
    "LUST": 0.65,
    "PANIC_GRIEF": 0.85,
    "PLAY": 0.30
  }
}
```

These values are clamped to `[0,1]` and represent Panksepp systems (here: high fear/rage/panic-grief, low playfulness).

## 2. Per-Turn Emotion Synthesis

When the therapist sends a message, LangGraph runs `update_emotional_state` with these steps:

1) **Context Event Detection** – `_detect_context_event` uses therapist text + safety flags + topic change to select an event:
   - `success_discussion`, `boundary`, `abandonment_cue`, `empathy`, or `neutral`.
2) **Salience Weight** – each event maps to a salience score; longer turns and topic changes slightly increase salience.
3) **Gaussian Noise** – per-emotion noise scaled by volatility and salience.
4) **Deterministic Modifiers** – event-specific deltas (e.g., `abandonment_cue` boosts FEAR and PANIC_GRIEF).
5) **Clamp + Smooth** – values are clamped and smoothed toward the previous turn based on salience. After multiple low-salience turns, the vector decays toward baseline.

The resulting vector is stored in `state.emotion_state` and drives the prompt. `emotion_intensity` is a weighted top-two average.

## 3. Prompt Builder Integration

The prompt exposes only the dominant 1–3 systems (softmax + floor), then instructs the model to follow those systems:

```
• Dominant affect systems: PANIC_GRIEF (0.82), RAGE (0.78), FEAR (0.79)
• Affect intensity: 0.82 (high tension and emotions close to the surface)
- Follow affect drivers: Panic/Grief → Separation distress...; Rage → Irritable edge...; Fear → Hypervigilant...
```

Muted emotions are hidden so the agent doesn’t drift into unrepresentative affect.

## 4. Tone Classification (Telemetry)

Separately from the vector, `update_memory()` uses a lightweight LLM classifier to label the patient's reply as one of the Panksepp systems. This label is used for telemetry (API and logs), while the emotion vector continues to drive prompt tone.

## 5. End-to-End Turn

Therapist input: **“We’re almost at time, but I’m proud of how you held your needs with your partner this week.”**

- Event detection: `success_discussion`.
- Modifiers: SEEKING +0.10, PLAY +0.10.
- The dominant systems remain PANIC_GRIEF/RAGE/FEAR, so the response is guarded but receptive.

