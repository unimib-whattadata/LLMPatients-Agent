# Emotion Dynamics Walkthrough

This document explains how PsyLLM models a patient's momentary emotional state each turn and how that affects the prompt the LLM receives.

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
    "SADNESS": 0.85,
    "PLAY": 0.30
  }
}
```

These values are clamped to `[0,1]` and stored in `PatientProfile.EmotionDynamics`. They represent the archetypal Panksepp systems for the persona (here, Juanita: high fear/rage/sadness, low playfulness).

## 2. Per-Turn Emotion Synthesis

When the therapist sends a message, LangGraph runs `update_emotional_state` with these steps:

1. **Context Event Detection** – the therapist text (or safety flags) is matched against `_detect_context_event`:
   - "I'm proud of how you handled that boundary" → `success_discussion`
   - "We need to end here for today" → `abandonment_cue`
   - No keywords + no safety flags → `neutral`

2. **Salience Weight** – each event maps to a salience score (`neutral=0.2`, `empathy=0.4`, `success_discussion=0.5`, `boundary=0.75`, `abandonment_cue=0.9`). This score controls how much Gaussian noise we add.

3. **Gaussian Noise** – for every emotion we sample `Normal(0, sigma)` where `sigma = base_sigma(volatility) * (0.25 + 0.75 * salience)`. Low-salience turns barely move; high-salience turns swing harder. Example for Juanita (`volatility=high → base_sigma=0.12`):

```
neutral salience 0.2 → multiplier 0.25 + 0.75*0.2 = 0.40 → sigma ≈ 0.048
boundary salience 0.75 → multiplier 0.25 + 0.75*0.75 = 0.8125 → sigma ≈ 0.0975
```

4. **Deterministic Modifiers** – we then add fixed deltas per event:

| Event              | Modifiers                                             |
|--------------------|-------------------------------------------------------|
| `empathy`          | SADNESS −0.10, CARE +0.10                             |
| `boundary`         | RAGE +0.15, FEAR +0.10                                |
| `abandonment_cue`  | FEAR +0.20, SADNESS +0.15                             |
| `success_discussion` | SEEKING +0.10, PLAY +0.10                           |
| `neutral`          | (no change)                                           |

5. **Clamp and Smooth** – `baseline + noise + modifier` is clamped to `[0,1]` and exponentially smoothed with the previous turn's vector. The smoothing factor is `0.2 + salience * 0.6`. Neutral turns (salience 0.2) blend 32% of the new value with 68% of the old value; abandonment cues (salience 0.9) blend ~74% new with 26% old.

The resulting map is saved into `state.emotion_state`, `profile.emotion_state`, and is used to compute `emotion_intensity` (the max value) and `emotion_event`.

## 3. Prompt Builder Integration

`prompt_builder` now includes:

```
🎚️ Dominant Affective Systems
{
  "RAGE": "0.87",
  "FEAR": "0.79",
  "SADNESS": "0.81"
}
...
• Dominant affect systems: RAGE (0.87), SADNESS (0.81), FEAR (0.79)
• Affect intensity: 0.87 (high tension and emotions close to the surface)
• Therapist-triggered context event: boundary
...
- Follow affect drivers: RAGE (0.87) → Irritable, confrontational edge...; SADNESS ...
```

Only the top 1–3 systems are surfaced, so the LLM never leans on muted emotions (e.g., PLAY 0.14 is omitted). Morality and dysfunctional behavior sections are always included (unless their values are "Not reported") so the persona remembers its ethical boundaries and risks.

## 4. Full Turn Timeline

Therapist input: **“We’re almost at time, but I’m proud of how you held your needs with your partner this week.”**

Below is the exact order of nodes and what each one contributes for a single LangGraph invocation (Juanita, volatility `high`).

| Step | Node / Function | Internal Work |
| ---- | --------------- | ------------- |
| 1 | `sanitize_user_input` | Text is safe → `safe_user_input` unchanged, no safety flags. |
| 2 | `detect_intent_topic` | Embedding search flags *Relationships → BoundariesCommunication & RecurringDynamics*. |
| 3 | `hydrate_long_term_context` | Pulls 2 snippets about last session’s argument. |
| 4 | `update_emotional_state` | Detailed breakdown below. |
| 5 | `build_prompt` | Injects profile, morality/dysfunction sections, dominant affect systems, context bullet list. |
| 6 | `generate_response` | LLM receives the prompt and answers in character. |
| 7 | `update_memory` | Stores the new turn, writes telemetry to RunLogger, persists the latest emotion vector for smoothing next turn. |

### 4.1 Emotion Update Internals

1. **Event detection**: keywords “proud”/“held your needs” ⇒ `success_discussion` (salience 0.5).
2. **Noise**: `base_sigma(juanita=0.12)` × `(0.25 + 0.75 * 0.5) = 0.075`. Example draws (per dimension):
   - SEEKING noise +0.05
   - RAGE noise −0.07
   - FEAR noise −0.02
   - SADNESS noise −0.03
   - PLAY noise +0.02
3. **Deterministic modifiers**: +0.10 to SEEKING, +0.10 to PLAY.
4. **Baseline + noise + modifier**:
   - SEEKING: 0.70 + 0.05 + 0.10 → **0.85** (clamped)
   - RAGE:    0.85 − 0.07        → **0.78**
   - FEAR:    0.80 − 0.02        → **0.78**
   - SADNESS: 0.85 − 0.03        → **0.82**
   - PLAY:    0.30 + 0.02 + 0.10 → **0.42**
5. **Smoothing**: salience=0.5 ⇒ smoothing factor `0.2 + 0.6*0.5 = 0.5`. If the previous FEAR value was 0.80, the new FEAR becomes `0.80 + 0.5*(0.78 - 0.80) = 0.79`. Repeat for each dimension. The dominant systems stay SADNESS/RAGE/FEAR; SEEKING/PLAY rises but not enough to overtake them.
6. **State writes**: `state.emotion_state` holds the final smoothed map, `emotion_intensity` is the max value (0.82), `emotion_event` is `success_discussion`. These get persisted in the run log for future turns.

### 4.2 Prompt Surface

```
🎚️ Dominant Affective Systems
{
  "RAGE": "0.78",
  "SADNESS": "0.82",
  "FEAR": "0.79"
}

🧩 Context for This Turn
• Last discussed topic: Relationships → BoundariesCommunication & RecurringDynamics
• Current detected topic: Relationships → BoundariesCommunication & RecurringDynamics
• Dominant affect systems: SADNESS (0.82), RAGE (0.78), FEAR (0.79)
• Affect intensity: 0.82 (high tension and emotions close to the surface)
• Therapist-triggered context event: success_discussion
• Therapist's latest message: "We’re almost at time..."

✳️ Instruction (excerpt)
- Follow affect drivers: SADNESS (0.82) → Heavy, resigned tone; RAGE (0.78) → Irritable edge; FEAR (0.79) → hypervigilance. Avoid SEEKING/PLAY unless they rise above muted levels.
```

### 4.3 Outcome

Because the therapist praised Juanita, SEEKING/PLAY nudge upward—but not enough to unseat the dominant grief/anger/fear profile—so her reply remains guardedly hopeful: proud of herself yet anxious about the session ending. If the therapist had instead said “We need to stop here, goodbye,” the `abandonment_cue` modifier (salience 0.9) would quickly spike FEAR/SADNESS while the smoothing factor (≈0.74) prevents the next neutral turn from instantly snapping back.
