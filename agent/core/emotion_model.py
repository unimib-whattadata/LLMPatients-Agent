"""Compute momentary emotional states using baselines, stochastic noise, and contextual modifiers."""

from __future__ import annotations

import random
from typing import Dict, Optional

EMOTIONS = ["SEEKING", "RAGE", "FEAR", "CARE", "LUST", "SADNESS", "PLAY"]

VOLATILITY_SIGMA = {
    "low": 0.05,
    "medium": 0.08,
    "high": 0.12,
}

CONTEXT_MODIFIERS = {
    "empathy": {"SADNESS": -0.10, "CARE": 0.10},
    "boundary": {"RAGE": 0.15, "FEAR": 0.10},
    "abandonment_cue": {"FEAR": 0.20, "SADNESS": 0.15},
    "success_discussion": {"SEEKING": 0.10, "PLAY": 0.10},
    "neutral": {},
}

EVENT_SALIENCE = {
    "neutral": 0.2,
    "empathy": 0.4,
    "success_discussion": 0.5,
    "boundary": 0.75,
    "abandonment_cue": 0.9,
}


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Constrain a numeric value."""
    return max(lower, min(upper, value))


def _normalized_baseline(trait_baseline: Dict[str, float]) -> Dict[str, float]:
    """Ensure baselines cover all emotions and respect the valid range."""
    normalized = {}
    for emotion in EMOTIONS:
        normalized[emotion] = clamp(trait_baseline.get(emotion, 0.0))
    return normalized


def sample_noise(volatility_level: str, *, salience: float = 1.0) -> Dict[str, float]:
    """
    Return Gaussian noise per emotion dimension.

    Distribution: Normal(0, sigma) where sigma depends on the patient's volatility level.
    The `salience` multiplier damps noise when the therapist turn is low-impact.
    """
    sigma = VOLATILITY_SIGMA.get(volatility_level, VOLATILITY_SIGMA["medium"])
    # Map salience in [0,1] → multiplier in [0.25, 1.0] so neutral turns barely move.
    salience = max(0.0, min(1.0, salience))
    effective_sigma = sigma * (0.25 + 0.75 * salience)
    return {emotion: random.gauss(0.0, effective_sigma) for emotion in EMOTIONS}


def context_adjustments(event: str) -> Dict[str, float]:
    """Fetch deterministic modifier deltas for the supplied context event."""
    return CONTEXT_MODIFIERS.get(event, CONTEXT_MODIFIERS["neutral"])


def compute_emotional_state(
    trait_baseline: Dict[str, float],
    *,
    volatility_level: str,
    event: str = "neutral",
    previous_state: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Combine baseline + noise + context modifiers to produce the momentary emotional map."""
    baseline = _normalized_baseline(trait_baseline)
    salience = EVENT_SALIENCE.get(event, EVENT_SALIENCE["neutral"])
    noise = sample_noise(volatility_level, salience=salience)
    modifiers = context_adjustments(event)

    target = {}
    for emotion in EMOTIONS:
        raw = baseline[emotion] + noise[emotion] + modifiers.get(emotion, 0.0)
        target[emotion] = clamp(raw)

    # Apply light counterweights so supportive systems can soften hot ones.
    target = _apply_counterweights(target)

    if not previous_state:
        return target

    smoothing = _smoothing_factor(salience)
    smoothed = {}
    for emotion in EMOTIONS:
        prev = clamp(previous_state.get(emotion, baseline[emotion]))
        smoothed[emotion] = clamp(prev + smoothing * (target[emotion] - prev))
    return smoothed


def _smoothing_factor(salience: float) -> float:
    """Convert salience (0–1) into a smoothing factor favoring stability."""
    salience = max(0.0, min(1.0, salience))
    # Neutral turns → ~0.2 (slow drift), high-salience → up to 0.8 (faster response).
    return 0.2 + salience * 0.6


def _apply_counterweights(vector: Dict[str, float]) -> Dict[str, float]:
    """Dampen hot systems slightly if their counterweights are present."""
    softened = dict(vector)
    if softened.get("RAGE", 0.0) > 0.6 and softened.get("CARE", 0.0) > 0.4:
        softened["RAGE"] = clamp(softened["RAGE"] - 0.03)
    if softened.get("RAGE", 0.0) > 0.6 and softened.get("PLAY", 0.0) > 0.25:
        softened["RAGE"] = clamp(softened["RAGE"] - 0.02)
    if softened.get("SADNESS", 0.0) > 0.6 and softened.get("PLAY", 0.0) > 0.3:
        softened["SADNESS"] = clamp(softened["SADNESS"] - 0.02)
    return softened
