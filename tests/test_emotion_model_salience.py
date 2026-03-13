import unittest


from agent.core import emotion_model


class TestEmotionModelSalience(unittest.TestCase):
    def test_salience_override_controls_smoothing(self):
        baseline = {emotion: 0.5 for emotion in emotion_model.EMOTIONS}
        previous = dict(baseline)

        original_sample_noise = emotion_model.sample_noise
        try:
            emotion_model.sample_noise = (
                lambda volatility_level, *, salience=1.0: {emotion: 0.0 for emotion in emotion_model.EMOTIONS}
            )

            low_salience = emotion_model.compute_emotional_state(
                baseline,
                volatility_level="medium",
                event="boundary",
                previous_state=previous,
                salience=0.0,
            )
            high_salience = emotion_model.compute_emotional_state(
                baseline,
                volatility_level="medium",
                event="boundary",
                previous_state=previous,
                salience=1.0,
            )

            target = emotion_model.compute_emotional_state(
                baseline,
                volatility_level="medium",
                event="boundary",
                salience=1.0,
            )
            delta = target["RAGE"] - previous["RAGE"]
            low_expected = previous["RAGE"] + emotion_model._smoothing_factor(0.0) * delta
            high_expected = previous["RAGE"] + emotion_model._smoothing_factor(1.0) * delta
        finally:
            emotion_model.sample_noise = original_sample_noise

        self.assertLess(low_salience["RAGE"], high_salience["RAGE"])
        self.assertAlmostEqual(low_salience["RAGE"], low_expected, places=6)
        self.assertAlmostEqual(high_salience["RAGE"], high_expected, places=6)


if __name__ == "__main__":
    unittest.main()
