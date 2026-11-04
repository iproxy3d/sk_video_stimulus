from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from sk_video_stimulus import (
    EMOTION_ORDER,
    NormalizedVideoFeatures,
    OscillationConfig,
    OscillationFeatures,
    SpatialEncoderConfig,
    VideoFeatureNormalizer,
    VideoNeutralProfile,
    VideoNeutralProfileBuilder,
    VideoProtoEmotionMapper,
)


class NeutralProfileTests(unittest.TestCase):
    def test_profile_roundtrip(self) -> None:
        profile = VideoNeutralProfile(0.1, 0.2, 0.3, 0.4, 12, source="neutral.mp4")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            profile.save(path)
            self.assertEqual(VideoNeutralProfile.load(path), profile)

    def test_builder_ignores_warmup_and_averages_ready_samples(self) -> None:
        builder = VideoNeutralProfileBuilder()
        cold = OscillationFeatures(0.0, False, 1, 0, 0.0, 0.0, 0.0, 0.0)
        left = OscillationFeatures(1.0, True, 25, 2, 0.2, 0.4, 1.0, 0.1)
        right = OscillationFeatures(2.0, True, 50, 3, 0.4, 0.8, 3.0, 0.3)
        self.assertFalse(builder.update(cold))
        self.assertTrue(builder.update(left))
        self.assertTrue(builder.update(right))
        profile = builder.build(SpatialEncoderConfig(), OscillationConfig())
        self.assertAlmostEqual(profile.activity, 0.3)
        self.assertAlmostEqual(profile.anxiety, 0.6)
        self.assertAlmostEqual(profile.tempo, 2.0)
        self.assertAlmostEqual(profile.deviation, 0.2)
        self.assertEqual(profile.ready_samples, 2)

    def test_normalizer_centers_neutral_values(self) -> None:
        config = OscillationConfig()
        profile = VideoNeutralProfile(0.2, 0.4, 1.5, 0.25, 10)
        sample = OscillationFeatures(2.0, True, 50, 4, 0.2, 0.4, 1.5, 0.25)
        normalized = VideoFeatureNormalizer(profile, config).transform(sample)
        self.assertEqual(normalized, NormalizedVideoFeatures(0.0, 0.0, 0.0, 0.0))

    def test_normalizer_rejects_warmup_sample(self) -> None:
        profile = VideoNeutralProfile(0.0, 0.0, 0.0, 0.0, 1)
        sample = OscillationFeatures(0.0, False, 1, 0, 0.0, 0.0, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "warm-up"):
            VideoFeatureNormalizer(profile, OscillationConfig()).transform(sample)


class ProtoEmotionTests(unittest.TestCase):
    def test_mapper_returns_one_value_per_emotion(self) -> None:
        mapper = VideoProtoEmotionMapper()
        normalized = NormalizedVideoFeatures(0.4, 0.2, 0.3, 0.5)
        state = np.ones(len(EMOTION_ORDER), dtype=np.float32)
        result = mapper.transform(normalized, state, eta=0.25)
        self.assertEqual(result.potential.shape, (len(EMOTION_ORDER),))
        self.assertEqual(result.correction.shape, (len(EMOTION_ORDER),))
        np.testing.assert_allclose(result.correction, 0.125 * result.potential)

    def test_mapper_validates_state_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            VideoProtoEmotionMapper().transform(
                NormalizedVideoFeatures(0.0, 0.0, 0.0, 0.0),
                np.zeros(3),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
