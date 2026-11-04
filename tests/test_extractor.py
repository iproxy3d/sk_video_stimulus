#!/usr/bin/env python3
"""Dependency-light tests for video_feature_extractor.py."""

from __future__ import annotations

import math
import unittest

import numpy as np

from sk_video_stimulus import (
    FrameSignalExtractor,
    OscillationConfig,
    OscillationFeatureExtractor,
    SpatialEncoderConfig,
    VideoFeatureExtractor,
)


class FrameSignalExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SpatialEncoderConfig(
            image_size=256,
            region_rows=4,
            region_cols=4,
            fps=25.0,
            baseline_tau_seconds=1.0,
            input_color_order="gray",
        )

    def test_static_video_stays_at_zero(self) -> None:
        extractor = FrameSignalExtractor(self.config)
        frame = np.full((256, 256), 127, dtype=np.uint8)
        values = [extractor.process_frame(frame, timestamp=i / 25.0) for i in range(8)]
        self.assertTrue(all(abs(value) < 1e-7 for value in values))

    def test_asymmetric_change_creates_signed_response(self) -> None:
        extractor = FrameSignalExtractor(self.config)
        dark = np.zeros((256, 256), dtype=np.uint8)
        changed = dark.copy()
        changed[64:80, 64:80] = 255
        first = extractor.process_frame(dark, timestamp=0.0)
        impulse = extractor.process_frame(changed, timestamp=1 / 25.0)
        held = extractor.process_frame(changed, timestamp=2 / 25.0)
        returned = extractor.process_frame(dark, timestamp=3 / 25.0)
        self.assertEqual(first, 0.0)
        self.assertGreater(abs(impulse), 0.0)
        self.assertGreater(abs(held), 0.0)  # slow channel remains active
        self.assertGreater(abs(returned), 0.0)
        self.assertLessEqual(abs(impulse), 1.0)

    def test_state_roundtrip_is_exact(self) -> None:
        frame_a = np.zeros((256, 256), dtype=np.uint8)
        frame_b = frame_a.copy()
        frame_b[16:48, 32:64] = 255
        left = FrameSignalExtractor(self.config)
        left.process_frame(frame_a, timestamp=0.0)
        left.process_frame(frame_b, timestamp=0.04)
        state = left.state_dict()

        right = FrameSignalExtractor(self.config)
        right.load_state_dict(state)
        next_left = left.process_frame(frame_a, timestamp=0.08)
        next_right = right.process_frame(frame_a, timestamp=0.08)
        self.assertAlmostEqual(next_left, next_right, places=7)


class OscillationFeatureExtractorTests(unittest.TestCase):
    def test_periodic_signal_produces_tempo(self) -> None:
        fps = 25.0
        analyzer = OscillationFeatureExtractor(
            OscillationConfig(
                fps=fps,
                window_seconds=4.0,
                warmup_seconds=1.0,
                peak_threshold=0.1,
                min_peak_distance_seconds=0.1,
            )
        )
        result = None
        for index in range(int(5 * fps)):
            time = index / fps
            value = 0.6 * math.sin(2.0 * math.pi * 2.0 * time)
            result = analyzer.update(value, timestamp=time)
        assert result is not None
        self.assertTrue(result.ready)
        self.assertGreater(result.tempo, 3.0)
        self.assertLess(result.tempo, 5.0)
        self.assertGreater(result.anxiety, 0.0)
        self.assertGreater(result.deviation, 0.35)

    def test_pipeline_returns_one_sample_per_frame(self) -> None:
        extractor = VideoFeatureExtractor(
            SpatialEncoderConfig(input_color_order="gray"),
            OscillationConfig(),
        )
        frame = np.zeros((256, 256), dtype=np.uint8)
        samples = [extractor.process_frame(frame) for _ in range(30)]
        self.assertEqual([sample.frame_index for sample in samples], list(range(30)))
        self.assertTrue(samples[-1].ready)
        self.assertEqual(samples[-1].x_t, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
