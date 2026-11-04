#!/usr/bin/env python3
"""Streaming extraction of low-level video stimulus features.

The module implements the video branch described in emotional-state-agents:

    frame -> 15 spatial components per region
          -> fast/slow temporal differences
          -> one signed global winner x_t
          -> oscillation features: activity, anxiety, tempo, deviation

The frame encoder is intentionally lightweight. With the default 256x256
input and a 4x4 region grid, every region is 64x64 and every internal cell is
16x16. NumPy therefore computes all cell sums with one reshape and reduction;
there are no Python loops over pixels.

Dependencies for the core classes:

    python -m pip install numpy pillow

OpenCV is needed only by the command-line video/camera reader:

    python -m pip install opencv-python

Example:

    python video_feature_extractor.py input.mp4 --output features.csv

The output x_t is divided by the fixed theoretical range of a region and is
therefore normally in [-1, 1]. This is a fixed geometric scale, not adaptive
normalization from the current video.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Iterable, Literal, Sequence

import numpy as np
from PIL import Image


ColorOrder = Literal["rgb", "bgr", "gray"]


@dataclass(frozen=True)
class SpatialEncoderConfig:
    """Configuration of the per-frame spatial and temporal encoder."""

    image_size: int = 256
    region_rows: int = 4
    region_cols: int = 4
    inner_grid: int = 4
    fps: float = 25.0
    baseline_tau_seconds: float = 1.0
    peripheral_weight: float = 0.65
    input_color_order: ColorOrder = "bgr"
    normalize_output: bool = True

    def validate(self) -> None:
        if self.image_size < 16:
            raise ValueError("image_size must be at least 16")
        if self.region_rows < 1 or self.region_cols < 1:
            raise ValueError("region grid dimensions must be positive")
        if self.inner_grid != 4:
            raise ValueError("the current 15-component basis requires inner_grid=4")
        row_divisor = self.region_rows * self.inner_grid
        col_divisor = self.region_cols * self.inner_grid
        if self.image_size % row_divisor or self.image_size % col_divisor:
            raise ValueError(
                "image_size must be divisible by region_rows*4 and region_cols*4"
            )
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.baseline_tau_seconds <= 0:
            raise ValueError("baseline_tau_seconds must be positive")
        if not 0 < self.peripheral_weight <= 1:
            raise ValueError("peripheral_weight must be in (0, 1]")
        if self.input_color_order not in {"rgb", "bgr", "gray"}:
            raise ValueError("input_color_order must be rgb, bgr, or gray")


@dataclass(frozen=True)
class OscillationConfig:
    """Configuration of the sliding-window oscillation analyzer."""

    fps: float = 25.0
    window_seconds: float = 4.0
    warmup_seconds: float = 1.0
    peak_threshold: float = 0.04
    min_peak_distance_seconds: float = 0.12

    def validate(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if not 0 <= self.warmup_seconds <= self.window_seconds:
            raise ValueError("warmup_seconds must be in [0, window_seconds]")
        if not 0 <= self.peak_threshold <= 1:
            raise ValueError("peak_threshold must be in [0, 1]")
        if self.min_peak_distance_seconds < 0:
            raise ValueError("min_peak_distance_seconds must be non-negative")


@dataclass(frozen=True)
class OscillationFeatures:
    """Four raw video proto-features produced from the current signal window.

    activity is signed and bounded to [-1, 1]. Deviation is RMS signal power
    in [0, 1] for a normalized input. Tempo is the number of significant
    positive/negative extrema per second. Anxiety combines signal power,
    direction changes and irregularity; it is intentionally left on a raw
    non-negative scale for later calibration against a fixed video profile.
    """

    timestamp: float
    ready: bool
    samples: int
    events: int
    activity: float
    anxiety: float
    tempo: float
    deviation: float


@dataclass(frozen=True)
class VideoFeatureSample:
    """One streaming result for one input frame."""

    frame_index: int
    timestamp: float
    x_t: float
    ready: bool
    activity: float
    anxiety: float
    tempo: float
    deviation: float
    events: int


VIDEO_FEATURE_NAMES = ("activity", "anxiety", "tempo", "deviation")

# The order is the same as in the emotional-state model and its audio matrix.
EMOTION_ORDER = (
    "disappointment",
    "sadness",
    "annoyance",
    "neutral",
    "disapproval",
    "realization",
    "nervousness",
    "approval",
    "joy",
    "anger",
    "embarrassment",
    "caring",
    "remorse",
    "disgust",
    "grief",
    "confusion",
    "relief",
    "desire",
    "admiration",
    "optimism",
    "fear",
    "love",
    "excitement",
    "curiosity",
    "amusement",
    "surprise",
    "gratitude",
    "pride",
)

# Rows are emotions, columns are activity/anxiety/tempo. The symbols reproduce
# the interpretation table used for audio. ``±`` is not an unknown sign: it
# ignores direction and contributes the absolute feature magnitude ``|s_j|``.
_VIDEO_MATRIX_SYMBOLS: dict[str, tuple[str, str, str]] = {
    "disappointment": ("-", "0", "0"),
    "sadness": ("-", "0", "-"),
    "annoyance": ("+", "+", "+"),
    "neutral": ("0", "0", "0"),
    "disapproval": ("-", "0", "±"),
    "realization": ("0", "-", "-"),
    "nervousness": ("+", "+", "+"),
    "approval": ("0", "-", "-"),
    "joy": ("+", "±", "+"),
    "anger": ("+", "+", "+"),
    "embarrassment": ("±", "+", "±"),
    "caring": ("±", "-", "±"),
    "remorse": ("-", "0", "-"),
    "disgust": ("-", "0", "±"),
    "grief": ("-", "0", "-"),
    "confusion": ("±", "+", "±"),
    "relief": ("-", "-", "-"),
    "desire": ("+", "±", "±"),
    "admiration": ("+", "-", "0"),
    "optimism": ("+", "-", "0"),
    "fear": ("+", "+", "+"),
    "love": ("+", "±", "±"),
    "excitement": ("+", "±", "+"),
    "curiosity": ("+", "±", "+"),
    "amusement": ("+", "0", "0"),
    "surprise": ("+", "±", "+"),
    "gratitude": ("0", "-", "0"),
    "pride": ("+", "0", "0"),
}


@dataclass(frozen=True)
class VideoNeutralProfile:
    """Mean raw features measured on one reference neutral video."""

    activity: float
    anxiety: float
    tempo: float
    deviation: float
    ready_samples: int
    source: str | None = None
    created_at_utc: str | None = None
    spatial_config: dict[str, Any] | None = None
    oscillation_config: dict[str, Any] | None = None
    format_version: int = 1

    def validate(self) -> None:
        if self.format_version != 1:
            raise ValueError(f"unsupported profile format: {self.format_version}")
        if self.ready_samples < 1:
            raise ValueError("neutral profile must contain at least one ready sample")
        values = np.asarray(
            [self.activity, self.anxiety, self.tempo, self.deviation],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("neutral profile contains a non-finite value")

    def save(self, path: str | Path) -> Path:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "VideoNeutralProfile":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        profile = cls(**data)
        profile.validate()
        return profile

    def assert_compatible(
        self,
        spatial_config: SpatialEncoderConfig,
        oscillation_config: OscillationConfig,
    ) -> None:
        """Reject a profile produced by a materially different extractor."""

        if self.spatial_config is not None and self.spatial_config != asdict(
            spatial_config
        ):
            raise ValueError(
                "neutral profile was created with a different spatial configuration"
            )
        if self.oscillation_config is not None and self.oscillation_config != asdict(
            oscillation_config
        ):
            raise ValueError(
                "neutral profile was created with a different oscillation configuration"
            )


class VideoNeutralProfileBuilder:
    """Accumulate ready feature windows and save their arithmetic mean."""

    def __init__(self) -> None:
        self._values: list[tuple[float, float, float, float]] = []

    @property
    def ready_samples(self) -> int:
        return len(self._values)

    def update(self, sample: VideoFeatureSample | OscillationFeatures) -> bool:
        if not sample.ready:
            return False
        self._values.append(
            (
                float(sample.activity),
                float(sample.anxiety),
                float(sample.tempo),
                float(sample.deviation),
            )
        )
        return True

    def build(
        self,
        spatial_config: SpatialEncoderConfig,
        oscillation_config: OscillationConfig,
        source: str | None = None,
    ) -> VideoNeutralProfile:
        if not self._values:
            raise ValueError(
                "neutral video produced no ready samples; increase its duration "
                "or reduce warmup_seconds"
            )
        means = np.asarray(self._values, dtype=np.float64).mean(axis=0)
        return VideoNeutralProfile(
            activity=float(means[0]),
            anxiety=float(means[1]),
            tempo=float(means[2]),
            deviation=float(means[3]),
            ready_samples=len(self._values),
            source=source,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            spatial_config=asdict(spatial_config),
            oscillation_config=asdict(oscillation_config),
        )


@dataclass(frozen=True)
class NormalizedVideoFeatures:
    """Calibrated matrix input s_video and non-negative stimulus strength."""

    activity: float
    anxiety: float
    tempo: float
    deviation: float

    def matrix_input(self) -> np.ndarray:
        return np.asarray(
            [self.activity, self.anxiety, self.tempo], dtype=np.float32
        )


class VideoFeatureNormalizer:
    """Center raw video features on a fixed non-zero neutral profile."""

    def __init__(
        self,
        profile: VideoNeutralProfile,
        oscillation_config: OscillationConfig,
    ) -> None:
        profile.validate()
        oscillation_config.validate()
        self.profile = profile
        self.config = oscillation_config

        # Fixed bounds follow from the analyzer, not from the current video.
        # This keeps a neutral reference from automatically absorbing unusual
        # stimuli into its normalization range.
        if self.config.min_peak_distance_seconds > 0:
            distance_limit = 1.0 / self.config.min_peak_distance_seconds
        else:
            distance_limit = self.config.fps / 2.0
        self.tempo_scale = max(
            min(self.config.fps / 2.0, distance_limit), 1e-6
        )
        self.anxiety_scale = max(
            self.config.fps + self.tempo_scale + 3.0, 1e-6
        )

    def transform(
        self,
        sample: VideoFeatureSample | OscillationFeatures,
    ) -> NormalizedVideoFeatures:
        if not sample.ready:
            raise ValueError("cannot normalize a feature window before warm-up")

        # activity itself is in [-1, 1], so the difference of two valid values
        # is in [-2, 2]. Other scales are fixed analyzer bounds.
        activity = (float(sample.activity) - self.profile.activity) / 2.0
        anxiety = (float(sample.anxiety) - self.profile.anxiety) / self.anxiety_scale
        tempo = (float(sample.tempo) - self.profile.tempo) / self.tempo_scale

        # Both a more dynamic and a less dynamic scene can be a stimulus.
        # Deviation has no direction in M and remains non-negative.
        deviation = abs(float(sample.deviation) - self.profile.deviation)
        return NormalizedVideoFeatures(
            activity=float(np.clip(activity, -1.0, 1.0)),
            anxiety=float(np.clip(anxiety, -1.0, 1.0)),
            tempo=float(np.clip(tempo, -1.0, 1.0)),
            deviation=float(np.clip(deviation, 0.0, 1.0)),
        )


@dataclass(frozen=True)
class VideoProtoEmotionResult:
    normalized: NormalizedVideoFeatures
    potential: np.ndarray
    correction: np.ndarray


class VideoProtoEmotionMapper:
    """Apply the audio-compatible interpretation matrix to video features."""

    def __init__(self) -> None:
        signed_weights = {"-": -1.0, "0": 0.0, "+": 1.0, "±": 0.0}
        self.matrix = np.asarray(
            [
                [
                    signed_weights[
                        _VIDEO_MATRIX_SYMBOLS[emotion][feature_index]
                    ]
                    for emotion in EMOTION_ORDER
                ]
                for feature_index in range(3)
            ],
            dtype=np.float32,
        )
        self.magnitude_matrix = np.asarray(
            [
                [
                    float(_VIDEO_MATRIX_SYMBOLS[emotion][feature_index] == "±")
                    for emotion in EMOTION_ORDER
                ]
                for feature_index in range(3)
            ],
            dtype=np.float32,
        )

    def transform(
        self,
        normalized: NormalizedVideoFeatures,
        previous_state: Sequence[float] | np.ndarray,
        eta: float = 1.0,
    ) -> VideoProtoEmotionResult:
        state = np.asarray(previous_state, dtype=np.float32)
        if state.shape != (len(EMOTION_ORDER),):
            raise ValueError(
                f"previous_state must have shape ({len(EMOTION_ORDER)},)"
            )
        if not np.all(np.isfinite(state)) or not math.isfinite(eta):
            raise ValueError("previous_state and eta must be finite")
        matrix_input = normalized.matrix_input()
        potential = (
            matrix_input @ self.matrix
            + np.abs(matrix_input) @ self.magnitude_matrix
        )
        correction = (
            float(eta) * normalized.deviation * state * potential
        ).astype(np.float32, copy=False)
        return VideoProtoEmotionResult(
            normalized=normalized,
            potential=potential.astype(np.float32, copy=False),
            correction=correction,
        )


def _build_spatial_masks() -> np.ndarray:
    """Return the 15 non-DC signed 4x4 masks as float32 [15, 4, 4]."""

    basis = np.asarray(
        [
            [1, 1, 1, 1],
            [1, 1, -1, -1],
            [1, -1, -1, 1],
            [1, -1, 1, -1],
        ],
        dtype=np.float32,
    )
    masks = []
    for row_index in range(4):
        for col_index in range(4):
            if row_index == 0 and col_index == 0:
                continue
            masks.append(np.outer(basis[row_index], basis[col_index]))
    return np.stack(masks, axis=0).astype(np.float32, copy=False)


def _build_region_weights(
    rows: int,
    cols: int,
    peripheral_weight: float,
) -> np.ndarray:
    """Create a fixed radial center-priority map with max weight exactly 1."""

    y = (np.arange(rows, dtype=np.float32) + 0.5) / rows - 0.5
    x = (np.arange(cols, dtype=np.float32) + 0.5) / cols - 0.5
    yy, xx = np.meshgrid(y, x, indexing="ij")
    distance = np.sqrt(xx * xx + yy * yy)
    minimum = float(distance.min())
    maximum = float(distance.max())
    if math.isclose(minimum, maximum):
        weights = np.ones_like(distance, dtype=np.float32)
    else:
        closeness = 1.0 - (distance - minimum) / (maximum - minimum)
        weights = peripheral_weight + (1.0 - peripheral_weight) * closeness
    return weights.reshape(-1).astype(np.float32, copy=False)


class FrameSignalExtractor:
    """Convert each frame into one signed strongest-change value x_t."""

    def __init__(self, config: SpatialEncoderConfig | None = None) -> None:
        self.config = config or SpatialEncoderConfig()
        self.config.validate()
        self.masks = _build_spatial_masks()
        self.region_weights = _build_region_weights(
            self.config.region_rows,
            self.config.region_cols,
            self.config.peripheral_weight,
        )
        region_height = self.config.image_size // self.config.region_rows
        region_width = self.config.image_size // self.config.region_cols
        self.region_area = float(region_height * region_width)
        self.previous_mu: np.ndarray | None = None
        self.baseline_mu: np.ndarray | None = None
        self.previous_timestamp: float | None = None

    @property
    def region_count(self) -> int:
        return self.config.region_rows * self.config.region_cols

    def reset(self) -> None:
        self.previous_mu = None
        self.baseline_mu = None
        self.previous_timestamp = None

    def _resize_gray(self, frame: Any, color_order: ColorOrder) -> np.ndarray:
        """Return a 256x256 float32 grayscale image in [0, 1]."""

        if isinstance(frame, Image.Image):
            image = frame.convert("L")
            image = image.resize(
                (self.config.image_size, self.config.image_size),
                Image.Resampling.BILINEAR,
            )
            return np.asarray(image, dtype=np.float32) / 255.0

        array = np.asarray(frame)
        if array.ndim == 2:
            gray = array
        elif array.ndim == 3 and array.shape[2] in {1, 3, 4}:
            if array.shape[2] == 1:
                gray = array[..., 0]
            else:
                rgb = array[..., :3]
                if color_order == "bgr":
                    rgb = rgb[..., ::-1]
                elif color_order == "gray":
                    raise ValueError("gray color order requires a 2D or one-channel frame")
                if np.issubdtype(rgb.dtype, np.floating):
                    maximum = float(np.nanmax(rgb)) if rgb.size else 0.0
                    if maximum <= 1.0:
                        rgb = np.clip(rgb, 0.0, 1.0) * 255.0
                    rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
                else:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                image = Image.fromarray(rgb, mode="RGB").convert("L")
                image = image.resize(
                    (self.config.image_size, self.config.image_size),
                    Image.Resampling.BILINEAR,
                )
                return np.asarray(image, dtype=np.float32) / 255.0
        else:
            raise ValueError(
                "frame must be a PIL image or an array shaped [H,W], [H,W,1/3/4]"
            )

        gray = np.asarray(gray)
        if np.issubdtype(gray.dtype, np.floating):
            maximum = float(np.nanmax(gray)) if gray.size else 0.0
            if maximum <= 1.0:
                prepared = np.clip(gray, 0.0, 1.0).astype(np.float32)
            else:
                prepared = (np.clip(gray, 0.0, 255.0) / 255.0).astype(np.float32)
            image = Image.fromarray(prepared, mode="F")
            image = image.resize(
                (self.config.image_size, self.config.image_size),
                Image.Resampling.BILINEAR,
            )
            return np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)

        image = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), mode="L")
        image = image.resize(
            (self.config.image_size, self.config.image_size),
            Image.Resampling.BILINEAR,
        )
        return np.asarray(image, dtype=np.float32) / 255.0

    def encode_frame_to_mu(
        self,
        frame: Any,
        color_order: ColorOrder | None = None,
    ) -> np.ndarray:
        """Encode one image into [regions, 15] signed spatial components."""

        order = color_order or self.config.input_color_order
        gray = self._resize_gray(frame, order)
        rows = self.config.region_rows
        cols = self.config.region_cols
        inner = self.config.inner_grid
        cell_height = self.config.image_size // (rows * inner)
        cell_width = self.config.image_size // (cols * inner)

        # [region_y, cell_y, pixel_y, region_x, cell_x, pixel_x]
        cells = gray.reshape(
            rows,
            inner,
            cell_height,
            cols,
            inner,
            cell_width,
        ).sum(axis=(2, 5), dtype=np.float32)
        # [region_y, region_x, cell_y, cell_x] -> [R, 4, 4]
        masses = cells.transpose(0, 2, 1, 3).reshape(self.region_count, inner, inner)
        return np.einsum("rab,iab->ri", masses, self.masks, optimize=True).astype(
            np.float32,
            copy=False,
        )

    def _baseline_alpha(self, timestamp: float | None) -> float:
        if timestamp is not None and self.previous_timestamp is not None:
            dt = timestamp - self.previous_timestamp
            if dt <= 0:
                raise ValueError("timestamps must be strictly increasing")
        else:
            dt = 1.0 / self.config.fps
        return 1.0 - math.exp(-dt / self.config.baseline_tau_seconds)

    def process_frame(
        self,
        frame: Any,
        timestamp: float | None = None,
        color_order: ColorOrder | None = None,
    ) -> float:
        """Process one frame and return one signed global winner x_t.

        Fast and slow candidates, all fixed regions, and all 15 spatial
        components participate in one global argmax by absolute magnitude.
        The winner's sign is preserved.
        """

        current_mu = self.encode_frame_to_mu(frame, color_order=color_order)
        if self.previous_mu is None:
            self.previous_mu = current_mu.copy()
            self.baseline_mu = current_mu.copy()
            self.previous_timestamp = timestamp
            return 0.0

        assert self.baseline_mu is not None
        fast = current_mu - self.previous_mu
        slow = current_mu - self.baseline_mu
        weights = self.region_weights[:, None]
        candidates = np.stack((fast * weights, slow * weights), axis=0)
        winner_flat = int(np.argmax(np.abs(candidates)))
        x_raw = float(candidates.reshape(-1)[winner_flat])

        alpha = self._baseline_alpha(timestamp)
        self.baseline_mu = (
            (1.0 - alpha) * self.baseline_mu + alpha * current_mu
        ).astype(np.float32, copy=False)
        self.previous_mu = current_mu.copy()
        self.previous_timestamp = timestamp

        if not self.config.normalize_output:
            return x_raw
        # Every non-DC mask has 8 positive and 8 negative cells. The maximum
        # difference between two frames is therefore one full region area.
        return float(np.clip(x_raw / self.region_area, -1.0, 1.0))

    def state_dict(self) -> dict[str, Any]:
        return {
            "previous_mu": (
                None if self.previous_mu is None else self.previous_mu.copy()
            ),
            "baseline_mu": (
                None if self.baseline_mu is None else self.baseline_mu.copy()
            ),
            "previous_timestamp": self.previous_timestamp,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        previous = state.get("previous_mu")
        baseline = state.get("baseline_mu")
        expected_shape = (self.region_count, 15)
        if (previous is None) != (baseline is None):
            raise ValueError("previous_mu and baseline_mu must both be set or both be None")
        if previous is not None:
            previous = np.asarray(previous, dtype=np.float32)
            baseline = np.asarray(baseline, dtype=np.float32)
            if previous.shape != expected_shape or baseline.shape != expected_shape:
                raise ValueError(f"state tensors must have shape {expected_shape}")
            self.previous_mu = previous.copy()
            self.baseline_mu = baseline.copy()
        else:
            self.previous_mu = None
            self.baseline_mu = None
        timestamp = state.get("previous_timestamp")
        self.previous_timestamp = None if timestamp is None else float(timestamp)


class OscillationFeatureExtractor:
    """Extract activity/anxiety/tempo/deviation from a rolling x_t window."""

    def __init__(self, config: OscillationConfig | None = None) -> None:
        self.config = config or OscillationConfig()
        self.config.validate()
        self.buffer: Deque[tuple[float, float]] = deque()
        self._next_timestamp = 0.0

    def reset(self) -> None:
        self.buffer.clear()
        self._next_timestamp = 0.0

    def _timestamp(self, timestamp: float | None) -> float:
        if timestamp is None:
            result = self._next_timestamp
            self._next_timestamp += 1.0 / self.config.fps
            return result
        result = float(timestamp)
        if self.buffer and result <= self.buffer[-1][0]:
            raise ValueError("timestamps must be strictly increasing")
        self._next_timestamp = result + 1.0 / self.config.fps
        return result

    def _find_events(self, times: np.ndarray, values: np.ndarray) -> np.ndarray:
        if len(values) < 3:
            return np.empty(0, dtype=np.int64)
        center = values[1:-1]
        positive = (
            (center > values[:-2])
            & (center >= values[2:])
            & (center >= self.config.peak_threshold)
        )
        negative = (
            (center < values[:-2])
            & (center <= values[2:])
            & (center <= -self.config.peak_threshold)
        )
        candidates = np.flatnonzero(positive | negative) + 1
        if len(candidates) < 2 or self.config.min_peak_distance_seconds == 0:
            return candidates.astype(np.int64, copy=False)

        selected: list[int] = []
        for index in candidates.tolist():
            if not selected:
                selected.append(index)
                continue
            if times[index] - times[selected[-1]] >= self.config.min_peak_distance_seconds:
                selected.append(index)
            elif abs(values[index]) > abs(values[selected[-1]]):
                selected[-1] = index
        return np.asarray(selected, dtype=np.int64)

    @staticmethod
    def _zero_crossing_rate(times: np.ndarray, values: np.ndarray) -> float:
        nonzero = values[np.abs(values) > 1e-8]
        if len(nonzero) < 2:
            return 0.0
        duration = max(float(times[-1] - times[0]), 1e-8)
        crossings = int(np.count_nonzero(np.signbit(nonzero[1:]) != np.signbit(nonzero[:-1])))
        return crossings / duration

    def update(
        self,
        x_t: float,
        timestamp: float | None = None,
    ) -> OscillationFeatures:
        current_time = self._timestamp(timestamp)
        value = float(np.clip(x_t, -1.0, 1.0))
        self.buffer.append((current_time, value))
        cutoff = current_time - self.config.window_seconds
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()

        times = np.fromiter((item[0] for item in self.buffer), dtype=np.float64)
        values = np.fromiter((item[1] for item in self.buffer), dtype=np.float32)
        duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
        ready = duration >= self.config.warmup_seconds
        if not ready:
            return OscillationFeatures(
                timestamp=current_time,
                ready=False,
                samples=len(values),
                events=0,
                activity=0.0,
                anxiety=0.0,
                tempo=0.0,
                deviation=0.0,
            )

        events = self._find_events(times, values)
        deviation = float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))

        # Signed mean change plus a smaller contour-trend term. Multiplication
        # by duration makes the slope contribution independent of window units.
        centered_time = times - float(times.mean())
        denominator = float(np.dot(centered_time, centered_time))
        if denominator > 1e-12:
            slope = float(np.dot(centered_time, values - float(values.mean())) / denominator)
        else:
            slope = 0.0
        activity = float(np.clip(float(values.mean()) + 0.25 * slope * duration, -1.0, 1.0))

        tempo = len(events) / max(duration, 1e-8)
        zero_crossing_rate = self._zero_crossing_rate(times, values)
        alternation = 0.0
        interval_cv = 0.0
        if len(events) >= 2:
            event_values = values[events]
            alternation = float(
                np.mean(np.signbit(event_values[1:]) != np.signbit(event_values[:-1]))
            )
            intervals = np.diff(times[events])
            mean_interval = float(intervals.mean())
            if len(intervals) >= 2 and mean_interval > 1e-8:
                interval_cv = float(intervals.std(ddof=0) / mean_interval)

        # Raw non-negative instability score. It intentionally remains
        # uncalibrated; a fixed deployment profile maps it to the matrix input.
        anxiety = deviation * (
            zero_crossing_rate
            + tempo * alternation
            + min(interval_cv, 3.0)
        )
        return OscillationFeatures(
            timestamp=current_time,
            ready=True,
            samples=len(values),
            events=len(events),
            activity=activity,
            anxiety=float(anxiety),
            tempo=float(tempo),
            deviation=float(np.clip(deviation, 0.0, 1.0)),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "buffer": list(self.buffer),
            "next_timestamp": self._next_timestamp,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        buffer = deque()
        previous_time: float | None = None
        for raw_time, raw_value in state.get("buffer", []):
            time = float(raw_time)
            value = float(raw_value)
            if previous_time is not None and time <= previous_time:
                raise ValueError("saved buffer timestamps must be strictly increasing")
            buffer.append((time, value))
            previous_time = time
        self.buffer = buffer
        self._next_timestamp = float(state.get("next_timestamp", 0.0))


class VideoFeatureExtractor:
    """Complete streaming frame -> x_t -> four proto-feature pipeline."""

    def __init__(
        self,
        spatial_config: SpatialEncoderConfig | None = None,
        oscillation_config: OscillationConfig | None = None,
    ) -> None:
        spatial = spatial_config or SpatialEncoderConfig()
        oscillation = oscillation_config or OscillationConfig(fps=spatial.fps)
        if not math.isclose(spatial.fps, oscillation.fps):
            raise ValueError("spatial and oscillation FPS must match")
        self.frame_encoder = FrameSignalExtractor(spatial)
        self.oscillation = OscillationFeatureExtractor(oscillation)
        self.frame_index = 0

    def reset(self) -> None:
        self.frame_encoder.reset()
        self.oscillation.reset()
        self.frame_index = 0

    def process_frame(
        self,
        frame: Any,
        timestamp: float | None = None,
        color_order: ColorOrder | None = None,
    ) -> VideoFeatureSample:
        if timestamp is None:
            timestamp = self.frame_index / self.frame_encoder.config.fps
        x_t = self.frame_encoder.process_frame(
            frame,
            timestamp=timestamp,
            color_order=color_order,
        )
        features = self.oscillation.update(x_t, timestamp=timestamp)
        sample = VideoFeatureSample(
            frame_index=self.frame_index,
            timestamp=float(timestamp),
            x_t=x_t,
            ready=features.ready,
            activity=features.activity,
            anxiety=features.anxiety,
            tempo=features.tempo,
            deviation=features.deviation,
            events=features.events,
        )
        self.frame_index += 1
        return sample

    def state_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "frame_encoder": self.frame_encoder.state_dict(),
            "oscillation": self.oscillation.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.frame_index = int(state["frame_index"])
        self.frame_encoder.load_state_dict(state["frame_encoder"])
        self.oscillation.load_state_dict(state["oscillation"])


def process_frames(
    frames: Iterable[Any],
    extractor: VideoFeatureExtractor,
    color_order: ColorOrder | None = None,
) -> Iterable[VideoFeatureSample]:
    for frame in frames:
        yield extractor.process_frame(frame, color_order=color_order)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="video path or integer camera index")
    parser.add_argument("--output", type=Path, default=Path("video_features.csv"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--region-grid", type=int, default=4)
    parser.add_argument("--fps", type=float, default=None, help="override source FPS")
    parser.add_argument("--baseline-tau", type=float, default=1.0)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--peak-threshold", type=float, default=0.04)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means no limit")
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument(
        "--calibrate-profile",
        type=Path,
        help="treat input as neutral reference video and save mean profile JSON",
    )
    profile_group.add_argument(
        "--profile",
        type=Path,
        help="load a neutral profile and append normalized matrix inputs to CSV",
    )
    return parser.parse_args()


def _open_video_source(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _run_cli(args: argparse.Namespace) -> int:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "OpenCV is required for video/camera input: "
            "python -m pip install opencv-python"
        ) from exc

    capture = cv2.VideoCapture(_open_video_source(args.input))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video source: {args.input}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = args.fps or (source_fps if source_fps > 1e-6 else 25.0)
    spatial = SpatialEncoderConfig(
        image_size=args.image_size,
        region_rows=args.region_grid,
        region_cols=args.region_grid,
        fps=fps,
        baseline_tau_seconds=args.baseline_tau,
        input_color_order="bgr",
    )
    oscillation = OscillationConfig(
        fps=fps,
        window_seconds=args.window_seconds,
        warmup_seconds=args.warmup_seconds,
        peak_threshold=args.peak_threshold,
    )
    extractor = VideoFeatureExtractor(spatial, oscillation)
    profile_builder = VideoNeutralProfileBuilder() if args.calibrate_profile else None
    normalizer = None
    if args.profile:
        profile = VideoNeutralProfile.load(args.profile)
        profile.assert_compatible(spatial, oscillation)
        normalizer = VideoFeatureNormalizer(profile, oscillation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in VideoFeatureSample.__dataclass_fields__.values()]
    if normalizer is not None:
        fieldnames.extend(
            ["s_activity", "s_anxiety", "s_tempo", "normalized_deviation"]
        )

    processed = 0
    try:
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = processed / fps
                sample = extractor.process_frame(frame, timestamp=timestamp)
                if profile_builder is not None:
                    profile_builder.update(sample)
                row = asdict(sample)
                if normalizer is not None and sample.ready:
                    normalized = normalizer.transform(sample)
                    row.update(
                        {
                            "s_activity": normalized.activity,
                            "s_anxiety": normalized.anxiety,
                            "s_tempo": normalized.tempo,
                            "normalized_deviation": normalized.deviation,
                        }
                    )
                writer.writerow(row)
                processed += 1
                if args.max_frames and processed >= args.max_frames:
                    break
    finally:
        capture.release()
    if profile_builder is not None:
        profile = profile_builder.build(spatial, oscillation, source=args.input)
        profile.save(args.calibrate_profile)
        print(f"neutral ready samples: {profile.ready_samples}")
        print(f"neutral profile: {args.calibrate_profile.resolve()}")
    print(f"processed frames: {processed}")
    print(f"fps: {fps:.6g}")
    print(f"features: {args.output.resolve()}")
    return 0


def main() -> int:
    return _run_cli(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
