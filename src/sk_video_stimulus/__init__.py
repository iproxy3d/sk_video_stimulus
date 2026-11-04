"""Low-level visual stimulus dynamics extracted from video streams."""

from .extractor import (
    EMOTION_ORDER,
    VIDEO_FEATURE_NAMES,
    FrameSignalExtractor,
    NormalizedVideoFeatures,
    OscillationConfig,
    OscillationFeatureExtractor,
    OscillationFeatures,
    SpatialEncoderConfig,
    VideoFeatureExtractor,
    VideoFeatureNormalizer,
    VideoFeatureSample,
    VideoNeutralProfile,
    VideoNeutralProfileBuilder,
    VideoProtoEmotionMapper,
    VideoProtoEmotionResult,
    process_frames,
)

__version__ = "0.1.0"

__all__ = [
    "EMOTION_ORDER",
    "VIDEO_FEATURE_NAMES",
    "FrameSignalExtractor",
    "NormalizedVideoFeatures",
    "OscillationConfig",
    "OscillationFeatureExtractor",
    "OscillationFeatures",
    "SpatialEncoderConfig",
    "VideoFeatureExtractor",
    "VideoFeatureNormalizer",
    "VideoFeatureSample",
    "VideoNeutralProfile",
    "VideoNeutralProfileBuilder",
    "VideoProtoEmotionMapper",
    "VideoProtoEmotionResult",
    "process_frames",
]
