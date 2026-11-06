"""Низкоуровневая динамика зрительного стимула из видеопотока."""

from importlib.metadata import PackageNotFoundError, version

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

try:
    __version__ = version("sk-video-stimulus")
except PackageNotFoundError:  # Запуск непосредственно из дерева исходников.
    __version__ = "0+unknown"

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
