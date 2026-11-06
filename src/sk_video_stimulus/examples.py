"""Запускаемые после установки примеры видеофайла и веб-камеры."""

from __future__ import annotations

import argparse
import time
from typing import Any

from .extractor import OscillationConfig, SpatialEncoderConfig, VideoFeatureExtractor


def _cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - зависит от окружения пользователя
        raise SystemExit(
            "Для примеров установите OpenCV: "
            "python -m pip install 'sk-video-stimulus[video]'"
        ) from exc
    return cv2


def video_main() -> int:
    """Обработать видеофайл и раз в секунду печатать текущие признаки."""

    parser = argparse.ArgumentParser(description="Пример обработки видеофайла")
    parser.add_argument("video", help="путь к видеофайлу")
    parser.add_argument("--max-frames", type=int, default=0, help="0 — обработать всё")
    args = parser.parse_args()
    cv2 = _cv2()

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"Не удалось открыть видео: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    extractor = VideoFeatureExtractor(
        SpatialEncoderConfig(fps=fps, input_color_order="bgr"),
        OscillationConfig(fps=fps),
    )
    processed = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            sample = extractor.process_frame(frame, timestamp=processed / fps)
            if sample.ready and processed % max(1, round(fps)) == 0:
                print(
                    f"t={sample.timestamp:7.2f}  x_t={sample.x_t:+.4f}  "
                    f"activity={sample.activity:+.4f}  anxiety={sample.anxiety:.4f}  "
                    f"tempo={sample.tempo:.3f}  deviation={sample.deviation:.4f}"
                )
            processed += 1
            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        capture.release()
    print(f"Обработано кадров: {processed}; FPS: {fps:.3f}")
    return 0


def webcam_main() -> int:
    """Обрабатывать веб-камеру и показывать значения поверх изображения."""

    parser = argparse.ArgumentParser(description="Пример обработки веб-камеры")
    parser.add_argument("--camera", type=int, default=0, help="индекс камеры")
    parser.add_argument("--fps", type=float, default=25.0, help="частота обработки")
    args = parser.parse_args()
    cv2 = _cv2()

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"Не удалось открыть камеру {args.camera}")
    extractor = VideoFeatureExtractor(
        SpatialEncoderConfig(fps=args.fps, input_color_order="bgr"),
        OscillationConfig(fps=args.fps),
    )
    started_at = time.monotonic()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            sample = extractor.process_frame(frame, timestamp=time.monotonic() - started_at)
            lines = [
                f"x_t {sample.x_t:+.3f}",
                f"activity {sample.activity:+.3f}",
                f"anxiety {sample.anxiety:.3f}",
                f"tempo {sample.tempo:.2f}",
                f"deviation {sample.deviation:.3f}",
                "ready" if sample.ready else "warm-up",
            ]
            for index, label in enumerate(lines):
                cv2.putText(
                    frame,
                    label,
                    (14, 28 + index * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (70, 230, 170),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow("sk_video_stimulus — Q для выхода", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0
