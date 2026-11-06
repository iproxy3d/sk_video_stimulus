#!/usr/bin/env python3
"""Минимальный пример потоковой обработки видеофайла."""

from __future__ import annotations

import argparse

import cv2

from sk_video_stimulus import OscillationConfig, SpatialEncoderConfig, VideoFeatureExtractor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="путь к видеофайлу")
    parser.add_argument("--max-frames", type=int, default=0, help="0 — обработать всё")
    args = parser.parse_args()

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


if __name__ == "__main__":
    raise SystemExit(main())
