#!/usr/bin/env python3
"""Минимальный пример обработки кадров веб-камеры с предпросмотром."""

from __future__ import annotations

import argparse
import time

import cv2

from sk_video_stimulus import OscillationConfig, SpatialEncoderConfig, VideoFeatureExtractor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="индекс камеры")
    parser.add_argument("--fps", type=float, default=25.0, help="частота обработки")
    args = parser.parse_args()

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
            for index, text in enumerate(lines):
                cv2.putText(
                    frame, text, (14, 28 + index * 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (70, 230, 170), 2, cv2.LINE_AA,
                )
            cv2.imshow("sk_video_stimulus — Q для выхода", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
