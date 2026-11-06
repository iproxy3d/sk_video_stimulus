#!/usr/bin/env python3
"""Тесты HTTP-контракта локального dashboard."""

from __future__ import annotations

import io
import unittest

from PIL import Image

from sk_video_stimulus.dashboard import create_app


def _jpeg(value: int = 80) -> io.BytesIO:
    stream = io.BytesIO()
    Image.new("RGB", (64, 64), (value, value, value)).save(stream, format="JPEG")
    stream.seek(0)
    return stream


class DashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        app = create_app()
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_index_is_available(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("sk_video_stimulus", response.get_data(as_text=True))

    def test_reset_and_process_frame(self) -> None:
        reset = self.client.post("/api/reset", json={"fps": 25})
        self.assertEqual(reset.status_code, 200)
        session_id = reset.get_json()["session_id"]

        first = self.client.post(
            "/api/frame",
            data={
                "session_id": session_id,
                "timestamp": "0.0",
                "frame": (_jpeg(), "frame.jpg"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(first.status_code, 200)
        payload = first.get_json()
        self.assertEqual(payload["frame_index"], 0)
        self.assertIn("activity", payload)
        self.assertIn("deviation", payload)

    def test_invalid_fps_is_rejected(self) -> None:
        response = self.client.post("/api/reset", json={"fps": 0})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
