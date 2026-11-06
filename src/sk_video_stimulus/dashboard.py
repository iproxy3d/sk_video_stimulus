"""Локальный web dashboard для потоковой демонстрации пакета."""

from __future__ import annotations

import argparse
import math
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from io import BytesIO
from typing import Any

from PIL import Image, UnidentifiedImageError

from .extractor import OscillationConfig, SpatialEncoderConfig, VideoFeatureExtractor


@dataclass
class _DashboardSession:
    extractor: VideoFeatureExtractor
    touched_at: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _make_extractor(fps: float) -> VideoFeatureExtractor:
    spatial = SpatialEncoderConfig(fps=fps, input_color_order="rgb")
    oscillation = OscillationConfig(fps=fps)
    return VideoFeatureExtractor(spatial, oscillation)


def create_app() -> Any:
    """Создать Flask-приложение без запуска сервера.

    Функция вынесена отдельно, чтобы dashboard можно было встроить в WSGI-сервер
    и тестировать через стандартный Flask test client.
    """

    try:
        from flask import Flask, jsonify, render_template, request
    except ImportError as exc:  # pragma: no cover - зависит от окружения пользователя
        raise RuntimeError(
            "Для dashboard установите дополнительную зависимость: "
            "python -m pip install 'sk-video-stimulus[demo]'"
        ) from exc

    app = Flask(__name__, template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = 3 * 1024 * 1024
    sessions: dict[str, _DashboardSession] = {}
    sessions_lock = threading.Lock()

    def clean_old_sessions() -> None:
        cutoff = time.monotonic() - 60 * 60
        with sessions_lock:
            expired = [key for key, value in sessions.items() if value.touched_at < cutoff]
            for key in expired:
                del sessions[key]

    @app.get("/")
    def index() -> str:
        return render_template("dashboard.html")

    @app.post("/api/reset")
    def reset() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            fps = float(payload.get("fps", 12.0))
        except (TypeError, ValueError):
            return jsonify(error="fps должен быть числом"), 400
        if not math.isfinite(fps) or not 1.0 <= fps <= 60.0:
            return jsonify(error="fps должен находиться в диапазоне от 1 до 60"), 400

        clean_old_sessions()
        session_id = uuid.uuid4().hex
        with sessions_lock:
            sessions[session_id] = _DashboardSession(_make_extractor(fps))
        return jsonify(session_id=session_id, fps=fps)

    @app.post("/api/frame")
    def frame() -> Any:
        session_id = request.form.get("session_id", "")
        with sessions_lock:
            session = sessions.get(session_id)
        if session is None:
            return jsonify(error="Сессия не найдена. Нажмите «Запустить» ещё раз."), 404

        uploaded = request.files.get("frame")
        if uploaded is None:
            return jsonify(error="Не передан кадр"), 400
        try:
            timestamp = float(request.form.get("timestamp", ""))
        except (TypeError, ValueError):
            return jsonify(error="Некорректная временная метка"), 400
        if not math.isfinite(timestamp) or timestamp < 0:
            return jsonify(error="Временная метка должна быть неотрицательной"), 400

        try:
            raw = uploaded.read()
            image = Image.open(BytesIO(raw))
            image.load()
            if image.width * image.height > 20_000_000:
                return jsonify(error="Слишком большое разрешение кадра"), 413
            image = image.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError):
            return jsonify(error="Переданные данные не являются изображением"), 400

        try:
            with session.lock:
                sample = session.extractor.process_frame(
                    image,
                    timestamp=timestamp,
                    color_order="rgb",
                )
                session.touched_at = time.monotonic()
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(asdict(sample))

    @app.errorhandler(413)
    def payload_too_large(_error: Exception) -> Any:
        return jsonify(error="Кадр превышает допустимый размер 3 МБ"), 413

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Локальный web dashboard sk-video-stimulus"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    app = create_app()
    print(f"Dashboard: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
