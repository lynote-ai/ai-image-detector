from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from .config import DEFAULT_BACKEND, DEFAULT_HF_MODEL_ID
from .model import create_detector


def create_app(
    *,
    threshold: float = 0.5,
    device: str | None = None,
    backend: str = DEFAULT_BACKEND,
    weight_path: str | Path | None = None,
    hf_model: str = DEFAULT_HF_MODEL_ID,
):
    try:
        from fastapi import FastAPI, File, UploadFile
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install ai-image-detector[api].") from exc

    detector = create_detector(
        backend,
        device=device,
        threshold=threshold,
        weight_path=weight_path,
        hf_model=hf_model,
    )
    app = FastAPI(title="AI Image Detector", version="0.2.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": detector.model_info()}

    @app.post("/detect")
    async def detect(file: UploadFile = File(...)) -> dict:
        content = await file.read()
        with Image.open(io.BytesIO(content)) as image:
            result = detector.predict_image(image)
        payload = result.as_dict()
        payload["filename"] = file.filename
        return payload

    return app
