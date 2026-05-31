from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from aidetector.cli import app
from aidetector.types import DetectionResult


class BrightnessDetector:
    threshold = 0.5

    def predict_images(self, images, paths=None):
        paths = paths or [None] * len(images)
        results = []
        for image, path in zip(images, paths, strict=True):
            probability_ai = image.convert("L").resize((1, 1)).getpixel((0, 0)) / 255
            results.append(
                DetectionResult(
                    path=path,
                    label="ai" if probability_ai >= self.threshold else "real",
                    probability_ai=probability_ai,
                    probability_real=1 - probability_ai,
                    confidence=max(probability_ai, 1 - probability_ai),
                    raw_score=probability_ai,
                    backend="brightness",
                )
            )
        return results

    def predict_image(self, image, path: Path | None = None):
        return self.predict_images([image], [path])[0]

    def predict_path(self, path):
        with Image.open(path) as image:
            return self.predict_image(image, Path(path))

    def model_info(self):
        return {"backend": "brightness", "threshold": self.threshold}


def test_detect_json_output(monkeypatch, tmp_path):
    monkeypatch.setattr("aidetector.cli.create_detector", lambda *args, **kwargs: BrightnessDetector())
    Image.new("RGB", (8, 8), "white").save(tmp_path / "fake.png")

    result = CliRunner().invoke(app, ["detect", str(tmp_path), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout.strip())
    assert payload["label"] == "ai"
    assert payload["backend"] == "brightness"


def test_benchmark_folder_writes_report(monkeypatch, tmp_path):
    monkeypatch.setattr("aidetector.cli.create_detector", lambda *args, **kwargs: BrightnessDetector())
    (tmp_path / "real").mkdir()
    (tmp_path / "ai").mkdir()
    Image.new("RGB", (8, 8), "black").save(tmp_path / "real" / "real.png")
    Image.new("RGB", (8, 8), "white").save(tmp_path / "ai" / "fake.png")
    output = tmp_path / "report.json"

    result = CliRunner().invoke(app, ["benchmark-folder", str(tmp_path), "--output", str(output)])

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metrics"]["accuracy"] == 1.0
    assert payload["model"]["backend"] == "brightness"
