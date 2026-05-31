from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from aidetector.cli import app
from aidetector.evaluation import ImageSample
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


def test_benchmark_calibrated_folder_writes_report(monkeypatch, tmp_path):
    monkeypatch.setattr("aidetector.cli.create_detector", lambda *args, **kwargs: BrightnessDetector())
    (tmp_path / "real").mkdir()
    (tmp_path / "ai").mkdir()
    Image.new("RGB", (8, 8), "black").save(tmp_path / "real" / "real-1.png")
    Image.new("RGB", (8, 8), "black").save(tmp_path / "real" / "real-2.png")
    Image.new("RGB", (8, 8), "white").save(tmp_path / "ai" / "fake-1.png")
    Image.new("RGB", (8, 8), "white").save(tmp_path / "ai" / "fake-2.png")
    output = tmp_path / "calibrated-report.json"

    result = CliRunner().invoke(
        app,
        [
            "benchmark-calibrated-folder",
            str(tmp_path),
            "--backend",
            "hf",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["test"]["metrics"]["accuracy"] == 1.0
    assert payload["model"]["backend"] == "brightness"


def test_benchmark_tiny_genimage_local_writes_group_report(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "aidetector.cli.collect_tiny_genimage_parquet_samples",
        lambda *args, **kwargs: [
            ImageSample(Image.new("RGB", (8, 8), "black"), 0, sample_id="r1", metadata={"generator": "Real", "source_shard": "s1"}),
            ImageSample(Image.new("RGB", (8, 8), "black"), 0, sample_id="r2", metadata={"generator": "Real", "source_shard": "s1"}),
            ImageSample(Image.new("RGB", (8, 8), "white"), 1, sample_id="f1", metadata={"generator": "SD14", "source_shard": "s1"}),
            ImageSample(Image.new("RGB", (8, 8), "white"), 1, sample_id="f2", metadata={"generator": "SD14", "source_shard": "s1"}),
        ],
    )
    monkeypatch.setattr("aidetector.cli.create_detector", lambda *args, **kwargs: BrightnessDetector())
    output = tmp_path / "tiny-report.json"
    dummy_parquet = tmp_path / "dummy.parquet"
    dummy_parquet.write_text("x", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "benchmark-tiny-genimage-local",
            str(dummy_parquet),
            "--backend",
            "hf",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["groups"]["generator"]["SD14"]["accuracy"] == 1.0
    assert payload["dataset"]["kind"] == "tiny-genimage-local"
