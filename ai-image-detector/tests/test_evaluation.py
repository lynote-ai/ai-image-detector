from __future__ import annotations

from pathlib import Path

from PIL import Image

from aidetector.evaluation import ImageSample, compute_metrics, evaluate_folder, evaluate_samples, roc_auc
from aidetector.types import DetectionResult


class BrightnessDetector:
    threshold = 0.5

    def predict_image(self, image, path: Path | None = None):
        return self.predict_images([image], [path])[0]

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

    def predict_path(self, path):
        with Image.open(path) as image:
            return self.predict_image(image, Path(path))

    def model_info(self):
        return {"backend": "brightness", "threshold": self.threshold}


def test_compute_metrics_balanced_binary_case():
    metrics = compute_metrics(
        [0, 0, 1, 1],
        [0.1, 0.8, 0.7, 0.9],
        dataset="toy",
        threshold=0.5,
        seconds=2.0,
    )

    assert metrics.n_samples == 4
    assert metrics.true_positive == 2
    assert metrics.false_positive == 1
    assert metrics.accuracy == 0.75
    assert metrics.precision == 2 / 3
    assert metrics.recall == 1.0
    assert metrics.images_per_second == 2.0


def test_roc_auc_handles_ties():
    assert roc_auc([0, 1], [0.1, 0.9]) == 1.0
    assert roc_auc([0, 1], [0.5, 0.5]) == 0.5
    assert roc_auc([1, 1], [0.5, 0.6]) is None


def test_evaluate_samples_with_fake_detector():
    samples = [
        ImageSample(Image.new("RGB", (8, 8), "black"), 0, sample_id="real"),
        ImageSample(Image.new("RGB", (8, 8), "white"), 1, sample_id="ai"),
    ]

    report = evaluate_samples(
        BrightnessDetector(),
        samples,
        dataset_name="toy",
        batch_size=2,
    )

    assert report.metrics.accuracy == 1.0
    assert report.metrics.n_real == 1
    assert report.metrics.n_ai == 1
    assert report.predictions[0]["sample_id"] == "real"


def test_evaluate_folder_layout(tmp_path):
    real_dir = tmp_path / "nature"
    fake_dir = tmp_path / "ai"
    real_dir.mkdir()
    fake_dir.mkdir()
    Image.new("RGB", (8, 8), "black").save(real_dir / "real.png")
    Image.new("RGB", (8, 8), "white").save(fake_dir / "fake.png")

    report = evaluate_folder(
        BrightnessDetector(),
        tmp_path,
        real_dir="nature",
        fake_dir="ai",
        batch_size=2,
    )

    assert report.metrics.accuracy == 1.0
    assert report.dataset["real_dir"] == "nature"
