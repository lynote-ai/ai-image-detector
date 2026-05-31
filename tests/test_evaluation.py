from __future__ import annotations

from pathlib import Path

from PIL import Image

from aidetector.evaluation import (
    ImageSample,
    build_combined_predictions,
    combine_scores,
    compute_metrics,
    evaluate_folder,
    evaluate_samples,
    group_prediction_rows_against_reference,
    group_prediction_rows,
    roc_auc,
    search_hybrid_weight_threshold,
    split_samples_balanced,
)
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


def test_split_samples_balanced_preserves_both_classes():
    samples = [
        ImageSample(Image.new("RGB", (8, 8), "black"), 0, sample_id="real-1"),
        ImageSample(Image.new("RGB", (8, 8), "black"), 0, sample_id="real-2"),
        ImageSample(Image.new("RGB", (8, 8), "white"), 1, sample_id="ai-1"),
        ImageSample(Image.new("RGB", (8, 8), "white"), 1, sample_id="ai-2"),
    ]

    calibration, test = split_samples_balanced(samples, calibration_fraction=0.5, seed="test")

    assert len(calibration) == 2
    assert len(test) == 2
    assert {sample.label for sample in calibration} == {0, 1}
    assert {sample.label for sample in test} == {0, 1}


def test_hybrid_search_and_combined_predictions():
    y_true = [0, 0, 1, 1]
    univfd_scores = [0.2, 0.4, 0.45, 0.55]
    hf_scores = [0.1, 0.7, 0.8, 0.9]

    search = search_hybrid_weight_threshold(y_true, univfd_scores, hf_scores, alpha_step=0.25)

    assert search["univfd_weight"] is not None
    combined_scores = combine_scores(
        univfd_scores,
        hf_scores,
        univfd_weight=float(search["univfd_weight"]),
    )
    predictions = build_combined_predictions(
        [
            {"path": "a", "truth": "real", "probability_ai": 0.1, "probability_real": 0.9, "label": "real", "confidence": 0.9, "raw_score": 0.1, "backend": "x", "sample_id": "a"},
            {"path": "b", "truth": "real", "probability_ai": 0.7, "probability_real": 0.3, "label": "ai", "confidence": 0.7, "raw_score": 0.7, "backend": "x", "sample_id": "b"},
            {"path": "c", "truth": "ai", "probability_ai": 0.8, "probability_real": 0.2, "label": "ai", "confidence": 0.8, "raw_score": 0.8, "backend": "x", "sample_id": "c"},
            {"path": "d", "truth": "ai", "probability_ai": 0.9, "probability_real": 0.1, "label": "ai", "confidence": 0.9, "raw_score": 0.9, "backend": "x", "sample_id": "d"},
        ],
        combined_scores,
        threshold=float(search["threshold"]),
        backend="hybrid",
    )

    assert predictions[0]["backend"] == "hybrid"
    assert all("probability_real" in row for row in predictions)


def test_group_prediction_rows_builds_generator_metrics():
    rows = [
        {"truth": "real", "probability_ai": 0.1, "generator": "Real"},
        {"truth": "ai", "probability_ai": 0.8, "generator": "SD14"},
        {"truth": "ai", "probability_ai": 0.9, "generator": "SD14"},
    ]

    grouped = group_prediction_rows(rows, field="generator", threshold=0.5)

    assert grouped["SD14"]["accuracy"] == 1.0
    assert grouped["Real"]["n_real"] == 1


def test_group_prediction_rows_against_reference_builds_binary_slices():
    rows = [
        {"truth": "real", "probability_ai": 0.1, "generator": "Real"},
        {"truth": "real", "probability_ai": 0.2, "generator": "Real"},
        {"truth": "ai", "probability_ai": 0.9, "generator": "SD14"},
        {"truth": "ai", "probability_ai": 0.8, "generator": "SD14"},
    ]

    grouped = group_prediction_rows_against_reference(
        rows,
        field="generator",
        reference_value="Real",
        threshold=0.5,
    )

    assert grouped["SD14"]["accuracy"] == 1.0
    assert grouped["SD14"]["n_real"] == 2
    assert grouped["SD14"]["n_ai"] == 2
