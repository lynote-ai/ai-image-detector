from __future__ import annotations

from PIL import Image

from aidetector.model import iter_images
from aidetector.types import DetectionResult, EvaluationMetrics


def test_iter_images_recurses_and_filters(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    Image.new("RGB", (4, 4), "red").save(tmp_path / "a.jpg")
    Image.new("RGB", (4, 4), "blue").save(nested / "b.png")
    (tmp_path / "note.txt").write_text("not an image", encoding="utf-8")

    paths = iter_images(tmp_path, recursive=True)

    assert [path.name for path in paths] == ["a.jpg", "b.png"]
    assert iter_images(tmp_path / "note.txt") == []


def test_detection_result_serializes_path_and_backend(tmp_path):
    result = DetectionResult(
        path=tmp_path / "image.png",
        label="ai",
        probability_ai=0.81234567,
        probability_real=0.18765433,
        confidence=0.81234567,
        raw_score=1.23456789,
        backend="test",
    )

    payload = result.as_dict()

    assert payload["path"].endswith("image.png")
    assert payload["probability_ai"] == 0.812346
    assert payload["backend"] == "test"


def test_evaluation_metrics_serializes_auc_none():
    metrics = EvaluationMetrics(
        dataset="toy",
        n_samples=1,
        n_real=1,
        n_ai=0,
        threshold=0.5,
        accuracy=1.0,
        balanced_accuracy=0.5,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        roc_auc=None,
        true_positive=0,
        true_negative=1,
        false_positive=0,
        false_negative=0,
        seconds=0.1,
        images_per_second=10.0,
    )

    assert metrics.as_dict()["roc_auc"] is None
    assert metrics.as_dict()["confusion"]["true_negative"] == 1
