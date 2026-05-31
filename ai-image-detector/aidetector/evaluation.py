from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from PIL import Image, ImageOps

from .model import Detector, iter_images
from .types import EvaluationMetrics, EvaluationReport


@dataclass(frozen=True)
class ImageSample:
    image: Image.Image | None
    label: int
    path: Path | None = None
    sample_id: str | None = None


def parse_binary_label(value: Any, fake_label: str | int = "1") -> int:
    """Map a dataset label to 1 for AI/fake and 0 for real."""
    if _label_key(value) == _label_key(fake_label):
        return 1
    fake_tokens = {"fake", "ai", "aigc", "synthetic", "generated"}
    real_tokens = {"real", "human", "authentic", "natural", "nature"}
    key = _label_key(value)
    if key in fake_tokens:
        return 1
    if key in real_tokens:
        return 0
    return 0


def collect_folder_samples(
    root: str | Path,
    *,
    real_dir: str = "real",
    fake_dir: str = "ai",
    recursive: bool = True,
    max_per_class: int | None = None,
) -> list[ImageSample]:
    root = Path(root)
    real_paths = iter_images(root / real_dir, recursive=recursive)
    fake_paths = iter_images(root / fake_dir, recursive=recursive)
    if max_per_class is not None:
        real_paths = real_paths[:max_per_class]
        fake_paths = fake_paths[:max_per_class]
    return [ImageSample(None, 0, path) for path in real_paths] + [
        ImageSample(None, 1, path) for path in fake_paths
    ]


def evaluate_folder(
    detector: Detector,
    root: str | Path,
    *,
    real_dir: str = "real",
    fake_dir: str = "ai",
    recursive: bool = True,
    max_per_class: int | None = None,
    batch_size: int = 16,
) -> EvaluationReport:
    samples = collect_folder_samples(
        root,
        real_dir=real_dir,
        fake_dir=fake_dir,
        recursive=recursive,
        max_per_class=max_per_class,
    )
    return evaluate_samples(
        detector,
        samples,
        dataset_name=f"folder:{Path(root)}",
        batch_size=batch_size,
        dataset_info={
            "kind": "folder",
            "root": str(root),
            "real_dir": real_dir,
            "fake_dir": fake_dir,
            "recursive": recursive,
            "max_per_class": max_per_class,
        },
    )


def evaluate_hf_dataset(
    detector: Detector,
    dataset_name: str,
    *,
    split: str = "validation",
    image_field: str = "image",
    label_field: str = "label",
    fake_label: str = "1",
    max_samples: int | None = None,
    batch_size: int = 16,
    streaming: bool = False,
    shuffle_seed: int | None = None,
    trust_remote_code: bool = False,
) -> EvaluationReport:
    samples = iter_hf_samples(
        dataset_name,
        split=split,
        image_field=image_field,
        label_field=label_field,
        fake_label=fake_label,
        max_samples=max_samples,
        streaming=streaming,
        shuffle_seed=shuffle_seed,
        trust_remote_code=trust_remote_code,
    )
    return evaluate_samples(
        detector,
        samples,
        dataset_name=f"hf:{dataset_name}/{split}",
        batch_size=batch_size,
        dataset_info={
            "kind": "huggingface",
            "dataset": dataset_name,
            "split": split,
            "image_field": image_field,
            "label_field": label_field,
            "fake_label": fake_label,
            "max_samples": max_samples,
            "streaming": streaming,
            "shuffle_seed": shuffle_seed,
        },
    )


def iter_hf_samples(
    dataset_name: str,
    *,
    split: str,
    image_field: str,
    label_field: str,
    fake_label: str = "1",
    max_samples: int | None = None,
    streaming: bool = False,
    shuffle_seed: int | None = None,
    trust_remote_code: bool = False,
) -> Iterator[ImageSample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install datasets to run Hugging Face benchmarks.") from exc

    dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=streaming,
        trust_remote_code=trust_remote_code,
    )
    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed)

    for index, item in enumerate(dataset):
        if max_samples is not None and index >= max_samples:
            break
        image = _coerce_image(item[image_field])
        label = parse_binary_label(item[label_field], fake_label=fake_label)
        yield ImageSample(
            image=image,
            label=label,
            path=None,
            sample_id=f"{dataset_name}:{split}:{index}",
        )


def evaluate_samples(
    detector: Detector,
    samples: Iterable[ImageSample],
    *,
    dataset_name: str,
    batch_size: int = 16,
    dataset_info: dict[str, Any] | None = None,
) -> EvaluationReport:
    start = time.perf_counter()
    y_true: list[int] = []
    y_score: list[float] = []
    y_pred: list[int] = []
    predictions: list[dict[str, Any]] = []

    batch_images: list[Image.Image] = []
    batch_paths: list[Path | None] = []
    batch_labels: list[int] = []
    batch_ids: list[str | None] = []

    def flush() -> None:
        if not batch_images:
            return
        results = detector.predict_images(batch_images, paths=batch_paths)
        for result, true_label, sample_id in zip(results, batch_labels, batch_ids, strict=True):
            predicted = 1 if result.probability_ai >= detector.threshold else 0
            y_true.append(true_label)
            y_score.append(result.probability_ai)
            y_pred.append(predicted)
            row = result.as_dict()
            row["truth"] = "ai" if true_label else "real"
            row["sample_id"] = sample_id
            predictions.append(row)
        batch_images.clear()
        batch_paths.clear()
        batch_labels.clear()
        batch_ids.clear()

    for sample in samples:
        image = _load_sample_image(sample)
        batch_images.append(image)
        batch_paths.append(sample.path)
        batch_labels.append(int(sample.label))
        batch_ids.append(sample.sample_id or (str(sample.path) if sample.path else None))
        if len(batch_images) >= batch_size:
            flush()
    flush()

    seconds = time.perf_counter() - start
    metrics = compute_metrics(
        y_true,
        y_score,
        y_pred,
        dataset=dataset_name,
        threshold=detector.threshold,
        seconds=seconds,
    )
    return EvaluationReport(
        metrics=metrics,
        model=detector.model_info(),
        dataset=dataset_info or {"name": dataset_name},
        predictions=predictions,
    )


def compute_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    y_pred: Sequence[int] | None = None,
    *,
    dataset: str = "dataset",
    threshold: float = 0.5,
    seconds: float = 0.0,
) -> EvaluationMetrics:
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")
    if y_pred is None:
        y_pred = [1 if score >= threshold else 0 for score in y_score]
    if len(y_pred) != len(y_true):
        raise ValueError("y_pred and y_true must have the same length")

    tp = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == 1 and pred == 1)
    tn = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == 0 and pred == 0)
    fp = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == 0 and pred == 1)
    fn = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == 1 and pred == 0)

    n_samples = len(y_true)
    n_ai = sum(1 for value in y_true if value == 1)
    n_real = n_samples - n_ai
    accuracy = _safe_div(tp + tn, n_samples)
    true_positive_rate = _safe_div(tp, tp + fn)
    true_negative_rate = _safe_div(tn, tn + fp)
    balanced_accuracy = (true_positive_rate + true_negative_rate) / 2
    precision = _safe_div(tp, tp + fp)
    recall = true_positive_rate
    f1 = _safe_div(2 * precision * recall, precision + recall)
    images_per_second = _safe_div(n_samples, seconds)
    threshold_sweep = best_threshold_metrics(y_true, y_score)

    return EvaluationMetrics(
        dataset=dataset,
        n_samples=n_samples,
        n_real=n_real,
        n_ai=n_ai,
        threshold=threshold,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc(y_true, y_score),
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        seconds=seconds,
        images_per_second=images_per_second,
        best_threshold=threshold_sweep["threshold"],
        best_accuracy=threshold_sweep["accuracy"],
        best_balanced_accuracy=threshold_sweep["balanced_accuracy"],
        best_f1=threshold_sweep["f1"],
    )


def roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float | None:
    positives = sum(1 for value in y_true if value == 1)
    negatives = sum(1 for value in y_true if value == 0)
    if positives == 0 or negatives == 0:
        return None

    pairs = sorted(zip(y_score, y_true, strict=True), key=lambda item: item[0])
    rank_sum_positive = 0.0
    rank = 1
    index = 0
    while index < len(pairs):
        tie_end = index + 1
        while tie_end < len(pairs) and pairs[tie_end][0] == pairs[index][0]:
            tie_end += 1
        average_rank = (rank + rank + (tie_end - index) - 1) / 2
        for _, label in pairs[index:tie_end]:
            if label == 1:
                rank_sum_positive += average_rank
        rank += tie_end - index
        index = tie_end

    return (rank_sum_positive - positives * (positives + 1) / 2) / (positives * negatives)


def best_threshold_metrics(y_true: Sequence[int], y_score: Sequence[float]) -> dict[str, float | None]:
    if not y_true:
        return {"threshold": None, "accuracy": None, "balanced_accuracy": None, "f1": None}

    best: dict[str, float | None] = {
        "threshold": None,
        "accuracy": None,
        "balanced_accuracy": -1.0,
        "f1": None,
    }
    for threshold in sorted(set(float(score) for score in y_score)):
        y_pred = [1 if score >= threshold else 0 for score in y_score]
        tp = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == 1 and pred == 1)
        tn = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == 0 and pred == 0)
        fp = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == 0 and pred == 1)
        fn = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == 1 and pred == 0)
        accuracy = _safe_div(tp + tn, len(y_true))
        recall = _safe_div(tp, tp + fn)
        specificity = _safe_div(tn, tn + fp)
        balanced_accuracy = (recall + specificity) / 2
        precision = _safe_div(tp, tp + fp)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        candidate = (balanced_accuracy, accuracy, f1)
        incumbent = (
            float(best["balanced_accuracy"] or -1.0),
            float(best["accuracy"] or -1.0),
            float(best["f1"] or -1.0),
        )
        if candidate > incumbent:
            best = {
                "threshold": threshold,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "f1": f1,
            }
    return best


def write_report(report: EvaluationReport, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def _coerce_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return ImageOps.exif_transpose(value).convert("RGB")
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            with Image.open(io.BytesIO(value["bytes"])) as image:
                return ImageOps.exif_transpose(image).convert("RGB")
        if value.get("path") is not None:
            with Image.open(value["path"]) as image:
                return ImageOps.exif_transpose(image).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def _load_sample_image(sample: ImageSample) -> Image.Image:
    if sample.image is not None:
        return _coerce_image(sample.image)
    if sample.path is None:
        raise ValueError("sample must provide either image or path")
    with Image.open(sample.path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _label_key(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip().lower()


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
