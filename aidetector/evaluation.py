from __future__ import annotations

import io
import json
import hashlib
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
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class PredictionRun:
    y_true: list[int]
    y_score: list[float]
    y_pred: list[int]
    predictions: list[dict[str, Any]]
    seconds: float


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


def collect_tiny_genimage_parquet_samples(
    parquet_paths: Sequence[str | Path],
    *,
    max_per_class_per_shard: int | None = None,
    generators: set[str] | None = None,
) -> list[ImageSample]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install pyarrow or datasets eval extras.") from exc

    samples: list[ImageSample] = []
    allowed_generators = {value.lower() for value in generators} if generators else None
    for parquet_path_like in parquet_paths:
        parquet_path = Path(parquet_path_like)
        table = pq.read_table(parquet_path)
        metadata = table.schema.metadata or {}
        hf_metadata = metadata.get(b"huggingface")
        label_names: list[str] | None = None
        generator_names: list[str] | None = None
        if hf_metadata:
            payload = json.loads(hf_metadata.decode("utf-8"))
            features = payload.get("info", {}).get("features", {})
            label_names = features.get("label", {}).get("names")
            generator_names = features.get("generator", {}).get("names")

        rows = table.to_pylist()
        per_class_counts = {0: 0, 1: 0}
        for index, row in enumerate(rows):
            label = int(row["label"])
            generator_name = _class_name(int(row.get("generator", 0)), generator_names)
            generator_key = generator_name.lower()
            if allowed_generators and generator_key not in allowed_generators:
                continue
            if max_per_class_per_shard is not None and per_class_counts[label] >= max_per_class_per_shard:
                continue
            per_class_counts[label] += 1
            samples.append(
                ImageSample(
                    image=_coerce_image(row["image"]),
                    label=label,
                    sample_id=f"{parquet_path.name}:{index}",
                    metadata={
                        "source_shard": parquet_path.name,
                        "generator": generator_name,
                        "generator_id": int(row.get("generator", 0)),
                        "label_name": _class_name(label, label_names),
                    },
                )
            )
    return samples


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
    run = collect_predictions(detector, samples, batch_size=batch_size)
    metrics = compute_metrics(
        run.y_true,
        run.y_score,
        run.y_pred,
        dataset=dataset_name,
        threshold=detector.threshold,
        seconds=run.seconds,
    )
    return EvaluationReport(
        metrics=metrics,
        model=detector.model_info(),
        dataset=dataset_info or {"name": dataset_name},
        predictions=run.predictions,
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


def collect_predictions(
    detector: Detector,
    samples: Iterable[ImageSample],
    *,
    batch_size: int = 16,
) -> PredictionRun:
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
            metadata = _metadata_for_sample_id(sample_id)
            if metadata is not None:
                row.update(metadata)
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
        if sample.metadata is not None and batch_ids[-1] is not None:
            _SAMPLE_METADATA[batch_ids[-1]] = dict(sample.metadata)
        if len(batch_images) >= batch_size:
            flush()
    flush()

    return PredictionRun(
        y_true=y_true,
        y_score=y_score,
        y_pred=y_pred,
        predictions=predictions,
        seconds=time.perf_counter() - start,
    )


def split_samples_balanced(
    samples: Sequence[ImageSample],
    *,
    calibration_fraction: float = 0.5,
    seed: str = "aidetect",
) -> tuple[list[ImageSample], list[ImageSample]]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")

    groups = {0: [], 1: []}
    for sample in samples:
        groups[int(sample.label)].append(sample)

    calibration: list[ImageSample] = []
    test: list[ImageSample] = []
    for label, group in groups.items():
        if len(group) < 2:
            raise ValueError(f"Need at least 2 samples for label {label} to split calibration/test.")
        ranked = sorted(group, key=lambda sample: _stable_sample_key(sample, seed))
        calibration_count = round(len(ranked) * calibration_fraction)
        calibration_count = min(max(calibration_count, 1), len(ranked) - 1)
        calibration.extend(ranked[:calibration_count])
        test.extend(ranked[calibration_count:])
    return calibration, test


def best_threshold_metrics(y_true: Sequence[int], y_score: Sequence[float]) -> dict[str, float | None]:
    return search_threshold(y_true, y_score, objective="balanced_accuracy")


def search_threshold(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    objective: str = "balanced_accuracy",
    min_recall: float | None = None,
) -> dict[str, float | None]:
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
        if min_recall is not None and recall < min_recall:
            continue
        candidate = _objective_tuple(
            objective=objective,
            accuracy=accuracy,
            balanced_accuracy=balanced_accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
        )
        incumbent = (
            float(best["balanced_accuracy"] or -1.0),
            float(best["accuracy"] or -1.0),
            float(best["f1"] or -1.0),
            -1.0,
        ) if objective == "balanced_accuracy" and min_recall is None else best.get("_candidate", (-1.0, -1.0, -1.0, -1.0))
        if candidate > incumbent:
            best = {
                "threshold": threshold,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "_candidate": candidate,
            }
    best.pop("_candidate", None)
    return best


def search_blend_weight_threshold(
    y_true: Sequence[int],
    primary_scores: Sequence[float],
    secondary_scores: Sequence[float],
    *,
    alpha_step: float = 0.05,
    objective: str = "balanced_accuracy",
    min_recall: float | None = None,
) -> dict[str, float | None]:
    if len(y_true) != len(primary_scores) or len(y_true) != len(secondary_scores):
        raise ValueError("All sequences must have the same length")
    if not 0.0 < alpha_step <= 1.0:
        raise ValueError("alpha_step must be between 0 and 1")

    best: dict[str, float | None] = {
        "primary_weight": None,
        "threshold": None,
        "accuracy": None,
        "balanced_accuracy": -1.0,
        "f1": None,
    }
    steps = round(1.0 / alpha_step)
    for step in range(steps + 1):
        alpha = min(step * alpha_step, 1.0)
        combined_scores = [
            alpha * primary_score + (1.0 - alpha) * secondary_score
            for primary_score, secondary_score in zip(primary_scores, secondary_scores, strict=True)
        ]
        threshold_metrics = search_threshold(
            y_true,
            combined_scores,
            objective=objective,
            min_recall=min_recall,
        )
        candidate = _objective_tuple(
            objective=objective,
            accuracy=float(threshold_metrics["accuracy"] or -1.0),
            balanced_accuracy=float(threshold_metrics["balanced_accuracy"] or -1.0),
            precision=float(threshold_metrics.get("precision") or -1.0),
            recall=float(threshold_metrics.get("recall") or -1.0),
            f1=float(threshold_metrics["f1"] or -1.0),
        )
        incumbent = (
            float(best["balanced_accuracy"] or -1.0),
            float(best["accuracy"] or -1.0),
            float(best["f1"] or -1.0),
            -1.0,
        ) if objective == "balanced_accuracy" and min_recall is None else best.get("_candidate", (-1.0, -1.0, -1.0, -1.0))
        if candidate > incumbent:
            best = {
                "primary_weight": alpha,
                "threshold": threshold_metrics["threshold"],
                "accuracy": threshold_metrics["accuracy"],
                "balanced_accuracy": threshold_metrics["balanced_accuracy"],
                "f1": threshold_metrics["f1"],
                "precision": threshold_metrics.get("precision"),
                "recall": threshold_metrics.get("recall"),
                "_candidate": candidate,
            }
    best.pop("_candidate", None)
    return best


def search_hybrid_weight_threshold(
    y_true: Sequence[int],
    univfd_scores: Sequence[float],
    hf_scores: Sequence[float],
    *,
    alpha_step: float = 0.05,
    objective: str = "balanced_accuracy",
    min_recall: float | None = None,
) -> dict[str, float | None]:
    result = search_blend_weight_threshold(
        y_true,
        univfd_scores,
        hf_scores,
        alpha_step=alpha_step,
        objective=objective,
        min_recall=min_recall,
    )
    if result.get("primary_weight") is not None:
        result["univfd_weight"] = result["primary_weight"]
    return result


def combine_scores(
    univfd_scores: Sequence[float],
    hf_scores: Sequence[float],
    *,
    univfd_weight: float,
) -> list[float]:
    if len(univfd_scores) != len(hf_scores):
        raise ValueError("univfd_scores and hf_scores must have the same length")
    return [
        univfd_weight * univfd_score + (1.0 - univfd_weight) * hf_score
        for univfd_score, hf_score in zip(univfd_scores, hf_scores, strict=True)
    ]


def build_combined_predictions(
    predictions: Sequence[dict[str, Any]],
    scores: Sequence[float],
    *,
    threshold: float,
    backend: str,
) -> list[dict[str, Any]]:
    if len(predictions) != len(scores):
        raise ValueError("predictions and scores must have the same length")
    combined: list[dict[str, Any]] = []
    for row, score in zip(predictions, scores, strict=True):
        updated = dict(row)
        updated["probability_ai"] = round(score, 6)
        updated["probability_real"] = round(1.0 - score, 6)
        updated["label"] = "ai" if score >= threshold else "real"
        updated["confidence"] = round(max(score, 1.0 - score), 6)
        updated["backend"] = backend
        combined.append(updated)
    return combined


def metrics_from_prediction_rows(
    rows: Sequence[dict[str, Any]],
    *,
    threshold: float,
    dataset: str,
) -> dict[str, Any]:
    y_true = [1 if row["truth"] == "ai" else 0 for row in rows]
    y_score = [float(row["probability_ai"]) for row in rows]
    metrics = compute_metrics(y_true, y_score, dataset=dataset, threshold=threshold, seconds=0.0)
    payload = metrics.as_dict()
    payload["images_per_second"] = None
    payload["seconds"] = None
    return payload


def group_prediction_rows(
    rows: Sequence[dict[str, Any]],
    *,
    field: str,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        grouped.setdefault(str(value), []).append(row)
    return {
        key: metrics_from_prediction_rows(group_rows, threshold=threshold, dataset=f"{field}:{key}")
        for key, group_rows in sorted(grouped.items())
    }


def group_prediction_rows_against_reference(
    rows: Sequence[dict[str, Any]],
    *,
    field: str,
    reference_value: str,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    reference_rows = [row for row in rows if str(row.get(field)) == reference_value]
    if not reference_rows:
        return {}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = row.get(field)
        if value is None or str(value) == reference_value:
            continue
        grouped.setdefault(str(value), []).append(row)
    return {
        key: metrics_from_prediction_rows(
            [*reference_rows, *group_rows],
            threshold=threshold,
            dataset=f"{field}:{key}-vs-{reference_value}",
        )
        for key, group_rows in sorted(grouped.items())
    }


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


_SAMPLE_METADATA: dict[str, dict[str, Any]] = {}


def _metadata_for_sample_id(sample_id: str | None) -> dict[str, Any] | None:
    if sample_id is None:
        return None
    return _SAMPLE_METADATA.get(sample_id)


def _stable_sample_key(sample: ImageSample, seed: str) -> str:
    identity = sample.sample_id or (str(sample.path) if sample.path else repr(sample.image))
    return hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).hexdigest()


def _class_name(index: int, names: list[str] | None) -> str:
    if names is not None and 0 <= index < len(names):
        return str(names[index])
    return str(index)


def _objective_tuple(
    *,
    objective: str,
    accuracy: float,
    balanced_accuracy: float,
    precision: float,
    recall: float,
    f1: float,
) -> tuple[float, float, float, float]:
    if objective == "f1":
        return (f1, balanced_accuracy, accuracy, recall)
    if objective == "recall":
        return (recall, balanced_accuracy, f1, accuracy)
    if objective == "precision":
        return (precision, balanced_accuracy, f1, accuracy)
    return (balanced_accuracy, accuracy, f1, -abs(precision - recall))
