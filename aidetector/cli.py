from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

import typer
from PIL import Image, ImageOps
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from .config import (
        DEFAULT_BACKEND,
        DEFAULT_HF_MODEL_ID,
        DEFAULT_HYBRID_UNIVFD_WEIGHT,
        DEFAULT_HYBRID_PLUS_PRIMARY_WEIGHT,
        DEFAULT_MODEL_NAME,
        DEFAULT_ULTRA_PRIMARY_WEIGHT,
        DEFAULT_PRETRAINED,
)
from .evaluation import (
    build_combined_predictions,
    search_blend_weight_threshold,
    collect_folder_samples,
    collect_predictions,
    collect_tiny_genimage_parquet_samples,
    combine_scores,
    compute_metrics,
    evaluate_folder,
    evaluate_hf_dataset,
    group_prediction_rows_against_reference,
    group_prediction_rows,
    search_hybrid_weight_threshold,
    search_threshold,
    split_samples_balanced,
    write_report,
)
from .model import create_detector, iter_images
from .types import DetectionResult, EvaluationReport

app = typer.Typer(help="Simple AI-generated image detector powered by UnivFD/CLIP.")
console = Console()
VALID_OPTIMIZE_METRICS = {"balanced_accuracy", "f1", "recall", "precision"}


@app.command()
def detect(
    path: Path = typer.Argument(..., help="Image file or directory."),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Scan folders recursively."),
    threshold: float = typer.Option(0.5, help="AI probability threshold."),
    device: Optional[str] = typer.Option(None, help="cuda, mps, cpu, or auto when omitted."),
    backend: str = typer.Option(
        DEFAULT_BACKEND,
        help="Detector backend: univfd, hf, nonescape-mini, nonescape-full, sentry-convnext-small, hybrid, hybrid-plus, or ultra.",
    ),
    weight_path: Optional[Path] = typer.Option(None, help="Optional local UnivFD fc_weights.pth path."),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME, help="OpenCLIP model name for UnivFD."),
    pretrained: str = typer.Option(DEFAULT_PRETRAINED, help="OpenCLIP pretrained tag for UnivFD."),
    hf_model: str = typer.Option(DEFAULT_HF_MODEL_ID, help="Hugging Face model id for --backend hf."),
    hybrid_univfd_weight: float = typer.Option(
        DEFAULT_HYBRID_UNIVFD_WEIGHT,
        min=0.0,
        max=1.0,
        help="Blend weight for UnivFD when --backend hybrid.",
    ),
    hybrid_plus_primary_weight: float = typer.Option(
        DEFAULT_HYBRID_PLUS_PRIMARY_WEIGHT,
        min=0.0,
        max=1.0,
        help="Blend weight for hybrid when --backend hybrid-plus.",
    ),
    ultra_primary_weight: float = typer.Option(
        DEFAULT_ULTRA_PRIMARY_WEIGHT,
        min=0.0,
        max=1.0,
        help="Blend weight for hybrid-plus when --backend ultra.",
    ),
    batch_size: int = typer.Option(16, min=1, help="Batch size for inference."),
    json_output: bool = typer.Option(False, "--json", help="Print JSON lines instead of a table."),
    csv_output: Optional[Path] = typer.Option(None, "--csv", help="Save CSV report."),
) -> None:
    """Detect whether an image, or images in a folder, are AI-generated."""
    paths = iter_images(path, recursive=recursive)
    if not paths:
        raise typer.BadParameter(f"No supported images found at {path}")

    detector = create_detector(
        backend,
        device=device,
        threshold=threshold,
        weight_path=weight_path,
        model_name=model_name,
        pretrained=pretrained,
        hf_model=hf_model,
        hybrid_univfd_weight=hybrid_univfd_weight,
        hybrid_plus_primary_weight=hybrid_plus_primary_weight,
        ultra_primary_weight=ultra_primary_weight,
    )
    results = _predict_paths(detector, paths, batch_size=batch_size, quiet=json_output)

    if csv_output:
        _write_csv(results, csv_output)
        console.print(f"Saved CSV report to {csv_output}")

    if json_output:
        for result in results:
            print(json.dumps(result.as_dict(), ensure_ascii=False))
        return

    _print_detection_table(results)
    console.print("[yellow]Note:[/yellow] AI image detection is probabilistic; do not use it as sole proof.")


@app.command("benchmark-folder")
def benchmark_folder_command(
    root: Path = typer.Argument(..., help="Dataset root containing real/fake folders."),
    real_dir: str = typer.Option("real", help="Folder name containing real images."),
    fake_dir: str = typer.Option("ai", help="Folder name containing AI/fake images."),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Scan folders recursively."),
    max_per_class: Optional[int] = typer.Option(None, min=1, help="Optional cap per class."),
    threshold: float = typer.Option(0.5, help="AI probability threshold."),
    device: Optional[str] = typer.Option(None, help="cuda, mps, cpu, or auto when omitted."),
    backend: str = typer.Option(
        DEFAULT_BACKEND,
        help="Detector backend: univfd, hf, nonescape-mini, nonescape-full, sentry-convnext-small, hybrid, hybrid-plus, or ultra.",
    ),
    weight_path: Optional[Path] = typer.Option(None, help="Optional local UnivFD fc_weights.pth path."),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME, help="OpenCLIP model name for UnivFD."),
    pretrained: str = typer.Option(DEFAULT_PRETRAINED, help="OpenCLIP pretrained tag for UnivFD."),
    hf_model: str = typer.Option(DEFAULT_HF_MODEL_ID, help="Hugging Face model id for --backend hf."),
    hybrid_univfd_weight: float = typer.Option(
        DEFAULT_HYBRID_UNIVFD_WEIGHT,
        min=0.0,
        max=1.0,
        help="Blend weight for UnivFD when --backend hybrid.",
    ),
    hybrid_plus_primary_weight: float = typer.Option(
        DEFAULT_HYBRID_PLUS_PRIMARY_WEIGHT,
        min=0.0,
        max=1.0,
        help="Blend weight for hybrid when --backend hybrid-plus.",
    ),
    ultra_primary_weight: float = typer.Option(
        DEFAULT_ULTRA_PRIMARY_WEIGHT,
        min=0.0,
        max=1.0,
        help="Blend weight for hybrid-plus when --backend ultra.",
    ),
    batch_size: int = typer.Option(16, min=1, help="Batch size for inference."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write JSON benchmark report."),
) -> None:
    """Evaluate on an image-folder benchmark."""
    detector = create_detector(
        backend,
        device=device,
        threshold=threshold,
        weight_path=weight_path,
        model_name=model_name,
        pretrained=pretrained,
        hf_model=hf_model,
        hybrid_univfd_weight=hybrid_univfd_weight,
        hybrid_plus_primary_weight=hybrid_plus_primary_weight,
        ultra_primary_weight=ultra_primary_weight,
    )
    report = evaluate_folder(
        detector,
        root,
        real_dir=real_dir,
        fake_dir=fake_dir,
        recursive=recursive,
        max_per_class=max_per_class,
        batch_size=batch_size,
    )
    _finish_benchmark(report, output)


@app.command("benchmark-hf")
def benchmark_hf_command(
    dataset_name: str = typer.Argument(..., help="Hugging Face dataset name."),
    split: str = typer.Option("validation", help="Dataset split."),
    image_field: str = typer.Option("image", help="Image column name."),
    label_field: str = typer.Option("label", help="Binary label column name."),
    fake_label: str = typer.Option("1", help="Value in label column that means AI/fake."),
    max_samples: Optional[int] = typer.Option(None, min=1, help="Optional sample cap."),
    streaming: bool = typer.Option(False, help="Use streaming dataset loading."),
    shuffle_seed: Optional[int] = typer.Option(None, help="Shuffle before sampling."),
    trust_remote_code: bool = typer.Option(False, help="Allow remote dataset code."),
    threshold: float = typer.Option(0.5, help="AI probability threshold."),
    device: Optional[str] = typer.Option(None, help="cuda, mps, cpu, or auto when omitted."),
    backend: str = typer.Option(
        DEFAULT_BACKEND,
        help="Detector backend: univfd, hf, nonescape-mini, nonescape-full, sentry-convnext-small, hybrid, hybrid-plus, or ultra.",
    ),
    weight_path: Optional[Path] = typer.Option(None, help="Optional local UnivFD fc_weights.pth path."),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME, help="OpenCLIP model name for UnivFD."),
    pretrained: str = typer.Option(DEFAULT_PRETRAINED, help="OpenCLIP pretrained tag for UnivFD."),
    hf_model: str = typer.Option(DEFAULT_HF_MODEL_ID, help="Hugging Face model id for --backend hf."),
    hybrid_univfd_weight: float = typer.Option(
        DEFAULT_HYBRID_UNIVFD_WEIGHT,
        min=0.0,
        max=1.0,
        help="Blend weight for UnivFD when --backend hybrid.",
    ),
    hybrid_plus_primary_weight: float = typer.Option(
        DEFAULT_HYBRID_PLUS_PRIMARY_WEIGHT,
        min=0.0,
        max=1.0,
        help="Blend weight for hybrid when --backend hybrid-plus.",
    ),
    ultra_primary_weight: float = typer.Option(
        DEFAULT_ULTRA_PRIMARY_WEIGHT,
        min=0.0,
        max=1.0,
        help="Blend weight for hybrid-plus when --backend ultra.",
    ),
    batch_size: int = typer.Option(16, min=1, help="Batch size for inference."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write JSON benchmark report."),
) -> None:
    """Evaluate on a Hugging Face image dataset."""
    detector = create_detector(
        backend,
        device=device,
        threshold=threshold,
        weight_path=weight_path,
        model_name=model_name,
        pretrained=pretrained,
        hf_model=hf_model,
        hybrid_univfd_weight=hybrid_univfd_weight,
        hybrid_plus_primary_weight=hybrid_plus_primary_weight,
        ultra_primary_weight=ultra_primary_weight,
    )
    report = evaluate_hf_dataset(
        detector,
        dataset_name,
        split=split,
        image_field=image_field,
        label_field=label_field,
        fake_label=fake_label,
        max_samples=max_samples,
        batch_size=batch_size,
        streaming=streaming,
        shuffle_seed=shuffle_seed,
        trust_remote_code=trust_remote_code,
    )
    _finish_benchmark(report, output)


@app.command("benchmark-calibrated-folder")
def benchmark_calibrated_folder_command(
    root: Path = typer.Argument(..., help="Dataset root containing real/fake folders."),
    real_dir: str = typer.Option("real", help="Folder name containing real images."),
    fake_dir: str = typer.Option("ai", help="Folder name containing AI/fake images."),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Scan folders recursively."),
    max_per_class: Optional[int] = typer.Option(None, min=2, help="Optional cap per class."),
    calibration_fraction: float = typer.Option(
        0.5,
        min=0.05,
        max=0.95,
        help="Fraction of each class reserved for calibration.",
    ),
    split_seed: str = typer.Option("aidetect", help="Seed used for deterministic calibration/test split."),
    threshold: float = typer.Option(0.5, help="Fallback threshold before calibration."),
    device: Optional[str] = typer.Option(None, help="cuda, mps, cpu, or auto when omitted."),
    backend: str = typer.Option(
        "hybrid",
        help="Backend to calibrate: univfd, hf, nonescape-mini, nonescape-full, sentry-convnext-small, hybrid, hybrid-plus, or ultra.",
    ),
    weight_path: Optional[Path] = typer.Option(None, help="Optional local UnivFD fc_weights.pth path."),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME, help="OpenCLIP model name for UnivFD."),
    pretrained: str = typer.Option(DEFAULT_PRETRAINED, help="OpenCLIP pretrained tag for UnivFD."),
    hf_model: str = typer.Option(DEFAULT_HF_MODEL_ID, help="Hugging Face model id for HF or hybrid."),
    hybrid_univfd_weight: float = typer.Option(
        DEFAULT_HYBRID_UNIVFD_WEIGHT,
        min=0.0,
        max=1.0,
        help="Initial blend weight for UnivFD when backend=hybrid.",
    ),
    hybrid_alpha_step: float = typer.Option(
        0.05,
        min=0.01,
        max=0.5,
        help="Grid step for hybrid weight search during calibration.",
    ),
    hybrid_plus_primary_weight: float = typer.Option(
        DEFAULT_HYBRID_PLUS_PRIMARY_WEIGHT,
        min=0.0,
        max=1.0,
        help="Initial blend weight for hybrid when backend=hybrid-plus.",
    ),
    ultra_primary_weight: float = typer.Option(
        DEFAULT_ULTRA_PRIMARY_WEIGHT,
        min=0.0,
        max=1.0,
        help="Initial blend weight for hybrid-plus when backend=ultra.",
    ),
    optimize_metric: str = typer.Option(
        "balanced_accuracy",
        help="Calibration objective: balanced_accuracy, f1, recall, or precision.",
    ),
    min_recall: Optional[float] = typer.Option(
        None,
        min=0.0,
        max=1.0,
        help="Optional minimum calibration recall constraint for threshold search.",
    ),
    batch_size: int = typer.Option(16, min=1, help="Batch size for inference."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write JSON calibration benchmark report."),
) -> None:
    """Calibrate threshold on a held-out split, then evaluate on a disjoint test split."""
    _validate_optimize_metric(optimize_metric)
    samples = collect_folder_samples(
        root,
        real_dir=real_dir,
        fake_dir=fake_dir,
        recursive=recursive,
        max_per_class=max_per_class,
    )
    if len(samples) < 4:
        raise typer.BadParameter("Need at least two real and two AI images for calibrated benchmarking.")

    calibration_samples, test_samples = split_samples_balanced(
        samples,
        calibration_fraction=calibration_fraction,
        seed=split_seed,
    )
    report = _run_calibrated_folder_benchmark(
        calibration_samples=calibration_samples,
        test_samples=test_samples,
        backend=backend,
        threshold=threshold,
        device=device,
        weight_path=weight_path,
        model_name=model_name,
        pretrained=pretrained,
        hf_model=hf_model,
        hybrid_univfd_weight=hybrid_univfd_weight,
        hybrid_plus_primary_weight=hybrid_plus_primary_weight,
        ultra_primary_weight=ultra_primary_weight,
        hybrid_alpha_step=hybrid_alpha_step,
        optimize_metric=optimize_metric,
        min_recall=min_recall,
        batch_size=batch_size,
        dataset_info={
            "kind": "folder-calibrated",
            "root": str(root),
            "real_dir": real_dir,
            "fake_dir": fake_dir,
            "recursive": recursive,
            "max_per_class": max_per_class,
            "calibration_fraction": calibration_fraction,
            "split_seed": split_seed,
            "calibration_size": len(calibration_samples),
            "test_size": len(test_samples),
        },
    )
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"Saved calibrated benchmark report to {output}")
    _print_calibrated_benchmark(report)


@app.command("benchmark-tiny-genimage-local")
def benchmark_tiny_genimage_local_command(
    parquet_paths: list[Path] = typer.Argument(..., help="One or more local Tiny-GenImage parquet shards."),
    calibration_fraction: float = typer.Option(
        0.5,
        min=0.05,
        max=0.95,
        help="Fraction of each class reserved for calibration.",
    ),
    split_seed: str = typer.Option("aidetect", help="Seed used for deterministic calibration/test split."),
    max_per_class_per_shard: Optional[int] = typer.Option(
        None,
        min=1,
        help="Optional cap per class inside each parquet shard.",
    ),
    generators: Optional[str] = typer.Option(
        None,
        help="Comma-separated generator filter, for example 'Midjourney,SD14'.",
    ),
    threshold: float = typer.Option(0.5, help="Fallback threshold before calibration."),
    device: Optional[str] = typer.Option(None, help="cuda, mps, cpu, or auto when omitted."),
    backend: str = typer.Option(
        "univfd",
        help="Backend to calibrate: univfd, hf, nonescape-mini, nonescape-full, sentry-convnext-small, hybrid, hybrid-plus, or ultra.",
    ),
    weight_path: Optional[Path] = typer.Option(None, help="Optional local UnivFD fc_weights.pth path."),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME, help="OpenCLIP model name for UnivFD."),
    pretrained: str = typer.Option(DEFAULT_PRETRAINED, help="OpenCLIP pretrained tag for UnivFD."),
    hf_model: str = typer.Option(DEFAULT_HF_MODEL_ID, help="Hugging Face model id for HF or hybrid."),
    hybrid_univfd_weight: float = typer.Option(
        DEFAULT_HYBRID_UNIVFD_WEIGHT,
        min=0.0,
        max=1.0,
        help="Initial blend weight for UnivFD when backend=hybrid.",
    ),
    hybrid_alpha_step: float = typer.Option(
        0.05,
        min=0.01,
        max=0.5,
        help="Grid step for hybrid weight search during calibration.",
    ),
    hybrid_plus_primary_weight: float = typer.Option(
        DEFAULT_HYBRID_PLUS_PRIMARY_WEIGHT,
        min=0.0,
        max=1.0,
        help="Initial blend weight for hybrid when backend=hybrid-plus.",
    ),
    ultra_primary_weight: float = typer.Option(
        DEFAULT_ULTRA_PRIMARY_WEIGHT,
        min=0.0,
        max=1.0,
        help="Initial blend weight for hybrid-plus when backend=ultra.",
    ),
    optimize_metric: str = typer.Option(
        "balanced_accuracy",
        help="Calibration objective: balanced_accuracy, f1, recall, or precision.",
    ),
    min_recall: Optional[float] = typer.Option(
        None,
        min=0.0,
        max=1.0,
        help="Optional minimum calibration recall constraint for threshold search.",
    ),
    batch_size: int = typer.Option(16, min=1, help="Batch size for inference."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write JSON report."),
) -> None:
    """Benchmark one or more local Tiny-GenImage parquet shards with per-generator summaries."""
    _validate_optimize_metric(optimize_metric)
    generator_filter = None
    if generators:
        generator_filter = {value.strip() for value in generators.split(",") if value.strip()}

    samples = collect_tiny_genimage_parquet_samples(
        parquet_paths,
        max_per_class_per_shard=max_per_class_per_shard,
        generators=generator_filter,
    )
    if len(samples) < 4:
        raise typer.BadParameter("Need at least two real and two AI samples after filtering.")

    calibration_samples, test_samples = split_samples_balanced(
        samples,
        calibration_fraction=calibration_fraction,
        seed=split_seed,
    )
    report = _run_calibrated_folder_benchmark(
        calibration_samples=calibration_samples,
        test_samples=test_samples,
        backend=backend,
        threshold=threshold,
        device=device,
        weight_path=weight_path,
        model_name=model_name,
        pretrained=pretrained,
        hf_model=hf_model,
        hybrid_univfd_weight=hybrid_univfd_weight,
        hybrid_plus_primary_weight=hybrid_plus_primary_weight,
        ultra_primary_weight=ultra_primary_weight,
        hybrid_alpha_step=hybrid_alpha_step,
        optimize_metric=optimize_metric,
        min_recall=min_recall,
        batch_size=batch_size,
        dataset_info={
            "kind": "tiny-genimage-local",
            "parquet_paths": [str(path) for path in parquet_paths],
            "calibration_fraction": calibration_fraction,
            "split_seed": split_seed,
            "max_per_class_per_shard": max_per_class_per_shard,
            "generator_filter": sorted(generator_filter) if generator_filter else None,
            "calibration_size": len(calibration_samples),
            "test_size": len(test_samples),
        },
        group_fields=("generator", "source_shard"),
    )
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"Saved Tiny-GenImage benchmark report to {output}")
    _print_calibrated_benchmark(report)


@app.command("prepare-tiny-genimage")
def prepare_tiny_genimage_command(
    output_dir: Path = typer.Argument(..., help="Output folder with real/ai subfolders."),
    repo_id: str = typer.Option("TheKernel01/Tiny-GenImage", help="Hugging Face dataset repo."),
    filename: str = typer.Option(
        "data/validation-00000-of-00004.parquet",
        help="Parquet shard inside the dataset repo.",
    ),
    local_parquet: Optional[Path] = typer.Option(
        None,
        help="Use an already-downloaded parquet shard instead of fetching from Hugging Face.",
    ),
    max_per_class: int = typer.Option(20, min=1, help="Images to export per class."),
    overwrite: bool = typer.Option(False, help="Overwrite existing exported images."),
) -> None:
    """Download a Tiny-GenImage shard and export a balanced folder benchmark."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise typer.BadParameter("Install eval extras first: pip install 'ai-image-detector[eval]'") from exc

    if local_parquet is not None:
        parquet_path = str(local_parquet)
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise typer.BadParameter("Install huggingface_hub or pass --local-parquet.") from exc
        parquet_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)
    dataset = load_dataset("parquet", data_files={"data": parquet_path}, split="data")
    real_dir = output_dir / "real"
    fake_dir = output_dir / "ai"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    exported = {"real": 0, "ai": 0}
    label_names = getattr(dataset.features.get("label"), "names", None)
    generator_names = getattr(dataset.features.get("generator"), "names", None)
    for index, item in enumerate(dataset):
        label_value = int(item["label"])
        class_name = _tiny_genimage_class_name(label_value, label_names)
        if class_name not in exported or exported[class_name] >= max_per_class:
            continue
        generator = _safe_name(_class_label_name(item.get("generator"), generator_names))
        filename_out = f"{index:06d}-{generator}.jpg"
        output_path = (real_dir if class_name == "real" else fake_dir) / filename_out
        if output_path.exists() and not overwrite:
            exported[class_name] += 1
        else:
            item["image"].convert("RGB").save(output_path, quality=95)
            exported[class_name] += 1
        if all(count >= max_per_class for count in exported.values()):
            break

    if any(count < max_per_class for count in exported.values()):
        raise typer.BadParameter(
            f"Could only export {exported}; try a different shard or lower --max-per-class."
        )
    console.print(
        f"Exported Tiny-GenImage folder benchmark to {output_dir} "
        f"({exported['real']} real, {exported['ai']} ai)."
    )


@app.command()
def serve(
    threshold: float = typer.Option(0.5, help="AI probability threshold."),
    device: Optional[str] = typer.Option(None, help="cuda, mps, cpu, or auto when omitted."),
    backend: str = typer.Option(
        DEFAULT_BACKEND,
        help="Detector backend: univfd, hf, nonescape-mini, nonescape-full, sentry-convnext-small, hybrid, hybrid-plus, or ultra.",
    ),
    weight_path: Optional[Path] = typer.Option(None, help="Optional local UnivFD fc_weights.pth path."),
    hf_model: str = typer.Option(DEFAULT_HF_MODEL_ID, help="Hugging Face model id for --backend hf."),
    hybrid_univfd_weight: float = typer.Option(
        DEFAULT_HYBRID_UNIVFD_WEIGHT,
        min=0.0,
        max=1.0,
        help="Blend weight for UnivFD when --backend hybrid.",
    ),
    hybrid_plus_primary_weight: float = typer.Option(
        DEFAULT_HYBRID_PLUS_PRIMARY_WEIGHT,
        min=0.0,
        max=1.0,
        help="Blend weight for hybrid when --backend hybrid-plus.",
    ),
    ultra_primary_weight: float = typer.Option(
        DEFAULT_ULTRA_PRIMARY_WEIGHT,
        min=0.0,
        max=1.0,
        help="Blend weight for hybrid-plus when --backend ultra.",
    ),
) -> None:
    """Launch a local Gradio web UI."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise typer.BadParameter("Install web extras first: pip install 'ai-image-detector[web]'") from exc

    detector = create_detector(
        backend,
        device=device,
        threshold=threshold,
        weight_path=weight_path,
        hf_model=hf_model,
        hybrid_univfd_weight=hybrid_univfd_weight,
        hybrid_plus_primary_weight=hybrid_plus_primary_weight,
        ultra_primary_weight=ultra_primary_weight,
    )

    def predict(image):
        if image is None:
            return {"error": "Please upload an image."}
        result = detector.predict_image(image)
        return result.as_dict()

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload image"),
        outputs=gr.JSON(label="Detection result"),
        title="AI Image Detector",
        description="AI-generated image detector. Results are probabilistic.",
    )
    demo.launch()


@app.command("api")
def api_command(
    host: str = typer.Option("127.0.0.1", help="Host to bind."),
    port: int = typer.Option(8000, help="Port to bind."),
    threshold: float = typer.Option(0.5, help="AI probability threshold."),
    device: Optional[str] = typer.Option(None, help="cuda, mps, cpu, or auto when omitted."),
    backend: str = typer.Option(
        DEFAULT_BACKEND,
        help="Detector backend: univfd, hf, nonescape-mini, nonescape-full, sentry-convnext-small, hybrid, hybrid-plus, or ultra.",
    ),
    weight_path: Optional[Path] = typer.Option(None, help="Optional local UnivFD fc_weights.pth path."),
    hf_model: str = typer.Option(DEFAULT_HF_MODEL_ID, help="Hugging Face model id for --backend hf."),
    hybrid_univfd_weight: float = typer.Option(
        DEFAULT_HYBRID_UNIVFD_WEIGHT,
        min=0.0,
        max=1.0,
        help="Blend weight for UnivFD when --backend hybrid.",
    ),
    hybrid_plus_primary_weight: float = typer.Option(
        DEFAULT_HYBRID_PLUS_PRIMARY_WEIGHT,
        min=0.0,
        max=1.0,
        help="Blend weight for hybrid when --backend hybrid-plus.",
    ),
    ultra_primary_weight: float = typer.Option(
        DEFAULT_ULTRA_PRIMARY_WEIGHT,
        min=0.0,
        max=1.0,
        help="Blend weight for hybrid-plus when --backend ultra.",
    ),
) -> None:
    """Launch a FastAPI service."""
    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter("Install API extras first: pip install 'ai-image-detector[api]'") from exc

    from .api import create_app

    fastapi_app = create_app(
        threshold=threshold,
        device=device,
        backend=backend,
        weight_path=weight_path,
        hf_model=hf_model,
        hybrid_univfd_weight=hybrid_univfd_weight,
        hybrid_plus_primary_weight=hybrid_plus_primary_weight,
        ultra_primary_weight=ultra_primary_weight,
    )
    uvicorn.run(fastapi_app, host=host, port=port)


def _predict_paths(detector, paths: list[Path], *, batch_size: int, quiet: bool) -> list[DetectionResult]:
    results: list[DetectionResult] = []
    batch_images = []
    batch_paths = []
    iterator = tqdm(paths, disable=quiet or len(paths) == 1)
    for image_path in iterator:
        try:
            with Image.open(image_path) as image:
                batch_images.append(ImageOps.exif_transpose(image).convert("RGB"))
            batch_paths.append(image_path)
            if len(batch_images) >= batch_size:
                results.extend(detector.predict_images(batch_images, paths=batch_paths))
                batch_images = []
                batch_paths = []
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Failed:[/red] {image_path} ({exc})")
    if batch_images:
        results.extend(detector.predict_images(batch_images, paths=batch_paths))
    return results


def _write_csv(results: list[DetectionResult], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "path",
            "label",
            "probability_ai",
            "probability_real",
            "confidence",
            "raw_score",
            "backend",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([r.as_dict() for r in results])


def _print_detection_table(results: list[DetectionResult]) -> None:
    table = Table(title="AI Image Detection Results")
    table.add_column("Image")
    table.add_column("Label")
    table.add_column("AI Prob", justify="right")
    table.add_column("Real Prob", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Backend")
    for result in results:
        row_style = "red" if result.label == "ai" else "green"
        table.add_row(
            str(result.path),
            result.label.upper(),
            f"{result.probability_ai:.3f}",
            f"{result.probability_real:.3f}",
            f"{result.confidence:.3f}",
            result.backend,
            style=row_style,
        )
    console.print(table)


def _finish_benchmark(report: EvaluationReport, output: Path | None) -> None:
    if output:
        write_report(report, output)
        console.print(f"Saved benchmark report to {output}")
    metrics = report.metrics
    table = Table(title="Benchmark Results")
    table.add_column("Dataset")
    table.add_column("N", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Balanced Acc", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("ROC AUC", justify="right")
    table.add_column("Images/s", justify="right")
    table.add_row(
        metrics.dataset,
        str(metrics.n_samples),
        f"{metrics.accuracy:.3f}",
        f"{metrics.balanced_accuracy:.3f}",
        f"{metrics.f1:.3f}",
        "n/a" if metrics.roc_auc is None else f"{metrics.roc_auc:.3f}",
        f"{metrics.images_per_second:.2f}",
    )
    console.print(table)


def _run_calibrated_folder_benchmark(
    *,
    calibration_samples,
    test_samples,
    backend: str,
    threshold: float,
    device: Optional[str],
    weight_path: Optional[Path],
    model_name: str,
    pretrained: str,
    hf_model: str,
    hybrid_univfd_weight: float,
    hybrid_plus_primary_weight: float,
    ultra_primary_weight: float,
    hybrid_alpha_step: float,
    optimize_metric: str,
    min_recall: Optional[float],
    batch_size: int,
    dataset_info: dict[str, Any],
    group_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    backend = backend.lower()
    if backend in {"ultra", "hybrid-ultra", "sentry-plus"}:
        primary_detector = create_detector(
            "hybrid-plus",
            device=device,
            threshold=threshold,
            weight_path=weight_path,
            model_name=model_name,
            pretrained=pretrained,
            hf_model=hf_model,
            hybrid_univfd_weight=hybrid_univfd_weight,
            hybrid_plus_primary_weight=hybrid_plus_primary_weight,
            ultra_primary_weight=ultra_primary_weight,
        )
        secondary_detector = create_detector(
            "sentry-convnext-small",
            device=device,
            threshold=threshold,
            weight_path=weight_path,
            model_name=model_name,
            pretrained=pretrained,
            hf_model=hf_model,
            hybrid_univfd_weight=hybrid_univfd_weight,
            hybrid_plus_primary_weight=hybrid_plus_primary_weight,
            ultra_primary_weight=ultra_primary_weight,
        )
        calibration_primary = collect_predictions(primary_detector, calibration_samples, batch_size=batch_size)
        calibration_secondary = collect_predictions(secondary_detector, calibration_samples, batch_size=batch_size)
        search = search_blend_weight_threshold(
            calibration_primary.y_true,
            calibration_primary.y_score,
            calibration_secondary.y_score,
            alpha_step=hybrid_alpha_step,
            objective=optimize_metric,
            min_recall=min_recall,
        )
        selected_weight = float(search["primary_weight"] or ultra_primary_weight)
        selected_threshold = float(search["threshold"] or threshold)
        calibration_scores = combine_scores(
            calibration_primary.y_score,
            calibration_secondary.y_score,
            univfd_weight=selected_weight,
        )
        test_primary = collect_predictions(primary_detector, test_samples, batch_size=batch_size)
        test_secondary = collect_predictions(secondary_detector, test_samples, batch_size=batch_size)
        test_scores = combine_scores(
            test_primary.y_score,
            test_secondary.y_score,
            univfd_weight=selected_weight,
        )
        calibration_metrics = compute_metrics(
            calibration_primary.y_true,
            calibration_scores,
            dataset="calibration",
            threshold=selected_threshold,
            seconds=calibration_primary.seconds + calibration_secondary.seconds,
        )
        test_metrics = compute_metrics(
            test_primary.y_true,
            test_scores,
            dataset="test",
            threshold=selected_threshold,
            seconds=test_primary.seconds + test_secondary.seconds,
        )
        calibration_predictions = build_combined_predictions(
            calibration_primary.predictions,
            calibration_scores,
            threshold=selected_threshold,
            backend="ultra",
        )
        test_predictions = build_combined_predictions(
            test_primary.predictions,
            test_scores,
            threshold=selected_threshold,
            backend="ultra",
        )
        model = {
            "backend": "ultra",
            "selected_threshold": selected_threshold,
            "selected_primary_weight": selected_weight,
            "search_alpha_step": hybrid_alpha_step,
            "optimize_metric": optimize_metric,
            "min_recall": min_recall,
            "components": {
                "primary": primary_detector.model_info(),
                "secondary": secondary_detector.model_info(),
            },
        }
        search_summary = search
    elif backend in {"hybrid-plus", "hybrid_plus", "stacked"}:
        primary_detector = create_detector(
            "hybrid",
            device=device,
            threshold=threshold,
            weight_path=weight_path,
            model_name=model_name,
            pretrained=pretrained,
            hf_model=hf_model,
            hybrid_univfd_weight=hybrid_univfd_weight,
            hybrid_plus_primary_weight=hybrid_plus_primary_weight,
            ultra_primary_weight=ultra_primary_weight,
        )
        secondary_detector = create_detector(
            "nonescape-mini",
            device=device,
            threshold=threshold,
            weight_path=weight_path,
            model_name=model_name,
            pretrained=pretrained,
            hf_model=hf_model,
            hybrid_univfd_weight=hybrid_univfd_weight,
            hybrid_plus_primary_weight=hybrid_plus_primary_weight,
            ultra_primary_weight=ultra_primary_weight,
        )
        calibration_primary = collect_predictions(primary_detector, calibration_samples, batch_size=batch_size)
        calibration_secondary = collect_predictions(secondary_detector, calibration_samples, batch_size=batch_size)
        search = search_blend_weight_threshold(
            calibration_primary.y_true,
            calibration_primary.y_score,
            calibration_secondary.y_score,
            alpha_step=hybrid_alpha_step,
            objective=optimize_metric,
            min_recall=min_recall,
        )
        selected_weight = float(search["primary_weight"] or hybrid_plus_primary_weight)
        selected_threshold = float(search["threshold"] or threshold)
        calibration_scores = combine_scores(
            calibration_primary.y_score,
            calibration_secondary.y_score,
            univfd_weight=selected_weight,
        )
        test_primary = collect_predictions(primary_detector, test_samples, batch_size=batch_size)
        test_secondary = collect_predictions(secondary_detector, test_samples, batch_size=batch_size)
        test_scores = combine_scores(
            test_primary.y_score,
            test_secondary.y_score,
            univfd_weight=selected_weight,
        )
        calibration_metrics = compute_metrics(
            calibration_primary.y_true,
            calibration_scores,
            dataset="calibration",
            threshold=selected_threshold,
            seconds=calibration_primary.seconds + calibration_secondary.seconds,
        )
        test_metrics = compute_metrics(
            test_primary.y_true,
            test_scores,
            dataset="test",
            threshold=selected_threshold,
            seconds=test_primary.seconds + test_secondary.seconds,
        )
        calibration_predictions = build_combined_predictions(
            calibration_primary.predictions,
            calibration_scores,
            threshold=selected_threshold,
            backend="hybrid-plus",
        )
        test_predictions = build_combined_predictions(
            test_primary.predictions,
            test_scores,
            threshold=selected_threshold,
            backend="hybrid-plus",
        )
        model = {
            "backend": "hybrid-plus",
            "selected_threshold": selected_threshold,
            "selected_primary_weight": selected_weight,
            "search_alpha_step": hybrid_alpha_step,
            "optimize_metric": optimize_metric,
            "min_recall": min_recall,
            "components": {
                "primary": primary_detector.model_info(),
                "secondary": secondary_detector.model_info(),
            },
        }
        search_summary = search
    elif backend in {"hybrid", "ensemble"}:
        univfd_detector = create_detector(
            "univfd",
            device=device,
            threshold=threshold,
            weight_path=weight_path,
            model_name=model_name,
            pretrained=pretrained,
            hf_model=hf_model,
            hybrid_plus_primary_weight=hybrid_plus_primary_weight,
            ultra_primary_weight=ultra_primary_weight,
        )
        hf_detector = create_detector(
            "hf",
            device=device,
            threshold=threshold,
            weight_path=weight_path,
            model_name=model_name,
            pretrained=pretrained,
            hf_model=hf_model,
            hybrid_plus_primary_weight=hybrid_plus_primary_weight,
            ultra_primary_weight=ultra_primary_weight,
        )
        calibration_univfd = collect_predictions(univfd_detector, calibration_samples, batch_size=batch_size)
        calibration_hf = collect_predictions(hf_detector, calibration_samples, batch_size=batch_size)
        search = search_hybrid_weight_threshold(
            calibration_univfd.y_true,
            calibration_univfd.y_score,
            calibration_hf.y_score,
            alpha_step=hybrid_alpha_step,
            objective=optimize_metric,
            min_recall=min_recall,
        )
        selected_weight = float(search["univfd_weight"] or hybrid_univfd_weight)
        selected_threshold = float(search["threshold"] or threshold)
        calibration_scores = combine_scores(
            calibration_univfd.y_score,
            calibration_hf.y_score,
            univfd_weight=selected_weight,
        )
        test_univfd = collect_predictions(univfd_detector, test_samples, batch_size=batch_size)
        test_hf = collect_predictions(hf_detector, test_samples, batch_size=batch_size)
        test_scores = combine_scores(
            test_univfd.y_score,
            test_hf.y_score,
            univfd_weight=selected_weight,
        )
        calibration_metrics = compute_metrics(
            calibration_univfd.y_true,
            calibration_scores,
            dataset="calibration",
            threshold=selected_threshold,
            seconds=calibration_univfd.seconds + calibration_hf.seconds,
        )
        test_metrics = compute_metrics(
            test_univfd.y_true,
            test_scores,
            dataset="test",
            threshold=selected_threshold,
            seconds=test_univfd.seconds + test_hf.seconds,
        )
        calibration_predictions = build_combined_predictions(
            calibration_univfd.predictions,
            calibration_scores,
            threshold=selected_threshold,
            backend="hybrid",
        )
        test_predictions = build_combined_predictions(
            test_univfd.predictions,
            test_scores,
            threshold=selected_threshold,
            backend="hybrid",
        )
        model = {
            "backend": "hybrid",
            "selected_threshold": selected_threshold,
            "selected_univfd_weight": selected_weight,
            "search_alpha_step": hybrid_alpha_step,
            "optimize_metric": optimize_metric,
            "min_recall": min_recall,
            "components": {
                "univfd": univfd_detector.model_info(),
                "hf": hf_detector.model_info(),
            },
        }
        search_summary = search
    else:
        detector = create_detector(
            backend,
            device=device,
            threshold=threshold,
            weight_path=weight_path,
            model_name=model_name,
            pretrained=pretrained,
            hf_model=hf_model,
            hybrid_univfd_weight=hybrid_univfd_weight,
            hybrid_plus_primary_weight=hybrid_plus_primary_weight,
            ultra_primary_weight=ultra_primary_weight,
        )
        calibration_run = collect_predictions(detector, calibration_samples, batch_size=batch_size)
        search = search_threshold(
            calibration_run.y_true,
            calibration_run.y_score,
            objective=optimize_metric,
            min_recall=min_recall,
        )
        selected_threshold = float(search["threshold"] or threshold)
        test_run = collect_predictions(detector, test_samples, batch_size=batch_size)
        calibration_metrics = compute_metrics(
            calibration_run.y_true,
            calibration_run.y_score,
            dataset="calibration",
            threshold=selected_threshold,
            seconds=calibration_run.seconds,
        )
        test_metrics = compute_metrics(
            test_run.y_true,
            test_run.y_score,
            dataset="test",
            threshold=selected_threshold,
            seconds=test_run.seconds,
        )
        calibration_predictions = build_combined_predictions(
            calibration_run.predictions,
            calibration_run.y_score,
            threshold=selected_threshold,
            backend=backend,
        )
        test_predictions = build_combined_predictions(
            test_run.predictions,
            test_run.y_score,
            threshold=selected_threshold,
            backend=backend,
        )
        model = detector.model_info() | {
            "selected_threshold": selected_threshold,
            "optimize_metric": optimize_metric,
            "min_recall": min_recall,
        }
        search_summary = search

    report = {
        "mode": "calibrated-folder",
        "dataset": dataset_info,
        "model": model,
        "calibration": {
            "metrics": calibration_metrics.as_dict(),
            "predictions": calibration_predictions,
        },
        "test": {
            "metrics": test_metrics.as_dict(),
            "predictions": test_predictions,
        },
        "search": search_summary,
    }
    if group_fields:
        threshold_value = float(report["test"]["metrics"]["threshold"])
        report["groups"] = {}
        for field in group_fields:
            report["groups"][field] = group_prediction_rows(
                report["test"]["predictions"],
                field=field,
                threshold=threshold_value,
            )
            if field == "generator":
                report["groups"]["generator_vs_real"] = group_prediction_rows_against_reference(
                    report["test"]["predictions"],
                    field="generator",
                    reference_value="Real",
                    threshold=threshold_value,
                )
    return report


def _print_calibrated_benchmark(report: dict[str, Any]) -> None:
    table = Table(title="Calibrated Benchmark Results")
    table.add_column("Split")
    table.add_column("N", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Balanced Acc", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("ROC AUC", justify="right")
    table.add_column("Images/s", justify="right")
    for split in ("calibration", "test"):
        metrics = report[split]["metrics"]
        table.add_row(
            split,
            str(metrics["n_samples"]),
            f"{metrics['threshold']:.3f}",
            f"{metrics['accuracy']:.3f}",
            f"{metrics['balanced_accuracy']:.3f}",
            f"{metrics['f1']:.3f}",
            "n/a" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.3f}",
            f"{metrics['images_per_second']:.2f}",
        )
    console.print(table)
    model = report["model"]
    if model.get("backend") == "hybrid":
        console.print(
            "Selected hybrid setting: "
            f"threshold={model['selected_threshold']:.3f}, "
            f"univfd_weight={model['selected_univfd_weight']:.3f}"
        )
    if model.get("backend") == "hybrid-plus":
        console.print(
            "Selected hybrid-plus setting: "
            f"threshold={model['selected_threshold']:.3f}, "
            f"hybrid_weight={model['selected_primary_weight']:.3f}"
        )
    if model.get("backend") == "ultra":
        console.print(
            "Selected ultra setting: "
            f"threshold={model['selected_threshold']:.3f}, "
            f"hybrid-plus_weight={model['selected_primary_weight']:.3f}"
        )


def _tiny_genimage_class_name(label_value: int, label_names: list[str] | None) -> str:
    if label_names and 0 <= label_value < len(label_names):
        label_name = label_names[label_value].lower()
        if label_name in {"fake", "ai", "synthetic", "generated"}:
            return "ai"
        if label_name in {"real", "natural"}:
            return "real"
    return "ai" if label_value == 1 else "real"


def _class_label_name(value, names: list[str] | None) -> str:
    if names is not None and value is not None:
        index = int(value)
        if 0 <= index < len(names):
            return names[index]
    return "unknown" if value is None else str(value)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "-" for character in value)


def _validate_optimize_metric(value: str) -> None:
    if value not in VALID_OPTIMIZE_METRICS:
        raise typer.BadParameter(
            f"Unsupported optimize metric: {value!r}. Choose from {sorted(VALID_OPTIMIZE_METRICS)}."
        )


if __name__ == "__main__":
    app()
