from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import typer
from PIL import Image, ImageOps
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from .config import DEFAULT_BACKEND, DEFAULT_HF_MODEL_ID, DEFAULT_MODEL_NAME, DEFAULT_PRETRAINED
from .evaluation import evaluate_folder, evaluate_hf_dataset, write_report
from .model import create_detector, iter_images
from .types import DetectionResult, EvaluationReport

app = typer.Typer(help="Simple AI-generated image detector powered by UnivFD/CLIP.")
console = Console()


@app.command()
def detect(
    path: Path = typer.Argument(..., help="Image file or directory."),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Scan folders recursively."),
    threshold: float = typer.Option(0.5, help="AI probability threshold."),
    device: Optional[str] = typer.Option(None, help="cuda, mps, cpu, or auto when omitted."),
    backend: str = typer.Option(DEFAULT_BACKEND, help="Detector backend: univfd or hf."),
    weight_path: Optional[Path] = typer.Option(None, help="Optional local UnivFD fc_weights.pth path."),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME, help="OpenCLIP model name for UnivFD."),
    pretrained: str = typer.Option(DEFAULT_PRETRAINED, help="OpenCLIP pretrained tag for UnivFD."),
    hf_model: str = typer.Option(DEFAULT_HF_MODEL_ID, help="Hugging Face model id for --backend hf."),
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
    backend: str = typer.Option(DEFAULT_BACKEND, help="Detector backend: univfd or hf."),
    weight_path: Optional[Path] = typer.Option(None, help="Optional local UnivFD fc_weights.pth path."),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME, help="OpenCLIP model name for UnivFD."),
    pretrained: str = typer.Option(DEFAULT_PRETRAINED, help="OpenCLIP pretrained tag for UnivFD."),
    hf_model: str = typer.Option(DEFAULT_HF_MODEL_ID, help="Hugging Face model id for --backend hf."),
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
    backend: str = typer.Option(DEFAULT_BACKEND, help="Detector backend: univfd or hf."),
    weight_path: Optional[Path] = typer.Option(None, help="Optional local UnivFD fc_weights.pth path."),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME, help="OpenCLIP model name for UnivFD."),
    pretrained: str = typer.Option(DEFAULT_PRETRAINED, help="OpenCLIP pretrained tag for UnivFD."),
    hf_model: str = typer.Option(DEFAULT_HF_MODEL_ID, help="Hugging Face model id for --backend hf."),
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


@app.command("prepare-tiny-genimage")
def prepare_tiny_genimage_command(
    output_dir: Path = typer.Argument(..., help="Output folder with real/ai subfolders."),
    repo_id: str = typer.Option("TheKernel01/Tiny-GenImage", help="Hugging Face dataset repo."),
    filename: str = typer.Option(
        "data/validation-00000-of-00004.parquet",
        help="Parquet shard inside the dataset repo.",
    ),
    max_per_class: int = typer.Option(20, min=1, help="Images to export per class."),
    overwrite: bool = typer.Option(False, help="Overwrite existing exported images."),
) -> None:
    """Download a Tiny-GenImage shard and export a balanced folder benchmark."""
    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise typer.BadParameter("Install eval extras first: pip install 'ai-image-detector[eval]'") from exc

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
    backend: str = typer.Option(DEFAULT_BACKEND, help="Detector backend: univfd or hf."),
    weight_path: Optional[Path] = typer.Option(None, help="Optional local UnivFD fc_weights.pth path."),
    hf_model: str = typer.Option(DEFAULT_HF_MODEL_ID, help="Hugging Face model id for --backend hf."),
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
    backend: str = typer.Option(DEFAULT_BACKEND, help="Detector backend: univfd or hf."),
    weight_path: Optional[Path] = typer.Option(None, help="Optional local UnivFD fc_weights.pth path."),
    hf_model: str = typer.Option(DEFAULT_HF_MODEL_ID, help="Hugging Face model id for --backend hf."),
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


if __name__ == "__main__":
    app()
