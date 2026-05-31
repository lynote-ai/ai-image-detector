# AI Image Detector Spec

## Product goal

Turn the MVP into a practical open-source package for detecting AI-generated images. The project should be simple for users (`aidetect image.jpg`) while exposing reproducible evaluation tools for benchmark datasets.

## Research summary

- Default model path: UnivFD / UniversalFakeDetect, a CLIP ViT-L/14 image encoder plus a tiny linear head. It is not the newest research idea, but it is a strong practical baseline with good cross-generator generalization, small task-specific weights, and an official open-source lineage.
- Recent SOTA direction: AIDE combines high-level CLIP features with low-level frequency/noise features and reports gains over previous methods on AIGCDetectBenchmark and GenImage. This is worth documenting as the research frontier, but it is heavier and less package-friendly for a first open-source release.
- Benchmark target: support folder benchmarks and Hugging Face datasets. Use Tiny-GenImage for a runnable GenImage-style smoke benchmark, and support CIFAKE/other datasets via generic dataset options.

## Functional requirements

- CLI:
  - Detect a single image or folder recursively.
  - Output rich table, JSONL, and CSV.
  - Benchmark folder datasets with `real`/`ai` directories.
  - Benchmark Hugging Face datasets with configurable split, image field, label field, fake label, sampling, and output JSON.
  - Launch optional Gradio UI.
  - Launch optional FastAPI service.
- Python API:
  - Predict from PIL images and filesystem paths.
  - Batch prediction with configurable batch size.
  - Reusable evaluation helpers with metrics.
- Model implementation:
  - UnivFD backend with OpenCLIP model loading and UniversalFakeDetect head loading.
  - Safe checkpoint loading where supported.
  - Deterministic preprocessing through OpenCLIP transforms.
  - Clear errors when optional dependencies are missing.
- Evaluation:
  - Accuracy, balanced accuracy, precision, recall, F1, ROC AUC when possible, confusion counts, latency, and throughput.
  - JSON report with model metadata, dataset metadata, and per-run parameters.
- Packaging/docs:
  - Update pyproject dependencies/extras.
  - Add README with research context, install, CLI/API/API server usage, benchmarks, limitations, and citations.
- Tests:
  - Unit tests for path scanning, result serialization, metrics, dataset label mapping, and CLI output using a fake detector.

## Non-goals

- Training a new detector from scratch in this pass.
- Claiming forensic certainty or legal proof.
- Downloading full million-scale GenImage by default.
