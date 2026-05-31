# AI Image Detector Implementation Plan

1. Restructure the package into small modules:
   - `model.py`: UnivFD detector and image iteration.
   - `evaluation.py`: metrics and folder/HF dataset evaluation.
   - `api.py`: FastAPI factory.
   - `cli.py`: Typer commands wired to detector/evaluator/API/UI.
2. Improve robustness:
   - Batch inference.
   - Optional dependency checks.
   - Safe checkpoint loading with compatibility fallback.
   - Device/model metadata on reports.
3. Add tests:
   - Lightweight tests use fake detectors and generated images to avoid network/model downloads.
   - CLI smoke tests check JSONL/CSV and benchmark JSON output.
4. Update docs:
   - Explain SOTA survey and chosen default.
   - Document commands for detection, API server, Gradio, and benchmarks.
   - Include reproducible Tiny-GenImage benchmark command and recorded result.
5. QA:
   - Create Python 3.12 virtualenv.
   - Install package with dev/eval/web/api extras as needed.
   - Run unit tests.
   - Run a benchmark on a bounded Tiny-GenImage validation sample.
   - Save benchmark artifact under `benchmarks/`.
