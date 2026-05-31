# AI Image Detector

A small, friendly open-source detector for AI-generated images. It is designed in the
`yt-dlp` / `rembg` spirit: install it, run one command, get a probability and a
reproducible report.

> AI image detection is probabilistic. Treat the output as one signal, not as proof.

## Model Choice

The default backend is **UnivFD / UniversalFakeDetect**: CLIP ViT-L/14 image
features plus a tiny linear fake/real head. This is a strong practical default
because the task-specific weight is tiny, the code path is understandable, and the
CVPR 2023 paper showed good cross-generator generalization compared with older
GAN-trained detectors.

Recent research has moved further. **AIDE** combines CLIP semantics with low-level
frequency/noise features and reports gains on GenImage and AIGCDetectBenchmark.
That is a good research target for a future backend, but UnivFD is currently the
simplest robust default for an installable open-source tool.

Useful references:

- UniversalFakeDetect paper: https://openaccess.thecvf.com/content/CVPR2023/html/Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_Generative_Models_CVPR_2023_paper.html
- UniversalFakeDetect code: https://github.com/WisconsinAIVision/UniversalFakeDetect
- AIDE paper: https://arxiv.org/abs/2406.19435
- GenImage benchmark: https://github.com/GenImage-Dataset/GenImage
- Tiny-GenImage runnable subset: https://huggingface.co/datasets/TheKernel01/Tiny-GenImage
- CIFAKE benchmark paper: https://huggingface.co/papers/2303.14126

## Install

Use Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras:

```bash
pip install -e '.[eval]'      # Hugging Face dataset benchmarks
pip install -e '.[hf]'        # generic Hugging Face image-classification backend
pip install -e '.[api]'       # FastAPI server
pip install -e '.[web]'       # Gradio UI
pip install -e '.[dev]'       # tests and linting
```

## CLI Usage

Detect one image:

```bash
aidetect detect image.jpg
```

Detect a folder recursively:

```bash
aidetect detect ./images --csv report.csv
```

JSON lines output:

```bash
aidetect detect ./images --json
```

Use a Hugging Face image-classification model instead of UnivFD:

```bash
aidetect detect image.jpg --backend hf --hf-model capcheck/ai-image-detection
```

## Python API

```python
from aidetector import create_detector

detector = create_detector("univfd", device="auto")
result = detector.predict_path("image.jpg")
print(result.as_dict())
```

## Web UI

```bash
pip install -e '.[web]'
aidetect serve
```

## FastAPI

```bash
pip install -e '.[api]'
aidetect api --host 127.0.0.1 --port 8000
```

Then call:

```bash
curl -F "file=@image.jpg" http://127.0.0.1:8000/detect
```

## Benchmarks

Evaluate a GenImage-style folder where `nature/` contains real images and `ai/`
contains generated images:

```bash
aidetect benchmark-folder /path/to/GenImage/Midjourney/val \
  --real-dir nature \
  --fake-dir ai \
  --output benchmarks/midjourney-val.json
```

Evaluate a Hugging Face dataset such as Tiny-GenImage:

```bash
pip install -e '.[eval]'
aidetect benchmark-hf TheKernel01/Tiny-GenImage \
  --split validation \
  --image-field image \
  --label-field label \
  --fake-label 1 \
  --max-samples 200 \
  --output benchmarks/tiny-genimage-univfd-200.json
```

The JSON report includes accuracy, balanced accuracy, precision, recall, F1, ROC
AUC, confusion counts, a diagnostic threshold sweep, model metadata, dataset
metadata, and per-image predictions.

If Hugging Face dataset metadata requests are flaky, download a parquet shard and
export a folder benchmark:

```bash
aidetect prepare-tiny-genimage .cache/tiny-genimage-validation-40 \
  --max-per-class 20

aidetect benchmark-folder .cache/tiny-genimage-validation-40 \
  --backend univfd \
  --real-dir real \
  --fake-dir ai \
  --output benchmarks/tiny-genimage-univfd-40.json
```

Current local smoke benchmark on Tiny-GenImage validation shard
`data/validation-00000-of-00004.parquet`, 20 real + 20 fake images:

| Backend | Threshold | Accuracy | Balanced Acc | F1 | ROC AUC | Images/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UnivFD / CLIP ViT-L/14 | 0.5 | 0.500 | 0.500 | 0.000 | 0.715 | 2.31 |
| capcheck/ai-image-detection | 0.5 | 0.600 | 0.600 | 0.692 | 0.743 | 32.03 |

The UnivFD score ranking has signal on this small sample, but its default 0.5
threshold is not calibrated for this subset. The JSON report includes a
diagnostic sample-selected threshold sweep; do not report that as held-out test
performance.

## Model Weights

On first use, the UnivFD backend downloads:

- CLIP ViT-L/14 OpenAI weights through `open_clip_torch`
- UniversalFakeDetect linear head from
  `siddharthksah/deepsafe-weights/universalfakedetect/fc_weights.pth`

You can also pass a local head checkpoint:

```bash
aidetect detect image.jpg --weight-path ./fc_weights.pth
```

## Development

```bash
pip install -e '.[dev,eval,hf,api]'
pytest
ruff check .
```

## Limitations

- No detector is universal. New generators, heavy recompression, screenshots,
  crops, edits, upscaling, and adversarial post-processing can change results.
- Benchmarks can overstate real-world reliability if the deployment data differs
  from the benchmark distribution.
- The tool currently detects whole-image synthetic likelihood. It does not localize
  edited regions.

## Citation

If this helps your work, cite the original UnivFD paper:

```bibtex
@InProceedings{Ojha_2023_CVPR,
  author = {Ojha, Utkarsh and Li, Yuheng and Lee, Yong Jae},
  title = {Towards Universal Fake Image Detectors That Generalize Across Generative Models},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2023},
  pages = {24480-24489}
}
```
