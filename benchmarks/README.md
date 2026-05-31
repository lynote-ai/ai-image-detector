# Benchmarks

These reports were generated locally from Tiny-GenImage validation shard
`data/validation-00000-of-00004.parquet`.

Smoke benchmark export:

```bash
aidetect prepare-tiny-genimage .cache/tiny-genimage-validation-40 --max-per-class 20
```

Calibrated hold-out export:

```bash
aidetect prepare-tiny-genimage .cache/tiny-genimage-validation-200 --max-per-class 100
```

Smoke results use 20 real and 20 AI-generated images.

| Report | Backend | Accuracy | Balanced Acc | F1 | ROC AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| `tiny-genimage-univfd-40.json` | UnivFD / CLIP ViT-L/14 | 0.500 | 0.500 | 0.000 | 0.715 |
| `tiny-genimage-capcheck-40.json` | capcheck/ai-image-detection | 0.600 | 0.600 | 0.692 | 0.743 |

Calibrated hold-out results use a deterministic calibration/test split over 100
real + 100 AI-generated images.

| Report | Backend | Test Accuracy | Test Balanced Acc | Test F1 | Test ROC AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| `tiny-genimage-univfd-calibrated-200.json` | UnivFD / CLIP ViT-L/14 | 0.760 | 0.760 | 0.721 | 0.811 |
| `tiny-genimage-hybrid-calibrated-200.json` | Hybrid (UnivFD 0.8 + HF 0.2) | 0.670 | 0.670 | 0.629 | 0.752 |
| `tiny-genimage-hf-calibrated-200.json` | capcheck/ai-image-detection | 0.580 | 0.580 | 0.580 | 0.610 |

The calibrated JSON files include calibration metrics, test metrics, selected
thresholds, model metadata, and per-image predictions for both splits. These
reports were produced on CPU in the current workspace.

Multi-shard calibrated hold-out results currently include 3 validation shards
with up to 100 real + 100 fake samples exported per shard.

| Report | Backend | Test N | Test Accuracy | Test Balanced Acc | Precision | Recall | Test F1 | Test ROC AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tiny-genimage-hybrid-multishard-600.json` | Hybrid (UnivFD 0.85 + HF 0.15) | 300 | 0.743 | 0.743 | 0.745 | 0.740 | 0.742 | 0.816 |
| `tiny-genimage-univfd-multishard-600.json` | UnivFD / CLIP ViT-L/14 | 300 | 0.690 | 0.690 | 0.806 | 0.500 | 0.617 | 0.784 |

This report also includes:

- `groups.source_shard`: per-shard calibrated test metrics
- `groups.generator_vs_real`: one-generator-vs-real binary slices
- `groups.generator`: raw slice metrics for reference only
