# Benchmarks

These reports were generated locally on Tiny-GenImage validation shard
`data/validation-00000-of-00004.parquet`, exported with:

```bash
aidetect prepare-tiny-genimage .cache/tiny-genimage-validation-40 --max-per-class 20
```

Results use 20 real and 20 AI-generated images.

| Report | Backend | Accuracy | Balanced Acc | F1 | ROC AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| `tiny-genimage-univfd-40.json` | UnivFD / CLIP ViT-L/14 | 0.500 | 0.500 | 0.000 | 0.715 |
| `tiny-genimage-capcheck-40.json` | capcheck/ai-image-detection | 0.600 | 0.600 | 0.692 | 0.743 |

The JSON files include per-image predictions, confusion counts, runtime, model
metadata, and a diagnostic threshold sweep selected on the evaluated sample.
