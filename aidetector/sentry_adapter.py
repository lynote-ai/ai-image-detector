from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image

from .config import SENTRY_CONVNEXT_SMALL_CONFIG, SENTRY_CONVNEXT_SMALL_FILE, SENTRY_REPO_ID
from .types import DetectionResult


class SentryConvNeXtDetector:
    backend_name = "sentry-convnext-small"

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        device: str | None = None,
        weight_path: str | Path | None = None,
    ) -> None:
        try:
            import torch
            import timm
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: install torch, timm, and huggingface_hub to use the Sentry backend."
            ) from exc

        self._torch = torch
        if device and device != "auto":
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.threshold = threshold

        if weight_path is None:
            weight_path = hf_hub_download(SENTRY_REPO_ID, SENTRY_CONVNEXT_SMALL_FILE)
        self.weight_file = Path(weight_path)
        self.config_file = hf_hub_download(SENTRY_REPO_ID, SENTRY_CONVNEXT_SMALL_CONFIG)
        self.model = timm.create_model("convnext_small", pretrained=False, num_classes=2)
        checkpoint = torch.load(str(self.weight_file), map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        converted = convert_sentry_convnext_state_dict(state_dict)
        missing, unexpected = self.model.load_state_dict(converted, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected Sentry checkpoint keys: {unexpected[:10]}")
        if missing:
            required_missing = [key for key in missing if not key.startswith("head.norm")]
            if required_missing:
                raise RuntimeError(f"Missing required Sentry checkpoint keys: {required_missing[:10]}")
        self.model.to(self.device).eval()

    def model_info(self) -> dict:
        return {
            "backend": self.backend_name,
            "device": str(self.device),
            "threshold": self.threshold,
            "weight_file": str(self.weight_file),
            "config_file": str(self.config_file),
            "repo_id": SENTRY_REPO_ID,
        }

    def predict_image(self, image: Image.Image, path: Path | None = None) -> DetectionResult:
        return self.predict_images([image], paths=[path])[0]

    def predict_images(
        self,
        images: Sequence[Image.Image],
        paths: Sequence[Path | None] | None = None,
    ) -> list[DetectionResult]:
        if not images:
            return []
        if paths is None:
            paths = [None] * len(images)
        if len(paths) != len(images):
            raise ValueError("paths must have the same length as images")

        torch = self._torch
        batch = torch.stack([preprocess_sentry_image(image) for image in images], dim=0).to(self.device)
        with torch.inference_mode():
            logits = self.model(batch).detach().cpu()
            probabilities = torch.softmax(logits, dim=-1)

        results: list[DetectionResult] = []
        for row, logit_row, path in zip(probabilities, logits, paths, strict=True):
            # Sentry fake-image checkpoints use class 0 for fake and class 1 for real.
            probability_ai = float(row[0].item())
            probability_real = float(row[1].item())
            label = "ai" if probability_ai >= self.threshold else "real"
            confidence = probability_ai if label == "ai" else probability_real
            results.append(
                DetectionResult(
                    path=path,
                    label=label,
                    probability_ai=probability_ai,
                    probability_real=probability_real,
                    confidence=confidence,
                    raw_score=float(logit_row[0].item()),
                    backend=self.backend_name,
                )
            )
        return results

    def predict_path(self, path: str | Path) -> DetectionResult:
        path = Path(path)
        with Image.open(path) as image:
            return self.predict_image(image, path=path)


def preprocess_sentry_image(image: Image.Image):
    import torch
    import torchvision.transforms as T

    transform = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0],
                std=[58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0],
            ),
        ]
    )
    tensor = transform(image.convert("RGB"))
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Expected torchvision transform to return a torch.Tensor")
    return tensor


def convert_sentry_convnext_state_dict(state_dict: dict) -> dict:
    converted = {}
    for key, value in state_dict.items():
        if not key.startswith(("backbone.", "head.")):
            continue
        new_key = key
        new_key = new_key.replace("backbone.downsample_layers.0.0.", "stem.0.")
        new_key = new_key.replace("backbone.downsample_layers.0.1.", "stem.1.")
        for stage_index in range(1, 4):
            new_key = new_key.replace(
                f"backbone.downsample_layers.{stage_index}.0.",
                f"stages.{stage_index}.downsample.0.",
            )
            new_key = new_key.replace(
                f"backbone.downsample_layers.{stage_index}.1.",
                f"stages.{stage_index}.downsample.1.",
            )
        for stage_index in range(4):
            new_key = new_key.replace(
                f"backbone.stages.{stage_index}.",
                f"stages.{stage_index}.blocks.",
            )
        new_key = new_key.replace(".depthwise_conv.", ".conv_dw.")
        new_key = new_key.replace(".pointwise_conv1.", ".mlp.fc1.")
        new_key = new_key.replace(".pointwise_conv2.", ".mlp.fc2.")
        new_key = new_key.replace("backbone.norm3.", "head.norm.")
        new_key = new_key.replace("head.fc.", "head.fc.")
        converted[new_key] = value
    return converted
