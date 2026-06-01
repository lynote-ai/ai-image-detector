from __future__ import annotations

from pathlib import Path
from typing import Sequence
from urllib.request import urlretrieve

from PIL import Image

from .config import APP_NAME, NONESCAPE_FULL_URL, NONESCAPE_MINI_URL
from .types import DetectionResult


def _nonescape_cache_dir() -> Path:
    return Path.home() / ".cache" / APP_NAME / "nonescape"


def _ensure_weight_file(url: str, filename: str, weight_path: str | Path | None) -> Path:
    if weight_path is not None:
        return Path(weight_path)
    cache_dir = _nonescape_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / filename
    if not target.exists():
        urlretrieve(url, target)
    return target


class _NonescapeBase:
    backend_name = "nonescape"

    def __init__(self, *, threshold: float, device: str | None = None) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install torch to use the Nonescape backend.") from exc
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

    def predict_image(self, image: Image.Image, path: Path | None = None) -> DetectionResult:
        return self.predict_images([image], paths=[path])[0]

    def predict_path(self, path: str | Path) -> DetectionResult:
        path = Path(path)
        with Image.open(path) as image:
            return self.predict_image(image, path=path)

    def model_info(self) -> dict:
        return {
            "backend": self.backend_name,
            "device": str(self.device),
            "threshold": self.threshold,
        }


# Adapted from the Apache-2.0 licensed aediliclabs/nonescape project.
def preprocess_nonescape_image(image: Image.Image):
    import torch
    import torchvision.transforms.v2 as T

    transform = T.Compose(
        [
            T.ToImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.JPEG(quality=100),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image.convert("RGB"))


def _create_nonescape_mini_model():
    import torch
    import torchvision.models as models

    class NonescapeClassifierMini(torch.nn.Module):
        def __init__(self, num_classes: int = 2, embedding_size: int = 1024, dropout: float = 0.2):
            super().__init__()
            self.backbone = models.efficientnet_v2_s(
                weights=None,
                num_classes=embedding_size,
                dropout=dropout,
            )
            self.head = torch.nn.Linear(embedding_size, num_classes)

        def forward(self, x):
            emb = self.backbone(x)
            logits = self.head(emb)
            return torch.nn.functional.softmax(logits, dim=-1)

    return NonescapeClassifierMini()


class NonescapeMiniDetector(_NonescapeBase):
    backend_name = "nonescape-mini"

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        device: str | None = None,
        weight_path: str | Path | None = None,
    ) -> None:
        super().__init__(threshold=threshold, device=device)
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install safetensors to use the Nonescape backend.") from exc

        weight_file = _ensure_weight_file(
            NONESCAPE_MINI_URL,
            "nonescape-mini-v0.safetensors",
            weight_path,
        )
        self.weight_file = weight_file
        self.model = _create_nonescape_mini_model()
        state_dict = load_file(str(weight_file))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

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
        batch = torch.stack([preprocess_nonescape_image(image) for image in images], dim=0).to(self.device)
        with torch.inference_mode():
            probabilities = self.model(batch).detach().cpu()

        results: list[DetectionResult] = []
        for row, path in zip(probabilities, paths, strict=True):
            probability_ai = float(row[1].item())
            probability_real = float(row[0].item())
            label = "ai" if probability_ai >= self.threshold else "real"
            confidence = probability_ai if label == "ai" else probability_real
            results.append(
                DetectionResult(
                    path=path,
                    label=label,
                    probability_ai=probability_ai,
                    probability_real=probability_real,
                    confidence=confidence,
                    raw_score=probability_ai,
                    backend=self.backend_name,
                )
            )
        return results

    def model_info(self) -> dict:
        return super().model_info() | {
            "variant": "mini",
            "weight_file": str(self.weight_file),
            "weight_url": NONESCAPE_MINI_URL,
        }


class NonescapeFullDetector(_NonescapeBase):
    backend_name = "nonescape-full"

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        device: str | None = None,
        weight_path: str | Path | None = None,
    ) -> None:
        super().__init__(threshold=threshold, device=device)
        try:
            import torch
            import torchvision.models as models
            from safetensors.torch import load_file
            from transformers import Dinov2Config, Dinov2Model
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: install torch, torchvision, transformers, and safetensors "
                "to use the full Nonescape backend."
            ) from exc

        weight_file = _ensure_weight_file(
            NONESCAPE_FULL_URL,
            "nonescape-v0.safetensors",
            weight_path,
        )
        self.weight_file = weight_file

        class NonescapeClassifier(torch.nn.Module):
            def __init__(self, num_classes: int = 2, num_heads: int = 16, num_queries: int = 128):
                super().__init__()
                self.embedding_size = 1024
                self.num_queries = num_queries
                self.vit_backbone = Dinov2Model(Dinov2Config.from_pretrained("facebook/dinov2-large"))
                self.query_net = models.efficientnet_v2_l(
                    weights=None,
                    num_classes=num_queries * self.embedding_size,
                )
                self.key_net = torch.nn.Linear(self.embedding_size, self.embedding_size)
                self.value_net = torch.nn.Linear(self.embedding_size, self.embedding_size)
                self.attention = torch.nn.MultiheadAttention(
                    self.embedding_size,
                    num_heads=num_heads,
                    batch_first=True,
                )
                self.head = torch.nn.Linear(self.embedding_size, num_classes)

            def forward(self, x):
                batch_size = x.shape[0]
                with torch.no_grad():
                    vit_output = self.vit_backbone(x)
                    vit_features = vit_output.last_hidden_state
                q = self.query_net(x).reshape(batch_size, self.num_queries, -1)
                k = self.key_net(vit_features)
                v = self.value_net(vit_features)
                emb, _ = self.attention(q, k, v)
                emb = emb.mean(dim=1)
                logits = self.head(emb.squeeze(1))
                return torch.nn.functional.softmax(logits, dim=-1)

        self.model = NonescapeClassifier()
        state_dict = load_file(str(weight_file))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

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
        batch = torch.stack([preprocess_nonescape_image(image) for image in images], dim=0).to(self.device)
        with torch.inference_mode():
            probabilities = self.model(batch).detach().cpu()

        results: list[DetectionResult] = []
        for row, path in zip(probabilities, paths, strict=True):
            probability_ai = float(row[1].item())
            probability_real = float(row[0].item())
            label = "ai" if probability_ai >= self.threshold else "real"
            confidence = probability_ai if label == "ai" else probability_real
            results.append(
                DetectionResult(
                    path=path,
                    label=label,
                    probability_ai=probability_ai,
                    probability_real=probability_real,
                    confidence=confidence,
                    raw_score=probability_ai,
                    backend=self.backend_name,
                )
            )
        return results

    def model_info(self) -> dict:
        return super().model_info() | {
            "variant": "full",
            "weight_file": str(self.weight_file),
            "weight_url": NONESCAPE_FULL_URL,
        }
