from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol, Sequence

from PIL import Image, ImageOps

from .config import (
    DEFAULT_BACKEND,
    DEFAULT_HF_MODEL_ID,
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED,
    FC_WEIGHT_PATH_IN_REPO,
    IMAGE_EXTENSIONS,
    MODEL_REPO_ID,
)
from .types import DetectionResult


class Detector(Protocol):
    threshold: float

    def predict_image(self, image: Image.Image, path: Path | None = None) -> DetectionResult:
        ...

    def predict_images(
        self,
        images: Sequence[Image.Image],
        paths: Sequence[Path | None] | None = None,
    ) -> list[DetectionResult]:
        ...

    def predict_path(self, path: str | Path) -> DetectionResult:
        ...

    def model_info(self) -> dict:
        ...


def _select_device(torch_module, requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    mps = getattr(torch_module.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _normalise_image(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def _load_torch_checkpoint(torch_module, weight_path: str | Path):
    try:
        return torch_module.load(str(weight_path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch_module.load(str(weight_path), map_location="cpu")


class AIImageDetector:
    """UnivFD-style detector: CLIP image encoder plus a linear fake/real head."""

    backend_name = "univfd"

    def __init__(
        self,
        device: str | None = None,
        threshold: float = 0.5,
        model_name: str = DEFAULT_MODEL_NAME,
        pretrained: str = DEFAULT_PRETRAINED,
        weight_path: str | Path | None = None,
        repo_id: str = MODEL_REPO_ID,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install torch to use the UnivFD backend.") from exc
        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install open_clip_torch to use the UnivFD backend.") from exc

        self._torch = torch
        self.device = torch.device(_select_device(torch, device))
        self.threshold = threshold
        self.model_name = model_name
        self.pretrained = pretrained
        self.repo_id = repo_id

        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.clip_model.eval()

        output_dim = getattr(getattr(self.clip_model, "visual", None), "output_dim", 768)
        self.head = torch.nn.Linear(int(output_dim), 1).to(self.device)
        self._load_head(weight_path)
        self.head.eval()

    def _load_head(self, weight_path: str | Path | None) -> None:
        if weight_path is None:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise RuntimeError(
                    "Missing dependency: install huggingface_hub or pass --weight-path."
                ) from exc
            weight_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=FC_WEIGHT_PATH_IN_REPO,
            )

        state = _load_torch_checkpoint(self._torch, weight_path)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        if isinstance(state, dict):
            cleaned = {}
            for key, value in state.items():
                key = key.replace("module.", "")
                key = key.replace("model.", "")
                key = key.replace("fc.", "")
                key = key.replace("head.", "")
                if key in {"weight", "bias"}:
                    cleaned[key] = value
            if cleaned:
                self.head.load_state_dict(cleaned, strict=True)
                return
            try:
                self.head.load_state_dict(state, strict=False)
                return
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Could not load UnivFD head weights from {weight_path}: {exc}") from exc
        raise RuntimeError(f"Unsupported checkpoint format: {type(state)!r}")

    def model_info(self) -> dict:
        return {
            "backend": self.backend_name,
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "head_repo": self.repo_id,
            "head_file": FC_WEIGHT_PATH_IN_REPO,
            "device": str(self.device),
            "threshold": self.threshold,
        }

    @property
    def _torch_inference_mode(self):
        return self._torch.inference_mode

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
        tensors = [self.preprocess(_normalise_image(image)) for image in images]
        batch = torch.stack(tensors, dim=0).to(self.device)
        with torch.inference_mode():
            features = self.clip_model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
            logits = self.head(features).flatten().float().detach().cpu()
            probabilities = torch.sigmoid(logits)

        results: list[DetectionResult] = []
        for probability_ai_tensor, logit_tensor, path in zip(probabilities, logits, paths, strict=True):
            probability_ai = float(probability_ai_tensor.item())
            probability_real = 1.0 - probability_ai
            label = "ai" if probability_ai >= self.threshold else "real"
            confidence = probability_ai if label == "ai" else probability_real
            results.append(
                DetectionResult(
                    path=path,
                    label=label,
                    probability_ai=probability_ai,
                    probability_real=probability_real,
                    confidence=confidence,
                    raw_score=float(logit_tensor.item()),
                    backend=self.backend_name,
                )
            )
        return results

    def predict_path(self, path: str | Path) -> DetectionResult:
        path = Path(path)
        with Image.open(path) as image:
            return self.predict_image(image, path=path)

    def predict_many(
        self,
        paths: Iterable[str | Path],
        batch_size: int = 16,
    ) -> list[DetectionResult]:
        results: list[DetectionResult] = []
        batch_images: list[Image.Image] = []
        batch_paths: list[Path] = []
        for path_like in paths:
            path = Path(path_like)
            with Image.open(path) as image:
                batch_images.append(_normalise_image(image))
            batch_paths.append(path)
            if len(batch_images) >= batch_size:
                results.extend(self.predict_images(batch_images, paths=batch_paths))
                batch_images = []
                batch_paths = []
        if batch_images:
            results.extend(self.predict_images(batch_images, paths=batch_paths))
        return results


class HuggingFaceImageDetector:
    """Generic Hugging Face image-classification backend."""

    backend_name = "hf"

    def __init__(
        self,
        model_id: str = DEFAULT_HF_MODEL_ID,
        device: str | None = None,
        threshold: float = 0.5,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install torch to use the Hugging Face backend.") from exc
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install transformers to use the HF backend.") from exc

        self._torch = torch
        self.device = torch.device(_select_device(torch, device))
        self.threshold = threshold
        self.model_id = model_id
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageClassification.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def model_info(self) -> dict:
        return {
            "backend": self.backend_name,
            "model_id": self.model_id,
            "device": str(self.device),
            "threshold": self.threshold,
            "id2label": {str(k): v for k, v in self.id2label.items()},
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
        normalised = [_normalise_image(image) for image in images]
        inputs = self.processor(images=normalised, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits.detach().cpu()
            probabilities = torch.softmax(logits, dim=-1)

        fake_indices = self._fake_label_indices()
        if not fake_indices:
            raise RuntimeError(f"Could not infer fake/AI label from model labels: {self.id2label}")

        results: list[DetectionResult] = []
        for row, logit_row, path in zip(probabilities, logits, paths, strict=True):
            probability_ai = float(row[fake_indices].sum().item())
            probability_real = 1.0 - probability_ai
            label = "ai" if probability_ai >= self.threshold else "real"
            confidence = probability_ai if label == "ai" else probability_real
            raw_score = float(logit_row[fake_indices].mean().item())
            results.append(
                DetectionResult(
                    path=path,
                    label=label,
                    probability_ai=probability_ai,
                    probability_real=probability_real,
                    confidence=confidence,
                    raw_score=raw_score,
                    backend=self.backend_name,
                )
            )
        return results

    def predict_path(self, path: str | Path) -> DetectionResult:
        path = Path(path)
        with Image.open(path) as image:
            return self.predict_image(image, path=path)

    def predict_many(
        self,
        paths: Iterable[str | Path],
        batch_size: int = 16,
    ) -> list[DetectionResult]:
        results: list[DetectionResult] = []
        batch_images: list[Image.Image] = []
        batch_paths: list[Path] = []
        for path_like in paths:
            path = Path(path_like)
            with Image.open(path) as image:
                batch_images.append(_normalise_image(image))
            batch_paths.append(path)
            if len(batch_images) >= batch_size:
                results.extend(self.predict_images(batch_images, paths=batch_paths))
                batch_images = []
                batch_paths = []
        if batch_images:
            results.extend(self.predict_images(batch_images, paths=batch_paths))
        return results

    def _fake_label_indices(self) -> list[int]:
        fake_tokens = ("fake", "ai", "aigc", "synthetic", "generated")
        real_tokens = ("real", "human", "authentic", "natural")
        fake_indices: list[int] = []
        for index, label in self.id2label.items():
            label_l = str(label).lower()
            is_fake = any(token in label_l for token in fake_tokens)
            is_real = any(token in label_l for token in real_tokens)
            if is_fake and not is_real:
                fake_indices.append(int(index))
        return fake_indices


def create_detector(
    backend: str = DEFAULT_BACKEND,
    *,
    device: str | None = None,
    threshold: float = 0.5,
    weight_path: str | Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: str = DEFAULT_PRETRAINED,
    hf_model: str = DEFAULT_HF_MODEL_ID,
) -> Detector:
    backend = backend.lower()
    if backend == "univfd":
        return AIImageDetector(
            device=device,
            threshold=threshold,
            model_name=model_name,
            pretrained=pretrained,
            weight_path=weight_path,
        )
    if backend in {"hf", "huggingface"}:
        return HuggingFaceImageDetector(model_id=hf_model, device=device, threshold=threshold)
    raise ValueError(f"Unsupported backend: {backend!r}. Choose 'univfd' or 'hf'.")


def iter_images(path: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_EXTENSIONS else []
    if not root.exists():
        return []
    globber = root.rglob if recursive else root.glob
    return sorted(p for p in globber("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
