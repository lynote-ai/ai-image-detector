from pathlib import Path

APP_NAME = "ai-image-detector"
MODEL_REPO_ID = "siddharthksah/deepsafe-weights"
FC_WEIGHT_PATH_IN_REPO = "universalfakedetect/fc_weights.pth"
DEFAULT_MODEL_NAME = "ViT-L-14"
DEFAULT_PRETRAINED = "openai"
DEFAULT_BACKEND = "univfd"
DEFAULT_HF_MODEL_ID = "capcheck/ai-image-detection"
DEFAULT_HYBRID_UNIVFD_WEIGHT = 0.25
DEFAULT_HYBRID_PLUS_PRIMARY_WEIGHT = 0.85
NONESCAPE_MINI_URL = "https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-mini-v0.safetensors"
NONESCAPE_FULL_URL = "https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-v0.safetensors"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def cache_dir() -> Path:
    return Path.home() / ".cache" / APP_NAME
