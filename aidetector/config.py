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
DEFAULT_ULTRA_PRIMARY_WEIGHT = 0.2
NONESCAPE_MINI_URL = "https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-mini-v0.safetensors"
NONESCAPE_FULL_URL = "https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-v0.safetensors"
SENTRY_REPO_ID = "InfImagine/Sentry_image_models"
SENTRY_CONVNEXT_SMALL_DIR = "convnext_small_4xb256_fake5m-lr4e-4"
SENTRY_CONVNEXT_SMALL_FILE = f"{SENTRY_CONVNEXT_SMALL_DIR}/epoch_15.pth"
SENTRY_CONVNEXT_SMALL_CONFIG = f"{SENTRY_CONVNEXT_SMALL_DIR}/convnext_small_4xb256_fake5m-lr4e-4.py"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def cache_dir() -> Path:
    return Path.home() / ".cache" / APP_NAME
