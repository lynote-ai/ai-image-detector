from .model import AIImageDetector, HuggingFaceImageDetector, create_detector
from .types import DetectionResult, EvaluationMetrics, EvaluationReport

__version__ = "0.2.0"

__all__ = [
    "AIImageDetector",
    "DetectionResult",
    "EvaluationMetrics",
    "EvaluationReport",
    "HuggingFaceImageDetector",
    "create_detector",
]
