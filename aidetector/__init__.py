from .model import AIImageDetector, HuggingFaceImageDetector, HybridImageDetector, create_detector
from .types import DetectionResult, EvaluationMetrics, EvaluationReport

__version__ = "0.3.0"

__all__ = [
    "AIImageDetector",
    "DetectionResult",
    "EvaluationMetrics",
    "EvaluationReport",
    "HuggingFaceImageDetector",
    "HybridImageDetector",
    "create_detector",
]
