from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DetectionResult:
    path: Path | None
    label: str
    probability_ai: float
    probability_real: float
    confidence: float
    raw_score: float
    backend: str = "univfd"

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path) if self.path else None,
            "label": self.label,
            "probability_ai": round(self.probability_ai, 6),
            "probability_real": round(self.probability_real, 6),
            "confidence": round(self.confidence, 6),
            "raw_score": round(self.raw_score, 6),
            "backend": self.backend,
        }


@dataclass(frozen=True)
class EvaluationMetrics:
    dataset: str
    n_samples: int
    n_real: int
    n_ai: int
    threshold: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    seconds: float
    images_per_second: float
    best_threshold: float | None = None
    best_accuracy: float | None = None
    best_balanced_accuracy: float | None = None
    best_f1: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "n_samples": self.n_samples,
            "n_real": self.n_real,
            "n_ai": self.n_ai,
            "threshold": self.threshold,
            "accuracy": round(self.accuracy, 6),
            "balanced_accuracy": round(self.balanced_accuracy, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "roc_auc": round(self.roc_auc, 6) if self.roc_auc is not None else None,
            "confusion": {
                "true_positive": self.true_positive,
                "true_negative": self.true_negative,
                "false_positive": self.false_positive,
                "false_negative": self.false_negative,
            },
            "seconds": round(self.seconds, 6),
            "images_per_second": round(self.images_per_second, 6),
            "threshold_sweep": {
                "best_threshold": round(self.best_threshold, 6)
                if self.best_threshold is not None
                else None,
                "best_accuracy": round(self.best_accuracy, 6)
                if self.best_accuracy is not None
                else None,
                "best_balanced_accuracy": round(self.best_balanced_accuracy, 6)
                if self.best_balanced_accuracy is not None
                else None,
                "best_f1": round(self.best_f1, 6) if self.best_f1 is not None else None,
                "note": "Diagnostic only: this threshold is selected on the evaluated sample.",
            },
        }


@dataclass(frozen=True)
class EvaluationReport:
    metrics: EvaluationMetrics
    model: dict[str, Any]
    dataset: dict[str, Any]
    predictions: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.as_dict(),
            "model": self.model,
            "dataset": self.dataset,
            "predictions": self.predictions,
        }
