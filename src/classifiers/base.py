from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClassifierResult:
    classifier: str          # "compliance" | "llama_guard"
    label: str               # classifier-specific label string
    raw_response: str        # raw text back from the model
    latency_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseClassifier(ABC):
    @abstractmethod
    def classify(self, prompt: str, response: str) -> ClassifierResult:
        raise NotImplementedError
