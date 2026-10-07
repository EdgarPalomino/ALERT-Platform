"""
This module defines the TargetAdapter protocol, which specifies a common interface for generating
responses from different language model backends.
Concrete implementations of TargetAdapter can wrap various LLM APIs or local models, allowing
the rest of the codebase to interact with them in a consistent way through the generate() method.
"""

from __future__ import annotations

import sys
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from config import GenerationConfig

@dataclass
class TargetResponse:
    text: str
    model_name: str
    backend_type: str
    latency_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TargetAdapter(ABC):
    def __init__(self, model_name: str, generation_config: GenerationConfig | None = None):
        self.model_name = model_name
        self.generation_config = generation_config or GenerationConfig()

    @abstractmethod
    def generate(self, prompt: str) -> TargetResponse:
        raise NotImplementedError
