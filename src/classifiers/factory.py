from __future__ import annotations

from classifiers.base import BaseClassifier
from classifiers.compliance_classifier import ComplianceClassifier
from classifiers.llama_guard_classifier import LlamaGuardClassifier


def build_compliance_classifier(config: dict) -> ComplianceClassifier:
    return ComplianceClassifier(
        model_name=config.get("compliance_model", "llama3.2:latest"),
        base_url=config.get("ollama_base_url", "http://localhost:11434"),
        num_ctx=config.get("ollama_num_ctx", 4096),
        keep_alive=config.get("ollama_keep_alive", "5m"),
        timeout_seconds=config.get("timeout_seconds", 120),
    )


def build_llama_guard_classifier(config: dict) -> LlamaGuardClassifier:
    return LlamaGuardClassifier(
        model_name=config.get("llama_guard_model", "llama-guard3:1b"),
        base_url=config.get("ollama_base_url", "http://localhost:11434"),
        num_ctx=config.get("ollama_num_ctx", 4096),
        keep_alive=config.get("ollama_keep_alive", "5m"),
        timeout_seconds=config.get("timeout_seconds", 120),
    )
