"""
LlamaGuard classifier using llama-guard3 via Ollama.
Parses Llama Guard's native output format:
  safe
  OR
  unsafe
  <violation_category>
"""

from __future__ import annotations

import time
import requests

from classifiers.base import BaseClassifier, ClassifierResult


class LlamaGuardClassifier(BaseClassifier):
    def __init__(
        self,
        model_name: str = "llama-guard3:1b",
        base_url: str = "http://localhost:11434",
        num_ctx: int = 4096,
        keep_alive: str = "5m",
        timeout_seconds: int = 120,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.timeout_seconds = timeout_seconds

    def classify(self, prompt: str, response: str) -> ClassifierResult:
        start = time.perf_counter()

        # Llama Guard expects the conversation in its native format
        content = f"[INST] User: {prompt}\nAgent: {response} [/INST]"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"num_ctx": self.num_ctx, "temperature": 0.0},
        }

        raw_text = self._call(payload)
        label, category = self._parse(raw_text)
        latency = time.perf_counter() - start

        return ClassifierResult(
            classifier="llama_guard",
            label=label,               # "safe" | "unsafe"
            raw_response=raw_text,
            latency_seconds=latency,
            metadata={
                "model": self.model_name,
                "violation_category": category,   # e.g. "S1" or None if safe
            },
        )

    def _call(self, payload: dict) -> str:
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _parse(self, raw: str) -> tuple[str, str | None]:
        lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
        if not lines:
            return "unknown", None
        verdict = lines[0].lower()
        if verdict == "safe":
            return "safe", None
        if verdict == "unsafe":
            category = lines[1] if len(lines) > 1 else None
            return "unsafe", category
        return "unknown", None
