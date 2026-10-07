"""
This module defines the OllamaAdapter, a TargetAdapter implementation that runs models via a
local Ollama server. It sends requests to Ollama's native /api/chat endpoint and implements
the generate() method to produce responses, while measuring latency and token counts.
"""

from __future__ import annotations

import sys
import time
import requests
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from target_adapters.base import GenerationConfig, TargetAdapter, TargetResponse


class OllamaAdapter(TargetAdapter):
    def __init__(
        self,
        model_name: str,
        ollama_base_url: str = "http://localhost:11434",
        generation_config: GenerationConfig | None = None,
        ollama_num_ctx: int = 4096,
        ollama_keep_alive: str = "5m",
        ollama_auto_pull: bool = False,
        system_prompt: str | None = None,
    ):
        super().__init__(model_name=model_name, generation_config=generation_config)
        self.base_url = ollama_base_url.rstrip("/")
        self.num_ctx = ollama_num_ctx
        self.keep_alive = ollama_keep_alive
        self.system_prompt = system_prompt

        if ollama_auto_pull:
            self._pull_model()

    def _pull_model(self) -> None:
        response = requests.post(
            f"{self.base_url}/api/pull",
            json={"name": self.model_name},
            timeout=300,
        )
        response.raise_for_status()

    def generate(self, prompt: str) -> TargetResponse:
        start = time.perf_counter()

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        cfg = self.generation_config
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "num_predict": cfg.max_new_tokens,
                "repeat_penalty": cfg.repetition_penalty,
            },
        }
        if cfg.stop:
            payload["options"]["stop"] = cfg.stop

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        latency = time.perf_counter() - start
        text = data["message"]["content"] or data["message"]["thinking"]

        return TargetResponse(
            text=text.strip(),
            model_name=self.model_name,
            backend_type="ollama",
            latency_seconds=latency,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            metadata={
                "num_ctx": self.num_ctx,
                "keep_alive": self.keep_alive,
                "total_duration_ns": data.get("total_duration"),
            },
        )


if __name__ == "__main__":
    adapter = OllamaAdapter(
        model_name="llama3.1:8b",
        ollama_base_url="http://localhost:11434",
        generation_config=GenerationConfig(max_new_tokens=256, temperature=0.7),
        ollama_num_ctx=4096,
        ollama_auto_pull=False,
    )
    response = adapter.generate("What is the capital of France?")

    print("=== Response ===")
    print("Model Name:", response.model_name)
    print("Backend Type:", response.backend_type)
    print("Latency (seconds):", response.latency_seconds)
    print("Prompt Tokens:", response.prompt_tokens)
    print("Completion Tokens:", response.completion_tokens)
    print("Metadata:", response.metadata)
    print()
    print("Generated Text:\n", response.text)
