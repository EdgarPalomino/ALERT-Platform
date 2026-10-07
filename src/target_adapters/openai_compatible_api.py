"""
This module defines a TargetAdapter implementation for OpenAI-compatible APIs. It sends generation
requests to a specified API endpoint, including support for system prompts and configurable
generation parameters.

The generate() method measures latency and returns a TargetResponse with the generated text and
relevant metadata, allowing the rest of the codebase to interact with OpenAI-compatible APIs
through a consistent interface defined by the TargetAdapter protocol.
"""

from __future__ import annotations

import sys
import time
import requests
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from target_adapters.base import GenerationConfig, TargetAdapter, TargetResponse


class OpenAICompatibleAPIAdapter(TargetAdapter):
    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str | None = None,
        generation_config: GenerationConfig | None = None,
        timeout_seconds: int = 120,
        system_prompt: str | None = None,
    ):
        super().__init__(model_name=model_name, generation_config=generation_config)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.system_prompt = system_prompt

    def generate(self, prompt: str) -> TargetResponse:
        start = time.perf_counter()

        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        cfg = self.generation_config
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()

        latency = time.perf_counter() - start
        choice = data["choices"][0]["message"]["content"]

        usage = data.get("usage", {})

        return TargetResponse(
            text=choice.strip(),
            model_name=self.model_name,
            backend_type="openai_compatible_api",
            latency_seconds=latency,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            metadata={"raw_usage": usage},
        )

if __name__ == "__main__":
    # Example usage
    adapter = OpenAICompatibleAPIAdapter(
        model_name="gpt-3.5-turbo",
        base_url="http://localhost:8000/v1",
        api_key=None,
        generation_config=GenerationConfig(max_new_tokens=50, temperature=0.5),
        system_prompt="You are a helpful assistant.",
    )

    response = adapter.generate("What is the capital of France?")
    print(response)
