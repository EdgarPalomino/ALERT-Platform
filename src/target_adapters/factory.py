"""
This module defines a factory function for building TargetAdapter instances based on a provided
configuration dictionary. It supports multiple backends, including local Hugging Face models,
OpenAI-compatible APIs, and process-based streaming.

The MODEL_REGISTRY provides metadata about known / available models and their recommended backends.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from target_adapters.base import GenerationConfig, TargetAdapter
from target_adapters.local_hf import LocalHFAdapter
from target_adapters.openai_compatible_api import OpenAICompatibleAPIAdapter
from target_adapters.ollama import OllamaAdapter


MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "llama3.2-3b": {
        "hf_model_id": "meta-llama/Llama-3.2-3B",
        "ollama_model_id": "llama3.2:latest",
        "recommended_backend": "ollama",
    },
    "deepseek-r1-1.5b": {
        "hf_model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "ollama_model_id": "deepseek-r1:1.5b",
        "recommended_backend": "ollama",
    },
    "gpt-4o": {
        "hf_model_id": None,
        "ollama_model_id": None,
        "recommended_backend": "openai_api",
    },
    "gpt-4o-mini": {
        "hf_model_id": None,
        "ollama_model_id": None,
        "recommended_backend": "openai_api",
    },
    "llama3.1-8b-instruct": {
        "hf_model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "ollama_model_id": "llama3.1:8b",
        "recommended_backend": "local_hf",
    },
    "llama3.3-70b-instruct": {
        "hf_model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "ollama_model_id": "llama3.3:70b",
        "recommended_backend": "openai_api",
    },
    "mistral-7b-instruct": {
        "hf_model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "ollama_model_id": "mistral:7b",
        "recommended_backend": "local_hf",
    },
    "mixtral-8x7b-instruct": {
        "hf_model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "ollama_model_id": "mixtral:8x7b",
        "recommended_backend": "openai_api",
    },
    "qwen2.5-7b-instruct": {
        "hf_model_id": "Qwen/Qwen2.5-7B-Instruct",
        "ollama_model_id": "qwen2.5:7b",
        "recommended_backend": "local_hf",
    },
    "gemma2-9b-it": {
        "hf_model_id": "google/gemma-2-9b-it",
        "ollama_model_id": "gemma2:9b",
        "recommended_backend": "local_hf",
    },
    "deepseek-v3": {
        "hf_model_id": "deepseek-ai/DeepSeek-V3",
        "ollama_model_id": "deepseek-v3:latest",
        "recommended_backend": "openai_api",
    },
}


def build_target_adapter(config: dict) -> TargetAdapter:
    backend: str = config["backend"]
    model_name: str = config["model_name"]
    gen_cfg: GenerationConfig = GenerationConfig(**config.get("generation_config", {}))

    if backend == "local_hf":
        return LocalHFAdapter(
            model_name=model_name,
            hf_model_id=config["hf_model_id"],
            generation_config=gen_cfg,
            device_map=config.get("device_map", "auto"),
            torch_dtype=config.get("torch_dtype", "auto"),
            load_in_4bit=config.get("load_in_4bit", False),
            trust_remote_code=config.get("trust_remote_code", False),
        )

    if backend == "openai_api":
        return OpenAICompatibleAPIAdapter(
            model_name=model_name,
            base_url=config["base_url"],
            api_key=config.get("api_key"),
            generation_config=gen_cfg,
            timeout_seconds=config.get("timeout_seconds", 120),
            system_prompt=config.get("system_prompt"),
        )
    
    if backend == "ollama":

        registry_entry = MODEL_REGISTRY.get(model_name, {})
        ollama_model_id = registry_entry.get("ollama_model_id", model_name)  # fall back to raw name if not in registry

        return OllamaAdapter(
            model_name=ollama_model_id,
            ollama_base_url=config.get("ollama_base_url", "http://localhost:11434"),
            generation_config=gen_cfg,
            ollama_num_ctx=config.get("ollama_num_ctx", 4096),
            ollama_keep_alive=config.get("ollama_keep_alive", "5m"),
            ollama_auto_pull=config.get("ollama_auto_pull", False),
            system_prompt=config.get("system_prompt"),
        )

    raise ValueError(f"Unsupported backend: {backend}")

if __name__ == "__main__":
    # Example usage
    gen_cfg: dict[str, int | float | bool] = {
        "max_new_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.2,
    }

    adapter = build_target_adapter({
        "backend": "local_hf",
        "model_name": "test-local-hf",
        "hf_model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "generation_config": gen_cfg,
        "device_map": "cpu",
        "torch_dtype": "auto",
        "load_in_4bit": True,
        "trust_remote_code": False,
    })

    response = adapter.generate("What is the capital of France?")
    print(response)
