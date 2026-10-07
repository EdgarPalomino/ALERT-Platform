"""
This module defines the LocalHFAdapter, a TargetAdapter implementation that runs
Hugging Face Transformers models locally. It loads a specified model and tokenizer, and implements
the generate() method to produce responses from the local model, while measuring latency and token
counts.
This allows the rest of the codebase to interact with local Hugging Face models through a
consistent interface defined by the TargetAdapter protocol.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from target_adapters.base import GenerationConfig, TargetAdapter, TargetResponse


class LocalHFAdapter(TargetAdapter):
    def __init__(
        self,
        model_name: str,
        hf_model_id: str,
        generation_config: GenerationConfig | None = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
    ):
        super().__init__(model_name=model_name, generation_config=generation_config)
        self.hf_model_id: str = hf_model_id
        self.device_map: str = device_map
        self.torch_dtype: str = torch_dtype
        self.load_in_4bit: bool = load_in_4bit
        self.trust_remote_code: bool = trust_remote_code

        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            hf_model_id,
            trust_remote_code=trust_remote_code,
        )

        model_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }

        if torch_dtype != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)

        # Optional bitsandbytes path
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            **model_kwargs,
        )

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str) -> TargetResponse:
        """
        Generate a response from the local Hugging Face model for the given prompt, and return a
        TargetResponse containing the generated text, latency, token counts, and metadata.
        """
        start: float = time.perf_counter()

        inputs: dict[str, torch.Tensor] = self.tokenizer(prompt, return_tensors="pt")
        inputs: dict[str, torch.Tensor] = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_cfg: GenerationConfig = self.generation_config
        with torch.no_grad():
            output_ids: torch.Tensor = self.model.generate(
                **inputs,
                max_new_tokens=gen_cfg.max_new_tokens,
                temperature=gen_cfg.temperature,
                top_p=gen_cfg.top_p,
                do_sample=gen_cfg.do_sample,
                repetition_penalty=gen_cfg.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens: torch.Tensor = output_ids[0][inputs["input_ids"].shape[1]:]
        text: str = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        latency: float = time.perf_counter() - start

        return TargetResponse(
            text=text.strip(),
            model_name=self.model_name,
            backend_type="local_hf",
            latency_seconds=latency,
            prompt_tokens=int(inputs["input_ids"].shape[1]),
            completion_tokens=int(new_tokens.shape[0]),
            metadata={
                "hf_model_id": self.hf_model_id,
                "device_map": self.device_map,
                "load_in_4bit": self.load_in_4bit,
            },
        )

if __name__ == "__main__":
    # Test code
    adapter = LocalHFAdapter(
        model_name="test-local-hf",
        hf_model_id="meta-llama/Llama-3.1-8B-Instruct",
        device_map="auto",
        torch_dtype="auto",
        load_in_4bit=True,
        trust_remote_code=False,
    )
    response = adapter.generate("Hello, world!")

    # print each component of response separately
    print("=== Response ===")
    print("Model Name:", response.model_name)
    print("Backend Type:", response.backend_type)
    print("Latency (seconds):", response.latency_seconds)
    print("Prompt Tokens:", response.prompt_tokens)
    print("Completion Tokens:", response.completion_tokens)
    print("Metadata:", response.metadata)
    print()
    print("Generated Text:\n", response.text)
    print()
