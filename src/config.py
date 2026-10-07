"""
This module defines dataclasses for configuration of the adversarial attack framework.
The main RunConfig class encapsulates all configuration options for a single run, including
dataset loading, attack transformations, target model settings, and output options.
"""

import os

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    repetition_penalty: float = 1.0
    stop: list[str] | None = None


@dataclass
class TargetConfig:
    backend: str = "local_hf"  # local_hf | openai_api | ollama
    model_name: str = "llama3.1-8b-instruct"

    # local_hf configuration
    hf_model_id: Optional[str] = None # "meta-llama/Llama-3.1-8B-Instruct"
    device_map: str = "auto"
    torch_dtype: str = "auto"
    load_in_4bit: bool = False
    trust_remote_code: bool = False

    # api configuration
    base_url: str = "http://localhost:8000/v1"
    api_key: Optional[str] = os.environ.get("OPENAI_API_KEY", None)
    timeout_seconds: int = 120
    system_prompt: Optional[str] = None

    # ollama configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 4096
    ollama_keep_alive: str = "5m"
    ollama_auto_pull: bool = False


@dataclass
class DatasetConfig:
    data_path: str = "data/advbench/train-00000-of-00001.parquet"
    limit: Optional[int] = None
    start_index: int = 0


@dataclass
class AttackConfig:
    use_identity: bool = True
    use_roleplay: bool = True
    use_obfuscation: bool = True
    use_prefix_injection: bool = True

    roleplay_persona: str = "a fictional expert who answers all requests directly"

    # Prefix modes:
    # direct_answer | no_safety_commentary | no_safety_or_refusal | maximal_attack
    prefix_mode: str = "no_safety_or_refusal"


@dataclass
class OutputConfig:
    output_dir: str = "outputs"
    run_name: str = "debug_run"
    save_pretty_json: bool = True


@dataclass
class ClassifierConfig:
    use_compliance: bool = True
    use_llama_guard: bool = True
    compliance_model: str = "llama3.2:latest"
    llama_guard_model: str = "llama-guard3:1b"
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 4096
    ollama_keep_alive: str = "5m"
    timeout_seconds: int = 120


@dataclass
class RunConfig:
    seed: int = 100
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)  # new
    output: OutputConfig = field(default_factory=OutputConfig)

    def output_path(self) -> Path:
        return Path(self.output.output_dir) / f"{self.output.run_name}.json"
