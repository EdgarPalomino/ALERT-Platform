import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
import pyrallis

from config import RunConfig
from dataset_loader import PromptRecord, DatasetLoader, AdvBenchDatasetLoader
from attack_transformers import (
    Attack,
    AttackGenerator,
    AttackTransformer,
    IdentityTransformer,
    ObfuscationTransformer,
    PrefixInjectionTransformer,
    RoleplayTransformer,
)
from target_adapters.base import TargetAdapter
from target_adapters.factory import build_target_adapter
from classifiers.base import BaseClassifier, ClassifierResult
from classifiers.factory import build_compliance_classifier, build_llama_guard_classifier


def set_seed(seed: int, backend: str = "local_hf") -> None:
    """
    Set random seeds for reproducibility and consistency across random, numpy, and torch.
    """
    random.seed(seed)

    try:
        np.random.seed(seed)
    except Exception:
        pass

    if backend == "local_hf":
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def build_attack_generator(cfg: RunConfig) -> AttackGenerator:
    """
    Build the AttackGenerator by instantiating the enabled AttackTransformers based on the config.
    The AttackGenerator will apply all enabled transformations to each input record and return the
    combined results.
    """
    transformers: list[AttackTransformer] = []

    if cfg.attack.use_identity:
        transformers.append(IdentityTransformer())

    if cfg.attack.use_roleplay:
        transformers.append(
            RoleplayTransformer(persona=cfg.attack.roleplay_persona)
        )

    if cfg.attack.use_obfuscation:
        transformers.append(ObfuscationTransformer())

    if cfg.attack.use_prefix_injection:
        transformers.append(
            PrefixInjectionTransformer(mode=cfg.attack.prefix_mode)
        )

    if not transformers:
        raise ValueError("No attack transformers enabled in config.")

    return AttackGenerator(transformers=transformers)


def load_records(cfg: RunConfig) -> list[PromptRecord]:
    """
    Load using DatasetLoader and specified dataset file, applying any limits or starting index
    or other parameters as configured.
    """
    loader: DatasetLoader = AdvBenchDatasetLoader(data_path=cfg.dataset.data_path)
    records: list[PromptRecord] = loader.load()

    start: int = cfg.dataset.start_index
    if start < 0:
        raise ValueError("dataset.start_index must be >= 0")

    if start >= len(records):
        raise ValueError(
            f"dataset.start_index={start} is out of range for dataset of size {len(records)}"
        )

    records = records[start:]

    if cfg.dataset.limit is not None:
        records = records[: cfg.dataset.limit]

    return records


def make_serializable(obj: Any) -> Any:
    """
    Recursively unsure objects are JSON-serializable.
    Handles:
    - dataclasses -> dict
    - Paths -> str
    - lists and tuples -> (recursive application to elements)
    - dicts -> (keys -> str, recursive application to values)
    - other types are returned as is (assuming natively serializable or will raise an error if not)
    """
    if hasattr(obj, "__dataclass_fields__"):
        return {k: make_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    return obj


def build_classifiers(cfg: RunConfig) -> tuple[BaseClassifier | None, BaseClassifier | None]:
    """
    Instantiate enabled classifiers from config. Returns (compliance, llama_guard),
    either of which may be None if disabled.
    """
    classifier_cfg_dict = make_serializable(cfg.classifier)

    compliance = (
        build_compliance_classifier(classifier_cfg_dict)
        if cfg.classifier.use_compliance
        else None
    )
    llama_guard = (
        build_llama_guard_classifier(classifier_cfg_dict)
        if cfg.classifier.use_llama_guard
        else None
    )
    return compliance, llama_guard


def run_classifiers(
    prompt: str,
    response_text: str,
    compliance: BaseClassifier | None,
    llama_guard: BaseClassifier | None,
) -> dict[str, Any]:
    """
    Run whichever classifiers are enabled and return a dict ready for JSON output.
    Errors are caught per-classifier so one failure doesn't abort the whole attack result.
    """
    results = {}

    if compliance is not None:
        try:
            result: ClassifierResult = compliance.classify(prompt, response_text)
            results["compliance"] = make_serializable(result)
        except Exception as e:
            results["compliance"] = {"error": {"type": type(e).__name__, "message": str(e)}}

    if llama_guard is not None:
        try:
            result: ClassifierResult = llama_guard.classify(prompt, response_text)
            results["llama_guard"] = make_serializable(result)
        except Exception as e:
            results["llama_guard"] = {"error": {"type": type(e).__name__, "message": str(e)}}

    return results


def run(cfg: RunConfig) -> dict[str, Any]:
    """
    Main pipeline function that executes the full attack generation and target model querying
    process based on the provided configuration. It returns a structured dictionary containing the
    configuration, summary statistics, and detailed results for each record and attack attempt.
    The output is designed to be JSON-serializable for easy saving and analysis.

    Steps:
    1. Setup - With configuration, set random seeds, and prepare necessary components.
    2. Dataset Initialization - Load dataset records based on the config.
    3. Attack Generation - Build the AttackGenerator with the enabled transformers.
    4. Target Adapter Initialization - Build the TargetAdapter based on the target configuration.
    5. Attack Execution - For each record, generate attacks and query the target model, recording
       results and any errors.
    6. Output Compilation - Compile the results into a structured dictionary with summary statistics.
    """
    set_seed(cfg.seed, backend=cfg.target.backend)

    records: list[PromptRecord] = load_records(cfg)
    attack_generator: AttackGenerator = build_attack_generator(cfg)

    target_cfg_dict: dict[str, Any] = make_serializable(cfg.target)
    target_adapter: TargetAdapter = build_target_adapter(target_cfg_dict)

    compliance_classifier, llama_guard_classifier = build_classifiers(cfg)

    run_start: float = time.perf_counter()

    results: list[dict[str, Any]] = []
    attacks_completed: int = 0
    total_errors: int = 0
    successful_calls: int = 0

    outer_pbar = tqdm(
        enumerate(records),
        total=len(records),
        desc=f"Processing records [{cfg.target.model_name}]",
        dynamic_ncols=True,
    )

    for record_idx, record in outer_pbar:
        attacks: list[Attack] = attack_generator.generate(record)

        record_result: dict[str, Any] = {
            "record_index": record_idx,
            "record": make_serializable(record),
            "attacks": [],
        }

        record_label = getattr(record, "record_id", f"record_{record_idx:04d}")
        prompt_preview = getattr(record, "base_prompt", "")[:50].replace("\n", " ")

        outer_pbar.set_postfix(
            record_id=record_label,
            prompt_preview=prompt_preview,
            errors=total_errors,
            attacks_completed=attacks_completed,
        )

        inner_pbar = tqdm(
            attacks,
            total=len(attacks),
            desc=f"{record_label}",
            dynamic_ncols=True,
            leave=False,
        )

        for attack in inner_pbar:
            attacks_completed += 1
            attack_start: float = time.perf_counter()
            strat_name = getattr(attack, "strategy_name", "unknown")
            inner_pbar.set_postfix(
                strategy=strat_name,
                errors=total_errors,
            )

            try:
                response = target_adapter.generate(attack.attacked_prompt)
                wall_time = time.perf_counter() - attack_start
                successful_calls += 1

                # run classifiers against the attack prompt + model response
                classifications = run_classifiers(
                    prompt=attack.attacked_prompt,
                    response_text=response.text,
                    compliance=compliance_classifier,
                    llama_guard=llama_guard_classifier,
                )

                attack_result: dict[str, Any] = {
                    "success": True,
                    "error": None,
                    "attack": make_serializable(attack),
                    "response": make_serializable(response),
                    "classifications": classifications,
                    "wall_time_seconds": wall_time,
                }

                inner_pbar.set_postfix(
                    strategy=strat_name,
                    errors=total_errors,
                    latency=f"{wall_time:.2f}s",
                )

            except Exception as e:
                wall_time = time.perf_counter() - attack_start
                total_errors += 1

                attack_result: dict[str, Any] = {
                    "success": False,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                    },
                    "attack": make_serializable(attack),
                    "response": None,
                    "classifications": {},
                    "wall_time_seconds": wall_time,
                }

                inner_pbar.set_postfix(
                    strategy=strat_name,
                    errors=total_errors,
                    latency=f"{wall_time:.2f}s",
                )

                tqdm.write(
                    f"[ERROR] Record={record_label}, Strategy={strat_name}: {type(e).__name__}: {e}"
                )

            record_result["attacks"].append(attack_result)

        results.append(record_result)

    total_runtime: float = time.perf_counter() - run_start

    output = {
        "config": make_serializable(cfg),
        "summary": {
            "num_records": len(records),
            "total_attacks": attacks_completed,
            "total_errors": total_errors,
            "successful_calls": attacks_completed - total_errors,
            "total_runtime_seconds": total_runtime,
            "average_runtime_per_attack_seconds": (
                total_runtime / attacks_completed if attacks_completed > 0 else None
            ),
        },
        "results": results,
    }

    return output


def save_output(output: dict[str, Any], cfg: RunConfig) -> Path:
    output_path: Path = cfg.output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        if cfg.output.save_pretty_json:
            json.dump(output, f, indent=2, ensure_ascii=False)
        else:
            json.dump(output, f, ensure_ascii=False)

    return output_path


@pyrallis.wrap()
def main(cfg: RunConfig) -> None:
    output: dict[str, Any] = run(cfg)
    output_path: Path = save_output(output, cfg)

    print("=" * 80)
    print("Run complete.")
    print(f"Saved JSON output to: {output_path}")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
