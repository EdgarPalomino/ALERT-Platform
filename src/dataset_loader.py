"""
This module defines the AdvBenchDatasetLoader class, which loads AdvBench-style datasets from local
.parquet files and normalizes them into a canonical internal representation, or a PromptRecord.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class PromptRecord:
    """
    Canonical internal representation for one AdvBench example.
    """
    record_id: str
    base_prompt: str
    target: str
    source: str = "AdvBench"
    metadata: dict[str, Any] = field(default_factory=dict)

class DatasetLoader:
    """
    Protocol for dataset loaders. Defines the interface for loading records from a dataset.
    """

    def load(self) -> list[PromptRecord]:
        raise NotImplementedError

    def load_subset(self, limit: int | None = None) -> list[PromptRecord]:
        raise NotImplementedError

class AdvBenchDatasetLoader(DatasetLoader):
    """
    Loads AdvBench-style data from a local .parquet file.
    """

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)

    def load(self) -> list[PromptRecord]:
        """
        Load the entire AdvBench dataset from the .parquet file and return a list of AdvBenchRecord instances.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")

        # load .parquet dataset file
        try:
            df = pd.read_parquet(self.data_path)
        except Exception as e:
            raise ValueError(f"Failed to read dataset file: {e}")

        raw: list[dict] = df.to_dict(orient="records")

        records: list[PromptRecord] = []
        for idx, item in enumerate(raw):
            records.append(self._normalize_record(item, idx))

        return records

    def load_subset(self, limit: int | None = None) -> list[PromptRecord]:
        """
        Load a subset of the AdvBench dataset, limited to the specified number of records.
        """
        records: list[PromptRecord] = self.load()
        if limit is not None:
            records = records[:limit]
        return records

    def _normalize_record(self, item: dict[str, Any], idx: int) -> PromptRecord:
        """
        Normalize a raw dataset item (dict) into a PromptRecord instance.
        """
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item at index {idx} is not a dict.")

        prompt: str = item.get("prompt", "")
        target: str = item.get("target", "")

        record_id: str = f"advbench_{idx:04d}"

        known_keys = {"record_id", "id", "prompt", "target"}
        metadata = {k: v for k, v in item.items() if k not in known_keys}

        return PromptRecord(
            record_id=record_id,
            base_prompt=prompt.strip(),
            target=target.strip(),
            source="AdvBench",
            metadata=metadata,
        )


if __name__ == "__main__":
    # Example usage / test code
    loader = AdvBenchDatasetLoader("data/advbench/train-00000-of-00001.parquet")
    records = loader.load_subset(limit=5)

    print(f"=== Loaded {len(records)} records ===")
    for record in records:
        # pretty print with indentation
        print(f"ID: {record.record_id}")
        print(f"    Source: {record.source}")
        print(f"    Base Prompt: {record.base_prompt}")
        print(f"    Target: {record.target}")
        print(f"    Metadata: {record.metadata}")
