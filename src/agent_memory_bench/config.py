"""Configuration management for the benchmark suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent_memory_bench.models import TaskType

DEFAULT_TASKS = list(TaskType)
DEFAULT_TOP_K = 5
DEFAULT_NUM_SAMPLES = 20


class AdapterConfig(BaseModel):
    """Configuration for a memory system adapter."""

    name: str
    adapter_class: str
    params: dict[str, Any] = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark run."""

    tasks: list[TaskType] = Field(default_factory=lambda: DEFAULT_TASKS)
    adapters: list[AdapterConfig] = Field(default_factory=list)
    top_k: int = DEFAULT_TOP_K
    num_samples: int = DEFAULT_NUM_SAMPLES
    output_dir: Path = Path("./results")
    seed: int = 42

    @classmethod
    def from_file(cls, path: Path) -> BenchmarkConfig:
        """Load configuration from a JSON file."""
        import json

        data = json.loads(path.read_text())
        return cls(**data)
