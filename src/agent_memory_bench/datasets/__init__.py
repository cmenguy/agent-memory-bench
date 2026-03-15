"""Dataset loading for benchmark tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class FactRecallEntry(BaseModel):
    """Schema for a fact recall dataset entry."""

    fact: str
    question: str
    answer: str


class TemporalEntry(BaseModel):
    """Schema for a temporal reasoning dataset entry."""

    old_statement: str
    new_statement: str
    question: str
    answer: str


class ContradictionEntry(BaseModel):
    """Schema for a contradiction detection dataset entry."""

    old_fact: str
    new_fact: str
    question: str
    answer: str


DATASETS_DIR = Path(__file__).parent


def load_dataset(filename: str, model: type[BaseModel], path: Path | None = None) -> list[Any]:
    """Load and validate a dataset from a JSON file.

    Args:
        filename: Default filename (e.g., "fact_recall.json").
        model: Pydantic model to validate each entry.
        path: Optional path to a custom dataset file. If None, uses the
              bundled default dataset.

    Returns:
        List of validated Pydantic model instances.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If entries fail validation.
    """
    file_path = Path(path) if path is not None else DATASETS_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    data = json.loads(file_path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Dataset must be a JSON array, got {type(data).__name__}")

    return [model(**entry) for entry in data]
