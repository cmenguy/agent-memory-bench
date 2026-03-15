"""Tests for dataset loading."""

from __future__ import annotations

import json

import pytest

from agent_memory_bench.core import (
    generate_contradiction_detection_task,
    generate_fact_recall_task,
    generate_temporal_reasoning_task,
)
from agent_memory_bench.datasets import (
    ContradictionEntry,
    FactRecallEntry,
    TemporalEntry,
    load_dataset,
)


class TestLoadDataset:
    def test_load_default_fact_recall(self):
        entries = load_dataset("fact_recall.json", FactRecallEntry)
        assert len(entries) == 20
        assert entries[0].fact == "The capital of France is Paris."

    def test_load_default_temporal(self):
        entries = load_dataset("temporal.json", TemporalEntry)
        assert len(entries) == 10

    def test_load_default_contradictions(self):
        entries = load_dataset("contradictions.json", ContradictionEntry)
        assert len(entries) == 10

    def test_load_custom_file(self, tmp_path):
        data = [
            {"fact": "Custom fact.", "question": "Custom Q?", "answer": "Custom A"},
        ]
        custom_file = tmp_path / "custom.json"
        custom_file.write_text(json.dumps(data))
        entries = load_dataset("unused.json", FactRecallEntry, path=custom_file)
        assert len(entries) == 1
        assert entries[0].fact == "Custom fact."

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent.json", FactRecallEntry, path="/tmp/no_such_file.json")

    def test_load_invalid_json(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json")
        with pytest.raises(json.JSONDecodeError):
            load_dataset("bad.json", FactRecallEntry, path=bad_file)

    def test_load_non_array(self, tmp_path):
        bad_file = tmp_path / "obj.json"
        bad_file.write_text('{"key": "value"}')
        with pytest.raises(ValueError, match="JSON array"):
            load_dataset("obj.json", FactRecallEntry, path=bad_file)

    def test_load_invalid_schema(self, tmp_path):
        data = [{"wrong_field": "value"}]
        bad_file = tmp_path / "schema.json"
        bad_file.write_text(json.dumps(data))
        with pytest.raises(Exception):  # Pydantic ValidationError
            load_dataset("schema.json", FactRecallEntry, path=bad_file)


class TestGeneratorsWithDataset:
    def test_fact_recall_from_custom_dataset(self, tmp_path):
        data = [
            {"fact": "Fact A.", "question": "Q A?", "answer": "A"},
            {"fact": "Fact B.", "question": "Q B?", "answer": "B"},
            {"fact": "Fact C.", "question": "Q C?", "answer": "C"},
        ]
        ds_file = tmp_path / "facts.json"
        ds_file.write_text(json.dumps(data))
        task = generate_fact_recall_task(num_samples=2, seed=42, dataset_path=ds_file)
        assert len(task.samples) == 2
        answers = {s.expected_answer for s in task.samples}
        assert answers.issubset({"A", "B", "C"})

    def test_temporal_from_custom_dataset(self, tmp_path):
        data = [
            {
                "old_statement": "Old val.",
                "new_statement": "New val.",
                "question": "What val?",
                "answer": "New",
            },
        ]
        ds_file = tmp_path / "temporal.json"
        ds_file.write_text(json.dumps(data))
        task = generate_temporal_reasoning_task(num_samples=1, seed=42, dataset_path=ds_file)
        assert len(task.samples) == 1
        assert task.samples[0].expected_answer == "New"

    def test_contradiction_from_custom_dataset(self, tmp_path):
        data = [
            {
                "old_fact": "Old fact.",
                "new_fact": "New fact.",
                "question": "Which fact?",
                "answer": "New",
            },
        ]
        ds_file = tmp_path / "contradictions.json"
        ds_file.write_text(json.dumps(data))
        task = generate_contradiction_detection_task(
            num_samples=1, seed=42, dataset_path=ds_file
        )
        assert len(task.samples) == 1
        assert task.samples[0].expected_answer == "New"

    def test_generators_still_work_without_dataset(self):
        """Ensure backward compatibility — generators work without dataset_path."""
        task = generate_fact_recall_task(num_samples=3, seed=42)
        assert len(task.samples) == 3
        task = generate_temporal_reasoning_task(num_samples=1, seed=42)
        assert len(task.samples) == 1
        task = generate_contradiction_detection_task(num_samples=3, seed=42)
        assert len(task.samples) == 3
