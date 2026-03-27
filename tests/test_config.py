"""Tests for configuration loading."""

from __future__ import annotations

import json

import pytest

from agent_memory_bench.config import BenchmarkConfig


class TestBenchmarkConfig:
    def test_from_file_valid(self, tmp_path):
        config_data = {
            "tasks": ["fact-recall", "temporal-reasoning"],
            "adapters": [
                {"name": "mem0", "adapter_class": "mem0", "params": {}},
            ],
            "num_samples": 50,
            "seed": 123,
            "output_dir": "./my-results",
        }
        config_file = tmp_path / "bench.json"
        config_file.write_text(json.dumps(config_data))

        cfg = BenchmarkConfig.from_file(config_file)
        assert len(cfg.tasks) == 2
        assert cfg.tasks[0].value == "fact-recall"
        assert cfg.tasks[1].value == "temporal-reasoning"
        assert len(cfg.adapters) == 1
        assert cfg.adapters[0].name == "mem0"
        assert cfg.num_samples == 50
        assert cfg.seed == 123

    def test_from_file_defaults(self, tmp_path):
        config_file = tmp_path / "minimal.json"
        config_file.write_text("{}")

        cfg = BenchmarkConfig.from_file(config_file)
        assert cfg.num_samples == 20
        assert cfg.seed == 42
        assert len(cfg.tasks) == 7  # all task types

    def test_from_file_invalid_json(self, tmp_path):
        config_file = tmp_path / "bad.json"
        config_file.write_text("not valid json {{{")

        with pytest.raises(json.JSONDecodeError):
            BenchmarkConfig.from_file(config_file)

    def test_from_file_invalid_task(self, tmp_path):
        config_data = {"tasks": ["nonexistent-task"]}
        config_file = tmp_path / "bad_task.json"
        config_file.write_text(json.dumps(config_data))

        with pytest.raises(ValueError):
            BenchmarkConfig.from_file(config_file)

    def test_adapter_config_with_params(self, tmp_path):
        config_data = {
            "adapters": [
                {
                    "name": "custom-mem0",
                    "adapter_class": "mem0",
                    "params": {"api_key": "test-key"},
                },
            ],
        }
        config_file = tmp_path / "adapters.json"
        config_file.write_text(json.dumps(config_data))

        cfg = BenchmarkConfig.from_file(config_file)
        assert cfg.adapters[0].params == {"api_key": "test-key"}

    def test_default_config_has_all_task_types(self):
        cfg = BenchmarkConfig()
        assert len(cfg.tasks) == 7
        task_values = {t.value for t in cfg.tasks}
        assert "fact-recall" in task_values
        assert "temporal-reasoning" in task_values
        assert "context-window-efficiency" in task_values
