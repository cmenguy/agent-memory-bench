"""Tests for CLI commands."""

from __future__ import annotations

import json

from click.testing import CliRunner

from agent_memory_bench.cli import main


class TestTasksCommand:
    def test_lists_all_tasks(self):
        runner = CliRunner()
        result = runner.invoke(main, ["tasks"])
        assert result.exit_code == 0
        assert "fact-recall" in result.output
        assert "temporal-reasoning" in result.output
        assert "contradiction-detection" in result.output
        assert "cross-conversation" in result.output
        assert "multi-hop-retrieval" in result.output
        assert "memory-update" in result.output
        assert "context-window-efficiency" in result.output

    def test_shows_descriptions(self):
        runner = CliRunner()
        result = runner.invoke(main, ["tasks"])
        assert result.exit_code == 0
        # Each task should have a description line
        assert "store facts" in result.output.lower() or "retriev" in result.output.lower()


class TestAdaptersCommand:
    def test_lists_adapters(self):
        runner = CliRunner()
        result = runner.invoke(main, ["adapters"])
        assert result.exit_code == 0
        assert "mem0" in result.output
        assert "memos" in result.output

    def test_shows_install_instructions(self):
        runner = CliRunner()
        result = runner.invoke(main, ["adapters"])
        assert result.exit_code == 0
        assert "pip install" in result.output


class TestRunCommand:
    def test_no_adapter_or_task_shows_error(self):
        runner = CliRunner()
        result = runner.invoke(main, ["run"])
        assert result.exit_code != 0

    def test_unknown_adapter_shows_error(self):
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--task", "fact-recall", "--adapter", "nonexistent"])
        assert result.exit_code != 0

    def test_missing_task_shows_error(self):
        runner = CliRunner()
        # Provide adapter but no task
        result = runner.invoke(main, ["run", "--adapter", "mem0"])
        assert result.exit_code != 0


class TestReportCommand:
    def test_reads_report_file(self, tmp_path):
        report_data = {
            "timestamp": "2026-01-01T00:00:00",
            "leaderboard": [
                {
                    "adapter": "test-adapter",
                    "task": "fact-recall",
                    "recall@5": 0.9,
                    "mrr": 0.8,
                    "latency_ms": 15.0,
                },
            ],
        }
        report_file = tmp_path / "report.json"
        report_file.write_text(json.dumps(report_data))

        runner = CliRunner()
        result = runner.invoke(main, ["report", str(report_file)])
        assert result.exit_code == 0
        assert "test-adapter" in result.output
        assert "fact-recall" in result.output

    def test_nonexistent_file_shows_error(self):
        runner = CliRunner()
        result = runner.invoke(main, ["report", "/nonexistent/path.json"])
        assert result.exit_code != 0


class TestMainGroup:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Agent Memory Benchmark Suite" in result.output

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()
