"""Tests for benchmark runner output formatting and report generation."""

from __future__ import annotations

import json

from rich.table import Table

from agent_memory_bench.models import BenchmarkReport, SampleResult, TaskResult, TaskType
from agent_memory_bench.runner import format_leaderboard, print_report, save_report


def _make_report() -> BenchmarkReport:
    """Create a known BenchmarkReport for testing."""
    return BenchmarkReport(
        task_results=[
            TaskResult(
                task_type=TaskType.FACT_RECALL,
                adapter_name="adapter_a",
                sample_results=[
                    SampleResult(
                        sample_id="s1",
                        retrieved=[],
                        recall_at_k=0.9,
                        mrr=0.8,
                        latency_ms=10.0,
                        store_latency_ms=5.0,
                    ),
                ],
                avg_recall_at_k=0.9,
                avg_mrr=0.8,
                avg_latency_ms=10.0,
                avg_store_latency_ms=5.0,
                num_samples=1,
                num_successes=1,
            ),
            TaskResult(
                task_type=TaskType.TEMPORAL_REASONING,
                adapter_name="adapter_b",
                sample_results=[
                    SampleResult(
                        sample_id="s2",
                        retrieved=[],
                        recall_at_k=0.7,
                        mrr=0.6,
                        latency_ms=20.0,
                        store_latency_ms=8.0,
                    ),
                ],
                avg_recall_at_k=0.7,
                avg_mrr=0.6,
                avg_latency_ms=20.0,
                avg_store_latency_ms=8.0,
                num_samples=1,
                num_successes=1,
            ),
        ],
    )


class TestFormatLeaderboard:
    def test_returns_rich_table(self):
        report = _make_report()
        table = format_leaderboard(report)
        assert isinstance(table, Table)

    def test_table_has_correct_title(self):
        report = _make_report()
        table = format_leaderboard(report)
        assert table.title == "Agent Memory Benchmark Leaderboard"

    def test_table_has_expected_columns(self):
        report = _make_report()
        table = format_leaderboard(report)
        column_names = [col.header for col in table.columns]
        assert "Adapter" in column_names
        assert "Task" in column_names
        assert "Recall@5" in column_names
        assert "MRR" in column_names
        assert "Latency (ms)" in column_names

    def test_table_has_rows(self):
        report = _make_report()
        table = format_leaderboard(report)
        assert table.row_count == 2

    def test_empty_report_returns_empty_table(self):
        report = BenchmarkReport()
        table = format_leaderboard(report)
        assert table.row_count == 0


class TestSaveReport:
    def test_saves_json_file(self, tmp_path):
        report = _make_report()
        output_path = tmp_path / "output" / "report.json"
        save_report(report, output_path)
        assert output_path.exists()

        data = json.loads(output_path.read_text())
        assert "timestamp" in data
        assert "leaderboard" in data
        assert len(data["leaderboard"]) == 2

    def test_creates_parent_directories(self, tmp_path):
        report = _make_report()
        output_path = tmp_path / "deep" / "nested" / "dir" / "report.json"
        save_report(report, output_path)
        assert output_path.exists()

    def test_leaderboard_contains_expected_fields(self, tmp_path):
        report = _make_report()
        output_path = tmp_path / "report.json"
        save_report(report, output_path)

        data = json.loads(output_path.read_text())
        row = data["leaderboard"][0]
        assert "adapter" in row
        assert "task" in row
        assert "recall@5" in row
        assert "mrr" in row
        assert "latency_ms" in row


class TestPrintReport:
    def test_does_not_raise(self, capsys):
        report = _make_report()
        print_report(report)
        # Should not raise; output goes to Rich console

    def test_empty_report_does_not_raise(self, capsys):
        report = BenchmarkReport()
        print_report(report)
