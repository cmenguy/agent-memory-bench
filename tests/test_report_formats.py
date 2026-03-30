"""Tests for Markdown and HTML report output formats."""

from __future__ import annotations

import json

from agent_memory_bench.models import BenchmarkReport, SampleResult, TaskResult, TaskType
from agent_memory_bench.runner import save_report


def _make_report() -> BenchmarkReport:
    """Create a known BenchmarkReport for testing."""
    return BenchmarkReport(
        task_results=[
            TaskResult(
                task_type=TaskType.FACT_RECALL,
                adapter_name="test-adapter",
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
        ],
    )


class TestSaveReportJSON:
    def test_produces_valid_json(self, tmp_path):
        report = _make_report()
        out = tmp_path / "report.json"
        save_report(report, out, fmt="json")
        data = json.loads(out.read_text())
        assert "timestamp" in data
        assert "leaderboard" in data
        assert len(data["leaderboard"]) == 1

    def test_default_format_is_json(self, tmp_path):
        report = _make_report()
        out = tmp_path / "report.json"
        save_report(report, out)
        data = json.loads(out.read_text())
        assert "leaderboard" in data


class TestSaveReportMarkdown:
    def test_produces_markdown_file(self, tmp_path):
        report = _make_report()
        out = tmp_path / "report.md"
        save_report(report, out, fmt="markdown")
        content = out.read_text()
        assert out.exists()
        assert "# Agent Memory Benchmark Leaderboard" in content

    def test_contains_table(self, tmp_path):
        report = _make_report()
        out = tmp_path / "report.md"
        save_report(report, out, fmt="markdown")
        content = out.read_text()
        assert "| Adapter" in content
        assert "test-adapter" in content
        assert "fact-recall" in content

    def test_contains_timestamp(self, tmp_path):
        report = _make_report()
        out = tmp_path / "report.md"
        save_report(report, out, fmt="markdown")
        content = out.read_text()
        assert "**Timestamp**" in content

    def test_contains_summary(self, tmp_path):
        report = _make_report()
        out = tmp_path / "report.md"
        save_report(report, out, fmt="markdown")
        content = out.read_text()
        assert "Summary by Adapter" in content
        assert "test-adapter" in content

    def test_github_flavored_table(self, tmp_path):
        report = _make_report()
        out = tmp_path / "report.md"
        save_report(report, out, fmt="markdown")
        content = out.read_text()
        # GitHub-flavored markdown uses pipes and dashes
        assert "|" in content
        assert "---" in content


class TestSaveReportHTML:
    def test_produces_html_file(self, tmp_path):
        report = _make_report()
        out = tmp_path / "report.html"
        save_report(report, out, fmt="html")
        content = out.read_text()
        assert out.exists()
        assert "<html>" in content.lower() or "<!doctype" in content.lower()

    def test_contains_adapter_data(self, tmp_path):
        report = _make_report()
        out = tmp_path / "report.html"
        save_report(report, out, fmt="html")
        content = out.read_text()
        assert "test-adapter" in content

    def test_contains_leaderboard_title(self, tmp_path):
        report = _make_report()
        out = tmp_path / "report.html"
        save_report(report, out, fmt="html")
        content = out.read_text()
        assert "Leaderboard" in content
