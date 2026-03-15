"""Tests for async adapter support."""

from __future__ import annotations

from agent_memory_bench.adapters.base import AsyncMemoryAdapter, MemoryAdapter
from agent_memory_bench.core import BenchmarkRunner
from agent_memory_bench.models import MemoryEntry, Query, RetrievedMemory


class MockAsyncAdapter(AsyncMemoryAdapter):
    """Mock async adapter for testing."""

    def __init__(self):
        self._memories: list[MemoryEntry] = []

    @property
    def name(self) -> str:
        return "mock-async"

    async def store(self, entry: MemoryEntry) -> None:
        self._memories.append(entry)

    async def retrieve(self, query: Query) -> list[RetrievedMemory]:
        results = []
        query_words = set(query.text.lower().split())
        for mem in self._memories:
            content_words = set(mem.content.lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                results.append(
                    RetrievedMemory(
                        content=mem.content,
                        score=overlap / max(len(query_words), 1),
                        metadata=mem.metadata,
                    )
                )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[: query.top_k]

    async def clear(self) -> None:
        self._memories.clear()


class TestAsyncMemoryAdapter:
    def test_is_not_sync_adapter(self):
        adapter = MockAsyncAdapter()
        assert not isinstance(adapter, MemoryAdapter)
        assert isinstance(adapter, AsyncMemoryAdapter)

    def test_name(self):
        adapter = MockAsyncAdapter()
        assert adapter.name == "mock-async"


class TestBenchmarkRunnerWithAsyncAdapter:
    def test_register_async_adapter(self):
        runner = BenchmarkRunner()
        adapter = MockAsyncAdapter()
        runner.register_adapter("async-test", adapter)
        assert "async-test" in runner._adapters

    def test_run_with_async_adapter(self):
        runner = BenchmarkRunner(num_samples=3)
        adapter = MockAsyncAdapter()
        runner.register_adapter("async-test", adapter)
        report = runner.run(tasks=["fact-recall"])
        assert len(report.task_results) == 1
        result = report.task_results[0]
        assert result.adapter_name == "mock-async"
        assert result.num_samples == 3

    def test_run_multiple_tasks_async(self):
        runner = BenchmarkRunner(num_samples=2)
        adapter = MockAsyncAdapter()
        runner.register_adapter("async-test", adapter)
        report = runner.run(tasks=["fact-recall", "temporal-reasoning"])
        assert len(report.task_results) == 2

    def test_async_latency_is_measured(self):
        runner = BenchmarkRunner(num_samples=1)
        adapter = MockAsyncAdapter()
        runner.register_adapter("async-test", adapter)
        report = runner.run(tasks=["fact-recall"])
        result = report.task_results[0]
        # Latency should be non-negative (actual time measured)
        assert result.avg_latency_ms >= 0
        assert result.avg_store_latency_ms >= 0

    def test_mixed_sync_and_async_adapters(self, in_memory_adapter):
        runner = BenchmarkRunner(num_samples=2)
        runner.register_adapter("sync", in_memory_adapter)
        runner.register_adapter("async", MockAsyncAdapter())
        report = runner.run(tasks=["fact-recall"])
        assert len(report.task_results) == 2
        adapter_names = {r.adapter_name for r in report.task_results}
        assert "in-memory" in adapter_names
        assert "mock-async" in adapter_names
