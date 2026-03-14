"""Basic usage example for agent-memory-bench.

This example demonstrates how to run benchmarks using the in-memory
test adapter. Replace it with a real adapter (Mem0, MemOS, etc.)
for actual benchmarking.
"""

from __future__ import annotations

from agent_memory_bench.adapters.base import MemoryAdapter
from agent_memory_bench.core import BenchmarkRunner
from agent_memory_bench.models import MemoryEntry, Query, RetrievedMemory
from agent_memory_bench.runner import print_report, save_report
from pathlib import Path


class SimpleInMemoryAdapter(MemoryAdapter):
    """A simple in-memory adapter for demonstration."""

    def __init__(self):
        self._store: list[MemoryEntry] = []

    @property
    def name(self) -> str:
        return "simple-in-memory"

    def store(self, entry: MemoryEntry) -> None:
        self._store.append(entry)

    def retrieve(self, query: Query) -> list[RetrievedMemory]:
        query_words = set(query.text.lower().split())
        scored = []
        for mem in self._store:
            content_words = set(mem.content.lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                scored.append((mem, overlap / max(len(query_words), 1)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            RetrievedMemory(content=m.content, score=s, metadata=m.metadata)
            for m, s in scored[: query.top_k]
        ]

    def clear(self) -> None:
        self._store.clear()


def main():
    # Create and register the adapter
    runner = BenchmarkRunner(seed=42, num_samples=5)
    runner.register_adapter("simple", SimpleInMemoryAdapter())

    # Run all available tasks
    print("Running benchmarks...")
    report = runner.run()

    # Print the leaderboard
    print_report(report)

    # Save to file
    output_path = Path("example_report.json")
    save_report(report, output_path)
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
