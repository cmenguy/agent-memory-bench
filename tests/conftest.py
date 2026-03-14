"""Shared test fixtures for agent-memory-bench."""

from __future__ import annotations

import pytest

from agent_memory_bench.adapters.base import MemoryAdapter
from agent_memory_bench.models import MemoryEntry, Query, RetrievedMemory


class InMemoryAdapter(MemoryAdapter):
    """Simple in-memory adapter for testing."""

    def __init__(self):
        self._memories: list[MemoryEntry] = []

    @property
    def name(self) -> str:
        return "in-memory"

    def store(self, entry: MemoryEntry) -> None:
        self._memories.append(entry)

    def retrieve(self, query: Query) -> list[RetrievedMemory]:
        # Naive keyword matching for testing
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

    def clear(self) -> None:
        self._memories.clear()


@pytest.fixture
def in_memory_adapter():
    """Provide a fresh in-memory adapter for each test."""
    return InMemoryAdapter()
