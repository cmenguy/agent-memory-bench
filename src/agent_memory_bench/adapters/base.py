"""Base adapter interface for memory systems."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agent_memory_bench.models import MemoryEntry, Query, RetrievedMemory


class MemoryAdapter(ABC):
    """Abstract base class for synchronous memory system adapters.

    All adapters must implement store, retrieve, and clear operations
    to be benchmarked by the suite.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this memory system."""

    @abstractmethod
    def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry in the system."""

    @abstractmethod
    def retrieve(self, query: Query) -> list[RetrievedMemory]:
        """Retrieve relevant memories for a query."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored memories. Called between benchmark tasks."""

    def setup(self) -> None:
        """Optional setup hook called before benchmark tasks begin."""

    def teardown(self) -> None:
        """Optional teardown hook called after all benchmark tasks complete."""


class AsyncMemoryAdapter(ABC):
    """Abstract base class for asynchronous memory system adapters.

    Use this for memory systems with async Python APIs to ensure fair
    latency benchmarking without synchronous wrapper overhead.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this memory system."""

    @abstractmethod
    async def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry in the system."""

    @abstractmethod
    async def retrieve(self, query: Query) -> list[RetrievedMemory]:
        """Retrieve relevant memories for a query."""

    @abstractmethod
    async def clear(self) -> None:
        """Clear all stored memories. Called between benchmark tasks."""

    async def setup(self) -> None:
        """Optional async setup hook called before benchmark tasks begin."""

    async def teardown(self) -> None:
        """Optional async teardown hook called after all benchmark tasks complete."""
