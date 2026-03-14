"""Adapter for the Mem0 memory system."""

from __future__ import annotations

from agent_memory_bench.adapters.base import MemoryAdapter
from agent_memory_bench.models import MemoryEntry, Query, RetrievedMemory


class Mem0Adapter(MemoryAdapter):
    """Adapter for Mem0 (https://github.com/mem0ai/mem0)."""

    def __init__(self, api_key: str | None = None, **kwargs):
        self._client = None
        self._api_key = api_key
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return "mem0"

    def setup(self) -> None:
        """Initialize the Mem0 client."""
        try:
            from mem0 import Memory

            self._client = Memory(**self._kwargs)
        except ImportError:
            raise ImportError(
                "mem0ai is not installed. Install it with: pip install agent-memory-bench[mem0]"
            )

    def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry in Mem0."""
        if self._client is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")
        self._client.add(
            entry.content,
            user_id=entry.user_id,
            metadata=entry.metadata,
        )

    def retrieve(self, query: Query) -> list[RetrievedMemory]:
        """Retrieve memories from Mem0."""
        if self._client is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")
        results = self._client.search(
            query.text,
            user_id=query.user_id,
            limit=query.top_k,
        )
        return [
            RetrievedMemory(
                content=r.get("memory", r.get("text", "")),
                score=r.get("score", 0.0),
                metadata=r.get("metadata", {}),
            )
            for r in results
        ]

    def clear(self) -> None:
        """Clear all Mem0 memories."""
        if self._client is not None:
            self._client.reset()
