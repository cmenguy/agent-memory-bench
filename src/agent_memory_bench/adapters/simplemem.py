"""Adapter for the SimpleMem memory system."""

from __future__ import annotations

from agent_memory_bench.adapters.base import MemoryAdapter
from agent_memory_bench.models import MemoryEntry, Query, RetrievedMemory


class SimpleMemAdapter(MemoryAdapter):
    """Adapter for SimpleMem (https://simplemem.com).

    SimpleMem provides a simple REST API for storing and retrieving memories.
    This adapter wraps the SimpleMem Python SDK to implement the MemoryAdapter
    interface.
    """

    def __init__(self, api_key: str | None = None, **kwargs):
        self._client = None
        self._api_key = api_key
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return "simplemem"

    def setup(self) -> None:
        """Initialize the SimpleMem client."""
        try:
            import simplemem

            self._client = simplemem.Client(api_key=self._api_key, **self._kwargs)
        except ImportError:
            raise ImportError(
                "simplemem is not installed. "
                "Install it with: pip install agent-memory-bench[simplemem]"
            )

    def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry in SimpleMem."""
        if self._client is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")
        self._client.store(
            content=entry.content,
            user_id=entry.user_id,
            metadata=entry.metadata,
        )

    def retrieve(self, query: Query) -> list[RetrievedMemory]:
        """Retrieve memories from SimpleMem."""
        if self._client is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")
        results = self._client.search(
            query=query.text,
            user_id=query.user_id,
            limit=query.top_k,
        )
        return [
            RetrievedMemory(
                content=r.get("content", r.get("text", "")),
                score=r.get("score", r.get("relevance", 0.0)),
                metadata=r.get("metadata", {}),
            )
            for r in results
        ]

    def clear(self) -> None:
        """Clear all SimpleMem memories."""
        if self._client is not None:
            self._client.clear()
