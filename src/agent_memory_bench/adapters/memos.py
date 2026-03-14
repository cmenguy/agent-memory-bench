"""Adapter for the MemOS memory system."""

from __future__ import annotations

from agent_memory_bench.adapters.base import MemoryAdapter
from agent_memory_bench.models import MemoryEntry, Query, RetrievedMemory


class MemOSAdapter(MemoryAdapter):
    """Adapter for MemOS (https://github.com/MemTensor/MemOS)."""

    def __init__(self, config_path: str | None = None, **kwargs):
        self._client = None
        self._config_path = config_path
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return "memos"

    def setup(self) -> None:
        """Initialize the MemOS client."""
        try:
            import memos

            self._client = memos.Client(config_path=self._config_path, **self._kwargs)
        except ImportError:
            raise ImportError(
                "memos-sdk is not installed. Install it with: pip install agent-memory-bench[memos]"
            )

    def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry in MemOS."""
        if self._client is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")
        self._client.store(
            content=entry.content,
            user_id=entry.user_id,
            session_id=entry.session_id,
            metadata=entry.metadata,
        )

    def retrieve(self, query: Query) -> list[RetrievedMemory]:
        """Retrieve memories from MemOS."""
        if self._client is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")
        results = self._client.search(
            query=query.text,
            user_id=query.user_id,
            top_k=query.top_k,
        )
        return [
            RetrievedMemory(
                content=r.get("content", ""),
                score=r.get("relevance", 0.0),
                metadata=r.get("metadata", {}),
            )
            for r in results
        ]

    def clear(self) -> None:
        """Clear all MemOS memories."""
        if self._client is not None:
            self._client.clear()
