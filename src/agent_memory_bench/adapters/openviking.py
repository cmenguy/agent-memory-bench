"""Adapter for the OpenViking context database."""

from __future__ import annotations

from agent_memory_bench.adapters.base import MemoryAdapter
from agent_memory_bench.models import MemoryEntry, Query, RetrievedMemory


class OpenVikingAdapter(MemoryAdapter):
    """Adapter for OpenViking (https://github.com/volcengine/OpenViking).

    OpenViking is a context database for AI agents that manages memory, resources,
    and skills through a file system paradigm. This adapter maps the MemoryAdapter
    interface to OpenViking's session-based store/search/clear operations.
    """

    def __init__(self, **kwargs):
        self._client = None
        self._kwargs = kwargs
        self._session_id: str | None = None

    @property
    def name(self) -> str:
        return "openviking"

    def setup(self) -> None:
        """Initialize the OpenViking client."""
        try:
            from openviking.sync_client import SyncOpenViking

            self._client = SyncOpenViking(**self._kwargs)
            self._client.initialize()
            session_info = self._client.create_session()
            self._session_id = session_info.get("session_id", session_info.get("id"))
        except ImportError:
            raise ImportError(
                "openviking is not installed. "
                "Install it with: pip install agent-memory-bench[openviking]"
            )

    def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry in OpenViking as a user message."""
        if self._client is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")
        self._client.add_message(
            session_id=self._session_id,
            role="user",
            content=entry.content,
        )
        self._client.commit_session(self._session_id)

    def retrieve(self, query: Query) -> list[RetrievedMemory]:
        """Retrieve memories from OpenViking via search."""
        if self._client is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")
        results = self._client.search(
            query=query.text,
            session_id=self._session_id,
            limit=query.top_k,
        )
        if isinstance(results, dict):
            items = results.get("results", results.get("items", []))
        elif isinstance(results, list):
            items = results
        else:
            items = []
        return [
            RetrievedMemory(
                content=r.get("content", r.get("text", "")),
                score=r.get("score", r.get("relevance", 0.0)),
                metadata=r.get("metadata", {}),
            )
            for r in items
        ]

    def clear(self) -> None:
        """Clear OpenViking memories by deleting and recreating the session."""
        if self._client is not None and self._session_id is not None:
            self._client.delete_session(self._session_id)
            session_info = self._client.create_session()
            self._session_id = session_info.get("session_id", session_info.get("id"))

    def teardown(self) -> None:
        """Clean up the OpenViking session."""
        if self._client is not None and self._session_id is not None:
            self._client.delete_session(self._session_id)
            self._session_id = None
