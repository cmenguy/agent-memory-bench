"""Adapter for the MemOS memory system."""

from __future__ import annotations

from agent_memory_bench.adapters.base import MemoryAdapter
from agent_memory_bench.models import MemoryEntry, Query, RetrievedMemory


class MemOSAdapter(MemoryAdapter):
    """Adapter for MemOS (https://github.com/MemTensor/MemOS).

    Requires: ``pip install MemoryOS>=2.0`` (PyPI package name: MemoryOS,
    import name: ``memos``).

    MemOS uses a ``MOS`` class as its main entry point, with memory organized
    into cubes (text, act, para, pref). This adapter uses ``MOS.chat()`` for
    retrieval since MemOS is designed around conversational memory access
    rather than direct store/search operations.

    Note: The MemOS API differs significantly from simple key-value memory
    systems. This adapter provides a best-effort mapping. For optimal results,
    configure MOS with a ``MOSConfig`` that matches your use case.
    """

    def __init__(self, config: dict | None = None, **kwargs):
        self._mos = None
        self._config = config
        self._kwargs = kwargs
        self._memories: list[MemoryEntry] = []

    @property
    def name(self) -> str:
        return "memos"

    def setup(self) -> None:
        """Initialize the MemOS MOS instance."""
        try:
            from memos import MOS

            if self._config is None:
                self._mos = MOS.simple()
            else:
                from memos.configs.mem_os import MOSConfig

                mos_config = MOSConfig(**self._config, **self._kwargs)
                self._mos = MOS(config=mos_config)
        except ImportError:
            raise ImportError(
                "MemoryOS is not installed. "
                "Install it with: pip install agent-memory-bench[memos]"
            )

    def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry.

        MemOS doesn't have a direct 'store' API — memories are ingested
        through chat interactions. This adapter accumulates entries and
        uses them as context during retrieval.
        """
        if self._mos is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")
        self._memories.append(entry)

    def retrieve(self, query: Query) -> list[RetrievedMemory]:
        """Retrieve memories using MemOS chat-based retrieval."""
        if self._mos is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")
        # Use MOS.chat() which searches memory and generates a response
        try:
            response = self._mos.chat(
                query=query.text,
                user_id=query.user_id,
            )
            # MOS.chat returns a string response; wrap it as a single result
            return [
                RetrievedMemory(
                    content=response,
                    score=1.0,
                    metadata={},
                )
            ]
        except Exception:
            return []

    def clear(self) -> None:
        """Clear accumulated memories."""
        self._memories.clear()
