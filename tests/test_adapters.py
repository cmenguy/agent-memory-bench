"""Tests for memory system adapters."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent_memory_bench.adapters.base import MemoryAdapter
from agent_memory_bench.models import MemoryEntry, Query, RetrievedMemory
from tests.conftest import InMemoryAdapter


class TestInMemoryAdapter:
    def test_name(self):
        adapter = InMemoryAdapter()
        assert adapter.name == "in-memory"

    def test_is_memory_adapter(self):
        adapter = InMemoryAdapter()
        assert isinstance(adapter, MemoryAdapter)

    def test_store_and_retrieve(self):
        adapter = InMemoryAdapter()
        adapter.store(MemoryEntry(content="The capital of France is Paris."))
        results = adapter.retrieve(Query(text="What is the capital of France?"))
        assert len(results) > 0
        assert "Paris" in results[0].content

    def test_clear(self):
        adapter = InMemoryAdapter()
        adapter.store(MemoryEntry(content="Some fact."))
        adapter.clear()
        results = adapter.retrieve(Query(text="Some fact"))
        assert len(results) == 0

    def test_retrieve_respects_top_k(self):
        adapter = InMemoryAdapter()
        for i in range(10):
            adapter.store(MemoryEntry(content=f"Fact number {i} about cats."))
        results = adapter.retrieve(Query(text="cats", top_k=3))
        assert len(results) <= 3

    def test_retrieve_scores_by_overlap(self):
        adapter = InMemoryAdapter()
        adapter.store(MemoryEntry(content="Python is a programming language."))
        adapter.store(MemoryEntry(content="Java is also a programming language."))
        adapter.store(MemoryEntry(content="Unrelated content about cooking."))
        results = adapter.retrieve(Query(text="programming language"))
        assert len(results) == 2
        for r in results:
            assert "programming" in r.content.lower()

    def test_retrieve_returns_retrieved_memory_objects(self):
        adapter = InMemoryAdapter()
        adapter.store(MemoryEntry(content="Test content.", metadata={"key": "value"}))
        results = adapter.retrieve(Query(text="Test"))
        assert len(results) == 1
        assert isinstance(results[0], RetrievedMemory)
        assert results[0].score > 0


class TestMem0AdapterMocked:
    def test_setup_initializes_client(self):
        mock_memory_class = MagicMock()
        mock_client = MagicMock()
        mock_memory_class.return_value = mock_client

        with patch.dict("sys.modules", {"mem0": MagicMock(Memory=mock_memory_class)}):
            from agent_memory_bench.adapters.mem0 import Mem0Adapter
            adapter = Mem0Adapter()
            adapter.setup()
            mock_memory_class.assert_called_once()
            assert adapter._client is mock_client

    def test_store_calls_add(self):
        from agent_memory_bench.adapters.mem0 import Mem0Adapter
        adapter = Mem0Adapter()
        adapter._client = MagicMock()
        entry = MemoryEntry(content="Test fact.", user_id="user1", metadata={"k": "v"})
        adapter.store(entry)
        adapter._client.add.assert_called_once_with(
            "Test fact.", user_id="user1", metadata={"k": "v"}
        )

    def test_store_without_setup_raises(self):
        from agent_memory_bench.adapters.mem0 import Mem0Adapter
        adapter = Mem0Adapter()
        with pytest.raises(RuntimeError, match="not initialized"):
            adapter.store(MemoryEntry(content="test"))

    def test_retrieve_calls_search(self):
        from agent_memory_bench.adapters.mem0 import Mem0Adapter
        adapter = Mem0Adapter()
        adapter._client = MagicMock()
        adapter._client.search.return_value = [
            {"memory": "result text", "score": 0.95, "metadata": {}},
        ]
        results = adapter.retrieve(Query(text="query", top_k=3))
        adapter._client.search.assert_called_once_with("query", user_id="benchmark-user", limit=3)
        assert len(results) == 1
        assert results[0].content == "result text"
        assert results[0].score == 0.95

    def test_clear_calls_reset(self):
        from agent_memory_bench.adapters.mem0 import Mem0Adapter
        adapter = Mem0Adapter()
        adapter._client = MagicMock()
        adapter.clear()
        adapter._client.reset.assert_called_once()

    def test_name(self):
        from agent_memory_bench.adapters.mem0 import Mem0Adapter
        assert Mem0Adapter().name == "mem0"


class TestMemOSAdapterMocked:
    def test_setup_initializes_client(self):
        mock_memos = MagicMock()
        mock_client = MagicMock()
        mock_memos.Client.return_value = mock_client

        with patch.dict("sys.modules", {"memos": mock_memos}):
            from agent_memory_bench.adapters.memos import MemOSAdapter
            adapter = MemOSAdapter(config_path="/tmp/config")
            adapter.setup()
            mock_memos.Client.assert_called_once_with(config_path="/tmp/config")
            assert adapter._client is mock_client

    def test_store_calls_store(self):
        from agent_memory_bench.adapters.memos import MemOSAdapter
        adapter = MemOSAdapter()
        adapter._client = MagicMock()
        entry = MemoryEntry(
            content="Test fact.", user_id="user1", session_id="sess1", metadata={"k": "v"}
        )
        adapter.store(entry)
        adapter._client.store.assert_called_once_with(
            content="Test fact.", user_id="user1", session_id="sess1", metadata={"k": "v"}
        )

    def test_store_without_setup_raises(self):
        from agent_memory_bench.adapters.memos import MemOSAdapter
        adapter = MemOSAdapter()
        with pytest.raises(RuntimeError, match="not initialized"):
            adapter.store(MemoryEntry(content="test"))

    def test_retrieve_calls_search(self):
        from agent_memory_bench.adapters.memos import MemOSAdapter
        adapter = MemOSAdapter()
        adapter._client = MagicMock()
        adapter._client.search.return_value = [
            {"content": "found", "relevance": 0.8, "metadata": {}},
        ]
        results = adapter.retrieve(Query(text="query", top_k=5))
        adapter._client.search.assert_called_once_with(
            query="query", user_id="benchmark-user", top_k=5
        )
        assert len(results) == 1
        assert results[0].content == "found"
        assert results[0].score == 0.8

    def test_clear_calls_clear(self):
        from agent_memory_bench.adapters.memos import MemOSAdapter
        adapter = MemOSAdapter()
        adapter._client = MagicMock()
        adapter.clear()
        adapter._client.clear.assert_called_once()

    def test_name(self):
        from agent_memory_bench.adapters.memos import MemOSAdapter
        assert MemOSAdapter().name == "memos"
