"""Shared test fixtures for agent-memory-bench."""

from __future__ import annotations

import pytest

from agent_memory_bench.adapters.inmemory import InMemoryAdapter


@pytest.fixture
def in_memory_adapter():
    """Provide a fresh in-memory adapter for each test."""
    return InMemoryAdapter()
