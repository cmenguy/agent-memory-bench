"""Data models for the benchmark suite."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Types of memory benchmark tasks."""

    FACT_RECALL = "fact-recall"
    TEMPORAL_REASONING = "temporal-reasoning"
    CROSS_CONVERSATION = "cross-conversation"
    CONTRADICTION_DETECTION = "contradiction-detection"
    CONTEXT_WINDOW_EFFICIENCY = "context-window-efficiency"
    MULTI_HOP_RETRIEVAL = "multi-hop-retrieval"
    MEMORY_UPDATE = "memory-update"


class MemoryEntry(BaseModel):
    """A single memory entry to store in the memory system."""

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: str = "default"
    user_id: str = "benchmark-user"


class Query(BaseModel):
    """A query to retrieve memories from the memory system."""

    text: str
    session_id: str = "default"
    user_id: str = "benchmark-user"
    top_k: int = 5


class RetrievedMemory(BaseModel):
    """A memory retrieved from the memory system."""

    content: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskSample(BaseModel):
    """A single sample within a benchmark task."""

    sample_id: str
    memories_to_store: list[MemoryEntry]
    query: Query
    expected_answer: str
    expected_retrieved_contents: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskDefinition(BaseModel):
    """Definition of a benchmark task."""

    task_type: TaskType
    name: str
    description: str
    samples: list[TaskSample]
    metrics: list[str] = Field(default_factory=lambda: ["recall@5", "mrr", "latency_ms"])


class SampleResult(BaseModel):
    """Result for a single benchmark sample."""

    sample_id: str
    retrieved: list[RetrievedMemory]
    recall_at_k: float = 0.0
    mrr: float = 0.0
    latency_ms: float = 0.0
    store_latency_ms: float = 0.0
    success: bool = True
    error: str | None = None


class TaskResult(BaseModel):
    """Aggregated result for a benchmark task."""

    task_type: TaskType
    adapter_name: str
    sample_results: list[SampleResult]
    avg_recall_at_k: float = 0.0
    avg_mrr: float = 0.0
    avg_latency_ms: float = 0.0
    avg_store_latency_ms: float = 0.0
    num_samples: int = 0
    num_successes: int = 0

    def compute_aggregates(self) -> None:
        """Compute aggregate metrics from sample results."""
        self.num_samples = len(self.sample_results)
        successful = [r for r in self.sample_results if r.success]
        self.num_successes = len(successful)
        if successful:
            self.avg_recall_at_k = sum(r.recall_at_k for r in successful) / len(successful)
            self.avg_mrr = sum(r.mrr for r in successful) / len(successful)
            self.avg_latency_ms = sum(r.latency_ms for r in successful) / len(successful)
            self.avg_store_latency_ms = sum(r.store_latency_ms for r in successful) / len(
                successful
            )


class BenchmarkReport(BaseModel):
    """Full benchmark report across all tasks and adapters."""

    timestamp: datetime = Field(default_factory=datetime.now)
    task_results: list[TaskResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_leaderboard(self) -> list[dict[str, Any]]:
        """Generate a leaderboard from the benchmark results."""
        rows: list[dict[str, Any]] = []
        for result in self.task_results:
            rows.append(
                {
                    "adapter": result.adapter_name,
                    "task": result.task_type.value,
                    "recall@5": round(result.avg_recall_at_k, 4),
                    "mrr": round(result.avg_mrr, 4),
                    "latency_ms": round(result.avg_latency_ms, 2),
                    "store_latency_ms": round(result.avg_store_latency_ms, 2),
                    "samples": result.num_samples,
                    "success_rate": (
                        round(result.num_successes / result.num_samples, 4)
                        if result.num_samples
                        else 0.0
                    ),
                }
            )
        return sorted(rows, key=lambda r: (-r["recall@5"], r["latency_ms"]))
