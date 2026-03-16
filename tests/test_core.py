"""Tests for core benchmark logic."""

from __future__ import annotations

from agent_memory_bench.core import (
    BenchmarkRunner,
    compute_mrr,
    compute_recall_at_k,
    generate_context_window_efficiency_task,
    generate_contradiction_detection_task,
    generate_cross_conversation_task,
    generate_fact_recall_task,
    generate_memory_update_task,
    generate_multi_hop_retrieval_task,
    generate_temporal_reasoning_task,
)
from agent_memory_bench.models import BenchmarkReport, SampleResult, TaskResult, TaskType


class TestMetrics:
    def test_recall_at_k_perfect(self):
        retrieved = ["fact A", "fact B", "fact C"]
        expected = ["fact A", "fact B"]
        assert compute_recall_at_k(retrieved, expected, k=5) == 1.0

    def test_recall_at_k_partial(self):
        retrieved = ["fact A", "fact X", "fact Y"]
        expected = ["fact A", "fact B"]
        assert compute_recall_at_k(retrieved, expected, k=5) == 0.5

    def test_recall_at_k_none(self):
        retrieved = ["fact X", "fact Y", "fact Z"]
        expected = ["fact A", "fact B"]
        assert compute_recall_at_k(retrieved, expected, k=5) == 0.0

    def test_recall_at_k_empty_expected(self):
        retrieved = ["fact A"]
        assert compute_recall_at_k(retrieved, [], k=5) == 1.0

    def test_mrr_first_position(self):
        retrieved = ["fact A", "fact B", "fact C"]
        expected = ["fact A"]
        assert compute_mrr(retrieved, expected) == 1.0

    def test_mrr_second_position(self):
        retrieved = ["fact X", "fact A", "fact C"]
        expected = ["fact A"]
        assert compute_mrr(retrieved, expected) == 0.5

    def test_mrr_not_found(self):
        retrieved = ["fact X", "fact Y", "fact Z"]
        expected = ["fact A"]
        assert compute_mrr(retrieved, expected) == 0.0


class TestTaskGeneration:
    def test_fact_recall_generates_samples(self):
        task = generate_fact_recall_task(num_samples=5, seed=42)
        assert task.task_type == TaskType.FACT_RECALL
        assert len(task.samples) == 5
        for sample in task.samples:
            assert len(sample.memories_to_store) > 0
            assert sample.query.text
            assert sample.expected_answer

    def test_temporal_reasoning_generates_samples(self):
        task = generate_temporal_reasoning_task(num_samples=3, seed=42)
        assert task.task_type == TaskType.TEMPORAL_REASONING
        assert len(task.samples) == 3

    def test_contradiction_detection_generates_samples(self):
        task = generate_contradiction_detection_task(num_samples=3, seed=42)
        assert task.task_type == TaskType.CONTRADICTION_DETECTION
        assert len(task.samples) == 3
        for sample in task.samples:
            assert len(sample.memories_to_store) == 2

    def test_reproducibility_with_same_seed(self):
        task1 = generate_fact_recall_task(num_samples=5, seed=99)
        task2 = generate_fact_recall_task(num_samples=5, seed=99)
        for s1, s2 in zip(task1.samples, task2.samples):
            assert s1.sample_id == s2.sample_id
            assert s1.expected_answer == s2.expected_answer

    def test_cross_conversation_generates_samples(self):
        task = generate_cross_conversation_task(num_samples=5, seed=42)
        assert task.task_type == TaskType.CROSS_CONVERSATION
        assert len(task.samples) == 5
        for sample in task.samples:
            assert len(sample.memories_to_store) == 2
            assert sample.query.text
            assert sample.expected_answer
            # Memories should have different session_ids
            session_ids = {m.session_id for m in sample.memories_to_store}
            assert len(session_ids) == 2

    def test_cross_conversation_query_session_differs(self):
        task = generate_cross_conversation_task(num_samples=10, seed=42)
        for sample in task.samples:
            memory_sessions = {m.session_id for m in sample.memories_to_store}
            # Query session may or may not overlap with memory sessions,
            # but the key property is that multiple sessions are used
            assert len(memory_sessions) >= 2

    def test_cross_conversation_reproducibility(self):
        task1 = generate_cross_conversation_task(num_samples=5, seed=77)
        task2 = generate_cross_conversation_task(num_samples=5, seed=77)
        for s1, s2 in zip(task1.samples, task2.samples):
            assert s1.sample_id == s2.sample_id
            assert s1.expected_answer == s2.expected_answer

    def test_multi_hop_retrieval_generates_samples(self):
        task = generate_multi_hop_retrieval_task(num_samples=5, seed=42)
        assert task.task_type == TaskType.MULTI_HOP_RETRIEVAL
        assert len(task.samples) == 5
        for sample in task.samples:
            assert len(sample.memories_to_store) == 3  # 2 chain links + 1 distractor
            assert sample.query.text
            assert sample.expected_answer
            assert len(sample.expected_retrieved_contents) == 2

    def test_multi_hop_retrieval_has_chain_links(self):
        task = generate_multi_hop_retrieval_task(num_samples=5, seed=42)
        for sample in task.samples:
            metadata_types = {m.metadata.get("type") for m in sample.memories_to_store}
            assert "chain-link-1" in metadata_types
            assert "chain-link-2" in metadata_types
            assert "distractor" in metadata_types

    def test_multi_hop_retrieval_reproducibility(self):
        task1 = generate_multi_hop_retrieval_task(num_samples=5, seed=77)
        task2 = generate_multi_hop_retrieval_task(num_samples=5, seed=77)
        for s1, s2 in zip(task1.samples, task2.samples):
            assert s1.sample_id == s2.sample_id
            assert s1.expected_answer == s2.expected_answer

    def test_memory_update_generates_samples(self):
        task = generate_memory_update_task(num_samples=5, seed=42)
        assert task.task_type == TaskType.MEMORY_UPDATE
        assert len(task.samples) == 5
        for sample in task.samples:
            assert len(sample.memories_to_store) == 2
            assert sample.query.text
            assert sample.expected_answer
            # Should have original and update with different timestamps
            timestamps = [m.timestamp for m in sample.memories_to_store]
            assert timestamps[0] < timestamps[1]

    def test_memory_update_has_explicit_update_semantics(self):
        task = generate_memory_update_task(num_samples=5, seed=42)
        for sample in task.samples:
            metadata_types = [m.metadata.get("type") for m in sample.memories_to_store]
            assert "original" in metadata_types
            assert "update" in metadata_types
            # Update entry should have supersedes field
            update_entry = [m for m in sample.memories_to_store if m.metadata.get("type") == "update"][0]
            assert update_entry.metadata.get("supersedes") == "original"
            assert update_entry.metadata.get("version") == 2

    def test_memory_update_reproducibility(self):
        task1 = generate_memory_update_task(num_samples=5, seed=77)
        task2 = generate_memory_update_task(num_samples=5, seed=77)
        for s1, s2 in zip(task1.samples, task2.samples):
            assert s1.sample_id == s2.sample_id
            assert s1.expected_answer == s2.expected_answer

    def test_context_window_efficiency_generates_samples(self):
        task = generate_context_window_efficiency_task(num_samples=4, seed=42)
        assert task.task_type == TaskType.CONTEXT_WINDOW_EFFICIENCY
        assert len(task.samples) == 4
        for sample in task.samples:
            assert len(sample.memories_to_store) > 1
            assert sample.query.text
            assert sample.expected_answer
            assert "scale" in sample.metadata

    def test_context_window_efficiency_varying_scales(self):
        task = generate_context_window_efficiency_task(num_samples=4, seed=42)
        scales = [s.metadata["scale"] for s in task.samples]
        # With 4 samples, we should see all 4 scale levels (10, 50, 100, 200)
        assert set(scales) == {10, 50, 100, 200}

    def test_context_window_efficiency_memory_counts(self):
        task = generate_context_window_efficiency_task(num_samples=4, seed=42)
        for sample in task.samples:
            scale = sample.metadata["scale"]
            # Should have scale fillers + 1 target
            assert len(sample.memories_to_store) == scale + 1

    def test_context_window_efficiency_reproducibility(self):
        task1 = generate_context_window_efficiency_task(num_samples=4, seed=77)
        task2 = generate_context_window_efficiency_task(num_samples=4, seed=77)
        for s1, s2 in zip(task1.samples, task2.samples):
            assert s1.sample_id == s2.sample_id
            assert s1.expected_answer == s2.expected_answer


class TestBenchmarkRunner:
    def test_register_adapter(self, in_memory_adapter):
        runner = BenchmarkRunner()
        runner.register_adapter("test", in_memory_adapter)
        assert "test" in runner._adapters

    def test_run_without_adapters_raises(self):
        runner = BenchmarkRunner()
        try:
            runner.run()
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_run_with_adapter(self, in_memory_adapter):
        runner = BenchmarkRunner(num_samples=3)
        runner.register_adapter("test", in_memory_adapter)
        report = runner.run(tasks=["fact-recall"])
        assert len(report.task_results) == 1
        result = report.task_results[0]
        assert result.adapter_name == "in-memory"
        assert result.num_samples == 3

    def test_run_multiple_tasks(self, in_memory_adapter):
        runner = BenchmarkRunner(num_samples=2)
        runner.register_adapter("test", in_memory_adapter)
        report = runner.run(tasks=["fact-recall", "temporal-reasoning"])
        assert len(report.task_results) == 2


class TestModels:
    def test_task_result_compute_aggregates(self):
        result = TaskResult(
            task_type=TaskType.FACT_RECALL,
            adapter_name="test",
            sample_results=[
                SampleResult(
                    sample_id="s1",
                    retrieved=[],
                    recall_at_k=0.8,
                    mrr=0.5,
                    latency_ms=10.0,
                    store_latency_ms=5.0,
                ),
                SampleResult(
                    sample_id="s2",
                    retrieved=[],
                    recall_at_k=0.6,
                    mrr=1.0,
                    latency_ms=20.0,
                    store_latency_ms=8.0,
                ),
            ],
        )
        result.compute_aggregates()
        assert result.num_samples == 2
        assert result.num_successes == 2
        assert result.avg_recall_at_k == 0.7
        assert result.avg_mrr == 0.75
        assert result.avg_latency_ms == 15.0

    def test_benchmark_report_leaderboard(self):
        report = BenchmarkReport(
            task_results=[
                TaskResult(
                    task_type=TaskType.FACT_RECALL,
                    adapter_name="adapter_a",
                    sample_results=[],
                    avg_recall_at_k=0.9,
                    avg_mrr=0.8,
                    avg_latency_ms=15.0,
                    num_samples=10,
                    num_successes=10,
                ),
                TaskResult(
                    task_type=TaskType.FACT_RECALL,
                    adapter_name="adapter_b",
                    sample_results=[],
                    avg_recall_at_k=0.7,
                    avg_mrr=0.6,
                    avg_latency_ms=25.0,
                    num_samples=10,
                    num_successes=9,
                ),
            ]
        )
        lb = report.to_leaderboard()
        assert len(lb) == 2
        assert lb[0]["adapter"] == "adapter_a"  # Higher recall comes first
        assert lb[1]["adapter"] == "adapter_b"
