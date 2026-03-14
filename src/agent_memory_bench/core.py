"""Core benchmark logic -- task definitions and evaluation."""

from __future__ import annotations

import random
import time
from datetime import datetime, timedelta

from agent_memory_bench.adapters.base import MemoryAdapter
from agent_memory_bench.models import (
    BenchmarkReport,
    MemoryEntry,
    Query,
    SampleResult,
    TaskDefinition,
    TaskResult,
    TaskSample,
    TaskType,
)


def compute_recall_at_k(
    retrieved: list[str], expected: list[str], k: int = 5
) -> float:
    """Compute recall@k: fraction of expected items found in top-k retrieved."""
    if not expected:
        return 1.0
    top_k = retrieved[:k]
    found = sum(1 for e in expected if any(e.lower() in r.lower() for r in top_k))
    return found / len(expected)


def compute_mrr(retrieved: list[str], expected: list[str]) -> float:
    """Compute Mean Reciprocal Rank over expected items."""
    if not expected:
        return 1.0
    reciprocal_ranks = []
    for exp in expected:
        for rank, ret in enumerate(retrieved, 1):
            if exp.lower() in ret.lower():
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def generate_fact_recall_task(num_samples: int = 20, seed: int = 42) -> TaskDefinition:
    """Generate a fact recall benchmark task.

    Tests whether the memory system can store facts and retrieve them
    when queried with related questions.
    """
    rng = random.Random(seed)
    facts = [
        ("The capital of France is Paris.", "What is the capital of France?", "Paris"),
        ("Python was created by Guido van Rossum in 1991.", "Who created Python?", "Guido van Rossum"),
        ("The speed of light is approximately 299,792,458 meters per second.", "What is the speed of light?", "299,792,458"),
        ("Mount Everest is 8,849 meters tall.", "How tall is Mount Everest?", "8,849"),
        ("The human body has 206 bones.", "How many bones does the human body have?", "206"),
        ("Water boils at 100 degrees Celsius at sea level.", "At what temperature does water boil?", "100"),
        ("The Great Wall of China is approximately 21,196 kilometers long.", "How long is the Great Wall of China?", "21,196"),
        ("DNA stands for deoxyribonucleic acid.", "What does DNA stand for?", "deoxyribonucleic acid"),
        ("The Amazon River is the largest river by volume.", "What is the largest river by volume?", "Amazon"),
        ("Jupiter is the largest planet in our solar system.", "What is the largest planet?", "Jupiter"),
        ("The first Moon landing was in 1969.", "When was the first Moon landing?", "1969"),
        ("Oxygen has the atomic number 8.", "What is the atomic number of oxygen?", "8"),
        ("Shakespeare wrote 37 plays.", "How many plays did Shakespeare write?", "37"),
        ("The Pacific Ocean is the largest ocean.", "What is the largest ocean?", "Pacific"),
        ("Einstein published the theory of general relativity in 1915.", "When was general relativity published?", "1915"),
        ("The Sahara is the largest hot desert.", "What is the largest hot desert?", "Sahara"),
        ("A marathon is 42.195 kilometers.", "How long is a marathon?", "42.195"),
        ("The Earth is approximately 4.5 billion years old.", "How old is the Earth?", "4.5 billion"),
        ("Beethoven composed 9 symphonies.", "How many symphonies did Beethoven compose?", "9"),
        ("Gold has the chemical symbol Au.", "What is the chemical symbol for gold?", "Au"),
    ]

    samples = []
    selected = rng.sample(facts, min(num_samples, len(facts)))
    for i, (fact, question, answer) in enumerate(selected):
        # Add distractor memories alongside the target fact
        distractors = rng.sample([f for f, _, _ in facts if f != fact], min(3, len(facts) - 1))
        memories = [
            MemoryEntry(content=fact, metadata={"type": "target"}),
        ] + [
            MemoryEntry(content=d, metadata={"type": "distractor"})
            for d in distractors
        ]
        samples.append(
            TaskSample(
                sample_id=f"fact-recall-{i:03d}",
                memories_to_store=memories,
                query=Query(text=question),
                expected_answer=answer,
                expected_retrieved_contents=[fact],
            )
        )

    return TaskDefinition(
        task_type=TaskType.FACT_RECALL,
        name="Fact Recall",
        description="Tests whether the memory system can store facts and retrieve them accurately.",
        samples=samples,
    )


def generate_temporal_reasoning_task(num_samples: int = 10, seed: int = 42) -> TaskDefinition:
    """Generate a temporal reasoning benchmark task.

    Tests whether the memory system handles time-ordered information correctly.
    """
    rng = random.Random(seed)
    base_time = datetime(2026, 1, 1)
    samples = []

    for i in range(num_samples):
        # Create a sequence of events
        events = [
            MemoryEntry(
                content=f"User set their favorite color to blue.",
                timestamp=base_time + timedelta(days=i * 10),
                metadata={"type": "preference", "version": 1},
            ),
            MemoryEntry(
                content=f"User changed their favorite color to green.",
                timestamp=base_time + timedelta(days=i * 10 + 5),
                metadata={"type": "preference", "version": 2},
            ),
        ]
        rng.shuffle(events)  # Store in random order

        samples.append(
            TaskSample(
                sample_id=f"temporal-{i:03d}",
                memories_to_store=events,
                query=Query(text="What is the user's current favorite color?"),
                expected_answer="green",
                expected_retrieved_contents=["User changed their favorite color to green."],
            )
        )

    return TaskDefinition(
        task_type=TaskType.TEMPORAL_REASONING,
        name="Temporal Reasoning",
        description="Tests whether the memory system correctly handles time-ordered updates.",
        samples=samples,
    )


def generate_contradiction_detection_task(num_samples: int = 10, seed: int = 42) -> TaskDefinition:
    """Generate a contradiction detection benchmark task.

    Tests whether the memory system can detect and handle contradictory information.
    """
    samples = []
    contradictions = [
        (
            "The project deadline is March 15th.",
            "The project deadline has been moved to April 1st.",
            "What is the project deadline?",
            "April 1st",
        ),
        (
            "The meeting is scheduled for 2pm.",
            "The meeting has been rescheduled to 4pm.",
            "When is the meeting?",
            "4pm",
        ),
        (
            "The budget for Q1 is $50,000.",
            "The Q1 budget has been revised to $75,000.",
            "What is the Q1 budget?",
            "$75,000",
        ),
        (
            "John is the team lead for Project Alpha.",
            "Sarah has replaced John as team lead for Project Alpha.",
            "Who is the team lead for Project Alpha?",
            "Sarah",
        ),
        (
            "The server runs on Python 3.9.",
            "The server has been upgraded to Python 3.12.",
            "What Python version does the server run?",
            "Python 3.12",
        ),
        (
            "The office is located on the 5th floor.",
            "The team has moved to the 8th floor.",
            "What floor is the office on?",
            "8th floor",
        ),
        (
            "The API rate limit is 100 requests per minute.",
            "The API rate limit has been increased to 500 requests per minute.",
            "What is the API rate limit?",
            "500 requests per minute",
        ),
        (
            "The database uses PostgreSQL 14.",
            "We migrated the database to PostgreSQL 16.",
            "What version of PostgreSQL is the database?",
            "PostgreSQL 16",
        ),
        (
            "The deploy target is AWS us-east-1.",
            "We switched the deploy target to AWS eu-west-1.",
            "What is the deploy target region?",
            "eu-west-1",
        ),
        (
            "The sprint duration is 2 weeks.",
            "We changed sprint duration to 3 weeks.",
            "How long is the sprint?",
            "3 weeks",
        ),
    ]

    for i, (old_fact, new_fact, question, answer) in enumerate(contradictions[:num_samples]):
        base_time = datetime(2026, 1, 1)
        samples.append(
            TaskSample(
                sample_id=f"contradiction-{i:03d}",
                memories_to_store=[
                    MemoryEntry(
                        content=old_fact,
                        timestamp=base_time,
                        metadata={"type": "original"},
                    ),
                    MemoryEntry(
                        content=new_fact,
                        timestamp=base_time + timedelta(days=7),
                        metadata={"type": "update"},
                    ),
                ],
                query=Query(text=question),
                expected_answer=answer,
                expected_retrieved_contents=[new_fact],
            )
        )

    return TaskDefinition(
        task_type=TaskType.CONTRADICTION_DETECTION,
        name="Contradiction Detection",
        description="Tests whether the memory system handles contradictory/updated information.",
        samples=samples,
    )


TASK_GENERATORS = {
    TaskType.FACT_RECALL: generate_fact_recall_task,
    TaskType.TEMPORAL_REASONING: generate_temporal_reasoning_task,
    TaskType.CONTRADICTION_DETECTION: generate_contradiction_detection_task,
}


class BenchmarkRunner:
    """Runs benchmark tasks against registered memory adapters."""

    def __init__(self, seed: int = 42, num_samples: int = 20):
        self._adapters: dict[str, MemoryAdapter] = {}
        self._seed = seed
        self._num_samples = num_samples

    def register_adapter(self, name: str, adapter: MemoryAdapter) -> None:
        """Register a memory adapter for benchmarking."""
        self._adapters[name] = adapter

    @property
    def available_tasks(self) -> list[TaskType]:
        """List of task types with implemented generators."""
        return list(TASK_GENERATORS.keys())

    def run_task(self, adapter: MemoryAdapter, task: TaskDefinition) -> TaskResult:
        """Run a single benchmark task against an adapter."""
        sample_results: list[SampleResult] = []

        for sample in task.samples:
            adapter.clear()
            try:
                # Store memories
                store_start = time.perf_counter()
                for entry in sample.memories_to_store:
                    adapter.store(entry)
                store_elapsed = (time.perf_counter() - store_start) * 1000

                # Retrieve
                retrieve_start = time.perf_counter()
                retrieved = adapter.retrieve(sample.query)
                retrieve_elapsed = (time.perf_counter() - retrieve_start) * 1000

                retrieved_contents = [r.content for r in retrieved]
                recall = compute_recall_at_k(
                    retrieved_contents, sample.expected_retrieved_contents
                )
                mrr = compute_mrr(retrieved_contents, sample.expected_retrieved_contents)

                sample_results.append(
                    SampleResult(
                        sample_id=sample.sample_id,
                        retrieved=retrieved,
                        recall_at_k=recall,
                        mrr=mrr,
                        latency_ms=retrieve_elapsed,
                        store_latency_ms=store_elapsed,
                    )
                )
            except Exception as e:
                sample_results.append(
                    SampleResult(
                        sample_id=sample.sample_id,
                        retrieved=[],
                        success=False,
                        error=str(e),
                    )
                )

        result = TaskResult(
            task_type=task.task_type,
            adapter_name=adapter.name,
            sample_results=sample_results,
        )
        result.compute_aggregates()
        return result

    def run(self, tasks: list[str | TaskType] | None = None) -> BenchmarkReport:
        """Run benchmarks for the specified tasks across all registered adapters."""
        if not self._adapters:
            raise ValueError("No adapters registered. Call register_adapter() first.")

        # Resolve task types
        task_types: list[TaskType] = []
        if tasks is None:
            task_types = self.available_tasks
        else:
            for t in tasks:
                if isinstance(t, str):
                    task_types.append(TaskType(t))
                else:
                    task_types.append(t)

        report = BenchmarkReport()
        for adapter_name, adapter in self._adapters.items():
            adapter.setup()
            try:
                for task_type in task_types:
                    generator = TASK_GENERATORS.get(task_type)
                    if generator is None:
                        continue
                    task_def = generator(
                        num_samples=self._num_samples, seed=self._seed
                    )
                    result = self.run_task(adapter, task_def)
                    report.task_results.append(result)
            finally:
                adapter.teardown()

        return report
