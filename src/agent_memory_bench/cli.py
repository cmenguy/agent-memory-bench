"""CLI entry point for the benchmark suite."""

from __future__ import annotations

from pathlib import Path

import click

from agent_memory_bench.core import BenchmarkRunner, TASK_GENERATORS
from agent_memory_bench.models import TaskType
from agent_memory_bench.runner import print_report, save_report


AVAILABLE_TASKS = [t.value for t in TASK_GENERATORS]


def _resolve_adapter(name: str):
    """Resolve adapter by name, importing and instantiating it."""
    match name:
        case "inmemory":
            from agent_memory_bench.adapters.inmemory import InMemoryAdapter
            return InMemoryAdapter()
        case "mem0":
            from agent_memory_bench.adapters.mem0 import Mem0Adapter
            return Mem0Adapter()
        case "memos":
            from agent_memory_bench.adapters.memos import MemOSAdapter
            return MemOSAdapter()
        case _:
            raise click.BadParameter(
                f"Unknown adapter: {name}. Available: inmemory, mem0, memos"
            )


@click.group()
@click.version_option(package_name="agent-memory-bench")
def main():
    """Agent Memory Benchmark Suite.

    A standardized benchmark suite for evaluating and comparing agent memory systems.
    """


@main.command()
@click.option("--task", "-t", multiple=True, help=f"Task(s) to run. Available: {AVAILABLE_TASKS}")
@click.option("--adapter", "-a", multiple=True, help="Adapter(s) to benchmark. Available: inmemory, mem0, memos")
@click.option("--all", "run_all", is_flag=True, help="Run all tasks against all adapters")
@click.option("--num-samples", "-n", default=20, help="Number of samples per task")
@click.option("--seed", "-s", default=42, help="Random seed for reproducibility")
@click.option("--output", "-o", type=click.Path(), help="Save report to JSON file")
def run(task, adapter, run_all, num_samples, seed, output):
    """Run benchmark tasks against memory system adapters."""
    runner = BenchmarkRunner(seed=seed, num_samples=num_samples)

    # Register adapters
    adapter_names = list(adapter) if adapter else (["inmemory", "mem0", "memos"] if run_all else [])
    if not adapter_names:
        raise click.UsageError("Specify --adapter or use --all to benchmark all adapters.")

    for name in adapter_names:
        try:
            a = _resolve_adapter(name)
            runner.register_adapter(name, a)
        except ImportError as e:
            click.echo(f"Warning: Skipping adapter '{name}': {e}", err=True)

    # Resolve tasks
    tasks = list(task) if task else (AVAILABLE_TASKS if run_all else [])
    if not tasks:
        raise click.UsageError("Specify --task or use --all to run all tasks.")

    click.echo(f"Running {len(tasks)} task(s) against {len(runner._adapters)} adapter(s)...")
    report = runner.run(tasks=tasks)
    print_report(report)

    if output:
        save_report(report, Path(output))
        click.echo(f"Report saved to {output}")


@main.command()
def tasks():
    """List available benchmark tasks."""
    click.echo("Available benchmark tasks:\n")
    for task_type, generator in TASK_GENERATORS.items():
        task_def = generator(num_samples=1)
        click.echo(f"  {task_type.value}")
        click.echo(f"    {task_def.description}")
        click.echo()


@main.command()
def adapters():
    """List available memory system adapters."""
    click.echo("Available adapters:\n")
    adapter_info = [
        ("inmemory", "In-Memory (keyword baseline) - built-in, no dependencies", "built-in"),
        ("mem0", "Mem0 - Universal memory layer for AI agents", "pip install agent-memory-bench[mem0]"),
        ("memos", "MemOS - Memory operating system for LLMs", "pip install agent-memory-bench[memos]"),
    ]
    for name, desc, install in adapter_info:
        click.echo(f"  {name}")
        click.echo(f"    {desc}")
        click.echo(f"    Install: {install}")
        click.echo()


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def report(input_file, output):
    """Generate a leaderboard report from a previous benchmark run."""
    import json
    from agent_memory_bench.models import BenchmarkReport

    data = json.loads(Path(input_file).read_text())
    click.echo(f"Leaderboard from {data.get('timestamp', 'unknown')}:\n")
    for row in data.get("leaderboard", []):
        click.echo(
            f"  {row['adapter']:>10s} | {row['task']:<30s} | "
            f"recall@5={row['recall@5']:.4f} | mrr={row['mrr']:.4f} | "
            f"latency={row['latency_ms']:.2f}ms"
        )


if __name__ == "__main__":
    main()
