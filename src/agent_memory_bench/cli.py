"""CLI entry point for the benchmark suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from agent_memory_bench.config import BenchmarkConfig
from agent_memory_bench.core import BenchmarkRunner, TASK_GENERATORS
from agent_memory_bench.models import TaskType
from agent_memory_bench.runner import print_report, save_report


AVAILABLE_TASKS = [t.value for t in TASK_GENERATORS]


def _resolve_adapter(name: str, **kwargs: Any):
    """Resolve adapter by name, importing and instantiating it."""
    match name:
        case "mem0":
            from agent_memory_bench.adapters.mem0 import Mem0Adapter
            return Mem0Adapter(**kwargs)
        case "memos":
            from agent_memory_bench.adapters.memos import MemOSAdapter
            return MemOSAdapter(**kwargs)
        case _:
            raise click.BadParameter(
                f"Unknown adapter: {name}. Available: mem0, memos"
            )


SAMPLE_CONFIG = {
    "tasks": ["fact-recall", "temporal-reasoning"],
    "adapters": [
        {"name": "mem0", "adapter_class": "mem0", "params": {}},
        {"name": "memos", "adapter_class": "memos", "params": {}},
    ],
    "num_samples": 50,
    "seed": 42,
    "output_dir": "./results",
}


@click.group()
@click.version_option(package_name="agent-memory-bench")
def main():
    """Agent Memory Benchmark Suite.

    A standardized benchmark suite for evaluating and comparing agent memory systems.
    """


@main.command()
@click.option("--task", "-t", multiple=True, help=f"Task(s) to run. Available: {AVAILABLE_TASKS}")
@click.option("--adapter", "-a", multiple=True, help="Adapter(s) to benchmark. Available: mem0, memos")
@click.option("--all", "run_all", is_flag=True, help="Run all tasks against all adapters")
@click.option("--num-samples", "-n", default=None, type=int, help="Number of samples per task")
@click.option("--seed", "-s", default=None, type=int, help="Random seed for reproducibility")
@click.option("--output", "-o", type=click.Path(), help="Save report to JSON file")
@click.option("--config", "-c", type=click.Path(exists=True), help="Benchmark config JSON file")
def run(task, adapter, run_all, num_samples, seed, output, config):
    """Run benchmark tasks against memory system adapters.

    When --config is provided, settings are loaded from the config file. CLI flags
    override config file values when both are provided.
    """
    # Load config file if provided
    cfg = BenchmarkConfig.from_file(Path(config)) if config else BenchmarkConfig()

    # CLI flags override config values
    effective_seed = seed if seed is not None else cfg.seed
    effective_num_samples = num_samples if num_samples is not None else cfg.num_samples
    effective_output = output if output else (str(cfg.output_dir) if config else None)

    runner = BenchmarkRunner(seed=effective_seed, num_samples=effective_num_samples)

    # Register adapters: CLI flags take precedence over config
    if adapter:
        adapter_names = list(adapter)
        for name in adapter_names:
            try:
                a = _resolve_adapter(name)
                runner.register_adapter(name, a)
            except ImportError as e:
                click.echo(f"Warning: Skipping adapter '{name}': {e}", err=True)
    elif run_all:
        for name in ["mem0", "memos"]:
            try:
                a = _resolve_adapter(name)
                runner.register_adapter(name, a)
            except ImportError as e:
                click.echo(f"Warning: Skipping adapter '{name}': {e}", err=True)
    elif cfg.adapters:
        for adapter_cfg in cfg.adapters:
            try:
                a = _resolve_adapter(adapter_cfg.adapter_class, **adapter_cfg.params)
                runner.register_adapter(adapter_cfg.name, a)
            except ImportError as e:
                click.echo(f"Warning: Skipping adapter '{adapter_cfg.name}': {e}", err=True)
    else:
        raise click.UsageError("Specify --adapter, use --all, or provide --config with adapters.")

    # Resolve tasks: CLI flags take precedence over config
    if task:
        tasks = list(task)
    elif run_all:
        tasks = AVAILABLE_TASKS
    elif config:
        tasks = [t.value for t in cfg.tasks]
    else:
        raise click.UsageError("Specify --task, use --all, or provide --config with tasks.")

    click.echo(f"Running {len(tasks)} task(s) against {len(runner._adapters)} adapter(s)...")
    report = runner.run(tasks=tasks)
    print_report(report)

    if effective_output:
        save_report(report, Path(effective_output))
        click.echo(f"Report saved to {effective_output}")


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


@main.command()
@click.option("--output", "-o", type=click.Path(), help="Write sample config to file instead of stdout")
def config(output):
    """Generate a sample benchmark configuration file."""
    config_json = json.dumps(SAMPLE_CONFIG, indent=2)
    if output:
        Path(output).write_text(config_json + "\n")
        click.echo(f"Sample config written to {output}")
    else:
        click.echo(config_json)


if __name__ == "__main__":
    main()
