"""Benchmark runner with output formatting and report generation."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from agent_memory_bench.core import BenchmarkRunner
from agent_memory_bench.models import BenchmarkReport


def format_leaderboard(report: BenchmarkReport) -> Table:
    """Format a benchmark report as a Rich table."""
    table = Table(title="Agent Memory Benchmark Leaderboard")
    table.add_column("Adapter", style="cyan", no_wrap=True)
    table.add_column("Task", style="magenta")
    table.add_column("Recall@5", justify="right", style="green")
    table.add_column("MRR", justify="right", style="green")
    table.add_column("Latency (ms)", justify="right", style="yellow")
    table.add_column("Store (ms)", justify="right", style="yellow")
    table.add_column("Success", justify="right", style="blue")

    for row in report.to_leaderboard():
        table.add_row(
            row["adapter"],
            row["task"],
            f"{row['recall@5']:.4f}",
            f"{row['mrr']:.4f}",
            f"{row['latency_ms']:.2f}",
            f"{row['store_latency_ms']:.2f}",
            f"{row['success_rate']:.0%}",
        )

    return table


def save_report(report: BenchmarkReport, output_path: Path) -> None:
    """Save a benchmark report to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp": report.timestamp.isoformat(),
        "leaderboard": report.to_leaderboard(),
        "metadata": report.metadata,
    }
    output_path.write_text(json.dumps(data, indent=2))


def print_report(report: BenchmarkReport) -> None:
    """Print a benchmark report to the console."""
    console = Console()
    table = format_leaderboard(report)
    console.print()
    console.print(table)
    console.print()

    leaderboard = report.to_leaderboard()
    if leaderboard:
        # Summary by adapter
        adapters: dict[str, list] = {}
        for row in leaderboard:
            adapters.setdefault(row["adapter"], []).append(row)

        console.print("[bold]Summary by Adapter:[/bold]")
        for adapter_name, rows in adapters.items():
            avg_recall = sum(r["recall@5"] for r in rows) / len(rows)
            avg_mrr = sum(r["mrr"] for r in rows) / len(rows)
            avg_latency = sum(r["latency_ms"] for r in rows) / len(rows)
            console.print(
                f"  {adapter_name}: recall@5={avg_recall:.4f}  "
                f"mrr={avg_mrr:.4f}  latency={avg_latency:.2f}ms"
            )
        console.print()
