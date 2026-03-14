# agent-memory-bench

A standardized benchmark suite for evaluating and comparing agent memory systems.

## Why

There are now 5+ competing agent memory solutions (Mem0, MemOS, OpenViking, SimpleMem, Wax) but no standardized benchmark to compare them. Developers have no objective way to evaluate which memory system is best for their use case -- retrieval accuracy, latency, cross-session persistence, or context window efficiency.

The agent memory space exploded in Q1 2026 with multiple well-funded projects competing. Without benchmarks, the community is choosing tools based on star counts rather than capability.

## What It Does

A comprehensive benchmark suite for agent memory systems, similar to what MTEB did for embeddings or BrowseComp for browsing. It defines standard tasks (fact recall, temporal reasoning, cross-conversation continuity, contradiction detection), provides synthetic and real-world datasets, and outputs standardized leaderboards.

## MVP Roadmap

- [ ] Define 5-7 core memory benchmark tasks with evaluation metrics
- [ ] Implement adapters for Mem0, MemOS, OpenViking, and SimpleMem
- [ ] Build a CLI runner that generates a standardized leaderboard report

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Run all benchmarks against all adapters
agent-memory-bench run --all

# Run a specific task against a specific adapter
agent-memory-bench run --task fact-recall --adapter mem0

# Generate a leaderboard report
agent-memory-bench report --output leaderboard.json
```

```python
# Example Python usage
from agent_memory_bench.core import BenchmarkRunner
from agent_memory_bench.adapters.mem0 import Mem0Adapter

runner = BenchmarkRunner()
runner.register_adapter("mem0", Mem0Adapter())
results = runner.run(tasks=["fact-recall", "temporal-reasoning"])
print(results.to_leaderboard())
```

## Development

```bash
# Clone and install in dev mode
git clone <repo-url>
cd agent-memory-bench
pip install -e ".[dev]"

# Run tests
pytest

# Run the CLI
agent-memory-bench --help
```

## Project Structure

```
OSS-N001/
  README.md
  LICENSE
  pyproject.toml
  .gitignore
  src/
    agent_memory_bench/
      __init__.py
      cli.py
      core.py
      models.py
      config.py
      runner.py
      adapters/
        __init__.py
        base.py
        mem0.py
        memos.py
  tests/
    __init__.py
    test_core.py
    conftest.py
  examples/
    basic_usage.py
```

## Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

## License

MIT
