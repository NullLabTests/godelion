# Contributing to Godelion

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/godelion.git
cd godelion
pip install -e ".[dev]"
pre-commit install
```

## Code Style

- We use `ruff` for linting and formatting
- Run `ruff check . && ruff format .` before committing
- Pre-commit hooks will run these automatically

## Testing

```bash
pytest tests/ -v
```

## Pull Requests

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Add or update tests
5. Run `pre-commit run --all-files`
6. Submit a PR with a clear description

## Adding Features

### New Tools
Create a file in `tools/` with `tool_info()` and `tool_function()`. See `tools/edit.py` for reference.

### New Benchmarks
Add a new benchmark directory following `swe_bench/` pattern, register in `config.yaml`.

### New Model Providers
Add provider support in `godelion/llm.py` in the `create_client()` function.

## Code of Conduct

Be respectful, constructive, and collaborative. This is open research.
