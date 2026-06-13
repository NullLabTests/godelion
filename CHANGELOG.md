# Changelog

All notable changes to Godelion are documented in this file.

## 0.1.0 - 2025-06-13

### 🎉 Initial Release

This is the first public release of **Godelion**, an open-ended evolutionary
self-improving coding agent system.

### 🚀 Major Features

#### Core System
- Darwinian evolution of coding agents with fitness-based selection
- Recursive self-improvement: the system rewrites its own source code
- Empirical validation against SWE-bench and Polyglot benchmarks
- Archive-based diversity preservation against premature convergence

#### Usability & Accessibility
- **Local model support**: `ollama`, `vllm`, and `lm_studio` providers via a unified interface
- **YAML configuration system**: Hierarchical, layered config with env var overrides
- **Configurable fallback models**: Automatic retry with fallback models on failure
- **Cost comparison guide**: Documented expected costs across provider options

#### Documentation & Onboarding
- Complete README with architecture diagram, quickstart, troubleshooting
- RSI seed explanation with detailed description of recursive self-improvement
- Configuration reference documenting every section and key setting
- Extension guide for adding tools, benchmarks, and custom algorithms
- Safety & ethics section with explicit warnings and usage guidelines

#### Code Quality & Packaging
- `pyproject.toml` modern Python packaging
- Package namespace `godelion/` with clean `run.py` entry point
- Pre-commit hooks for code quality
- CI configuration for linting and testing
- Type hints on key functions
- Comprehensive test suite

#### Safety & Guardrails
- Constitutional check framework for self-modifications
- Protected files preventing agent modification of critical config
- Human approval gates for safety-critical changes
- Docker network isolation (default `network_disabled: true`)
- Resource limits for Docker containers
- Full audit trail of all self-modifications
- Checkpointing with resumable runs

#### Error Handling & Robustness
- Proper error handling throughout (replaced bare `except: pass`)
- Graceful degradation on self-improvement failures
- Thread-safe dataset handling
- Comprehensive retry logic with backoff
- Container cleanup on failure

#### Analysis & Visualization
- Lineage tree plotting (Graphviz)
- Performance tracking over generations
- HTML analysis reports
- CSV export of run metrics

#### Performance
- Conditional Docker rebuilds
- Parallel worker improvements
- LLM response caching support

### Compatibility Notes
- Configuration via `config.yaml` / `config.local.yaml`
- Output directory: `./output_godelion/`
- Docker image: `godelion`
- Package namespace: `godelion.*`
