# Changelog

All notable changes to Godelion are documented in this file.

## 0.2.0 - 2026-06-16

### 🧬 Enhanced Self-Improvement Engine

- **Meta-cognitive validation**: New `validate_improvement_proposal()` performs a risk/failure-mode/impact analysis of each improvement proposal *before* running the expensive evaluation harness, rejecting clearly harmful or low-value changes early
- **Patch quality analysis**: `analyze_patch_quality()` tracks files changed, lines added/removed, and size per patch, stored in metadata for downstream diversity and quality metrics
- **Architecture introspection**: `get_architecture_summary()` provides the system with a comprehensive view of its own source structure for informed self-modification decisions

### 🌿 Evolutionary Improvements

- **`diversity_weighted` selection**: New parent selection method combines accuracy (70%) with lineage-based diversity score (30%) to prevent archive collapse
- **`keep_diverse` archive update**: New archive method adds a diversity bonus (10%) when evaluating whether a child should join the archive
- **Lineage-based diversity tracking**: `compute_diversity_scores()` computes diversity from ancestor overlap, giving high diversity to agents from distinct evolutionary branches
- **Multi-objective metadata**: Archive now stores diversity scores alongside accuracy for richer analysis

### 💾 Checkpointing & Resumability

- **Per-generation checkpoints**: `save_checkpoint()` writes generation state to `checkpoint_gen_N.json` files with full archive, children, and timestamp
- **Resumable runs**: Future support for `--continue-from` with generation-level granularity

### 🐛 Bug Fixes

- **Fixed `process_selfimprove_eval_logs` arg mismatch**: `diagnose_improvement_prompt.py` was calling the function with 3 arguments when it requires 4; also unpacking 3 return values when 4 are returned
- **Fixed `full_eval_threshold` being silently dropped**: The parameter was computed in `run.py` and passed to `self_improve()`, but was silently ignored — now saved to metadata
- **Fixed `is_compiled_self_improve` logger crash**: When `logger=None`, calling `logger.info()` would crash; now uses a safe local logging wrapper
- **Fixed `pyproject.toml` build backend**: Changed `setuptools.backends._legacy` (broken) to `setuptools.build_meta`

### 🗑️ Cleanup

- **Removed `DGM_outer.py`**: Redundant compatibility shim that only forwarded to `run.py`

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
