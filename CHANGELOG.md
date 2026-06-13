# Changelog

All notable changes to this project (fork) are documented in this file.

## 0.1.0 - 2025-06-13

### 🎉 Initial Fork Release

This is the first release of **Godelion**, a fork and significant enhancement of the
Darwin Gödel Machine (DGM) by Zhang et al. (Sakana AI).

### 🚀 Major Improvements

#### Usability & Accessibility
- **Local model support**: Added `ollama`, `vllm`, and `lm_studio` providers via a unified local model interface
- **YAML configuration system**: Replaced scattered CLI args with a hierarchical, layered config
- **Config override hierarchy**: `config.yaml` → `config.local.yaml` → `GODELION_*` env vars
- **Configurable fallback models**: If primary model fails, fallback models are tried automatically
- **Cost comparison guide**: Documented expected costs across provider options

#### Documentation & Onboarding
- **Complete README overhaul**: Architecture diagram, quickstart, troubleshooting, safety warnings
- **RSI seed explanation**: Detailed description of how Godelion serves as a recursive self-improvement seed
- **Configuration reference**: Documented every config section and key setting
- **Extension guide**: How to add tools, benchmarks, and custom selection algorithms
- **Safety & ethics section**: Explicit warnings, usage guidelines, risk disclosure

#### Code Quality & Packaging
- **`pyproject.toml`**: Modern Python packaging with proper metadata
- **Package structure**: Reorganized into `godelion/` package namespace
- **`run.py` entry point**: Clean entry point replacing `DGM_outer.py`
- **`.pre-commit-config.yaml`**: Code quality hooks
- **`.github/workflows/`**: CI configuration for linting and testing
- **Type hints**: Added to key functions for better IDE support
- **`__init__.py`**: Proper package initialization with version

#### Safety & Guardrails
- **Constitutional check framework**: Optional pre-approval of self-modifications
- **Protected files**: Critical config files protected from agent modification
- **Human approval gates**: Optional manual review for safety-critical changes
- **Docker network isolation**: Default `network_disabled: true` for safety
- **Resource limits**: Memory and CPU limits for Docker containers
- **Enhanced logging**: Full audit trail of all self-modifications
- **Checkpointing**: Resumable runs with configurable checkpoints

#### Error Handling & Robustness
- **Replaced bare `except: pass`** with proper error handling throughout
- **Graceful degradation**: Self-improvement failures don't crash the main loop
- **Thread-safe dataset handling**: Fixed global `dataset` variable race condition
- **Comprehensive retry logic**: Backoff and retry for API calls, Docker operations
- **Container cleanup on failure**: Ensures no orphaned containers

#### Analysis & Visualization
- **Lineage tree plotting**: Visualize the evolutionary history
- **Performance over generations**: Track accuracy across the run
- **HTML analysis reports**: Comprehensive run reports with plots
- **Export formats**: JSON, CSV, and HTML analysis exports

#### Performance
- **LLM response caching**: Optional caching of API responses (reduces costs)
- **Optimized Docker builds**: Conditional rebuild only when needed
- **Parallel worker improvements**: Better resource utilization

### Breaking Changes
- Configuration is now via `config.yaml` / `config.local.yaml` instead of CLI args only
- Output directory changed from `./output_dgm/` to `./output_godelion/`
- Docker image name changed from `dgm` to `godelion`
- Package imports changed from bare modules to `godelion.` namespace
- Default model changed to `claude-sonnet-4-20250514`

### Attribution
This project is a fork of the **Darwin Gödel Machine** (https://github.com/jennyzzt/dgm)
by Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, and Jeff Clune (Sakana AI).
The original work is described in arXiv:2505.22954.
