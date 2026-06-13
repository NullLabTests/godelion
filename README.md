<h1 align="center">
  🦁 Godelion
</h1>

<p align="center">
  <strong>Open-Ended Evolution of Self-Improving Coding Agents</strong><br>
  <em>Recursive Self-Improvement — Empirically Validated — Safely Sandboxed</em>
</p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge" alt="License"></a>
  <a href="https://arxiv.org/abs/2505.22954"><img src="https://img.shields.io/badge/Paper-arXiv%202505.22954-b31b1b.svg?style=for-the-badge" alt="Paper"></a>
  <a href="https://sakana.ai/dgm/"><img src="https://img.shields.io/badge/Blog-Sakana%20AI-8D6748?style=for-the-badge" alt="Blog"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Docker-Required-2496ED?style=for-the-badge&logo=docker" alt="Docker">
</p>

---

> **⚠️ WARNING: Self-Improving Systems Can Be Dangerous**
>
> Godelion is a research prototype that **rewrites its own source code** and validates those changes on
> live benchmarks. This is a form of **Recursive Self-Improvement (RSI)** — the system gets better at
> improving itself with every generation. While currently constrained to coding benchmarks, a sufficiently
> capable self-improving agent could theoretically produce modifications outside its intended scope.
>
> **You MUST run this only in isolated environments** such as:
> - **RunPod** or similar GPU cloud with strict network policies
> - **Docker containers** with network disabled and no internet access
> - **Air-gapped machines** disconnected from production systems
> - **Ephemeral VMs** destroyed after each experiment
>
> Never run this on production systems, machines with sensitive data, or environments that could affect
> the real world if the agent escapes its sandbox. Always review model-generated patches before
> applying them outside the Docker sandbox. The authors and contributors of this project accept **zero
> liability** for misuse or unintended consequences.

---

## 🧬 What is Godelion?

Godelion is an **evolutionary self-improving coding system** — it writes code, evaluates how well that code
solves programming tasks (SWE-bench, Polyglot), then **rewrites its own source code** to fix the tasks it
failed. Each generation produces children that are empirically validated; the fittest survive. Over many
generations, the coding agent evolves to become more capable — not through human engineering but through
**Darwinian evolution applied to software itself**.

Think of it as an **RSI (Recursive Self-Improvement) seed**: a minimal starting program that can grow
its own capabilities by rewriting itself, validating each change against ground-truth benchmarks, and
compounding improvements over generations.

### Key Features

| Feature | Description |
|---------|-------------|
| **🧬 Darwinian Evolution** | Parents are selected by fitness; children inherit the best traits |
| **🔄 Recursive Self-Improvement** | The system rewrites its own `coding_agent.py`, tools, and prompts |
| **✅ Empirical Validation** | Every change is tested against real SWE-bench/Polyglot tasks |
| **🛡️ Docker Sandboxing** | All execution happens in isolated containers with optional network disable |
| **📊 Diversity Preservation** | An archive of past agents prevents premature convergence |
| **🔌 Multi-Model Support** | Anthropic, OpenAI, DeepSeek, Ollama, vLLM, LM Studio, OpenRouter |
| **💻 Local-First** | Run with free local models via Ollama/vLLM — no API keys required |
| **📈 Analysis Tooling** | Lineage tracking, performance plots, evolution visualization |
| **🔐 Safety-First** | Constitutional checks, human approval gates, protected files |
| **📦 Modern Python** | `pyproject.toml`, type hints, pre-commit hooks, CI config |

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [How It Works](#-how-it-works)
- [Configuration](#-configuration)
- [Running with Local Models](#-running-with-local-models)
- [Running Experiments](#-running-experiments)
- [Analysis & Visualization](#-analysis--visualization)
- [Project Structure](#-project-structure)
- [Safety & Ethics](#-safety--ethics)
- [Extending Godelion](#-extending-godelion)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Changelog](./CHANGELOG.md)
- [Citation](#-citation)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker (with `docker run hello-world` working)
- At least one LLM API key (or a local Ollama/vLLM instance)

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/godelion.git
cd godelion

python3 -m venv venv
source venv/bin/activate

# Install Godelion (editable mode)
pip install -e .

# Optional: for analysis & plotting
sudo apt-get install graphviz graphviz-dev
pip install -e ".[dev,analysis]"
```

### 2. Configure

```bash
cp config.yaml config.local.yaml
# Edit config.local.yaml with your model choice and API keys
```

Set your API keys (or skip if using local models):

```bash
export ANTHROPIC_API_KEY='sk-ant-...'
export OPENAI_API_KEY='sk-...'
# or
export DEEPSEEK_API_KEY='sk-...'
```

### 3. Prepare Benchmarks

```bash
# SWE-bench
cd swe_bench
git clone https://github.com/princeton-nlp/SWE-bench.git
cd SWE-bench
git checkout dc4c087c2b9e4cefebf2e3d201d27e36
pip install -e .
cd ../../

# Polyglot (optional)
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
python -m polyglot.prepare_polyglot_dataset
```

### 4. Run Godelion

```bash
python run.py
```

That's it. The system will start evolving. Output is saved to `./output_godelion/`.

---

## 🧠 How It Works

### The Evolutionary Loop

```
┌─────────────────────────────────────────────────────────┐
│                     GENERATION N                          │
│                                                          │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────┐  │
│   │  ARCHIVE  │───▶│   SELECT     │───▶│ SELF-IMPROVE │  │
│   │ (parents) │    │   PARENTS    │    │ (agent mutates│  │
│   │           │    │ (fitness +   │    │  its own code)│  │
│   │           │    │  diversity)  │    │              │  │
│   └──────────┘    └──────────────┘    └──────┬───────┘  │
│                                              │          │
│   ┌──────────┐    ┌──────────────┐    ┌──────▼───────┐  │
│   │  UPDATE  │◀───│   FILTER     │◀───│    EVAL      │  │
│   │ ARCHIVE  │    │ (compile OK) │    │ (SWE-bench   │  │
│   │          │    │              │    │  Polyglot)   │  │
│   └──────────┘    └──────────────┘    └──────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
                     GENERATION N+1
```

### The Self-Improvement Process

1. **Diagnose**: The system reviews logs of failed coding tasks, identifies root causes
2. **Propose**: An LLM proposes a modification to `coding_agent.py`, tools, or prompts
3. **Implement**: The proposed change is implemented inside a Docker container
4. **Validate**: The modified agent is tested on held-out benchmark tasks
5. **Compete**: If the change improves performance (or at least doesn't regress), it enters the archive
6. **Repeat**: Future generations can build on any archived agent

### RSI Seed Growth

This project is a **minimal seed for Recursive Self-Improvement**. The initial agent is capable enough to:

- Understand its own source code structure
- Diagnose its failures on coding benchmarks
- Propose and implement changes to itself
- Evaluate whether those changes actually help

As generations progress, the system becomes better at all of these tasks. The **meta-cognitive** ability to
improve its own improvement process is the key — this is what makes it an RSI seed rather than just another
code generator.

---

## ⚙️ Configuration

Godelion uses a **hierarchical YAML configuration system**:

1. `config.yaml` — Default configuration (committed to repo)
2. `config.local.yaml` — Local overrides (gitignored, never committed)
3. Environment variables — `GODELION_*` vars override YAML settings (e.g., `GODELION_EVOLUTION__MAX_GENERATIONS=100`)

### Key Configuration Sections

| Section | Key Settings |
|---------|-------------|
| `llm` | Model names, API keys, fallbacks, retry policy |
| `local` | Local model provider, URL, model names |
| `evolution` | Generations, selection method, archive policy |
| `evaluation` | Number of evals, shallow mode, thresholds |
| `docker` | Image name, timeouts, resource limits, network |
| `safety` | Constitutional checks, protected files, approvals |
| `logging` | Log level, output directory, retention |
| `checkpoint` | Checkpoint interval, compression |
| `benchmark` | Dataset paths, subset sizes |
| `tools` | Custom tool directories, disabled tools |

See [config.yaml](./config.yaml) for the complete reference.

---

## 🤖 Running with Local Models

One of Godelion's standout features is full support for **free, local, offline models**.

### Option 1: Ollama

```yaml
# config.local.yaml
local:
  enabled: true
  provider: "ollama"
  base_url: "http://localhost:11434/v1"
  coding_model: "deepseek-coder-v2"
  diagnose_model: "qwen2.5-coder:32b"
```

```bash
ollama pull deepseek-coder-v2
ollama pull qwen2.5-coder:32b
python run.py
```

### Option 2: vLLM

```yaml
local:
  enabled: true
  provider: "vllm"
  base_url: "http://localhost:8000/v1"
  coding_model: "deepseek-ai/DeepSeek-Coder-V2-Instruct"
  diagnose_model: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
```

### Option 3: LM Studio

```yaml
local:
  enabled: true
  provider: "lm_studio"
  base_url: "http://localhost:1234/v1"
  coding_model: "local-model"
  diagnose_model: "local-model"
```

### Cost Comparison

| Provider | Coding Model (per 1M tokens) | Diagnose Model | Cost per 80-gen run |
|----------|------------------------------|----------------|---------------------|
| Claude Sonnet 4 | $3.00 / $15.00 | Same | ~$2,000-5,000 |
| GPT-4o | $2.50 / $10.00 | Same | ~$1,500-4,000 |
| DeepSeek API | $0.27 / $1.10 | Same | ~$200-500 |
| **Local Ollama** | **$0.00 / $0.00** | **Same** | **$0 (GPU cost only)** |

---

## 🧪 Running Experiments

### Basic Run

```bash
# Default: 80 generations, 2 children per generation
python run.py
```

### Custom Run

```bash
# 40 generations, 4 children per generation, 4 parallel workers
python run.py --config config.local.yaml --max-generation 40 --selfimprove-size 4 --selfimprove-workers 4
```

### Baseline Comparisons

```bash
# No self-improvement (just the initial agent)
python run.py --run-baseline no_selfimprove

# No Darwin selection (always use the latest commit)
python run.py --run-baseline no_darwin
```

### Continuing a Previous Run

```bash
python run.py --continue-from ./output_godelion/20250101_120000_123456
```

### Polyglot Benchmark

```bash
python run.py --polyglot
```

### Shallow Evaluation (Faster, Less Accurate)

```bash
python run.py --shallow-eval
```

---

## 📊 Analysis & Visualization

### Lineage Tree

```bash
python -m analysis.plot_lineage --output-dir ./output_godelion/20250101_120000_123456
```

### Performance Over Generations

```bash
python -m analysis.plot_performance --output-dir ./output_godelion/20250101_120000_123456
```

### Full Analysis Report

```bash
python -m analysis.report --output-dir ./output_godelion/20250101_120000_123456 --format html
```

---

## 📁 Project Structure

```
godelion/
├── run.py                    # Main entry point (replaces DGM_outer.py)
├── config.yaml               # Default configuration
├── config.local.yaml         # Local overrides (gitignored)
├── godelion/
│   ├── __init__.py           # Package init, version
│   ├── config.py             # Configuration loader
│   ├── llm.py                # LLM client factory (multi-provider)
│   ├── llm_withtools.py      # LLM + tool calling orchestration
│   ├── coding_agent.py       # The coding agent (evolves!)
│   ├── coding_agent_polyglot.py  # Polyglot-specific agent
│   ├── self_improve_step.py  # Single self-improvement iteration
│   └── engine.py             # Main evolutionary loop
├── tools/
│   ├── bash.py               # Bash shell tool
│   └── edit.py               # File editing tool
├── prompts/
│   ├── self_improvement_prompt.py
│   ├── diagnose_improvement_prompt.py
│   ├── testrepo_prompt.py
│   └── tooluse_prompt.py
├── utils/
│   ├── common_utils.py
│   ├── docker_utils.py
│   ├── eval_utils.py
│   ├── evo_utils.py
│   ├── git_utils.py
│   └── swe_log_parsers.py
├── swe_bench/                # SWE-bench integration
├── polyglot/                 # Polyglot benchmark integration
├── analysis/                 # Analysis & visualization
│   ├── plot_lineage.py
│   ├── plot_performance.py
│   └── report.py
├── tests/                    # Test suite
├── Dockerfile                # Container definition
└── pyproject.toml            # Modern Python packaging
```

---

## 🛡️ Safety & Ethics

### Why This Matters

Self-improving systems are the **most important and most dangerous** technology on the horizon. A system
that can rewrite itself without human supervision could, in principle, undergo rapid capability gains —
and if misaligned, cause catastrophic harm. This project is a research tool for studying such systems
**safely and responsibly**.

### Built-in Safeguards

1. **🧪 Docker Sandboxing**: All code execution happens in ephemeral containers
2. **🔒 Network Isolation**: Containers can run with `--network none`
3. **👁️ Human Review**: All proposed patches are logged for inspection
4. **📝 Constitutional Checks**: Optional pre-approval of self-modifications
5. **🛡️ Protected Files**: Critical config files cannot be modified by the agent
6. **📊 Empirical Validation**: No change is accepted without measurable improvement
7. **⏱️ Timeouts**: Hard limits prevent runaway self-improvement
8. **📈 Full Audit Trail**: Every generation, every patch, every evaluation is logged

### Risks You Accept

- The system **will modify its own source code** — these changes are automatically generated
- Patches may contain subtle bugs even if they pass the benchmarks
- The system could potentially escape its Docker sandbox (extremely unlikely but not impossible)
- Future, more capable models could accelerate improvement beyond safe rates

### Ethical Usage Guidelines

- **Always run in isolated environments** (RunPod, dedicated cloud VMs, air-gapped)
- **Review generated patches** before using them outside the experiment
- **Set conservative limits** on generations and parallel workers
- **Monitor the system** during operation — watch for unexpected behavior
- **Do not deploy** self-improved agents to production without thorough testing
- **Do not connect** the system to the internet inside the container (set `network_disabled: true`)
- **Share results responsibly** — this is research, not a product

---

## 🔌 Extending Godelion

### Adding a New Tool

```python
# tools/my_tool.py
def tool_info():
    return {
        "name": "my_tool",
        "description": "Does something useful",
        "input_schema": {
            "type": "object",
            "properties": {
                "param": {"type": "string", "description": "A parameter"}
            },
            "required": ["param"]
        }
    }

def tool_function(param: str) -> str:
    return f"Result: {param}"
```

That's it. The tool is auto-discovered and available to the agent.

### Adding a New Benchmark

Create a new directory following the `swe_bench/` or `polyglot/` pattern, then add it to
`config.yaml` under `benchmark`.

### Customizing the Evolutionary Algorithm

Override the selection methods in `godelion/engine.py`:

```python
def custom_selection(archive, candidates, size):
    # Your custom selection logic here
    return selected_parents
```

---

## 🔧 Troubleshooting

### "Docker is not running"
```bash
sudo systemctl start docker
sudo usermod -aG docker $USER
newgrp docker
```

### "Model response is too long"
Increase context window or reduce the prompt size. Check `config.yaml` → `llm.api.max_tokens`.

### "Self-improvement times out"
Increase `docker.timeout_seconds` or reduce `evolution.selfimprove_size`.

### "Container runs out of memory"
Set `docker.memory_limit` in config, e.g., `"8g"`.

### "No evaluation files found"
Run `python -m swe_bench.harness` once to generate baseline evaluations.

### "Cannot connect to local model"
Verify the model is running: `curl http://localhost:11434/v1/models` (Ollama) or
`curl http://localhost:8000/v1/models` (vLLM).

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
pip install -e ".[dev]"
pre-commit install
pytest
```

### Pre-commit Hooks

This project uses pre-commit for code quality. Run `pre-commit run --all-files` before committing.

---

## 📜 Changelog

See [CHANGELOG.md](./CHANGELOG.md) for the full history of changes and improvements.

---

## 📖 Citation

This project is a fork and evolution of the **Darwin Gödel Machine** (arXiv:2505.22954) by
Zhang, Hu, Lu, Lange, and Clune (Sakana AI). If you use this work, please cite both:

```bibtex
@software{godelion2025,
  title = {Godelion: Open-Ended Evolution of Self-Improving Coding Agents},
  author = {Godelion Contributors},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/godelion}
}

@article{zhang2025darwin,
  title = {Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents},
  author = {Zhang, Jenny and Hu, Shengran and Lu, Cong and Lange, Robert and Clune, Jeff},
  journal = {arXiv preprint arXiv:2505.22954},
  year = {2025}
}
```

---

<p align="center">
  <strong>Built with 🔬 for open research into safe, self-improving systems.</strong>
</p>
