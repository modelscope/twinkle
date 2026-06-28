# Auto-Research

Twinkle Auto is a terminal-based intelligent training assistant that lets you **control, monitor, and debug ML training through natural language**. It combines a chat-driven AI agent with an automated health monitor that can detect and fix training failures autonomously.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│ TwinkleAuto (asyncio chat loop)                          │
│                                                          │
│ Components:                                              │
│   AgentLoop  ─── LLM tool-calling loop                   │
│   TrainingMonitor ─── periodic health check & auto-fix   │
│   LocalConnection ─── file-system based communication   │
│   SkillManager ─── async plugin loading                 │
└──────────────────────────────────────────────────────────┘
```

## Installation & Launch

Auto is part of the `twinkle-client` package:

```bash
pip install twinkle-client
```

### Command-Line Usage

```bash
# Basic launch (uses default local Ollama endpoint)
twinkle-auto

# Specify LLM backend
twinkle-auto --llm-base-url http://localhost:11434/v1 --llm-model qwen3.5

# Attach to an existing training run
twinkle-auto --run-id my-grpo-run

# Use a remote API (e.g., OpenAI-compatible)
twinkle-auto --llm-base-url https://api.example.com/v1 --llm-api-key sk-xxx --llm-model gpt-4o

# Enable debug logging
twinkle-auto --verbose
```

Or run as a Python module:

```bash
python -m twinkle_client.auto
```

### CLI Options

| Option | Env Var | Default | Description |
|--------|---------|---------|-------------|
| `--run-id`, `-r` | `TWINKLE_AUTO_RUN_ID` | None | Attach to an existing training run |
| `--llm-base-url` | `TWINKLE_LLM_BASE_URL` | `http://localhost:11434/v1` | LLM API base URL |
| `--llm-model` | `TWINKLE_LLM_MODEL` | `qwen3.5` | LLM model name |
| `--llm-api-key` | `TWINKLE_LLM_API_KEY` | `not-needed` | LLM API key |
| `--verbose`, `-v` | `TWINKLE_AUTO_VERBOSE` | `False` | Enable DEBUG logging |
| `--version`, `-V` | — | — | Show version and exit |

## Chat Agent

The core of Auto is an **LLM-powered tool-calling agent** (`AgentLoop`) that processes natural language commands through an OpenAI-compatible API. The agent maintains conversation history with automatic pruning (last 50 messages) and supports up to 10 tool-calling rounds per interaction.

### What You Can Say

**Training lifecycle:**
- *"List my training runs"*
- *"Start a new GRPO training with Qwen3.5-4B on gsm8k"*
- *"Pause the current run"*
- *"Resume training"*
- *"Stop training"*

**Server management:**
- *"Start the server with Qwen3.5-4B and a Qwen3.5-72B sampler on 2 GPUs"*
- *"Shut down the server"*
- *"How many GPUs are available?"*

**Monitoring & analysis:**
- *"How is the training going?"*
- *"Show me the reward-related metrics"*
- *"Zoom into steps 100-200"*
- *"Reset the chart view"*

**Search:**
- *"Search for math datasets"*
- *"Find Qwen models on ModelScope"*

### Available Tools

The agent has access to 13 built-in tools:

| Tool | Description |
|------|-------------|
| `list_training_runs` | List all training runs |
| `get_training_status` | Get detailed status and recent metrics |
| `start_server` | Start Ray cluster + Twinkle Server (idempotent) |
| `shutdown_server` | Shut down server and release GPU resources |
| `start_training` | Create and launch a new training run |
| `select_run` | Switch monitoring to a different run |
| `pause_training` | Pause training (SIGKILL, server retains state) |
| `resume_training` | Resume by re-launching the client script |
| `stop_training` | Stop training (SIGTERM, saves checkpoint) |
| `update_script` | Update training script with version archiving |
| `list_supported_models` | Query server for available models |
| `search_datasets` | Search ModelScope for datasets |
| `search_models` | Search ModelScope for models |
| `zoom_metrics` | Adjust metrics chart view range |
| `select_metrics` | Choose which metrics to display (max 4) |
| `get_cluster_info` | Get GPU/cluster resource info |

### Server Startup

The `start_server` tool automates a multi-step pipeline:

1. **GPU detection** — `nvidia-smi` hardware scan
2. **GPU allocation** — partition GPUs between training model and samplers
3. **Config generation** — auto-create `server_config.yaml`
4. **Ray cluster startup** — multi-node GPU partitioning with isolated `CUDA_VISIBLE_DEVICES`
5. **Server launch** — start Twinkle Server as background process
6. **Health check** — poll `/api/v1/healthz` until ready

Multi-model topology is supported: 1 training model + N sampler/teacher models.

### Skills System

Auto supports extensible skill plugins loaded from three sources:

1. **Bundled skills** — shipped inside `twinkle_client/skills/bundled/`
2. **User-local skills** — `~/.cache/twinkle/auto/skills/local/`
3. **Community skills** — fetched from ModelScope (best-effort, 10s timeout)

Skills are loaded asynchronously after startup and injected into the agent's system prompt. The agent is usable immediately even before skills finish loading.

## Training Monitor (Auto-Fix)

The `TrainingMonitor` is a background service that runs every **30 seconds**, collecting all available signals about the current training run and feeding them to the LLM for analysis.

### Collected Signals

- **Process status**: alive / dead / unknown
- **output.log tail**: last 1500 chars (prioritizes tracebacks)
- **Metrics**: recent entries + first-half vs second-half trend analysis
- **Stall duration**: seconds since last metric was produced
- **Current train.py**: full script source (for accurate fixes)

### Decision Framework

The LLM classifies each check into one of three actions:

| Decision | When | Action |
|----------|------|--------|
| **LGTM** | Training progressing normally | No action |
| **WARNING** | Loss plateau, reward hacking, KL explosion, etc. | Relay observation to user |
| **FIX** | Script crashed, process dead with traceback | Auto-fix and restart |

### Auto-Fix Pipeline

When a FIX is needed:

1. LLM outputs diagnosis + complete fixed script
2. Monitor archives the old `train.py` as `train_v{N}.py`
3. Writes the fixed script as the new `train.py`
4. Re-launches training via `resume_training`
5. Resets stall tracking for the new attempt

Safety guardrails:
- Max **3 auto-fix attempts** per run (prevents infinite retry loops)
- Fix attempts are tracked per `run_id`
- Snapshot deduplication avoids re-analyzing unchanged states

## File-Based Connection

Auto communicates with training processes through the local filesystem:

```
~/.cache/twinkle/{run_id}/
├── meta.json       — run metadata (model_id, config, status, pid)
├── metrics.jsonl   — one JSON object per step (incremental)
├── output.log      — combined stdout+stderr from training
├── train.py        — current active training script
└── train_v{N}.py   — archived previous script versions
```

### Training Control Model

In Server Mode, the Twinkle Server retains all model/optimizer state in GPU memory:

- **Pause** = kill client process (SIGKILL) — server state preserved
- **Resume** = re-launch client script — seamlessly continues training
- **Stop** = SIGTERM — triggers checkpoint saving then exits
- **Shut down server** = releases GPU resources, **destroys** model state

## TrainingRuntime (Script Integration)

Training scripts use `TrainingRuntime` to integrate with Auto:

```python
from twinkle_client.auto.runtime import TrainingRuntime

rt = TrainingRuntime(run_id='my-grpo-run')
rt.start(model_id='Qwen/Qwen3.5-4B', config={'lr': 1e-5})
rt.register_graceful_shutdown(model, dataloader)

for step, batch in enumerate(dataloader):
    # ... training logic ...
    rt.log_metrics(step=step, loss=loss, reward=reward, grad_norm=gn, lr=lr)
    rt.log(f'Completed step {step}, loss={loss:.4f}')

rt.finish()
```

### Key Methods

| Method | Description |
|--------|-------------|
| `start(model_id, config, script_path)` | Initialize run directory and metadata |
| `log_metrics(**kwargs)` | Write metrics entry to `metrics.jsonl` |
| `log(message)` | Print log message (captured as `output.log`) |
| `get_resume_info()` | Get `last_step` for resuming from checkpoint |
| `finish(status)` | Mark training as finished, close files |
| `register_graceful_shutdown(model, dataloader)` | Register SIGTERM handler that saves checkpoint |

### Resume Support

`TrainingRuntime` automatically saves training progress to `meta.json` (throttled to every 5 seconds). Scripts can use `get_resume_info()` to resume from the last saved step:

```python
rt = TrainingRuntime(run_id='my-run')
resume = rt.get_resume_info()
global_step = resume['last_step']

if global_step > 0:
    dataloader.skip_consumed_samples(global_step * BATCH_SIZE)
    print(f'Resuming from step {global_step}')
```

### Graceful Shutdown

When `register_graceful_shutdown()` is called, a SIGTERM handler is installed that:

1. Saves model checkpoint (LoRA weights + optimizer state)
2. Saves dataloader position (`consumed_train_samples`)
3. Logs the checkpoint path
4. Marks training as `stopped` and exits

## Logging

All logs are written to `./auto.log` (current working directory):

- Rotated at 5MB with 3 backups
- **No console output** — all output goes to the log file
- Use `--verbose` for DEBUG level logging
