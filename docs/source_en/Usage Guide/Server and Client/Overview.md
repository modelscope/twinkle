# Overview

Twinkle provides a complete HTTP Server/Client architecture that supports deploying models as services and remotely calling them through clients to complete training, inference, and other tasks. This architecture decouples **model hosting (Server side)** and **training logic (Client side)**, allowing multiple users to share the same base model for training.

## Core Concepts

- **Server side**: Deployed based on Ray Serve, hosts model weights and inference/training computation. The Server is responsible for managing model loading, forward/backward propagation, weight saving, sampling inference, etc. A single Server simultaneously supports both Twinkle Client and Tinker Client connections.
- **Client side**: Runs locally, responsible for data preparation, training loop orchestration, hyperparameter configuration, etc. The Client communicates with the Server via HTTP, sending data and commands.

### Model Backends

Model loading supports three backends:

| Backend | backend | Description |
|---------|---------|-------------|
| **Transformers** | `transformers` | Based on HuggingFace Transformers, suitable for most scenarios |
| **Megatron** | `megatron` | Based on Megatron-LM, suitable for ultra-large-scale model training, supports more efficient parallelization strategies |
| **Mock** | `mock` | Numpy-only mock backend for CPU-only development and testing |

### Two Client Modes

| Client | Initialization Method | Description |
|--------|---------|------|
| **Twinkle Client** | `init_twinkle_client` | Native client, simply change `from twinkle import` to `from twinkle_client import` to migrate local training code to remote calls |
| **Tinker Client** | `init_tinker_client` | Patches Tinker SDK, allowing existing Tinker training code to be directly reused |

## How to Choose

### Client Mode Selection

| Scenario | Recommendation |
|------|------|
| Existing Twinkle local training code, want to switch to remote | Twinkle Client — only need to change import paths |
| Existing Tinker training code, want to reuse | Tinker Client — only need to initialize patch |
| New project | Twinkle Client — simpler API |

### Model Backend Selection

| Scenario | Recommendation |
|----------|----------------|
| 7B/14B and other medium-small scale models | Transformers backend (`backend: transformers`) |
| Ultra-large-scale models requiring advanced parallelization strategies | Megatron backend (`backend: megatron`) |
| Rapid experimentation and prototype verification | Transformers backend (`backend: transformers`) |
| CPU-only development/testing | Mock backend (`backend: mock`) |

## Cookbook Reference

Complete runnable examples are located in the `cookbook/` directory:

```
cookbook/
├── observability/                  # Observability (Grafana + OTLP)
│   ├── docker-compose.yaml         # One-command LGTM stack
│   └── README.md
├── client/
│   ├── server/                     # Server startup configuration
│   │   ├── transformer/            # Transformers backend
│   │   │   ├── run.sh
│   │   │   ├── server_config.yaml
│   │   │   └── server_config_e2e.yaml
│   │   ├── megatron/               # Megatron backend
│   │   │   ├── run.sh
│   │   │   ├── server_config.yaml
│   │   │   └── server_config_4b.yaml
│   │   └── mock/                   # Mock backend (CPU-only quick start)
│   │       └── server_config.yaml
├── twinkle/                        # Twinkle Client examples
│   ├── dpo.py                      # DPO training client
│   ├── embedding.py                # Embedding training client
│   ├── multi_modal.py              # Multi-modal training client
│   ├── multi_turn_rollout.py       # Multi-turn rollout client
│   ├── sample.py                   # Inference sampling client
│   ├── self_cognition.py           # Self-cognition training client
│   ├── short_math_grpo.py          # GRPO math training client
│   └── upload_to_hub.py            # Checkpoint upload client
└── tinker/                         # Tinker Client examples
    ├── dpo.py                      # DPO training client
    ├── lora.py                     # LoRA training client
    ├── multi_modal.py              # Multi-modal training client
    ├── sample.py                   # Inference sampling client
    ├── self_cognition.py           # Self-cognition training client
    ├── short_math_grpo.py          # GRPO math training client
    └── upload_to_hub.py            # Checkpoint upload client
```

Running steps:

```bash
# 1. Start Server first
twinkle-server launch -c cookbook/client/server/transformer/server_config.yaml

# 2. Configure the client in another terminal
export TWINKLE_SERVER_URL=http://localhost:8000
export TWINKLE_SERVER_TOKEN=EMPTY_TOKEN
export TWINKLE_MODEL_ID=Qwen/Qwen3.5-4B

# 3. Run a Tinker Client example
python cookbook/client/tinker/self_cognition.py

# Or use Twinkle Client
python cookbook/client/twinkle/self_cognition.py
```

The same examples work with the ModelScope managed service by changing only the environment:

```bash
export TWINKLE_SERVER_URL=https://www.modelscope.cn/twinkle
export TWINKLE_SERVER_TOKEN="$MODELSCOPE_TOKEN"
export TWINKLE_MODEL_ID=Qwen/Qwen3.6-27B
```
