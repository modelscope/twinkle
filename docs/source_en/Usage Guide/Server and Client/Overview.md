# Server and Client

Twinkle provides a complete HTTP Server/Client architecture that supports deploying models as services and remotely calling them through clients to complete training, inference, and other tasks. This architecture decouples **model hosting (Server side)** and **training logic (Client side)**, allowing multiple users to share the same base model for training.

## Core Concepts

- **Server side**: Deployed based on Ray Serve, hosts model weights and inference/training computation. The Server is responsible for managing model loading, forward/backward propagation, weight saving, sampling inference, etc.
- **Client side**: Runs locally, responsible for data preparation, training loop orchestration, hyperparameter configuration, etc. The Client communicates with the Server via HTTP, sending data and commands.

### Two Server Modes

Twinkle Server supports two protocol modes:

| Mode | server_type | Description |
|------|------------|------|
| **Twinkle Server** | `twinkle` | Native Twinkle protocol, used with `twinkle_client`, simpler API |
| **Tinker Compatible Server** | `tinker` | Compatible with Tinker protocol, used with `init_tinker_compat_client`, can reuse existing Tinker training code |

### Two Model Backends

Regardless of Server mode, model loading supports two backends:

| Backend | use_megatron | Description |
|------|-------------|------|
| **Transformers** | `false` | Based on HuggingFace Transformers, suitable for most scenarios |
| **Megatron** | `true` | Based on Megatron-LM, suitable for ultra-large-scale model training, supports more efficient parallelization strategies |

### Two Client Modes

| Client | Initialization Method | Description |
|--------|---------|------|
| **Twinkle Client** | `init_twinkle_client` | Native client, simply change `from twinkle import` to `from twinkle_client import` to migrate local training code to remote calls |
| **Tinker Compatible Client** | `init_tinker_compat_client` | Patches Tinker SDK, allowing existing Tinker training code to be directly reused |

## How to Choose

### Server Mode Selection

| Scenario | Recommendation |
|------|------|
| New project using Twinkle system | Twinkle Server (`server_type: twinkle`) |
| Existing Tinker training code, want to migrate to Twinkle | Tinker Compatible Server (`server_type: tinker`) |
| Need inference sampling functionality | Tinker Compatible Server (built-in Sampler support) |

### Client Mode Selection

| Scenario | Recommendation |
|------|------|
| Existing Twinkle local training code, want to switch to remote | Twinkle Client — only need to change import paths |
| Existing Tinker training code, want to reuse | Tinker Compatible Client — only need to initialize patch |
| New project | Twinkle Client — simpler API |

### Model Backend Selection

| Scenario | Recommendation |
|------|------|
| 7B/14B and other medium-small scale models | Transformers backend |
| Ultra-large-scale models requiring advanced parallelization strategies | Megatron backend |
| Rapid experimentation and prototype verification | Transformers backend |

## Cookbook Reference

Complete runnable examples are located in the `cookbook/client/` directory:

```
cookbook/client/
├── twinkle/                    # Twinkle native protocol examples
│   ├── transformer/            # Transformers backend
│   │   ├── server.py           # Startup script
│   │   ├── server_config.yaml  # Configuration file
│   │   └── lora.py             # LoRA training client
│   └── megatron/               # Megatron backend
│       ├── server.py
│       ├── server_config.yaml
│       └── lora.py
└── tinker/                     # Tinker compatible protocol examples
    ├── transformer/            # Transformers backend
    │   ├── server.py
    │   ├── server_config.yaml
    │   ├── lora.py             # LoRA training
    │   ├── sample.py           # Inference sampling
    │   └── self_congnition.py  # Self-cognition training+evaluation
    └── megatron/               # Megatron backend
        ├── server.py
        ├── server_config.yaml
        └── lora.py
```

Running steps:

```bash
# 1. Start Server first
python cookbook/client/twinkle/transformer/server.py

# 2. Run Client in another terminal
python cookbook/client/twinkle/transformer/lora.py
```
