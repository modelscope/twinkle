# Twinkle AI Coding Agent Guidelines

These instructions help AI agents work productively in this repo. Focus on concrete repo patterns and workflows.

## Big Picture
- **Goal:** Training and serving LLMs with multi-adapter LoRA, efficient data handling, and distributed execution across Ray and Torch.
- **Core Modules:**
  - Infrastructure & distributed orchestration: [src/twinkle/infra/__init__.py](src/twinkle/infra/__init__.py)
  - Device layout & platform abstraction: [src/twinkle/utils/platform.py](src/twinkle/utils/platform.py), [src/twinkle/utils/framework.py](src/twinkle/utils/framework.py)
  - Model stack (Transformers + Multi-LoRA): [src/twinkle/model/multi_lora_transformers.py](src/twinkle/model/multi_lora_transformers.py)
  - Sampler (vLLM integration): [src/twinkle/sampler/vllm_sampler.py](src/twinkle/sampler/vllm_sampler.py)
  - Losses & metrics: [src/twinkle/loss](src/twinkle/loss), [src/twinkle/metric](src/twinkle/metric)
  - Templates & preprocessing: [src/twinkle/template](src/twinkle/template), [src/twinkle/preprocessor](src/twinkle/preprocessor)
  - Model/Processor HTTP services via Ray Serve: [src/twinkle/server/twinkle](src/twinkle/server/twinkle)
  - Hub integrations (ModelScope/HF): [src/twinkle/hub/hub.py](src/twinkle/hub/hub.py)

## Architecture & Patterns
- **Lazy import surface:** [src/twinkle/__init__.py](src/twinkle/__init__.py) exposes a small, lazy API (`_LazyModule`), import public symbols from here when possible.
- **Distributed mode selection:** `twinkle.infra.initialize()` toggles between local and Ray modes. Ray mode requires `TWINKLE_MODE=ray` or `initialize(mode='ray', ...)`.
- **Remote execution decorators:**
  - `remote_class()` wraps classes for Ray placement; auto-injects `DeviceMesh` if missing.
  - `remote_function(dispatch='slice', execute='all', collect='none')` patches methods for distributed dispatch/collect.
  - See usage in [src/twinkle/model/multi_lora_transformers.py](src/twinkle/model/multi_lora_transformers.py) and [src/twinkle/sampler/vllm_sampler.py](src/twinkle/sampler/vllm_sampler.py).
- **Device topology:** Represented by `DeviceMesh`/`DeviceGroup`. Visualize with `twinkle.infra.get_device_placement()`; examples in [tests/infra/test_infra_graph.py](tests/infra/test_infra_graph.py).
- **Platform abstractions:** `GPU`/`NPU` selection via env and device discovery. Rank/world size read from env (`RANK`, `WORLD_SIZE`, etc.). See [src/twinkle/utils/platform.py](src/twinkle/utils/platform.py).
- **Hub usage:** `HubOperation` routes to HF or ModelScope by `hf://` or `ms://` prefixes. Dataset/model download/push helpers in [src/twinkle/hub/hub.py](src/twinkle/hub/hub.py).
- **Plugin loading:** Use `Plugin.load_plugin(id, Base)` for remote code from hubs; guarded by `trust_remote_code()` to prevent unsafe execution. See [src/twinkle/utils/plugin.py](src/twinkle/utils/plugin.py).
- **Multi-LoRA conventions:**
  - `MultiLoraTransformersModel` wraps a base Transformers model via `MultiAdapter` to manage multiple LoRA adapters.
  - FSDP is unsupported for Multi-LoRA (`fsdp_world_size == 1` enforced). Adapter params are strictly controlled to avoid training base weights.
  - Adapter ops are routed through remote functions and grouped by DP process groups.

## Developer Workflows
- **Install:** Python 3.11+. Install with Poetry or pip.
  - Poetry: `poetry install --with transformers,ray`
  - Pip (editable): `pip install -e .[transformers,ray]`
- **Run tests:**
  - Unit tests: `python -m unittest tests/infra/test_infra_graph.py`
- **Local single-process dev:**
  - Initialize infra: `twinkle.initialize(mode='local', seed=42)`
  - Inspect device placement: call `twinkle.infra.get_device_placement()`.
- **Ray Serve demo (HTTP services):**
  - Config and launcher: [cookbook/client/tinker/megatron/server.py](https://github.com/modelscope/twinkle/blob/main/cookbook/client/tinker/megatron/server.py), [cookbook/client/tinker/megatron/server_config.yaml](https://github.com/modelscope/twinkle/blob/main/cookbook/client/tinker/megatron/server_config.yaml)
  - Start:
    - `cd cookbook/client/tinker/megatron`
    - `python server.py`
    - Endpoints print on startup (default `localhost:8000` or `https://www.modelscope.cn/twinkle`).
  - Model app binds `MultiLoraTransformersModel` and exposes routes like `/add_adapter_to_model`, `/forward`, `/calculate_loss`, etc. See [src/twinkle/server/twinkle/model.py](src/twinkle/server/twinkle/model.py).
- **vLLM inference:** Use `VLLMEngine` with engine args; LoRA weight sync via `patch.vllm_lora_weights`. See [src/twinkle/sampler/vllm_engine.py](src/twinkle/sampler/vllm_engine.py).

## Conventions & Gotchas
- **Safety:** Remote plugin code requires `trust_remote_code()` true; avoid loading arbitrary strings into adapter configs (enforced in Multi-LoRA).
- **Env-driven ranks:** Many utilities read ranks/world size from env; set `WORLD_SIZE`, `RANK`, `LOCAL_RANK` when using torchrun.
- **Determinism:** `seed_everything(seed, full_determinism)` controls CUDA/NPU determinism; may set envs like `CUDA_LAUNCH_BLOCKING`.
- **Adapter lifecycle:** Server auto-removes inactive adapters (heartbeat required); per-token adapter limits are enforced. See cleanup in [src/twinkle/server/twinkle/model.py](src/twinkle/server/twinkle/model.py).
- **Templates:** Tokenization/encode via `Template` (e.g., `Qwen3Template`), producing `InputFeature` for model forward. See [src/twinkle/template/base.py](src/twinkle/template/base.py).

## Examples
- **Visualize a custom mesh:** create `DeviceMesh` and call `get_device_placement()`; example in [tests/infra/test_infra_graph.py](tests/infra/test_infra_graph.py).
- **Add LoRA adapter via HTTP:** POST to `/add_adapter_to_model` with serialized `LoraConfig`; see server routes in [src/twinkle/server/twinkle/model.py](src/twinkle/server/twinkle/model.py).
- **Sample with vLLM:** Configure `vLLMSampler`, set `Template`/`Processor`, then `sample()` on `Trajectory` list; see [src/twinkle/sampler/vllm_sampler.py](src/twinkle/sampler/vllm_sampler.py).

---
Questions or gaps? Tell us where guidance is unclear (e.g., missing run scripts, Ray cluster setup), and we’ll refine this document.
