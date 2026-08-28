# Client-Orchestrated Async GRPO

The client owns the Dataset, Reward, Advantage, rollout partitions, staleness,
training schedule, and policy publication. The server exposes shared
Multi-LoRA Model, vLLM Sampler, and TransferQueue DataPlane components without
owning the GRPO loop.

This example deploys a dedicated Qwen3.5-4B GRPO server. It does not combine
different RL algorithms in one deployment. For the YAML-managed multi-tenant
runtime, see [`cookbook/rl`](../../rl/README.md).

## Resources

The server configuration uses two GPUs:

| Component | GPUs | Purpose |
|---|---:|---|
| Multi-LoRA Model | 1 | GRPO forward, backward, optimizer step, and checkpoint save |
| Async vLLM Sampler | 1 | Shared continuous-batched rollout generation |

The DataPlane keeps token tensors and sampled log-probabilities server-side.
The client reads decoded completions for Reward and Advantage, appends those
values to the same `DataRef`, and sends only references to the Model.

## Quick start

Install the server, client, and async-RL dependencies:

```bash
pip install -e '.[async-rl,client,server]'
```

Terminal 1 — start the dedicated component server:

```bash
export TWINKLE_LOCAL_MODEL_PATH=/absolute/path/to/Qwen3.5-4B
export CUDA_VISIBLE_DEVICES=0,1
bash cookbook/client/async_rl/run_server.sh
```

The server script starts a local Ray cluster when needed, validates
`server_config.yaml`, and launches Ray Serve on port 8000.

Terminal 2 — start one GRPO client:

```bash
export TWINKLE_SERVER_URL=http://127.0.0.1:8000
export TWINKLE_SERVER_TOKEN=EMPTY_TOKEN
export TWINKLE_TEMPLATE_MODEL_ID=/absolute/path/to/Qwen3.5-4B
export TWINKLE_DATASET_ID=/absolute/path/to/gsm8k
python cookbook/client/async_rl/client_orchestrated_grpo.py
```

`TWINKLE_TEMPLATE_MODEL_ID` is the tokenizer/template model path used in the
client process. The template class is fixed to `Qwen3_5Template`; a model ID
such as `Qwen/Qwen3.5-4B` is not a template class name.

## Execution model

Each DataLoader batch becomes a private client-side `_RolloutPartition` bound
to one immutable adapter checkpoint:

```text
DataLoader
    -> Rollout Worker
    -> async vLLM Sampler
    -> DataPlane DataRef
    -> Advantage Worker
    -> Trainer Worker
    -> save and publish policy
```

The workers are independent asyncio tasks:

- Prompt groups are submitted separately through
  `sampler.asample_to_data_plane()`.
- A complete group is rewarded and made trainable without waiting for the rest
  of its partition.
- The Trainer consumes `TRAIN_MINI_BATCH_SIZE / NUM_GENERATIONS` ready groups
  at a time.
- Partitions train and publish in FIFO order.
- A policy version advances only after every group in its partition has
  trained and the new checkpoint has been saved.

`MAX_STALENESS` controls the number of live partitions. Setting it to zero
still permits rollout/training overlap inside one partition; values above zero
also permit cross-partition overlap.

## Training metrics

After every `clip_grad_and_step()`, the Trainer calls:

```python
model.calculate_metric(is_training=True)
```

and prints the returned metrics to client stdout:

```text
optimizer_step=1 grad_norm=0.42 learning_rate=2e-05 loss=0.031
```

Partition publication is logged separately:

```text
partition=0 policy=1 optimizer_step=8 staleness=0
```

## Client settings

| Environment variable | Default | Purpose |
|---|---|---|
| `TWINKLE_SERVER_URL` | `http://localhost:8000` | Server base URL |
| `TWINKLE_SERVER_TOKEN` | `EMPTY_TOKEN` | Request authentication token |
| `TWINKLE_TEMPLATE_MODEL_ID` | `ms://Qwen/Qwen3.5-4B` | Client tokenizer/template source |
| `TWINKLE_DATASET_ID` | `ms://modelscope/gsm8k` | GSM8K dataset source |
| `TWINKLE_ADAPTER_NAME` | `client-grpo` | LoRA adapter name |
| `TWINKLE_MAX_PARTITIONS` | `100` | Maximum DataLoader batches admitted |
| `TWINKLE_MAX_STALENESS` | `2` | Maximum extra live partitions |
| `TWINKLE_ROLLOUT_CONCURRENCY` | `8` | Concurrent prompt-group submissions |
| `TWINKLE_NUM_GENERATIONS` | `4` | Generations per prompt group |
| `TWINKLE_BATCH_SIZE` | `8` | Prompt groups per partition |
| `TWINKLE_TRAIN_MINI_BATCH_SIZE` | `8` | Samples per optimizer step |
| `TWINKLE_MICRO_BATCH_SIZE` | `4` | Model micro-batch size |
| `TWINKLE_MAX_TOKENS_PER_MICRO_BATCH` | `4096` | Dynamic batching token limit |

The following must hold:

```text
TWINKLE_TRAIN_MINI_BATCH_SIZE % TWINKLE_NUM_GENERATIONS == 0
TWINKLE_BATCH_SIZE % (
    TWINKLE_TRAIN_MINI_BATCH_SIZE / TWINKLE_NUM_GENERATIONS
) == 0
```

## Files

| File | Role |
|---|---|
| `client_orchestrated_grpo.py` | Client-side async GRPO orchestration |
| `run_server.sh` | Start Ray and the dedicated component server |
| `server_config.yaml` | Qwen3.5-4B Model, Sampler, DataPlane, Gateway, and Processor deployment |

## Troubleshooting

- **Client receives HTTP 404 for the model** — use the provided server config;
  its public route is fixed to `Qwen/Qwen3.5-4B`, matching the client.
- **The process tries to access ModelScope while offline** — set both
  `TWINKLE_LOCAL_MODEL_PATH` on the server and
  `TWINKLE_TEMPLATE_MODEL_ID`/`TWINKLE_DATASET_ID` on the client to local
  paths.
- **No loss is printed** — confirm the Model request completed and look for an
  `optimizer_step=...` line. Metrics are calculated after every optimizer
  step, not after each rollout group.
- **Only one GPU is active** — verify Ray sees two GPUs and that Model and
  Sampler placement groups were assigned to different devices.
