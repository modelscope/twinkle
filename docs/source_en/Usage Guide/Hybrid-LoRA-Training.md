# Hybrid LoRA Training and Service Deployment

Hybrid LoRA is a training approach between LoRA and full fine-tuning. It trains the full weights of a small number of selected modules while continuing to use LoRA for the remaining target modules.

It is useful when standard LoRA does not provide enough adaptation capacity but full fine-tuning is too expensive.

## 1. What is Hybrid LoRA?

### Training method

Hybrid LoRA divides candidate modules into two groups:

- `S_FFT`: modules whose full weights are trained (Full Fine-Tuning, or FFT).
- Remaining target modules: modules trained with low-rank LoRA updates.

The forward computation for the two groups can be simplified as follows:

```text
S_FFT module:  W = W_fft
LoRA module:   W = W_base + scaling * B * A
```

LoRA and FFT parameters use independent learning rates:

```text
LoRA A/B:         lr_lora
Full FFT weights: lr_fft
```

### Comparison with LoRA and full fine-tuning

| Training method | Number of trainable parameters | Adaptation capacity | Resource cost |
|---|---:|---|---|
| LoRA | Low | Limited by the LoRA rank | Low |
| Hybrid LoRA | Medium | Selected modules can train their full weights | Medium |
| Full fine-tuning | All parameters | Every module can be updated | High |

## 2. Selecting FFT modules

The central Hybrid LoRA configuration is `S_FFT`, the set of modules that train their full weights.

You can define `S_FFT` manually or generate it with Twinkle's spectral allocation method. This method analyzes the base model weights:

1. Select candidate attention and MLP linear modules from the Transformer decoder.
2. Run SVD on each candidate weight and analyze its singular-value distribution.
3. Score each module using its effective rank, energy coverage at the LoRA rank, condition number, and spectral decay rate.
4. Within the parameter budget specified by `fft_ratio`, assign the highest-scoring modules to `S_FFT` first.

`fft_ratio` is the target ratio of FFT parameters to all candidate-module parameters. For example, `fft_ratio=0.1` attempts to select FFT modules within a 10% parameter budget.

The generated allocation file looks like this:

```json
{
  "method": "spectral_hybrid_lora",
  "model_id": "ms://Qwen/Qwen3.5-9B",
  "s_fft": [
    "model.layers.0.self_attn.q_proj",
    "model.layers.8.mlp.down_proj"
  ],
  "s_lora": [
    "model.layers.0.self_attn.k_proj"
  ],
  "r": 64,
  "lora_alpha": 128,
  "fft_ratio": 0.1
}
```

The allocation is used differently by the two training modes:

- Single-adapter training uses `s_fft` as the PEFT `modules_to_save` and `s_lora` as `target_modules`.
- The multi-tenant training service reads only `s_fft`. Modules selected by the client's `target_modules` that are not in `s_fft` automatically use LoRA.

Spectral allocation is a useful starting point. In practice, adjust `fft_ratio` or edit `s_fft` manually based on task quality and available GPU memory.

## 3. Training with Hybrid LoRA

Twinkle supports two workflows:

- **Single-adapter training** for local experiments, algorithm validation, and individual training jobs.
- **Multi-tenant training service** for multiple clients sharing one base model.

### 3.1 Single-adapter training

The examples are located at:

```text
cookbook/transformers/spectral_hybrid_lora.py
cookbook/transformers/spectral_hybrid_lora.sh
```

Single-adapter training uses the standard `TransformersModel` and PEFT `modules_to_save`. Its core configuration is equivalent to:

```python
config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=s_lora,
    modules_to_save=s_fft,
)

model.add_adapter_to_model('default', config)
```

`spectral_hybrid_lora.py` uses `--generate-allocation-only` to distinguish allocation generation from training:

- With `--generate-allocation-only`, it analyzes the base model and writes the allocation without loading a dataset or starting training.
- Without the flag, it loads the existing JSON specified by `--spectral-config` and starts training. It reports an error if the file does not exist.

Generating an allocation requires a complete, unsharded copy of the base model weights, so use a single process:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
  cookbook/transformers/spectral_hybrid_lora.py \
  --model-id ms://Qwen/Qwen3.5-9B \
  --output-dir ./output/spectral_hybrid_lora \
  --spectral-config ./output/spectral_hybrid_lora/config.json \
  --lora-r 64 \
  --spectral-r 64 \
  --spectral-alpha 128 \
  --spectral-fft-ratio 0.1 \
  --generate-allocation-only
```

After generating the allocation, remove `--generate-allocation-only` and supply the dataset, learning rates, and other training arguments. You can also use `spectral_hybrid_lora.sh` as a multi-GPU training example.

### 3.2 Deploying a multi-tenant training service

The service only needs an allocation containing `s_fft`:

```json
{
  "s_fft": [
    "model.layers.0.self_attn.q_proj",
    "model.layers.8.mlp.down_proj"
  ]
}
```

The Qwen3.5-9B example configuration is located at:

```text
cookbook/client/server/transformer/server_config_hybrid_qwen3_5_9b.yaml
```

The relevant model configuration is:

```yaml
args:
  backend: transformers
  model_id: "ms://Qwen/Qwen3.5-9B"

  max_loras: 3
  max_r: 64
  target_modules: all-linear

  hybrid:
    allocation_path: /shared/config/qwen3_5_9b_hybrid_allocation.json
    default_lr_lora: 2.5e-5
    default_lr_fft: 1.0e-6
```

- `max_loras`: number of adapter slots that the service can host at the same time.
- `max_r`: maximum rank of each preallocated LoRA slot.
- `target_modules`: server-side LoRA preallocation scope.
- `allocation_path`: shared `S_FFT` module set for all Hybrid adapters.

Each Hybrid adapter has independent LoRA parameters and FFT weights. Regular LoRA adapters are not affected by `S_FFT` and can still use LoRA on those layers.

The Hybrid service must create FFT slots before FSDP/DDP wrapping, so it does not support `memory_efficient_init=True`. FFT slots contain full weights; account for the resulting GPU memory cost when configuring `max_loras` and `S_FFT`.

Validate the configuration and start the service:

```bash
twinkle-server check-config \
  -c cookbook/client/server/transformer/server_config_hybrid_qwen3_5_9b.yaml

twinkle-server launch \
  -c cookbook/client/server/transformer/server_config_hybrid_qwen3_5_9b.yaml
```

### 3.3 Starting Hybrid LoRA training from a client

The complete example is located at:

```text
cookbook/client/twinkle/hybrid_self_cognition.py
```

Pass `adapter_mode='hybrid'` when creating the adapter:

```python
model = MultiLoraTransformersModel(model_id='ms://Qwen/Qwen3.5-9B')

config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules='all-linear',
)

model.add_adapter_to_model(
    'default',
    config,
    adapter_mode='hybrid',
)

model.set_optimizer(
    'Adam',
    lr_lora=2.5e-5,
    lr_fft=1e-6,
)
```

The client only selects the LoRA `target_modules`. The server uses `s_fft` from its allocation to determine which modules use FFT.

If `adapter_mode='hybrid'` is omitted, the service creates a regular LoRA adapter instead.

Run the example client:

```bash
export TWINKLE_MODEL_ID=Qwen/Qwen3.5-9B
export TWINKLE_SERVER_URL=http://127.0.0.1:8000
export TWINKLE_SERVER_TOKEN=EMPTY_TOKEN
export TWINKLE_SAVE_DIR=/tmp/twinkle_hybrid_sft_output
export TWINKLE_MAX_STEPS=20

python cookbook/client/twinkle/hybrid_self_cognition.py
```

To provide regular LoRA and Hybrid LoRA from the same service, select the appropriate `adapter_mode` in each adapter creation request. You do not need to deploy two copies of the base model.
