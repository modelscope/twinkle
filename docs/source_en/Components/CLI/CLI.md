# CLI

The CLI module provides a unified configuration system for Twinkle training scripts. It merges multiple configuration sources (environment variables, `.env` files, YAML configs, and command-line arguments) into a single `Args` dataclass with typed argument groups.

## Resolution Order

Configuration sources are applied in order (later wins):

1. **Dataclass defaults** — sensible out-of-the-box values
2. **`.env` file** — project-local overrides
3. **Environment variables** — `TWINKLE_` prefix or bare keys
4. **YAML config file** — `--config path/to/config.yaml`
5. **CLI overrides** — `--key value` (highest priority)

All keys are case-insensitive and dash/underscore equivalent.

## Quick Start

```python
from twinkle.cli import CLI

args = CLI.from_args()

# Access typed groups
print(args.model.model_id)
print(args.training.max_steps)
print(args.optimizer.learning_rate)

# Or get dictionaries for component construction
model_kwargs = args.get_model_args()
optimizer_kwargs = args.get_optimizer_args()
```

## Argument Groups

| Group | Class | Key Parameters |
|:------|:------|:---------------|
| model | `ModelArgs` | `model_id`, `mixed_precision`, `strategy`, `gradient_checkpointing` |
| lora | `LoraArgs` | `use_lora`, `lora_r`, `lora_alpha`, `lora_target_modules` |
| dataset | `DatasetArgs` | `dataset_id`, `subset_name`, `split`, `streaming` |
| template | `TemplateArgs` | `template_cls`, `max_length`, `truncation_strategy`, `enable_thinking` |
| training | `TrainingArgs` | `max_steps`, `batch_size`, `micro_batch_size`, `output_dir`, `save_steps` |
| optimizer | `OptimizerArgs` | `optimizer_cls`, `learning_rate`, `weight_decay`, `max_grad_norm` |
| scheduler | `SchedulerArgs` | `scheduler_cls`, `num_warmup_steps`, `t_max` |
| loss | `LossArgs` | `loss_cls`, `epsilon`, `beta`, `sft_weight` |
| sampler | `SamplerArgs` | `sampler_type`, `gpu_memory_utilization`, `tensor_parallel_size` |
| sampling | `SamplingArgs` | `max_tokens`, `temperature`, `top_k`, `top_p`, `num_samples` |
| infra | `InfraArgs` | `mode`, `nproc_per_node`, `model_gpus`, `sampler_gpus`, `dp_size` |
| server | `ServerArgs` | `config`, `host`, `port`, `ray_namespace` |
| rl | `RLArgs` | `num_generations`, `advantage_type`, `reward_fns` |
| checkpoint | `CheckpointArgs` | `save_optimizer`, `merge_and_sync`, `platform` |

## YAML Configuration

```yaml
# config.yaml
model_id: ms://Qwen/Qwen3.5-4B
mixed_precision: bf16
strategy: accelerate

use_lora: true
lora_r: 16
lora_alpha: 32

dataset_id: ms://swift/self-cognition
max_length: 4096

batch_size: 8
micro_batch_size: 2
max_steps: 200
learning_rate: 1e-5

mode: ray
nproc_per_node: 8
model_gpus: 4
sampler_gpus: 4
```

## Command-Line Usage

```bash
# Use with YAML config
python train.py --config config.yaml

# Override specific values
python train.py --config config.yaml --learning_rate 5e-6 --max_steps 500

# Boolean flags
python train.py --use_lora --no_gradient_checkpointing

# Without config file (all from CLI)
python train.py --model_id ms://Qwen/Qwen3.5-4B --batch_size 4
```

## Environment Variables

```bash
# TWINKLE_ prefix
export TWINKLE_MODEL_ID=ms://Qwen/Qwen3.5-4B
export TWINKLE_LEARNING_RATE=1e-5

# Or bare keys (when recognized)
export MODEL_ID=ms://Qwen/Qwen3.5-4B
```

## Field Aliases

Some fields support aliases for convenience:

- `learning_rate` ↔ `lr`
- `nproc_per_node` ↔ `num_gpus`
- `max_tokens` ↔ `max_new_tokens`
- `use_megatron=true` → `strategy=native_fsdp`

## Custom Config Sources

You can extend the CLI with custom configuration sources:

```python
from twinkle.cli.cli import ConfigSource, Args, ConfigResolver

class RemoteConfigSource(ConfigSource):
    def __init__(self, url: str):
        self.url = url

    def load(self) -> dict:
        import requests
        return requests.get(self.url).json()

# Apply custom source
args = Args()
resolver = ConfigResolver(args)
resolver.apply(RemoteConfigSource('http://config-server/my-config').load())
```
