# CLI 命令行配置

CLI 模块为 Twinkle 训练脚本提供统一的配置系统。它将多种配置来源（环境变量、`.env` 文件、YAML 配置、命令行参数）合并到一个带类型的 `Args` 数据类中。

## 配置优先级

配置按以下顺序应用（后者覆盖前者）：

1. **数据类默认值** — 开箱即用
2. **`.env` 文件** — 项目本地配置
3. **环境变量** — `TWINKLE_` 前缀或裸键名
4. **YAML 配置文件** — `--config path/to/config.yaml`
5. **命令行参数** — `--key value`（最高优先级）

所有键名不区分大小写，横杠和下划线等价。

## 快速开始

```python
from twinkle.cli import CLI

args = CLI.from_args()

# 访问类型化的参数组
print(args.model.model_id)
print(args.training.max_steps)
print(args.optimizer.learning_rate)

# 或获取字典用于组件构造
model_kwargs = args.get_model_args()
optimizer_kwargs = args.get_optimizer_args()
```

## 参数组

| 分组 | 类名 | 关键参数 |
|:-----|:-----|:---------|
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

## YAML 配置示例

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

## 命令行用法

```bash
# 使用 YAML 配置
python train.py --config config.yaml

# 覆盖特定值
python train.py --config config.yaml --learning_rate 5e-6 --max_steps 500

# 布尔标志
python train.py --use_lora --no_gradient_checkpointing

# 无配置文件（全部从命令行指定）
python train.py --model_id ms://Qwen/Qwen3.5-4B --batch_size 4
```

## 环境变量

```bash
# TWINKLE_ 前缀
export TWINKLE_MODEL_ID=ms://Qwen/Qwen3.5-4B
export TWINKLE_LEARNING_RATE=1e-5

# 或裸键名（当能识别时）
export MODEL_ID=ms://Qwen/Qwen3.5-4B
```

## 字段别名

部分字段支持别名：

- `learning_rate` ↔ `lr`
- `nproc_per_node` ↔ `num_gpus`
- `max_tokens` ↔ `max_new_tokens`
- `use_megatron=true` → `strategy=native_fsdp`

## 自定义配置源

你可以通过自定义配置源扩展 CLI：

```python
from twinkle.cli.cli import ConfigSource, Args, ConfigResolver

class RemoteConfigSource(ConfigSource):
    def __init__(self, url: str):
        self.url = url

    def load(self) -> dict:
        import requests
        return requests.get(self.url).json()

# 应用自定义配置源
args = Args()
resolver = ConfigResolver(args)
resolver.apply(RemoteConfigSource('http://config-server/my-config').load())
```
