# Hybrid LoRA 训练与服务部署

Hybrid LoRA 是一种介于 LoRA 和全参数微调之间的训练方式：少量指定模块训练完整权重，其余目标模块仍然使用 LoRA。

它适合这样的场景：纯 LoRA 的表达能力不足，但全参数微调的训练成本又太高。

## 1. Hybrid LoRA 是什么

### 训练方式

Hybrid LoRA 将候选模块分为两类：

- `S_FFT`：使用完整权重训练（Full Fine-Tuning，FFT）。
- 其余目标模块：使用 LoRA 训练低秩增量。

两类模块的前向计算可以简化为：

```text
S_FFT 模块：     W = W_fft
LoRA 模块：      W = W_base + scaling * B * A
```

LoRA 和 FFT 参数使用独立的学习率：

```text
LoRA A/B：       lr_lora
FFT 完整权重： lr_fft
```

### 与 LoRA、全参数微调的对比

| 训练方式 | 训练参数量 | 适配能力 | 资源开销 |
|---|---:|---|---|
| LoRA | 少 | 受 LoRA rank 限制 | 低 |
| Hybrid LoRA | 中等 | 关键模块可训练完整权重 | 中等 |
| 全参数微调 | 全部 | 所有模块均可更新 | 高 |

## 2. 如何选择 FFT 模块

Hybrid LoRA 的核心配置是 `S_FFT`，即哪些模块使用完整权重训练。

可以手工指定 `S_FFT`，也可以使用 Twinkle 提供的 spectral allocation 方法生成。该方法分析基座模型权重：

1. 从 Transformer decoder 中选出 attention 和 MLP 的候选 Linear 模块。
2. 对每个候选权重做 SVD，分析其奇异值分布。
3. 结合有效秩、LoRA rank 的能量覆盖率、条件数和频谱衰减速度，为每个模块计算分数。
4. 在 `fft_ratio` 给定的参数预算内，优先将高分模块分配到 `S_FFT`。

`fft_ratio` 表示 FFT 模块参数量占候选模块总参数量的目标比例。例如 `fft_ratio=0.1` 表示尝试在 10% 的参数预算内选择 FFT 模块。

生成的 allocation 文件类似：

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

其中：

- 单 adapter 训练使用 `s_fft` 作为 PEFT `modules_to_save`，使用 `s_lora` 作为 `target_modules`。
- 多租户训练服务只读取 `s_fft`。客户端 `target_modules` 选中且不属于 `s_fft` 的模块自动使用 LoRA。

spectral allocation 可以作为默认起点；实际使用时，仍建议结合任务效果和显存预算调整 `fft_ratio` 或手工修改 `s_fft`。

## 3. 如何进行训练

Twinkle 提供两种使用方式：

- **单 adapter 训练**：适合本地实验、算法验证和单个训练任务。
- **多租户训练服务**：适合多个客户端共享同一份基座模型。

### 3.1 单 adapter 训练

示例位于：

```text
cookbook/transformers/spectral_hybrid_lora.py
cookbook/transformers/spectral_hybrid_lora.sh
```

单 adapter 训练使用标准 `TransformersModel` 和 PEFT `modules_to_save`，核心配置等价于：

```python
config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=s_lora,
    modules_to_save=s_fft,
)

model.add_adapter_to_model('default', config)
```

`spectral_hybrid_lora.py` 使用 `--generate-allocation-only` 区分生成和训练：

- 传入 `--generate-allocation-only`：只分析基座权重并写入 allocation，不加载数据集，也不进行训练。
- 不传该参数：读取 `--spectral-config` 指定的已有 JSON 并开始训练；文件不存在时直接报错。

首次生成 allocation 需要加载未分片的完整基座权重，因此应使用单进程：

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

allocation 生成后，去掉 `--generate-allocation-only` 并补充数据集、学习率等训练参数即可开始训练，也可以直接参考 `spectral_hybrid_lora.sh` 的多卡训练命令。

### 3.2 部署多租户训练服务

服务端只需要一份包含 `s_fft` 的 allocation：

```json
{
  "s_fft": [
    "model.layers.0.self_attn.q_proj",
    "model.layers.8.mlp.down_proj"
  ]
}
```

Qwen3.5-9B 示例配置位于：

```text
cookbook/client/server/transformer/server_config_hybrid_qwen3_5_9b.yaml
```

关键配置如下：

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

- `max_loras`：服务可同时容纳的 adapter slot 数量。
- `max_r`：预分配 LoRA slot 的最大 rank。
- `target_modules`：服务端的 LoRA 预分配范围。
- `allocation_path`：所有 Hybrid adapter 共享的 `S_FFT` 模块集合。

每个 Hybrid adapter 都拥有独立的 LoRA 参数和 FFT 权重。普通 LoRA adapter 不受 `S_FFT` 影响，仍然可以在这些层上使用 LoRA。

Hybrid 服务需要在 FSDP/DDP 包装前创建 FFT slot，因此不支持 `memory_efficient_init=True`。由于 FFT slot 保存完整权重，配置 `max_loras` 和 `S_FFT` 时需要考虑显存开销。

检查并启动服务：

```bash
twinkle-server check-config \
  -c cookbook/client/server/transformer/server_config_hybrid_qwen3_5_9b.yaml

twinkle-server launch \
  -c cookbook/client/server/transformer/server_config_hybrid_qwen3_5_9b.yaml
```

### 3.3 使用 client 启动 Hybrid LoRA 训练

完整示例位于：

```text
cookbook/client/twinkle/hybrid_self_cognition.py
```

创建 adapter 时，显式传入 `adapter_mode='hybrid'`：

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

客户端只需要选择 LoRA `target_modules`，服务端会使用 allocation 中的 `s_fft` 决定哪些模块使用 FFT。

如果不传 `adapter_mode='hybrid'`，创建的就是普通 LoRA adapter。

启动示例 client：

```bash
export TWINKLE_MODEL_ID=Qwen/Qwen3.5-9B
export TWINKLE_SERVER_URL=http://127.0.0.1:8000
export TWINKLE_SERVER_TOKEN=EMPTY_TOKEN
export TWINKLE_SAVE_DIR=/tmp/twinkle_hybrid_sft_output
export TWINKLE_MAX_STEPS=20

python cookbook/client/twinkle/hybrid_self_cognition.py
```

当需要在同一服务中同时提供普通 LoRA 和 Hybrid LoRA 时，只需要在不同 adapter 的创建请求中选择对应的 `adapter_mode`，无需部署两份基座模型。
