# 支持的模型

Twinkle 支持任何兼容 HuggingFace Transformers 或 Megatron-LM 的模型。以下是经过测试的模型列表。

## 语言模型

| 模型系列 | 模型 ID | 参数量 | 特性 |
|:---------|:--------|:-------|:-----|
| Qwen 3.5 | `Qwen/Qwen3.5-0.6B` ~ `Qwen/Qwen3.5-235B-A22B` | 0.6B–235B | MoE、思考模式 |
| Qwen 2.5 | `Qwen/Qwen2.5-0.5B` ~ `Qwen/Qwen2.5-72B` | 0.5B–72B | Dense |
| DeepSeek V4 | `deepseek-ai/DeepSeek-V4` | 685B MoE | 自定义 DSML 编码 |
| DeepSeek R1 | `deepseek-ai/DeepSeek-R1` | 685B MoE | 推理 |
| LLaMA 3 | `meta-llama/Llama-3.3-70B-Instruct` | 8B–70B | Dense |
| Mistral | `mistralai/Mistral-7B-v0.3` | 7B | Dense |
| Yi | `01-ai/Yi-1.5-34B` | 6B–34B | Dense |
| GLM-4 | `THUDM/glm-4-9b-chat` | 9B | Dense |
| InternLM 2.5 | `internlm/internlm2_5-7b-chat` | 7B–20B | Dense |

## 视觉语言模型

| 模型系列 | 模型 ID | 特性 |
|:---------|:--------|:-----|
| Qwen 3.5 VL | `Qwen/Qwen3.5-VL-3B` ~ `Qwen/Qwen3.5-VL-72B` | 图片、视频 |
| Qwen 2.5 VL | `Qwen/Qwen2.5-VL-7B-Instruct` | 图片、视频 |
| InternVL 2.5 | `OpenGVLab/InternVL2_5-8B` | 图片 |

## 嵌入模型

| 模型系列 | 模型 ID | 训练方法 |
|:---------|:--------|:---------|
| Qwen3 Embedding | `Qwen/Qwen3-Embedding-0.6B` | InfoNCE 对比学习 |
| GTE | `thenlper/gte-large-zh` | InfoNCE 对比学习 |

## 模型加载

```python
from twinkle.model import TransformersModel

# 从 ModelScope 加载（ms:// 前缀）
model = TransformersModel(model_id='ms://Qwen/Qwen3.5-4B')

# 从 HuggingFace 加载（hf:// 前缀）
model = TransformersModel(model_id='hf://meta-llama/Llama-3.3-70B-Instruct')

# 本地路径
model = TransformersModel(model_id='/path/to/model')
```

## 框架支持

| 框架 | 类名 | 适用场景 |
|:-----|:-----|:---------|
| Transformers | `TransformersModel` | 通用训练（SFT、RLHF、DPO）|
| Transformers + Multi-LoRA | `MultiLoraTransformersModel` | 多租户训练 |
| Megatron-LM | `MegatronModel` | 大规模分布式预训练 |
| Megatron + Multi-LoRA | `MultiLoraMegatronModel` | 大规模多租户 |

## 精度支持

| 模式 | 说明 |
|:-----|:-----|
| `bf16` | BFloat16 混合精度（推荐 A100/H100）|
| `fp16` | Float16 混合精度（适用于旧 GPU）|
| `fp8` | FP8 精度（H100 + Transformer Engine）|
| `no` | 全精度（仅用于调试）|

## 并行策略

| 策略 | 配置键 | 说明 |
|:-----|:-------|:-----|
| FSDP | `strategy=accelerate` | Accelerate 管理的 FSDP（默认）|
| 原生 FSDP | `strategy=native_fsdp` | PyTorch 原生 FSDP |
| 张量并行 | `tp_size` | 跨 GPU 切分层 |
| 流水线并行 | `pp_size` | 切分模型阶段 |
| 数据并行 | `dp_size` | 复制模型，切分数据 |
| 序列并行 | `sequence_parallel` | 切分长序列 |
| 专家并行 | `ep_size` | MoE 专家分布 |
