# Supported Models

Twinkle supports any model compatible with HuggingFace Transformers or Megatron-LM. Below is a curated list of models tested with Twinkle.

## Language Models

| Model Family | Model IDs | Parameters | Features |
|:-------------|:----------|:-----------|:---------|
| Qwen 3.5 | `Qwen/Qwen3.5-0.6B` ~ `Qwen/Qwen3.5-235B-A22B` | 0.6B–235B | MoE, Thinking mode |
| Qwen 2.5 | `Qwen/Qwen2.5-0.5B` ~ `Qwen/Qwen2.5-72B` | 0.5B–72B | Dense |
| DeepSeek V4 | `deepseek-ai/DeepSeek-V4` | 685B MoE | Custom DSML encoding |
| DeepSeek R1 | `deepseek-ai/DeepSeek-R1` | 685B MoE | Reasoning |
| LLaMA 3 | `meta-llama/Llama-3.3-70B-Instruct` | 8B–70B | Dense |
| Mistral | `mistralai/Mistral-7B-v0.3` | 7B | Dense |
| Yi | `01-ai/Yi-1.5-34B` | 6B–34B | Dense |
| GLM-4 | `THUDM/glm-4-9b-chat` | 9B | Dense |
| InternLM 2.5 | `internlm/internlm2_5-7b-chat` | 7B–20B | Dense |

## Vision-Language Models

| Model Family | Model IDs | Features |
|:-------------|:----------|:---------|
| Qwen 3.5 VL | `Qwen/Qwen3.5-VL-3B` ~ `Qwen/Qwen3.5-VL-72B` | Image, Video |
| Qwen 2.5 VL | `Qwen/Qwen2.5-VL-7B-Instruct` | Image, Video |
| InternVL 2.5 | `OpenGVLab/InternVL2_5-8B` | Image |

## Embedding Models

| Model Family | Model IDs | Training Method |
|:-------------|:----------|:----------------|
| Qwen3 Embedding | `Qwen/Qwen3-Embedding-0.6B` | InfoNCE contrastive |
| GTE | `thenlper/gte-large-zh` | InfoNCE contrastive |

## Model Loading

Models can be loaded from ModelScope or HuggingFace:

```python
from twinkle.model import TransformersModel

# From ModelScope (ms:// prefix)
model = TransformersModel(model_id='ms://Qwen/Qwen3.5-4B')

# From HuggingFace (hf:// prefix)
model = TransformersModel(model_id='hf://meta-llama/Llama-3.3-70B-Instruct')

# Local path
model = TransformersModel(model_id='/path/to/model')
```

## Framework Support

| Framework | Class | Use Case |
|:----------|:------|:---------|
| Transformers | `TransformersModel` | General training (SFT, RLHF, DPO) |
| Transformers + Multi-LoRA | `MultiLoraTransformersModel` | Multi-tenant training |
| Megatron-LM | `MegatronModel` | Large-scale distributed pre-training |
| Megatron + Multi-LoRA | `MultiLoraMegatronModel` | Large-scale multi-tenant |

## Precision Support

| Mode | Description |
|:-----|:------------|
| `bf16` | BFloat16 mixed precision (recommended for A100/H100) |
| `fp16` | Float16 mixed precision (for older GPUs) |
| `fp8` | FP8 precision (H100 with Transformer Engine) |
| `no` | Full precision (debugging only) |

## Parallelism Strategies

| Strategy | Config Key | Description |
|:---------|:-----------|:------------|
| FSDP | `strategy=accelerate` | Accelerate-managed FSDP (default) |
| Native FSDP | `strategy=native_fsdp` | PyTorch native FSDP |
| Tensor Parallel | `tp_size` | Split layers across GPUs |
| Pipeline Parallel | `pp_size` | Split model stages |
| Data Parallel | `dp_size` | Replicate model, split data |
| Sequence Parallel | `sequence_parallel` | Split long sequences |
| Expert Parallel | `ep_size` | MoE expert distribution |
