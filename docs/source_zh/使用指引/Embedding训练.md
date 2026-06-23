# Embedding 模型训练

Twinkle 支持基于 InfoNCE 损失的对比学习 Embedding 模型训练，内置 in-batch negatives 和跨 rank 聚合。本文介绍如何使用 Twinkle 训练 Embedding 模型，包括带在线压缩的高级架构。

---

## 概述

Embedding 训练使用以下核心组件：

| 组件 | 职责 |
|:-----|:-----|
| `InfonceLoss` | 对比损失，支持 in-batch negatives |
| `EmbeddingMetric` | 追踪正/负对相似度和损失 |
| `TransformersModel` | 可训练的 Embedding 模型（LoRA 或全参） |
| `vLLMSampler` | 可选：在线压缩 condenser |
| `InputProcessor` | 将 anchor/positive 对处理为特征 |

### 数据格式

每个训练样本由 **(anchor, positive)** 对组成。在 Embedding 特征张量中：

```
embeddings: [anchor_0, positive_0, anchor_1, positive_1, ...]
labels:     [       1,         0,        1,          0, ...]
```

- `labels=1` 标记新分组的起始位置（anchor）
- `labels=0` 标记组内的 positive/negative

---

## 基础 Embedding 训练

使用 DDP 的最小化 Embedding 训练脚本：

```python
import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.dataloader import DataLoader
from twinkle.loss import InfonceLoss
from twinkle.metric import EmbeddingMetric
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.template import Qwen3_5Template

logger = get_logger()

# --- 配置 ---
MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
MODEL_GPUS = 4
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
TEMPERATURE = 0.07
EMB_MAX_LENGTH = 8192

# --- 初始化 ---
device_groups = [
    DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
]
model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
twinkle.initialize(mode='ray', nproc_per_node=MODEL_GPUS, groups=device_groups)

# --- 模型 ---
model = TransformersModel(
    model_id=MODEL_ID,
    device_mesh=model_mesh,
    remote_group='model',
    ddp_config={'find_unused_parameters': True},
)
model.set_processor(InputProcessor)
model.set_loss(InfonceLoss, temperature=TEMPERATURE, use_batch=True)
model.set_optimizer(optimizer_cls='AdamW', lr=LEARNING_RATE)
model.set_lr_scheduler(
    scheduler_cls='CosineWarmupScheduler',
    num_warmup_steps=200,
    num_training_steps=total_steps,
)
model.add_metric(EmbeddingMetric, is_training=True)

# --- 模板 ---
template = Qwen3_5Template(
    model_id=MODEL_ID,
    max_length=EMB_MAX_LENGTH,
    enable_thinking=False,
)

# --- 训练循环 ---
for step, batch in enumerate(dataloader):
    # batch: 包含 anchor/positive 对的特征列表
    model.forward_backward(inputs=batch, task='embedding')
    model.clip_grad_and_step(gradient_accumulation_steps=1)

    if step % 10 == 0:
        metric = model.calculate_metric(is_training=True)
        logger.info(f'Step {step}: {metric}')
```

### 关键参数

| 参数 | 推荐值 | 说明 |
|:----|:------|:-----|
| `temperature` | 0.05–0.1 | 越低对比越尖锐；0.07 保持梯度流动直至 cosine > 0.75 |
| `use_batch` | True | 启用跨样本 in-batch negatives 提升效率 |
| `hard_negatives` | None 或 7 | 固定每样本负例数量；None 使用全部 in-batch |
| `find_unused_parameters` | True | Embedding 模型必需（仅最后隐藏状态产生梯度） |

---

## 进阶：在线压缩架构

对于训练检索增强 Embedding，Twinkle 支持一种高级架构：冻结的 vLLM condenser 在训练过程中对文本进行在线压缩。

### 架构 (8 GPU)

```
┌─────────────────────────────────────────────────────────┐
│ GPU 0-3: 可训练 Embedding 模型 (LoRA)                    │
│   TransformersModel + InfonceLoss + EmbeddingMetric     │
├─────────────────────────────────────────────────────────┤
│ GPU 4-7: 冻结 vLLM Condenser                            │
│   vLLMSampler (在线文本压缩)                             │
└─────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
   Embedding 特征 ◄── Condenser 压缩后的文本
```

### 流水线

1. **预取**：加载超大批次 (batch_size × prefetch_multiplier)
2. **压缩**：将 (query, CoT) 对送入 vLLM condenser 生成密集摘要
3. **验证**：检查压缩质量；如截断则回退到外部 API
4. **编码**：通过模板将压缩文本转换为 Embedding 特征
5. **训练**：在小批次上用 InfoNCE 损失进行前向/反向

### 设备组配置

```python
device_groups = [
    DeviceGroup(name='model',
                ranks=list(range(MODEL_GPUS)),
                device_type='GPU'),
    DeviceGroup(name='condenser_sampler',
                ranks=list(range(MODEL_GPUS, MODEL_GPUS + CONDENSER_GPUS)),
                device_type='GPU'),
]
model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
condenser_mesh = DeviceMesh.from_sizes(world_size=CONDENSER_GPUS, dp_size=CONDENSER_GPUS)

twinkle.initialize(mode='ray', nproc_per_node=MODEL_GPUS + CONDENSER_GPUS, groups=device_groups)
```

### Condenser 采样器配置

```python
from twinkle.sampler import vLLMSampler
from twinkle.data_format import SamplingParams

condenser_sampler = vLLMSampler(
    model_id='ms://twinkle-kit/Qwen3.5-4B-CM-v2',
    engine_args={
        'gpu_memory_utilization': 0.8,
        'max_model_len': 32768,
    },
    device_mesh=condenser_mesh,
    remote_group='condenser_sampler',
)
condenser_sampler.set_template(
    'Qwen3_5Template',
    model_id='ms://twinkle-kit/Qwen3.5-4B-CM-v2',
    enable_thinking=False,
    truncation_strategy='delete',
    max_length=32768,
)

compress_params = SamplingParams(
    max_tokens=8192,
    temperature=0.2,
    top_p=0.5,
    num_samples=1,
)
```

### 压缩质量验证

脚本会验证 condenser 输出结构的完整性：

```python
def _is_truncated_compression(text: str, schema: str = 'new') -> bool:
    """拒绝不完整或 schema 退化的 condenser 输出。"""
    if not text or '## More' not in text or '## Summary' not in text:
        return True
    # 检查 ## More 段落有内容
    after_more = text.split('## More', 1)[1].strip()
    if not after_more:
        return True
    # 'new' schema：验证 Problem/Skill/Knowledge 标记
    if schema == 'new':
        summary_body = text.split('## Summary', 1)[1].split('## More', 1)[0]
        if not all(m in summary_body for m in ('Problem:', 'Skill:', 'Knowledge:')):
            return True
    return False
```

验证失败时，系统自动回退到外部 OpenAI 兼容 API：

```python
from twinkle_agentic.protocol.openai import OpenAI as OpenAIClient

api_client = OpenAIClient(
    model='qwen3.7-max',
    api_key=os.environ['COMPRESS_API_KEY'],
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
)
```

### ThreadPoolExecutor 预取

通过后台预取线程重叠压缩和训练，最大化 GPU 利用率：

```python
from concurrent.futures import ThreadPoolExecutor

prefetch_executor = ThreadPoolExecutor(max_workers=1)

batch_iter = iter(dataloader)
first = next(batch_iter)
future = prefetch_executor.submit(_sample_batch, first)

for raw_mega_batch in batch_iter:
    # 等待上一次压缩完成
    minibatches = future.result()
    # 在后台开始压缩下一批
    future = prefetch_executor.submit(_sample_batch, raw_mega_batch)

    # 在当前小批次上训练
    for mb in minibatches:
        model.forward_backward(inputs=mb, task='embedding')
        model.clip_grad_and_step(gradient_accumulation_steps=1)
```

---

## 配置参考

### 环境变量

| 变量 | 默认值 | 说明 |
|:----|:------|:-----|
| `MODEL_ID` | `ms://Qwen/Qwen3.5-4B` | 基础 Embedding 模型 |
| `CONDENSE_MODEL_ID` | `ms://twinkle-kit/Qwen3.5-4B-CM-v2` | Condenser 模型 |
| `MODEL_GPUS` | 4 | 可训练模型占用的 GPU 数 |
| `CONDENSER_SAMPLER_GPUS` | 4 | 冻结 condenser 占用的 GPU 数 |
| `BATCH_SIZE` | 32 | 每步有效批次大小 |
| `PREFETCH_BATCH_MULTIPLIER` | 8 | 超大批次 = BATCH_SIZE × 此值 |
| `RESUME_CHECKPOINT` | `` | 断点续训检查点路径 |
| `RESUME_STEP` | 0 | 断点续训起始步数 |
| `COMPRESS_API_KEY` | `` | 回退压缩 API 密钥 |
| `COMPRESS_BASE_URL` | DashScope | 回退 API 地址 |
| `COMPRESS_MODEL` | `qwen3.7-max` | 回退 API 模型 |
| `API_CONCURRENCY` | 8 | 最大并发 API 调用数 |
| `SAMPLER_TIMEOUT` | 300 | vLLM 超时后回退 API（秒） |

### 超参数

| 参数 | 值 | 备注 |
|:----|:---|:-----|
| 学习率 | 1e-5 | CosineWarmup（200 步预热） |
| Temperature | 0.07 | 保持对角对梯度直至 cosine > 0.75 |
| Embedding 最大长度 | 8192 | anchor/positive 的 token 限制 |
| 压缩最大 token 数 | 8192 | condenser 最大生成长度 |
| 梯度累积 | 1 | 可根据显存调整 |

---

## 容错机制

进阶脚本包含多种容错机制：

- **采样器超时重建**：vLLM 采样器超过 `SAMPLER_TIMEOUT` 后，自动杀死 Actor 并重建
- **API 回退**：截断或无效的压缩自动触发外部 API 调用
- **失败日志**：失败的压缩记录到 `failures.jsonl`，供离线 SFT 数据重新生成
- **响应日志**：所有压缩结果（vLLM 和 API）记录到 `responses.jsonl`，便于调试

---

## 监控指标

`EmbeddingMetric` 报告关键训练信号：

| 指标 | 含义 |
|:----|:-----|
| `pos_sim` | anchor-positive 平均余弦相似度（目标 > 0.8） |
| `neg_sim` | anchor-negative 平均相似度（目标 < 0.3） |
| `loss` | InfoNCE 损失值 |
| `grad_norm` | 梯度范数 |

健康的训练表现为 `pos_sim` 持续上升、`neg_sim` 稳定或下降。如果 `pos_sim` 过早饱和至 1.0 附近，应降低 temperature。
