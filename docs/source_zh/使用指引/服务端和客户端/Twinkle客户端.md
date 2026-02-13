# Twinkle 客户端

Twinkle Client 是原生客户端，设计理念是：**将 `from twinkle import` 改为 `from twinkle_client import`，即可将本地训练代码迁移为远端调用，原有训练逻辑无需改动**。

## 初始化

```python
from twinkle_client import init_twinkle_client

# 初始化客户端，连接到 Twinkle Server
client = init_twinkle_client(
    base_url='http://127.0.0.1:8000',   # Server 地址
    api_key='your-api-key'               # 认证令牌（可通过环境变量 TWINKLE_SERVER_TOKEN 设置）
)
```

初始化完成后，`client` 对象（`TwinkleClient`）提供以下管理功能：

```python
# 健康检查
client.health_check()

# 列出当前用户的训练运行
runs = client.list_training_runs(limit=20)

# 获取特定训练运行详情
run = client.get_training_run(run_id='xxx')

# 列出检查点
checkpoints = client.list_checkpoints(run_id='xxx')

# 获取检查点路径（用于恢复训练）
path = client.get_checkpoint_path(run_id='xxx', checkpoint_id='yyy')

# 获取最新检查点路径
latest_path = client.get_latest_checkpoint_path(run_id='xxx')
```

## 从本地代码迁移到远端

迁移非常简单，只需将 import 路径从 `twinkle` 替换为 `twinkle_client`：

```python
# 本地训练代码（原始）
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset
from twinkle.model import MultiLoraTransformersModel

# 远端训练代码（迁移后）
from twinkle_client.dataloader import DataLoader
from twinkle_client.dataset import Dataset
from twinkle_client.model import MultiLoraTransformersModel
```

训练循环、数据处理等逻辑完全不需要修改。

## 完整训练示例（Transformers 后端）

```python
import os
import dotenv
dotenv.load_dotenv('.env')

from peft import LoraConfig
from twinkle import get_logger
from twinkle.dataset import DatasetMeta

# 从 twinkle_client import 替代 twinkle，实现远端调用
from twinkle_client.dataloader import DataLoader
from twinkle_client.dataset import Dataset
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client import init_twinkle_client

logger = get_logger()

# Step 1: 初始化客户端
client = init_twinkle_client(
    base_url='http://127.0.0.1:8000',
    api_key=os.environ.get('MODELSCOPE_SDK_TOKEN')
)

# Step 2: 查询已有训练运行（可选，用于恢复训练）
runs = client.list_training_runs()
resume_path = None
for run in runs:
    checkpoints = client.list_checkpoints(run.training_run_id)
    for checkpoint in checkpoints:
        logger.info(checkpoint.model_dump_json(indent=2))
        # 取消注释以从检查点恢复：
        # resume_path = checkpoint.twinkle_path

# Step 3: 准备数据集
dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition'))

# 设置 chat 模板，使数据匹配模型的输入格式
dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct', max_length=512)

# 数据预处理：替换占位符为自定义名称
dataset.map('SelfCognitionProcessor',
            init_args={'model_name': 'twinkle模型', 'model_author': 'twinkle团队'})

# 编码数据集为模型可用的 token
dataset.encode(batched=True)

# 创建 DataLoader
dataloader = DataLoader(dataset=dataset, batch_size=8)

# Step 4: 配置模型
model = MultiLoraTransformersModel(model_id='ms://Qwen/Qwen2.5-7B-Instruct')

# 配置 LoRA
lora_config = LoraConfig(target_modules='all-linear')
model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2)

# 设置模板、处理器、损失函数
model.set_template('Template')
model.set_processor('InputProcessor', padding_side='right')
model.set_loss('CrossEntropyLoss')

# 设置优化器和学习率调度器
model.set_optimizer('AdamW', lr=1e-4)
model.set_lr_scheduler('LinearLR')

# Step 5: 恢复训练（可选）
if resume_path:
    logger.info(f'Resuming training from {resume_path}')
    model.load(resume_path, load_optimizer=True)

# Step 6: 训练循环
for step, batch in enumerate(dataloader):
    # 前向传播 + 反向传播
    output = model.forward_backward(inputs=batch)

    if step % 2 == 0:
        logger.info(f'Step {step // 2}, loss: {output}')

    # 梯度裁剪
    model.clip_grad_norm(1.0)

    # 优化器更新
    model.step()

    # 梯度清零
    model.zero_grad()

    # 学习率调度
    model.lr_step()

# Step 7: 保存检查点
twinkle_path = model.save(name=f'step-{step}', save_optimizer=True)
logger.info(f"Saved checkpoint: {twinkle_path}")

# Step 8: 上传到 ModelScope Hub（可选）
model.upload_to_hub(
    checkpoint_dir=twinkle_path,
    hub_model_id='your-username/your-model-name',
    async_upload=False
)
```

## Megatron 后端的差异

使用 Megatron 后端时，客户端代码的主要差异：

```python
# Megatron 后端不需要显式设置 loss（由 Megatron 内部计算）
# model.set_loss('CrossEntropyLoss')  # 不需要

# 优化器和 LR 调度器使用 Megatron 内置默认值
model.set_optimizer('default', lr=1e-4)
model.set_lr_scheduler('default', lr_decay_steps=1000, max_lr=1e-4)
```

其余数据处理、训练循环、检查点保存等代码完全相同。
