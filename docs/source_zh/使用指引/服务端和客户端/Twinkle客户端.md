# Twinkle 客户端

Twinkle Client 是原生客户端，设计理念是：**将 `from twinkle import` 改为 `from twinkle_client import`，即可将本地训练代码迁移为远端调用，原有训练逻辑无需改动**。

## 初始化

```python
from twinkle import init_twinkle_client

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

迁移时保留数据处理和训练循环；初始化远端客户端后，通过 `client.model(...)` 创建显式绑定该客户端的远端模型：

```python
# 本地训练代码（原始）
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset
from twinkle.model import MultiLoraTransformersModel

# 远端训练代码（迁移后）
# DataLoader 和 Dataset 使用本地 twinkle 或远端 twinkle_client 均可
from twinkle.dataloader import DataLoader        # 或 from twinkle_client.dataloader import DataLoader
from twinkle.dataset import Dataset              # 或 from twinkle_client.dataset import Dataset
from twinkle import init_twinkle_client

client = init_twinkle_client(base_url=base_url, api_key=api_key)
model = client.model(f'ms://{base_model}')
```

训练循环、数据处理等逻辑完全不需要修改；推荐使用 `client.model(...)`，而不是直接构造 `MultiLoraTransformersModel(...)`，以确保模型包装器显式复用当前客户端的 transport、会话与认证上下文。

## 完整训练示例（Transformers 后端）

```python
import dotenv
dotenv.load_dotenv('.env')

from peft import LoraConfig
from twinkle import get_logger
from twinkle.dataset import DatasetMeta
from twinkle import init_twinkle_client

# DataLoader 和 Dataset 使用本地 twinkle 或远端 twinkle_client 均可
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset

logger = get_logger()

base_model = 'Qwen/Qwen3.5-4B'
base_url = 'http://localhost:8000'
api_key = 'EMPTY_API_KEY'

# Step 1: 初始化客户端
client = init_twinkle_client(base_url=base_url, api_key=api_key)

# 列出服务器支持的模型
print('Available models:')
for item in client.get_server_capabilities().supported_models:
    print('- ' + item.model_name)

# Step 2: 查询已有训练运行（可选，用于恢复训练）
runs = client.list_training_runs()
resume_path = None
for run in runs:
    logger.info(run.model_dump_json(indent=2))
    checkpoints = client.list_checkpoints(run.training_run_id)
    for checkpoint in checkpoints:
        logger.info(checkpoint.model_dump_json(indent=2))
        # 取消注释以从检查点恢复：
        # resume_path = checkpoint.twinkle_path

# Step 3: 准备数据集
# data_slice 可限制加载的数据量
dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))

# 设置 chat 模板，使数据匹配模型的输入格式
dataset.set_template('Qwen3_5Template', model_id=f'ms://{base_model}', max_length=512)

# 数据预处理：替换占位符为自定义名称
dataset.map('SelfCognitionProcessor',
            init_args={'model_name': 'twinkle模型', 'model_author': 'ModelScope社区'})

# 编码数据集为模型可用的 token
dataset.encode(batched=True)
# 数据量大时可用 num_proc 多进程加速：
# dataset.encode(batched=True, num_proc=8)
# 使用 twinkle_client.dataset 时，encode 是通过 HTTP 调用远端服务，
# 默认 600 秒超时，可用 timeout 参数按需调大：
# dataset.encode(batched=True, num_proc=8, timeout=3600)

# 创建 DataLoader
dataloader = DataLoader(dataset=dataset, batch_size=4)

# Step 4: 通过当前 client 创建并绑定远端模型
model = client.model(f'ms://{base_model}')

# 配置 LoRA：对所有线性层应用低秩适配器
lora_config = LoraConfig(target_modules='all-linear')
# gradient_accumulation_steps=2 表示累积 2 个 micro-batch 的梯度后再执行一次优化器更新
model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2)

# 设置模板、处理器、损失函数
model.set_template('Qwen3_5Template')
model.set_processor('InputProcessor', padding_side='right')
model.set_loss('CrossEntropyLoss')

# 设置优化器（如果服务器使用 Megatron 后端，仅支持 Adam 优化器）
model.set_optimizer('Adam', lr=1e-4)

# 设置学习率调度器（如果服务器使用 Megatron 后端，不支持 LR 调度器）
# model.set_lr_scheduler('LinearLR')

# Step 5: 恢复训练（可选）
start_step = 0
if resume_path:
    logger.info(f'Resuming from checkpoint {resume_path}')
    progress = model.resume_from_checkpoint(resume_path)
    dataloader.resume_from_checkpoint(progress['consumed_train_samples'])
    start_step = progress['cur_step']

# Step 6: 训练循环
logger.info(model.get_train_configs().model_dump())

for epoch in range(3):
    logger.info(f'Starting epoch {epoch}')
    for cur_step, batch in enumerate(dataloader, start=start_step + 1):
        # 前向传播 + 反向传播
        model.forward_backward(inputs=batch)

        # 梯度裁剪 + 优化器更新（等价于依次调用 clip_grad_norm / step / zero_grad / lr_step）
        model.clip_grad_and_step()

        # 每 2 步打印一次指标（与 gradient_accumulation_steps 对齐）
        if cur_step % 2 == 0:
            metric = model.calculate_metric(is_training=True)
            logger.info(f'Current is step {cur_step} of {len(dataloader)}, metric: {metric.result}')

    # Step 7: 保存检查点
    twinkle_path = model.save(
        name=f'twinkle-epoch-{epoch}',
        save_optimizer=True,
        consumed_train_samples=dataloader.get_state()['consumed_train_samples'],
    )
    logger.info(f'Saved checkpoint: {twinkle_path}')

# Step 8: 上传到 ModelScope Hub（可选）
# 服务端始终在后台执行上传；当前调用会通过 future layer 等待上传完成或抛出失败。
# 不要传 async_upload 或 poll_interval：两者仅为兼容旧调用保留，已废弃且无效果。
# YOUR_USER_NAME = "your_username"
# hub_model_id = f'{YOUR_USER_NAME}/twinkle-self-cognition'
# model.upload_to_hub(
#     checkpoint_dir=twinkle_path,
#     hub_model_id=hub_model_id,
# )
```

Twinkle Client 场景下，推荐的断点续训流程是：

1. 先通过 `client.list_checkpoints(...)` 或 `client.get_latest_checkpoint_path(...)` 获取已有 checkpoint 路径。
2. 调用 `model.resume_from_checkpoint(resume_path)` 恢复权重、优化器、调度器、随机数状态和训练进度元数据。
3. 使用返回结果中的 `consumed_train_samples` 调用 `dataloader.resume_from_checkpoint(...)`，跳过已经训练过的数据。

完整示例可直接参考 `cookbook/client/twinkle/self_cognition.py`。

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

(client-async-grpo)=
## 客户端异步 GRPO

客户端可以在采样的同时进行训练。完成 [Twinkle 安装](../安装.md) 后，选择一个示例，按下面的步骤启动即可：

| 场景 | 示例 |
|---|---|
| 单轮数学问答（GSM8K） | [client_orchestrated_grpo.py](https://github.com/modelscope/twinkle/blob/main/cookbook/client/async_rl/client_orchestrated_grpo.py) |
| 带工具调用的多轮代码修复（纯文本） | [multi_turn_grpo.py](https://github.com/modelscope/twinkle/blob/main/cookbook/client/async_rl/multi_turn_grpo.py) |

### 1. 启动服务端

示例使用 Qwen3.5-4B，训练后端为 Transformers，采样后端为 vLLM，需要两张 GPU。准备好本地模型目录后，在仓库根目录执行：

```bash
pip install -e '.[async-rl,client,server]'
export TWINKLE_LOCAL_MODEL_PATH=/absolute/path/to/Qwen3.5-4B
export CUDA_VISIBLE_DEVICES=0,1
bash cookbook/client/async_rl/run_server.sh
```

服务默认监听 `http://127.0.0.1:8000`。如果已有可用的服务端，可以跳过启动步骤；服务配置见 [server_config.yaml](https://github.com/modelscope/twinkle/blob/main/cookbook/client/async_rl/server_config.yaml)。

### 2. 运行单轮数学任务

保持服务端运行，在另一个终端中从仓库根目录执行：

```bash
export TWINKLE_SERVER_URL=http://127.0.0.1:8000
export TWINKLE_SERVER_TOKEN=EMPTY_TOKEN
export TWINKLE_TEMPLATE_MODEL_ID=/absolute/path/to/Qwen3.5-4B
python cookbook/client/async_rl/client_orchestrated_grpo.py
```

示例默认下载 GSM8K 数据集。离线运行时，额外设置 `TWINKLE_DATASET_ID=/absolute/path/to/gsm8k`；连接已有服务端时，将地址和令牌换成对应配置。模板目录应与服务端模型匹配。

### 3. 运行多轮代码任务（可选）

多轮示例需要已有的 AgentENV 服务和可用的 Python 环境模板。沿用上面的客户端配置，执行：

```bash
pip install 'e2b>=2.7'
export AGENTENV_API_URL=http://your-agentenv-gateway:8080
export AGENTENV_TEMPLATE=your-python-template
python cookbook/client/async_rl/multi_turn_grpo.py
```

示例会让模型读取文件、运行测试并修复代码，再根据测试结果训练。每次尝试使用独立的工作区，初始化、评分和关闭环境都已在示例中处理；环境准备见 [Agentic RL 部署与训练](../Agentic%20RL部署与训练.md)。

### 修改为自己的任务

- 数学任务：修改脚本中的数据集和奖励函数。
- 代码任务：修改 `multi_turn_grpo.py` 顶部的 `TASK`，包括待修复文件、测试和提示词。
- 先使用示例默认训练参数；需要调整时，再修改脚本顶部的配置。训练日志会输出奖励和指标，并保存 LoRA 检查点。

## 可训练多轮 Rollout（ClientMultiTurnRollout）

需要自行编写多轮训练循环时，可使用 `twinkle_client.rollout.ClientMultiTurnRollout`。它支持纯文本工具调用，返回可直接用于训练的完整轨迹。

在已创建 rollout、轨迹、工具和策略检查点的异步训练循环中：

```python
rolled = await rollout.acall(
    trajectories,
    tool_manager=tool_managers,
    adapter_uri=policy_uri,
)
```

同步调用使用 `rollout(...)`，参数和结果相同。每条轨迹使用独立环境对应的 `ToolManager`；rollout 不会关闭你传入的环境。完整同步示例见 [multi_turn_rollout.py](https://github.com/modelscope/twinkle/blob/main/cookbook/client/twinkle/multi_turn_rollout.py)。
