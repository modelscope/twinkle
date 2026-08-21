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
# DataLoader 和 Dataset 使用本地 twinkle 或远端 twinkle_client 均可
from twinkle.dataloader import DataLoader        # 或 from twinkle_client.dataloader import DataLoader
from twinkle.dataset import Dataset              # 或 from twinkle_client.dataset import Dataset
from twinkle_client.model import MultiLoraTransformersModel
```

训练循环、数据处理等逻辑完全不需要修改。

## 完整训练示例（Transformers 后端）

```python
import dotenv
dotenv.load_dotenv('.env')

from peft import LoraConfig
from twinkle import get_logger
from twinkle.dataset import DatasetMeta
from twinkle_client import init_twinkle_client

# DataLoader 和 Dataset 使用本地 twinkle 或远端 twinkle_client 均可
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset
from twinkle_client.model import MultiLoraTransformersModel

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

# Step 4: 配置模型
model = MultiLoraTransformersModel(model_id=f'ms://{base_model}')

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
# YOUR_USER_NAME = "your_username"
# hub_model_id = f'{YOUR_USER_NAME}/twinkle-self-cognition'
# model.upload_to_hub(
#     checkpoint_dir=twinkle_path,
#     hub_model_id=hub_model_id,
#     async_upload=False
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

## 可训练多轮 Rollout（ClientMultiTurnRollout）

前面的示例都是单轮训练。如果你要做**带工具调用的多轮 agentic RL**（如 GRPO），并且需要产出可直接用于训练的 token 级对齐信息，可使用 `twinkle_client.rollout.ClientMultiTurnRollout`。它在客户端侧驱动 “采样 → 调用工具 → 拼接上下文 → 再采样” 的多轮循环，每轮采样走 HTTP（`/twinkle/sample`），每条 trajectory 产出带 `logprobs` 的可训练结果，可直接用于 GRPO 等 RL 训练。

### 依赖与约束

- **本地 Template**：bridge token 拼接（渲染工具轮 + 下一轮生成提示）需要在客户端本地持有一个 `Template` 实例。
- **vLLMSampler**：指向服务端 Sampler 服务的客户端采样器。
- **ToolManager**（可选）：注册你的工具；若某条 trajectory 产生了 tool_calls 但未提供 tool_manager，会在派发时抛 `ValueError`。
- **`num_samples=1`**：当前每条 trajectory 只采样一次。做 GRPO group 时，把同一个 prompt 复制成 `NUM_GENERATIONS` 条独立 trajectory 即可。

### 最小示例

```python
from peft import LoraConfig
from twinkle import init_twinkle_client
from twinkle.advantage import GRPOAdvantage
from twinkle.data_format import SamplingParams
from twinkle.template import Qwen3_5Template
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client.rollout import ClientMultiTurnRollout
from twinkle_client.sampler import vLLMSampler

MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
NUM_GENERATIONS = 2   # GRPO group size（rollout 每条采样 num_samples=1）

init_twinkle_client(base_url='http://127.0.0.1:8000', api_key='EMPTY_TOKEN')

# 训练模型（GRPO）
model = MultiLoraTransformersModel(model_id=MODEL_ID)
model.add_adapter_to_model('default', LoraConfig(target_modules='all-linear', r=16, lora_alpha=32))
model.set_loss('GRPOLoss', epsilon=0.2)
model.set_optimizer('Adam', lr=1e-5)
model.set_processor('InputProcessor')
model.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)

# 客户端采样器（HTTP）
sampler = vLLMSampler(model_id=MODEL_ID)
sampler.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)

# 多轮 rollout：需要本地 Template（bridge 拼接）与 ToolManager
rollout_template = Qwen3_5Template(model_id=MODEL_ID, max_length=8192, enable_thinking=False)
rollout_template.truncation_strategy = 'delete'
tool_manager = ToolManager([MyCalculatorTool()])   # 你的工具

rollout = ClientMultiTurnRollout(
    sampler=sampler,
    template=rollout_template,
    tool_manager=tool_manager,
    sampling_params=SamplingParams(max_tokens=512, num_samples=1, logprobs=1, temperature=1.0, top_p=0.95),
    max_turns=4,
)
advantage_fn = GRPOAdvantage()

for step in range(3):
    # 1. 批量多轮 rollout：每个 prompt 复制成 NUM_GENERATIONS 条 trajectory
    trajectories = build_trajectories(tool_manager.tool_infos())  # 见 cookbook
    rolled = rollout(trajectories, tool_manager=tool_manager)

    # 2. 读回 token 级 logprobs（top-1）与 reward
    all_inputs, all_old_logps = [], []
    for traj in rolled:
        all_old_logps.append([lp[0][1] for lp in (traj.get('logprobs') or [])])
        all_inputs.append(traj)
    rewards = compute_rewards(rolled)   # 见 cookbook

    # 3. GRPO 优势（组内相对）
    advantages = advantage_fn(rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()

    # 4. 策略更新
    model.forward_backward(inputs=all_inputs, advantages=advantages, old_logps=all_old_logps)
    model.clip_grad_and_step()
```

### 输出字段

每条返回的 trajectory 在原 dict 基础上追加以下顶层字段：

| 字段 | 含义 |
|:----|:-----|
| `messages` | 完整多轮对话（含 assistant 的 tool_calls 与 tool 响应轮） |
| `logprobs` | 各可训练 token 的 top-1 logprob；本轮未采样 logprobs 时为 `None` |
| `turns` | 实际经历的轮数（`<= max_turns`） |
| `stop_reason` | `'stop'` / `'length'` / `'max_turns'` 之一 |
| `truncated` | 是否因 `max_turns` 或长度上限被截断 |

### 常见错误

- 某条 trajectory 触发了工具调用，但没有提供 `tool_manager` → 抛 `ValueError`。构造时或按调用传入 `tool_manager` 即可。
- 采样器的网络 / 超时错误会**原样抛出**（不被吞掉），重试、退避请在你的循环外层处理。

完整可运行示例见 `cookbook/client/twinkle/multi_turn_rollout.py`。
