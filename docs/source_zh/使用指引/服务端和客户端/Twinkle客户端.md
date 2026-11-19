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
| 工具多轮 Blackjack（StreamingDriver，显式同步训练循环） | [streaming_multi_turn_grpo.py](https://github.com/modelscope/twinkle/blob/main/cookbook/client/async_rl/streaming_multi_turn_grpo.py) |
| E2B sandbox 中的 Blackjack（复用相同训练循环） | [e2b_multi_turn_grpo.py](https://github.com/modelscope/twinkle/blob/main/cookbook/client/async_rl/e2b_multi_turn_grpo.py) |

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

### 3. 运行 Blackjack 流式多轮示例（可选）

`streaming_multi_turn_grpo.py` 用 Blackjack（21 点）演示客户端多轮异步 GRPO：模型调用 `play(hit/stand)`，环境返回手牌和终局奖励，完整样本组进入训练。本地环境直接调用 `blackjack_game.py` 中的 OpenSpiel 游戏，无需环境服务或 OpenEnv 适配包。

客户端运行 `StreamingDriver` 和 `MultiTurnRollout`：Driver 调度 episode 并交付完整组，Rollout 管理多轮交互和环境会话，`HTTPSampler` 将每轮生成请求发给服务端。训练调用同样通过客户端模型接口发往服务端。

沿用上面的 Twinkle 服务和客户端配置，安装本地游戏依赖后运行：

```bash
pip install 'open_spiel==2.0.2'
python cookbook/client/async_rl/streaming_multi_turn_grpo.py
```

客户端需要与当前分支匹配的 Swift 模板依赖（提供 `swift.dev.template_contract`），并使用与服务端一致的 tokenizer/template。可先检查：

```bash
python -c "from twinkle.template import Qwen3_5Template"
```

同组成员使用相同的环境 seed，各自持有独立动作历史；模型采样不固定 seed。环境实现会话式 `Env` 的 `_open/_step`，由 Rollout 自动调用 `reset/step/close`。牌局结束状态在会话关闭前读取，累计奖励保存在轨迹的 `rewards.env` 中；未完成或截断的成员使整组在交付前被丢弃。

先读 `train()` 的组装，再读 `_train_loop()`：`for rows, facts in driver` 取完整组 → 将环境奖励接入命名 scores → `GRPOAdvantage` 生成 `advantages/old_logps` → `forward_backward` → `clip_grad_and_step` → `ClientCheckpointManager.sync_weights()`。整组同分时跳过参数更新；短运行不保证产生更新或 reward 上升。

本例每消费一个 batch 就发布一次快照，包括零 advantage 的 batch，以推进 Driver 的采样额度。因此发布版本数与实际更新次数分别统计：`MAX_STEPS` 限制已确认的 optimizer 更新，Driver 的 `max_steps` 则是交付批次数，本例不设置后者。每次运行自动生成检查点名称前缀，避免覆盖其他运行的快照。

`ClientCheckpointManager` 保存服务端 LoRA 并切换 `HTTPSampler` 默认 URI。已发出的请求继续使用原 URI，后续请求读取新 URI；同一条多轮轨迹可以跨发布版本。Rollout 不设置固定 `adapter_path`。

退出时先关闭 Driver，脚本中的 `_RolloutTasks` 等待已提交的 step/stop Future 返回，然后 `RolloutResources` 关闭 Rollout、环境组件和客户端。这个辅助类只管理待结束的调用，多轮交互仍由通用 Rollout 执行。模型训练和保存操作继续使用 SDK 的中断保护；`model.completed_step_calls` 记录成功收到回复的更新次数。等待本地调用结束不等同于远端强制取消。

采样固定 `temperature=1`、`top_p=1`、`top_k=-1`、`repetition_penalty=1`，保证采样 logprobs 与训练侧概率口径一致。共享 `server_config.yaml` 仍部署 DataPlane，服务端仍需 `[async-rl]` 依赖。

### 4. 在 E2B 官方 sandbox 中运行同一任务（可选）

`e2b_multi_turn_grpo.py` 用 `AgentEnv` 连接 E2B 官方服务，每条正在执行的 episode 独占一个 sandbox 会话；游戏历史、发牌和奖励都在 sandbox 中计算。模型仍只看到 `play(hit/stand)`，由通用 `MultiTurnRollout`、`HTTPSampler` 和 `StreamingDriver` 编排。训练循环直接复用上一例的 `train(env_factory=...)`，没有另建训练框架或评分器，也没有额外的 Agent harness。

先按 [AgentEnv 官方 E2B 连通检查](../../组件/Agentic/Envs.md) 验证创建、执行和释放。这一步不需要训练服务，可以先排除密钥、模板和网络问题。

保持前面启动的 Twinkle 服务运行，在仓库根目录的 `.env` 中配置两套服务的凭据。`TWINKLE_SERVER_TOKEN` 用于训练/采样服务，`E2B_API_KEY` 用于云 sandbox，两者不能互换：

```dotenv
TWINKLE_SERVER_URL=http://127.0.0.1:8000
TWINKLE_SERVER_TOKEN=EMPTY_TOKEN
TWINKLE_MODEL_ID=Qwen/Qwen3.5-4B
E2B_API_KEY=your_e2b_api_key
# 离线客户端还需设置与服务端模型匹配的本地模板目录：
# TWINKLE_TEMPLATE_MODEL_ID=/absolute/path/to/Qwen3.5-4B
```

将地址、令牌和密钥替换为自己的配置，`.env` 不要提交。使用官方服务时，删除 `.env` 中的 `E2B_API_URL`、`E2B_SANDBOX_URL`、`E2B_DOMAIN` 覆盖配置，并确认终端中也没有设置这些变量。然后运行：

```bash
pip install -e '.[client,e2b]'
python cookbook/client/async_rl/e2b_multi_turn_grpo.py
```

默认使用 E2B 的 `base` 模板，在可写的 `/home/user/blackjack` 中安装 `open_spiel==2.0.2` 并上传同目录的 `blackjack_game.py`。客户端无需安装 OpenEnv/OpenSpiel。可通过 `E2B_TEMPLATE` 指定已预装该版本 OpenSpiel 的自定义模板，减少每条轨迹的初始化耗时。环境默认最多同时持有 `(MAX_STALENESS + 1) × NUM_GENERATIONS = 8` 个会话，应与账户的 sandbox 并发额度匹配。

读代码时，按下面三个文件的职责看即可：

直接运行入口即可，示例会组装 rollout 和 driver。任务提示词和工具来自 `BlackjackTask`；更换任务时修改环境会话 hooks、任务输入和奖励接入。

| 文件 | 负责什么 |
| --- | --- |
| `streaming_multi_turn_grpo.py` | `train()` 连接服务并建立 rollout/driver；`_train_loop()` 展示优势、反向传播、optimizer step 和策略发布。组装任务提示词、环境会话和 `tools()`，游戏规则由 OpenSpiel 提供。 |
| `e2b_multi_turn_grpo.py` | `E2BBlackjackEnv` 实现会话式 `_open/_step` hooks，由 `AgentEnv` 管理 sandbox，最后用 `train(env_factory=E2BBlackjackEnv)` 切换环境。 |
| `blackjack_game.py` | 两个环境共用的 Blackjack 任务定义（`BlackjackTask.system_prompt` / `tools()`），同时也是上传到 sandbox 的游戏程序，调用 OpenSpiel；在 sandbox 的 `game.json` 保存 seed/动作历史，返回手牌、奖励和终态。它不参与训练。 |

运行时：driver 提交独立牌局 → rollout 请求远端 sampler → 模型调用 `play` → AgentEnv 在 E2B 中执行游戏 → 完整组计算优势并训练 → 发布的新 URI 用于后续生成请求。同组 seed 相同、环境会话独立；同一条轨迹的后续轮次可以使用新版本。

`env_factory(max_sessions=...)` 每次训练创建一个环境组件，组件为每条 episode 建立独立会话。两个后端共用提示词、`play` 工具、终局检查和奖励规则。更换任务时，实现新的环境 hooks，并调整 `_build_driver()` 的任务输入和 `_train_loop()` 的 scores；通用 `MultiTurnRollout` 支持多个工具。

规模由 `NUM_GENERATIONS`、`MAX_STALENESS`、`MAX_GROUPS`、`MAX_STEPS` 控制：每组 episode 数、允许落后的发布版本数、有限任务组数和最多实际更新次数。示例每批交付一个完整组、每批发布一次。减小 `MAX_STALENESS` 会缩小初始采样窗口及环境会话上限，同时减少可用于重叠的后续工作。

episode 完成后关闭会话；`AgentEnv` 清理工作目录并保留健康 sandbox 供后续会话复用。退出时先停止 Driver 并等待已提交的调用，再关闭 Rollout、环境会话和空闲 sandbox。示例执行固定游戏工具，不执行模型生成的代码，也不提供即时远端强制取消。

游戏进程失败或返回损坏结果时会中止训练，不会作为零奖励或普通未完成牌局跳过。模型给出无效工具参数则返回提示，允许其纠正。

| 现象 | 先检查什么 |
| --- | --- |
| E2B 鉴权或模板错误 | `.env` 中的 `E2B_API_KEY`、`E2B_TEMPLATE` 和端点覆盖；先单独重跑连通检查。官方默认模板为 `base`。 |
| sandbox 数量/额度不足 | 调小 `MAX_STALENESS` 或 `NUM_GENERATIONS`；默认会话上限为 `2 × 4 = 8`，并检查账户额度。 |
| 初始化慢 | 新会话创建或复用 sandbox，并检查/安装 OpenSpiel；可使用预装依赖且支持该工作目录的 `E2B_TEMPLATE`。 |
| 连通检查成功，训练连接失败 | 检查 `TWINKLE_SERVER_URL/TOKEN`、服务端模型和本地模板目录；E2B 连通不代表训练服务就绪。 |
| 运行了牌局但没有 optimizer 更新 | 检查日志中的 `zero advantage` 和 Driver 的 `dropped_sample/dropped_group` 统计；不要把有限尝试内的合法跳过当作连接失败。 |

### 修改为自己的任务

- 数学任务：修改脚本中的数据集和奖励函数。
- 流式工具任务：从 Blackjack 示例的环境、工具和环境 reward 接缝替换任务，保留完整组、优势、训练和发布流程。
- 先使用示例默认训练参数；需要调整时，再修改脚本顶部的配置。训练日志会输出奖励和指标，并保存 LoRA 检查点。

## 客户端多轮 Rollout

当前 CS 模式使用 `twinkle_agentic.rollout.MultiTurnRollout` 搭配 `twinkle_client.rollout.HTTPSampler`，权重发布使用 `ClientCheckpointManager`。旧的 `ClientMultiTurnRollout` 和 `training_groups` 已删除。

需要自定义多轮异步训练时，从上面的 [streaming_multi_turn_grpo.py](https://github.com/modelscope/twinkle/blob/main/cookbook/client/async_rl/streaming_multi_turn_grpo.py) 开始：保留 Driver 和 HTTP 连接，替换任务输入、环境和奖励。`MultiTurnRollout.step()` 推进一轮交互，Driver 在完整组就绪后交付训练。
