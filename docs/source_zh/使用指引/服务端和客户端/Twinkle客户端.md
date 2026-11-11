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

`streaming_multi_turn_grpo.py` 复用已有多轮示例的 Blackjack（21 点）任务：模型调用 `play(hit/stand)`，环境返回手牌和最终奖励，再按完整组计算 GRPO 优势、训练并发布新策略。OpenEnv/OpenSpiel 在客户端进程中运行，无需启动环境服务，也不执行模型生成的代码。

`ClientMultiTurnRollout` 控制每条任务的模型/工具对话，`StreamingDriver` 调度并发任务并收集完整训练组；本例没有额外的 Agent harness。进程内 OpenEnv、远程 OpenEnvClient 和 AgentENV 都适配统一的 `Env` 接口，进程/OS 隔离能力取决于具体环境后端及其部署。

安装客户端环境依赖后，沿用上面的 Twinkle 服务和客户端配置：

```bash
pip install 'openenv==0.11.0' 'open_spiel==2.0.2'
pip install 'openenv-openspiel-env @ git+https://github.com/huggingface/OpenEnv.git@9c6376390489f28e0e545cf74a277064e7edd52f#subdirectory=envs/openspiel_env'
python cookbook/client/async_rl/streaming_multi_turn_grpo.py
```

运行前需要配置与当前分支匹配的 Swift 模板依赖（提供 `swift.dev.template_contract`），并使用与服务端一致的 tokenizer/template。可先检查模板是否能导入；该命令不创建模型或 sandbox：

```bash
python -c "from twinkle.template import Qwen3_5Template"
```

如果提示缺少模板契约，应先补齐匹配的 Swift 环境；`[client,e2b]` 选项负责客户端和 E2B 连接依赖，不代替 Swift 环境配置。

同组使用相同的环境 seed、独立环境和不同的模型采样 seed。未结束的牌局和整组同分的数据会跳过训练；脚本不保证短运行一定产生更新或 reward 上升。训练循环在 `_train_loop` 中直接展示，每个完整组执行一步；顶部常量控制组大小、缓冲组数和更新预算。

先读 `train()` 的连接和初始化，再读 `_train_loop()` 的完整组 → 优势计算 → `forward_backward` → `clip_grad_and_step` → 请求发布 → `commit`。`commit(0)` 表示本次没有更新，不消耗训练步数；发布请求在 driver 恢复推进后执行。更换任务时再看 `_build_driver()` 的环境和奖励适配。资源由 SDK 的 `RolloutResources` 作用域管理，示例不需要编写异常收尾逻辑。

每次调用 `train()` 都自动生成独立的检查点名称前缀，避免重复运行覆盖 `policy-0` 等快照，无需用户配置。此前缀只用于文件命名，不提供租户隔离，也不控制随机种子或断点恢复。

`rollout.streaming(driver)` 负责关闭 streaming admission、广播停止、尝试归集全部调用，再联归 driver；退出后该实例不再接受 streaming admission。底层 rollout 默认借用环境，不自动关闭调用方资源；本例通过 `with RolloutResources(init_twinkle_client) as resources` 显式把客户端、rollout 和环境交给 SDK 管理。`own_rollout()` 登记专用 rollout，`own_environment()` 在 `reset()` 前登记环境，初始化失败也可收尾；收集完成后用 `close_environment()` 提前释放该轨迹的环境。退出作用域时 SDK 等待调用、排干 rollout、关闭剩余环境和客户端，处理重复中断并保留原异常。执行状态未确认时保留资源句柄并报告错误，不能按“已经关闭”处理。

脚本固定 `temperature=1`、`top_p=1`、`top_k=-1`、`repetition_penalty=1`，使共享配置的 processed logprobs 与训练侧概率口径一致。共享 `server_config.yaml` 仍部署 DataPlane，服务端仍需 `[async-rl]` 依赖。

### 4. 在 E2B 官方 sandbox 中运行同一任务（可选）

`e2b_multi_turn_grpo.py` 用 `AgentEnv` 连接 E2B 官方服务，每条轨迹独占一个 sandbox；游戏历史、发牌和奖励都在 sandbox 中计算。模型仍只看到 `play(hit/stand)`，由相同的 `ClientMultiTurnRollout` 和 `StreamingDriver` 编排。训练循环直接复用上一例的 `train(env_factory=...)`，没有另建训练框架或评分器，也没有额外的 Agent harness。

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

默认使用 E2B 的 `base` 模板，在可写的 `/home/user/blackjack` 中安装 `open_spiel==2.0.2` 并上传同目录的 `blackjack_game.py`。客户端无需安装 OpenEnv/OpenSpiel。可通过 `E2B_TEMPLATE` 指定已预装该版本 OpenSpiel 的自定义模板，减少每条轨迹的初始化耗时。默认缓冲最多 12 条轨迹，应与账户的 sandbox 并发额度匹配。

读代码时，按下面三个文件的职责看即可：

直接运行入口即可，示例会组装 rollout 和 driver，首次使用无需手动创建这些组件；环境提供 `system_prompt` 和标准 `Env.tools()` 工具定义，更换任务时再查看 `_build_driver()` 中的 `submit/collect` 两个接缝。

| 文件 | 负责什么 |
| --- | --- |
| `streaming_multi_turn_grpo.py` | `train()` 连接服务并建立 rollout/driver；`_train_loop()` 展示优势、反向传播、optimizer step 和策略发布。从环境读取任务提示词和 `tools()`，不定义 Blackjack 规则。 |
| `e2b_multi_turn_grpo.py` | `E2BBlackjackEnv` 把标准 `reset/step/close` 接口接到 E2B，最后用 `train(env_factory=E2BBlackjackEnv)` 切换环境。 |
| `blackjack_game.py` | 两个环境共用的 Blackjack 任务定义（`BlackjackTask.system_prompt` / `tools()`），同时也是上传到 sandbox 的游戏程序，调用 OpenSpiel；在 sandbox 的 `game.json` 保存 seed/动作历史，返回手牌、奖励和终态。它不参与训练。 |

运行时：driver 提交独立牌局 → rollout 请求远端 sampler → 模型调用 `play` → AgentEnv 在 E2B 中执行游戏 → 完整组计算优势并训练 → 发布的新 URI 用于后续牌局。同组 seed 相同、环境独立；每条轨迹固定使用同一个策略 URI。

`env_factory` 用于切换同一 Blackjack 任务的本地或 E2B 执行后端；两个后端共用提示词、`play` 工具和奖励规则。修改任务时，先定义环境的提示词、工具与奖励，再调整 `_build_driver()` 的 `submit/collect`，训练主循环可以保留。此示例明确要求一个 `play` 工具；空工具、额外工具或不同工具名会报错，避免静默忽略工具。SDK 的 `ClientMultiTurnRollout` / `ToolManager` 仍支持多工具任务，不受此示例限制。

上手只需先改服务连接配置。需要调节规模时，再改共享训练脚本顶部的 `NUM_GENERATIONS`、`BUFFER_GROUPS`、`MAX_GROUPS`、`MAX_STEPS`：分别控制每组轨迹数、缓冲组数、有限尝试组数、最多更新次数。示例每次拉取一个完整组；整组同分会跳过，因此 `MAX_STEPS` 是上限。缩小 `BUFFER_GROUPS` 可以减少同时持有的 sandbox，代价是较少的采样/训练重叠。

轨迹收集完成后立即关闭其 sandbox；退出时先停止并排干 rollout，再关闭尚未收集的环境。取消采用协作停止，已发出的命令需等待返回。未确认执行结束时保留句柄并报错，sandbox 的 600 秒超时作为兜底，不宣称即时远端中止。该例演示真实 sandbox 工具调用，不执行模型生成的代码。

游戏进程失败或返回损坏结果时会中止训练，不会作为零奖励或普通未完成牌局跳过。模型给出无效工具参数则返回提示，允许其纠正。

| 现象 | 先检查什么 |
| --- | --- |
| E2B 鉴权或模板错误 | `.env` 中的 `E2B_API_KEY`、`E2B_TEMPLATE` 和端点覆盖；先单独重跑连通检查。官方默认模板为 `base`。 |
| sandbox 数量/额度不足 | 调小 `BUFFER_GROUPS`；默认预算为 `3 × 4 = 12` 条轨迹，并检查账户额度。 |
| 初始化慢 | 默认每条轨迹创建 sandbox 并检查/安装 OpenSpiel；可使用预装依赖且支持该工作目录的 `E2B_TEMPLATE`。 |
| 连通检查成功，训练连接失败 | 检查 `TWINKLE_SERVER_URL/TOKEN`、服务端模型和本地模板目录；E2B 连通不代表训练服务就绪。 |
| 运行了牌局但没有 optimizer 更新 | 检查日志中的 `zero advantage` / `truncated`；不要把有限尝试内的合法跳过当作连接失败。 |

### 修改为自己的任务

- 数学任务：修改脚本中的数据集和奖励函数。
- 流式工具任务：从 Blackjack 示例的环境、工具和环境 reward 接缝替换任务，保留完整组、优势、训练和发布流程。
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
