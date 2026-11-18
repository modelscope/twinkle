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

## 客户端训练学习路线

每一章都可以先用一批固定的留出题测基线，训练后再测同一批。把训练数据、评分和评测分开，换自己的任务时仍能判断哪一步起了作用。

| 顺序 | 练习 | 看懂什么 |
|---|---|---|
| 1. SFT | [自我认知微调](https://github.com/modelscope/twinkle/blob/main/cookbook/client/twinkle/self_cognition.py) | 用标准答案做下一 token 预测。看为什么只给助手回复打标签，以及如何用 LoRA 更新局部参数。 |
| 2. DPO（选学） | [偏好微调](https://github.com/modelscope/twinkle/blob/main/cookbook/client/twinkle/dpo.py) | 同一道题配较好和较差的回答，学习如何增加相对偏好。 |
| 3. GRPO | [短答案数学](https://github.com/modelscope/twinkle/blob/main/cookbook/client/twinkle/short_math_grpo.py) | 让模型对同一道题生成多个答案，用对错和长度给分，再用组内相对奖励决定更新方向。 |
| 4. 在线蒸馏 | [下方的短乘法练习](#client-online-distillation) | 保留学生自己的回复，对比整词表分布、逐 token advantage 和自蒸馏三种训练信号。 |
| 5. Agentic RL | [下方的 Blackjack](#client-async-grpo) | 将一条回复扩展为多轮轨迹。工具返回观察，环境给奖励，训练用实际生成的 token，采样与训练可以重叠。 |

换成自己的任务时，先替换题目和标准答案，再写能独立检查的奖励或参考解法。接入工具后，用一条完整轨迹核对模型看到的消息、工具返回和终止条件。逐步增加组大小和并发，用留出题查正确率、格式、工具成功率与输出成本。把这几章串成自己的项目时，让 SFT 先教会模型输出格式或工具调用，再从该权重出发做奖励训练，逐阶段保留 checkpoint 和同题评测。

(client-online-distillation)=
## 在线蒸馏入门

[同一份客户端脚本](https://github.com/modelscope/twinkle/blob/main/cookbook/client/twinkle/online_distillation.py) 用 Qwen3.5-0.8B 学习短乘法，关闭 thinking。先看 `train()` 的采样、更新、发布和评测，再看 `distillation_signal()` 如何为同一批回复构造训练信号。

| 方法 | 教师信号 | 服务端设备 |
|---|---|---|
| OPSD | 当前学生看参考解法，对原回复逐 token 打分。 | 两张卡，运行 Transformers 和 vLLM |
| OPD-RL | 冻结的 Qwen3.5-2B 对原回复打分。教师 logprob 减去学生采样 logprob，得到每个 token 的 advantage。 | 增加一张教师卡 |
| GKD | 同一冻结教师给出整词表分布。每步一题，控制全词表 logits 的 HTTP 传输量。 | 同 OPD-RL |

从包含此示例和 GKD HTTP 支持的 Twinkle 源码目录运行。CUDA 服务端使用 `python -m pip install -e '.[client,rl,server]'`。独立的客户端环境使用 `python -m pip install -e '.[client]'`。两端都需要 Swift dev 的模板接口，可安装下面的版本。使用 PPU 时先按设备文档配置专用 PyTorch 和 vLLM 环境。

```bash
python -m pip install 'ms-swift @ git+https://github.com/modelscope/ms-swift.git@e1ac89d24f47cb8fd73a7245dd79df0959aec284' 'qwen-vl-utils>=0.0.14'
modelscope download --model Qwen/Qwen3.5-0.8B --local_dir ./models/Qwen3.5-0.8B
export TWINKLE_DISTILL_STUDENT_PATH="$PWD/models/Qwen3.5-0.8B"
CUDA_VISIBLE_DEVICES=0,1 twinkle-server launch -c cookbook/client/server/transformer/distillation.yaml
```

等服务端预热 vLLM 并显示 `All applications started`。在另一个终端执行：

```bash
export TWINKLE_DISTILL_STUDENT_PATH="$PWD/models/Qwen3.5-0.8B"
python cookbook/client/twinkle/online_distillation.py --method opsd --steps 4
```

运行 OPD-RL 或 GKD 时，在两端准备 `Qwen/Qwen3.5-2B` 并通过 `TWINKLE_DISTILL_TEACHER_PATH` 指向本地模型目录。用 `CUDA_VISIBLE_DEVICES=0,1,2` 启动同一份[配置](https://github.com/modelscope/twinkle/blob/main/cookbook/client/server/transformer/distillation.yaml)，将 `--method` 改为 `opd-rl` 或 `gkd`。远程连接沿用上文的服务地址和令牌配置。如果两端的学生模型目录不同，在客户端另设 `TWINKLE_DISTILL_SERVER_STUDENT_PATH` 为服务端的学生目录。

比较 `eval=before` 和 `eval=after` 中同一留出题的准确率和输出长度。答案严格匹配单个整数，分数包含格式遵循。四步用来读懂训练流程。做效果对比时固定留出题、实际训练样本数和采样设置，并用不同种子复跑。

(client-async-grpo)=
## 客户端异步 GRPO

客户端可以在采样的同时进行训练。完成 [Twinkle 安装](../安装.md) 后，选择一个示例，按下面的步骤启动即可：

| 场景 | 示例 |
|---|---|
| 单轮数学问答（GSM8K） | [client_orchestrated_grpo.py](https://github.com/modelscope/twinkle/blob/main/cookbook/client/async_rl/client_orchestrated_grpo.py) |
| 工具多轮 Blackjack（StreamingDriver，显式同步训练循环） | [streaming_multi_turn_grpo.py](https://github.com/modelscope/twinkle/blob/main/cookbook/client/async_rl/streaming_multi_turn_grpo.py) |
| E2B sandbox 中的 Blackjack（复用相同训练循环） | [e2b_multi_turn_grpo.py](https://github.com/modelscope/twinkle/blob/main/cookbook/client/async_rl/e2b_multi_turn_grpo.py) |

### 连接服务并运行单轮数学任务

这些示例需要管理员提供可用于 GRPO 和采样的 Qwen3.5-4B 服务。单轮异步数学示例还需要服务端 DataPlane。[服务端参考 YAML](https://github.com/modelscope/twinkle/blob/main/cookbook/client/async_rl/server_config.yaml)

在客户端设置连接信息，从仓库根目录执行：

```bash
export TWINKLE_SERVER_URL=https://your-twinkle-server
export TWINKLE_SERVER_TOKEN=your-token
python cookbook/client/async_rl/client_orchestrated_grpo.py
```

默认使用 GSM8K。离线时可设置 `TWINKLE_DATASET_ID` 为本地数据目录，`TWINKLE_TEMPLATE_MODEL_ID` 为与服务模型匹配的 tokenizer 与模板目录。

### 运行 Blackjack 多轮 GRPO

模型调用 `play(hit/stand)`，环境返回手牌和终局奖励。`MultiTurnRollout` 把每轮实际生成的 token 和 logprob 记到轨迹上，`HTTPSampler` 负责把各轮采样请求发给 Twinkle 服务。同步示例收齐一组牌局再训练，流式示例由 `StreamingDriver` 调度完整组。

沿用上面的服务地址和令牌。默认模型是 `Qwen/Qwen3.5-4B`，可通过 `TWINKLE_MODEL_ID` 指定服务配置里的模型。客户端使用与服务一致的 tokenizer 和模板。服务模型名是别名，或客户端离线运行时，将 `TWINKLE_TEMPLATE_MODEL_ID` 设为对应的模型 ID 或本地模板目录。

先在仓库根目录安装公共客户端依赖。运行 E2B 版本也需要这一步。

```bash
python -m pip install -e '.[client]' python-dotenv \
  'ms-swift @ git+https://github.com/modelscope/ms-swift.git@e1ac89d24f47cb8fd73a7245dd79df0959aec284'
```

同步版和 OpenEnv 流式版还需安装本地牌局环境。

```bash
pip install 'openenv==0.11.0' 'open_spiel==2.0.2'
pip install 'openenv-openspiel-env @ git+https://github.com/huggingface/OpenEnv.git@9c6376390489f28e0e545cf74a277064e7edd52f#subdirectory=envs/openspiel_env'

# 在仓库根目录二选一
python -m cookbook.client.twinkle.multi_turn_rollout
python cookbook/client/async_rl/streaming_multi_turn_grpo.py
```

同组牌局从相同的发牌种子开始，各自维护独立状态。只使用终局完整的组，整组同分时跳过更新。`train_batch()` 展示优势计算、反向传播、参数更新和 LoRA 发布。结束时查看 `completed_groups`、`optimizer_updates` 和 `latest_lora`。如果更新次数为 0，`latest_lora` 指向初始策略。

### 在 E2B sandbox 里运行同一个任务

安装上面的公共客户端依赖后，在 `.env` 设置 `E2B_API_KEY`。使用官方 E2B 服务时，留空端点覆盖配置。

```bash
pip install -e '.[client,e2b]'
python cookbook/client/async_rl/e2b_multi_turn_grpo.py
```

该入口复用流式训练循环，每条轨迹有自己的 sandbox。发牌和奖励在 sandbox 内通过 OpenSpiel 计算，客户端只上传游戏程序。默认使用 `base` 模板，需在 sandbox 内安装 `open_spiel==2.0.2`。可用 `E2B_TEMPLATE` 指定预装该版本的模板以缩短启动时间。本机无需为此入口安装 OpenSpiel。

### 换成自己的工具任务

参考 `BlackjackEnv` 或 `E2BBlackjackEnv`，定义环境的工具、会话初始观察和奖励，把环境传给 `MultiTurnRollout`。同步调用 `rollout(trajectories)` 可一次拿到完整轨迹。改用流式调度时，把同一个 rollout 传给 `StreamingDriver` 并迭代 `(rows, facts)`。两种入口共用 `ClientCheckpointManager` 来发布新策略，使用 `RolloutResources` 按顺序关闭 driver、rollout 和环境，释放训练句柄后再关闭客户端。
