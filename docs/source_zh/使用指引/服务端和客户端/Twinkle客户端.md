# Twinkle 客户端

从客户端定义数据、训练目标和评测，Twinkle Server 负责运行模型、更新权重和保存检查点。连接服务需要地址、令牌和服务端公布的模型名。客户端可在 CPU 环境运行。

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

使用 `client.model(...)` 绑定当前客户端的会话与认证。数据处理和训练循环可以沿用该模型服务支持的操作；迁移前核对 loss、优化器、调度器和恢复能力。

## SFT 训练示例

本例需要启用 Qwen3.5-4B、LoRA、`CrossEntropyLoss` 和 `Adam` 的模型服务。

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

# 设置本例的优化器
model.set_optimizer('Adam', lr=1e-4)

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
# SDK 会在等待期间显示传输进度，完成后返回仓库 ID 和提交版本。
# YOUR_USER_NAME = "your_username"
# hub_model_id = f'{YOUR_USER_NAME}/twinkle-self-cognition'
# upload = model.upload_to_hub(
#     checkpoint_dir=twinkle_path,
#     hub_model_id=hub_model_id,
# )
# logger.info(f'Uploaded {upload.repo_id} at {upload.revision}')
```

可传入 `on_progress` 回调，在自己的界面显示已传字节、总量、已提交字节和提交状态。ModelScope 在文件传完或批次提交后更新进度。Hub 去重时，已传字节可能小于已提交字节。服务端通过 `server` 安装选项获取上传依赖。

如果需要断线后继续查询，调用 `model.upload_to_hub(..., client_upload_id=upload_id)` 前先保存一个唯一标识，例如 UUID。重连时使用相同的 Twinkle 凭据、标识、检查点和目标仓库。服务端仍保留该任务时，SDK 会继续等待或返回原有结果。如果任务已失败，使用新的标识发起上传。

Twinkle Client 场景下，推荐的断点续训流程是：

1. 先通过 `client.list_checkpoints(...)` 或 `client.get_latest_checkpoint_path(...)` 获取已有 checkpoint 路径。
2. 调用 `model.resume_from_checkpoint(resume_path)` 恢复权重、优化器、调度器、随机数状态和训练进度元数据。
3. 使用返回结果中的 `consumed_train_samples` 调用 `dataloader.resume_from_checkpoint(...)`，跳过已经训练过的数据。

完整示例可直接参考 `cookbook/client/twinkle/self_cognition.py`。

## 模型服务与训练能力

`client.model(...)` 使用统一的 HTTP 接口。模型名决定请求发往哪个服务，服务端配置决定该服务怎样运行。用 `client.get_server_capabilities().supported_models` 查看模型目录。当前返回的协议特性和限制属于服务级信息，模型支持哪些 loss、优化器和状态恢复仍需查该服务的说明。

优先用 `forward_backward` 完成前反向，用 `forward_only` 做独立打分。部分服务将训练前反向融合执行，因而无法单独调用训练前向或反向。优化器和调度器也有不同的可用选项。运行教程前确认模型服务提供所需能力，训练循环按目标和数据编写即可。

## 客户端训练学习路线

每章先回答一个训练问题，再看代码里哪条数据决定更新方向。固定一小批留出题，记录训练前后结果，再换成自己的任务。这种写法可以参考 Agentic-RL-Lab 的 OPSD 章节，它把做法、可运行步骤、逐 checkpoint 评测和结果局限放在一起。[Agentic-RL-Lab OPSD](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/opsd/readme.md)

| 顺序 | 练习和模型 | 训练问题 | 先看什么 |
|---|---|---|---|
| 1. SFT | 自我认知微调（`cookbook/client/twinkle/self_cognition.py`），Qwen3.5-4B | 给定标准答案，模型学哪些 token？ | 检查助手回复的标签，比较同一组问题的输出。 |
| 2. DPO（选学） | 偏好微调（`cookbook/client/twinkle/dpo.py`），Qwen3.5-4B | 同一道题有两个答案，怎样让模型偏向更好的那个？ | 检查成对回答，再测留出偏好对。 |
| 3. GRPO | 短答案数学（`cookbook/client/twinkle/short_math_grpo.py`），Qwen3.5-4B | 只有答案的奖励，怎么决定哪些采样结果值得加强？ | 看一组同题回复的分数和 advantage，再查正确率与长度。 |
| 4. 在线蒸馏 | [下方的短乘法练习](#client-online-distillation)，Qwen3.5-0.8B 学生，外部教师方法用 Qwen3.5-2B | 教师是谁、看到什么？它的分数怎样更新学生？ | 核对教师与学生打分的 token，再比较留出题。 |
| 5. Agentic RL | [下方的 Blackjack](#client-async-grpo)，Qwen3.5-4B | 一次训练样本变成工具交互轨迹后，奖励和标签落在哪里？ | 先读一条含工具调用、观察和终局奖励的轨迹，再查完成率与环境奖励。 |

用自己的任务串起这几章时，可以让 SFT 先教会输出格式或工具调用，再从该权重出发做奖励训练。保留同一批留出题，任务变化时先验证答案或工具结果，再调整奖励与模型。各示例可以独立运行。

(client-online-distillation)=
## 在线蒸馏入门

`cookbook/client/twinkle/online_distillation.py` 用短乘法演示三种方法。以 `17 × 23` 为例，学生先生成自己的回答。GKD 在外部教师选出的候选 token 上比较两边的分布，OPD-RL 只比较实际选中 token 的 logprob，OPSD 让当前学生读完参考解法，再给原回答打分。三个方法共用采样、更新、发布和评测。先读 `train()` 看这条主线，再读 `distillation_signal()` 看信号如何构造。

向服务管理员获取地址、令牌和兼容的模型服务。示例使用 `distill-student`（Qwen3.5-0.8B）。OPD-RL 和 GKD 另需 `distill-teacher`（冻结的 Qwen3.5-2B）。服务端参考配置位于 `cookbook/client/server/transformer/distillation.yaml`。

在与本指南同一版本的仓库根目录安装客户端，填入管理员提供的地址、令牌和模型名：

```bash
python -m pip install -e '.[client]'
export TWINKLE_SERVER_URL=https://your-twinkle-server
export TWINKLE_SERVER_TOKEN=your-token
python cookbook/client/twinkle/online_distillation.py --method opsd --model distill-student --steps 4
```

OPSD 从采样结果取原始回复 token，把它们连同带解法的提示交给服务端打分。参考 YAML 为训练和采样固定相同的模板。客户端可直接使用服务端解码的文本评测，只需 CPU 和客户端依赖。

跑 OPD-RL 或 GKD 时，仍需上面两项连接配置，以及本地的学生与教师模板文件和匹配的 Swift 版本：

```bash
python -m pip install 'ms-swift @ git+https://github.com/modelscope/ms-swift.git@e1ac89d24f47cb8fd73a7245dd79df0959aec284' 'qwen-vl-utils>=0.0.14'
export TWINKLE_DISTILL_STUDENT_PATH=/path/to/student-template-assets
export TWINKLE_DISTILL_TEACHER_PATH=/path/to/teacher-template-assets
export TWINKLE_DISTILL_SERVER_STUDENT_PATH=/server/path/to/student-model
python cookbook/client/twinkle/online_distillation.py --method opd-rl --model distill-student --steps 4
```

将方法改为 `gkd` 可运行 GKD。教师默认在每个位置选出最可能的 32 个 token，概率在这 32 个候选内归一化。用 `--gkd-top-k 64` 可以比较更多候选。比较 `eval=before` 和 `eval=after` 的同一批题，检查答对数和输出长度。四步用于学会跑流程。比较收益时需要匹配训练样本数和评测设置，并用多个种子复跑。

(client-async-grpo)=
## 客户端异步 GRPO

客户端可以在采样的同时进行训练。完成 [Twinkle 安装](../安装.md) 后，选择一个示例，按下面的步骤启动即可：

| 场景 | 示例 |
|---|---|
| 流式单轮数学问答（GSM8K） | `cookbook/client/async_rl/client_orchestrated_grpo.py` |
| 工具多轮 Blackjack（StreamingDriver，显式同步训练循环） | `cookbook/client/async_rl/streaming_multi_turn_grpo.py` |
| E2B sandbox 中的 Blackjack（复用相同训练循环） | `cookbook/client/async_rl/e2b_multi_turn_grpo.py` |

### 连接服务并运行单轮数学任务

这两份示例需要管理员提供可用于 GRPO 和采样的 Qwen3.5-4B 服务。客户端可复用下文 Blackjack 的安装命令。

在客户端设置连接信息，从仓库根目录执行：

```bash
export TWINKLE_SERVER_URL=https://your-twinkle-server
export TWINKLE_SERVER_TOKEN=your-token
TWINKLE_MAX_STEPS=6 python cookbook/client/twinkle/short_math_grpo.py

# 进阶练习：用 StreamingDriver 让采样和训练重叠
TWINKLE_MAX_STEPS=6 python cookbook/client/async_rl/client_orchestrated_grpo.py
```

两份示例都使用 GSM8K。同步版适合第一次读 GRPO 循环。答对得 1 分，答对后最多再加 0.25 分的简短奖励。例如同组奖励为 `[1.25, 1, 0, 0]`，前两条的优势为正，后两条为负。试着把简短奖励系数改成 0，再用同一批留出题比较正确率和输出 token 数。流式版只按答案正误打分。训练上一组时，`StreamingDriver` 可以继续采样和评分。这个示例的 `TWINKLE_MAX_STEPS` 限制交给训练的组数；组内奖励相同就跳过更新。

离线时，两份示例都可设置 `TWINKLE_DATASET_ID` 为本地数据目录，并用 `TWINKLE_TEMPLATE_MODEL_ID` 指定与服务模型匹配的 tokenizer 和模板目录。

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
