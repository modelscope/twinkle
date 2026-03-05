# Qwen3.5 训练最佳实践

本文以 Qwen3.5-4B 为例，演示 Twinkle 框架的核心能力：**一套组件化代码，从单卡训练到Client-Server环境**。

---

## 一、Twinkle 是什么

Twinkle 是一个面向生产的大模型训练框架。它的核心设计非常容易理解：**训练逻辑用 Python 代码表达，运行模式通过初始化参数切换**。

这意味着：
- 实验室里写的训练脚本，改一行代码就能多方式运行
- 全开放的算法定制能力
- 不需要维护多套代码来支持 torchrun、Ray、HTTP 等不同模式
- 算法工程师专注写训练逻辑，框架自动处理分布式通信

Twinkle 同时支持 **Transformers** 和 **Megatron** 后端，以及 **多租户 LoRA 训练**——多个用户共享一个基座模型，各自训练自己的适配器。

---

## 二、本地多卡训练

### 场景说明

本地 1~8 张 GPU/NPU 的训练场景。Twinkle 基于 PyTorch 原生接口，支持 FSDP2、DDP 等并行策略。

### 完整代码

```python
from peft import LoraConfig
from tqdm import tqdm

import twinkle
from twinkle import DeviceMesh, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

# 构造 device_mesh：fsdp=4, dp=2，共使用 8 张卡
device_mesh = DeviceMesh.from_sizes(fsdp_size=4, dp_size=2)
# 使用 torchrun 模式
twinkle.initialize(mode='local', global_device_mesh=device_mesh)

logger = get_logger()


def eval(model):
    # 验证集：100 条样本
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(100)))
    dataset.set_template('Template', model_id='ms://Qwen/Qwen3.5-4B')
    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    dataset.encode()
    dataloader = DataLoader(dataset=dataset, batch_size=8)
    for step, batch in tqdm(enumerate(dataloader)):
        model.forward_only(inputs=batch)
        model.calculate_loss()
    metrics = model.calculate_metric(is_training=False)
    return metrics


def train():
    # 训练集：1000 条样本
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(1000)))
    # 设置模板，准备编码
    dataset.set_template('Template', model_id='ms://Qwen/Qwen3.5-4B')
    # 数据预处理：替换自我认知数据中的占位符
    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    # 编码数据集
    dataset.encode()
    # 全局 batch size = 8，8 张卡每张处理 1 条
    dataloader = DataLoader(dataset=dataset, batch_size=8)
    # 加载模型
    model = TransformersModel(model_id='ms://Qwen/Qwen3.5-4B')
    model.model._no_split_modules = {'Qwen3_5DecoderLayer'}

    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules='all-linear')

    # 添加 LoRA 适配器，命名为 'default'
    # 注释掉这行即可切换到全参数训练
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2)
    # 为 LoRA 配置优化器
    model.set_optimizer(optimizer_cls='AdamW', lr=1e-4)
    # 配置学习率调度器
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler', num_warmup_steps=5, num_training_steps=len(dataloader))
    logger.info(get_device_placement())
    # 打印训练配置
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')
    loss_metric = 99.0
    # LoRA 训练：约 18G * 4 显存占用
    # 全参数训练：约 50G * 4 显存占用
    for step, batch in enumerate(dataloader):
        # 前向 + 反向传播
        model.forward_backward(inputs=batch)
        # 梯度裁剪 + 优化器步进
        model.clip_grad_and_step()
        if step % 20 == 0:
            # 打印训练指标
            metric = model.calculate_metric(is_training=True)
            logger.info(f'Current is step {step} of {len(dataloader)}, metric: {metric}')
        if step > 0 and step % 40 == 0:
            # 定期验证
            metrics = eval(model)
            logger.info(f'Eval metric: {metrics}')
            metrics['step'] = step
            # 保存最优检查点
            if loss_metric > float(metrics['loss']):
                model.save(f'checkpoint-{step}')
                loss_metric = float(metrics['loss'])
    model.save(f'last-checkpoint')


if __name__ == '__main__':
    train()
```

### 启动命令

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 fsdp2.py
```

### 关键设计说明

**DeviceMesh 并行策略**

```python
device_mesh = DeviceMesh.from_sizes(fsdp_size=4, dp_size=2)
```

4 路 FSDP 分片 + 2 路数据并行的混合并行。Qwen3.5-4B 在 bf16 精度下权重占用约8GB，LoRA 模式下单卡显存占用大约 18GB，8 张 A100/H100 流畅跑。

**梯度累积**

```python
model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2)
```

`gradient_accumulation_steps=2` 每 2 个 micro-batch 更新一次参数，等效于 batch size 翻倍。显存受限但又想要较大 batch 时很实用。

**算法过程外露**

所有训练关键过程——前向、反向、梯度裁剪、检简点保存——都直接写在主循环里，开发者对训练过程有完整的控制权。底层的分布式通信由 Twinkle infra 负责，切换 Ray 还是 torchrun 对主循环并无影响。

对于复杂算法而言，这一点尤为关键。

### RL 训练：Ray 模式下的强化学习实战

Twinkle 支持多种 RL 算法，包括 GRPO、RLOO、GSPO等。这里以 GRPO（Group Relative Policy Optimization）为例——它是 DeepSeek-R1 中使用的核心 RL 算法——展示如何在 Ray 模式下完成 RL 训练。

与 PPO 不同，GRPO 不需要单独训练一个价值模型，而是通过组内采样结果的相对奖励来估计优势函数，简化了训练流程并降低了显存开销。Twinkle 的 Ray 模式特别适合这类需要**模型与采样器分离部署**的 RL 算法。在下面的例子中，我们用 4 张卡跑模型训练，另外 4 张卡跑 vLLM 采样，两者通过 Ray 集群协调：

```python
from typing import List, Dict, Any
from peft import LoraConfig
import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.reward import GSM8KAccuracyReward, GSM8KFormatReward
from twinkle.sampler import vLLMSampler
from twinkle.template import Template
from twinkle.metric import CompletionRewardMetric
from twinkle.preprocessor.llm import GSM8KProcessor

logger = get_logger()

MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
MODEL_GPUS = 4      # 模型训练用 4 张卡
SAMPLER_GPUS = 4    # vLLM 采样用 4 张卡
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = 8     # 每组采样 8 个结果
MAX_NEW_TOKENS = 4096
LEARNING_RATE = 1e-5
MAX_STEPS = 200
BATCH_SIZE = 16
MINI_BATCH_SIZE = 16
MICRO_BATCH_SIZE = 2
ADAPTER_NAME = 'default'

def create_gsm8k_dataset():
    dataset = Dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train'))
    dataset.set_template('Template', model_id=MODEL_ID, max_length=2048)
    dataset.map(GSM8KProcessor())
    dataset.encode(add_generation_prompt=True)
    return dataset

def compute_rewards(trajectories: List[Dict[str, Any]]):
    accuracy_reward_fn = GSM8KAccuracyReward()
    format_reward_fn = GSM8KFormatReward()
    accuracy_rewards = accuracy_reward_fn(trajectories)
    format_rewards = format_reward_fn(trajectories)
    total_rewards = [a + f for a, f in zip(accuracy_rewards, format_rewards)]
    return total_rewards, format_rewards, accuracy_rewards

def main():
    # 模型和采样器分到不同的 GPU 组
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU'),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)

    # Ray 模式初始化
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    lora_config = LoraConfig(target_modules='all-linear', r=32, lora_alpha=64, lora_dropout=0.05)

    # 模型部署在 'model' 组
    model = TransformersModel(model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model')
    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=1)
    model.set_optimizer('AdamW', lr=LEARNING_RATE)
    model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)
    model.set_loss('GRPOLoss', epsilon=0.2)
    model.set_processor(InputProcessor)
    model.set_template('Template', model_id=MODEL_ID)

    # 采样器部署在 'sampler' 组
    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 4096,
            'max_lora_rank': 32,
            'enable_lora': False,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(Template, model_id=MODEL_ID)

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)

    dataloader = DataLoader(
        dataset=create_gsm8k_dataset,
        batch_size=BATCH_SIZE,
        min_batch_size=BATCH_SIZE,
        device_mesh=model_mesh,
        remote_group='model',
    )

    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()
    sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS)

    optim_step = 0
    logger.info(get_device_placement())

    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break
        metrics.reset()
        global_prompts = batch if isinstance(batch, list) else [batch]

        # 同步权重到采样器
        ckpt_manager.sync_weights(merge_and_sync=True)
        sampler.reset_prefix_cache()

        # 组采样：每个 prompt 采样 NUM_GENERATIONS 个结果
        sample_response = sampler.sample(
            global_prompts * NUM_GENERATIONS,
            sampling_params,
            num_samples=1,
        )

        all_input_data = []
        all_old_logps = []
        all_completion_lengths = []

        for sequence in sample_response.sequences:
            all_input_data.append(sequence.new_input_feature)
            all_old_logps.append(sequence.logprobs)
            all_completion_lengths.append(len(sequence.tokens))

        # 计算奖励
        total_rewards, format_rewards, accuracy_rewards = compute_rewards(all_input_data)
        metrics.accumulate(
            completion_lengths=all_completion_lengths,
            rewards={
                'total': total_rewards,
                'format': format_rewards,
                'accuracy': accuracy_rewards,
            },
        )

        # GRPO 优势估计：组内归一化
        advantages = advantage_fn(total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()

        # Mini-batch 训练
        total_completions = len(all_input_data)
        for mb_start in range(0, total_completions, MINI_BATCH_SIZE):
            mb_end = min(mb_start + MINI_BATCH_SIZE, total_completions)
            mb_inputs = all_input_data[mb_start:mb_end]
            mb_old_logps = all_old_logps[mb_start:mb_end]
            mb_advantages = advantages[mb_start:mb_end]

            model.forward_backward(
                inputs=mb_inputs,
                old_logps=mb_old_logps,
                advantages=mb_advantages,
                micro_batch_size=MICRO_BATCH_SIZE,
            )
            model.clip_grad_and_step()
            optim_step += 1

            if optim_step >= MAX_STEPS:
                break
            log_dict = metrics.calculate()
            log_dict.update(model.calculate_metric(is_training=True))
            metrics.reset()
            logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('grpo-gsm8k-checkpoint')

if __name__ == '__main__':
    main()
```

由于是Ray集群运行，所以启动只需要：

```shell
python train.py
```

**GRPO 训练的关键设计：**

1. **模型与采样器分离**：`DeviceGroup` 将 8 张卡分成两组，训练和采样互不干扰，采样流程可充分利用 vLLM 的高吞吐

2. **组采样策略**：`global_prompts * NUM_GENERATIONS` 让每个问题采样多个结果，通过组内相对奖励估计优势——不需要单独训练价值模型

3. **权重同步**：`ckpt_manager.sync_weights()` 在每次采样前将训练模型权重同步到 vLLM，确保采样始终使用最新策略

4. **算法组件外露**：`GRPOAdvantage` 和 `GRPOLoss` 直接注册到模型，可替换为其他 RL 算法组件而不需修改其他任何代码

这种写法的核心价值在于：整个 RL 训练流程——采样、奖励计算、优势估计、梯度更新——都展开在可见的 Python 主循环里，没有隐藏的魔法。不同 RL 算法的差异，往往只在于替换几个组件。

---

## 三、远程训练：Client-Server 架构

当算力资源和服务消费方分离时——企业内部训推平台、云服务商的 Serverless 训练服务——就需要把训练能力以 API 形式暴露出来。

Twinkle 支持两种 Client 接入方式：
- **Twinkle Client**：和本地训练 API 完全一致，适合需要精细控制的场景
- **Tinker Client**：兼容 [Tinker](https://github.com/thinking-machines-lab/tinker) 生态，调用方式更简洁

服务端只维护一份基座模型，多个客户端可并行训练各自的 LoRA 适配器。

### 3.1 Twinkle Client：细粒度控制

Twinkle Client 提供与本地训练几乎完全一致的 API，适合需要精细控制训练流程的场景。

```python
import dotenv
dotenv.load_dotenv('.env')

from peft import LoraConfig

from twinkle import get_logger
from twinkle.dataset import DatasetMeta
from twinkle_client import init_twinkle_client
from twinkle_client.dataloader import DataLoader
from twinkle_client.dataset import Dataset
from twinkle_client.model import MultiLoraTransformersModel

logger = get_logger()

# 初始化 Twinkle 客户端
client = init_twinkle_client(base_url='http://127.0.0.1:8000', api_key='EMPTY_TOKEN')

# 查询已有训练运行和检查点
runs = client.list_training_runs()
resume_path = None
for run in runs:
    logger.info(run.model_dump_json(indent=2))
    checkpoints = client.list_checkpoints(run.training_run_id)
    for checkpoint in checkpoints:
        logger.info(checkpoint.model_dump_json(indent=2))
        # 如需恢复训练，取消下面注释
        # resume_path = checkpoint.twinkle_path


def train():
    # 准备数据集
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))
    dataset.set_template('Template', model_id='ms://Qwen/Qwen3.5-4B', max_length=512)
    dataset.map('SelfCognitionProcessor', init_args={'model_name': 'twinkle模型', 'model_author': 'ModelScope社区'})
    dataset.encode(batched=True)
    dataloader = DataLoader(dataset=dataset, batch_size=4)

    # 配置模型
    model = MultiLoraTransformersModel(model_id='ms://Qwen/Qwen3.5-4B')

    lora_config = LoraConfig(target_modules='all-linear')
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2)
    model.set_template('Template')
    model.set_processor('InputProcessor', padding_side='right')
    model.set_loss('CrossEntropyLoss')
    model.set_optimizer('AdamW', lr=1e-4)
    model.set_lr_scheduler('LinearLR')

    # 恢复训练（如有检查点）
    if resume_path:
        logger.info(f'Resuming training from {resume_path}')
        model.load(resume_path, load_optimizer=True)

    logger.info(model.get_train_configs())

    for epoch in range(3):
        logger.info(f'Starting epoch {epoch}')
        for step, batch in enumerate(dataloader):
            # 前向 + 反向
            output = model.forward_backward(inputs=batch)

            if step % 2 == 0:
                logger.info(f'Current is step {step // 2}, loss: {output}')

            model.clip_grad_norm(1.0)
            model.step()
            model.zero_grad()
            model.lr_step()

        # 保存检查点
        twinkle_path = model.save(name=f'twinkle-epoch-{epoch}', save_optimizer=True)
        logger.info(f'Saved checkpoint: {twinkle_path}')


if __name__ == '__main__':
    train()
```

**Twinkle Client 的特点：**

- API 与本地训练完全一致，无额外学习成本
- 支持断点续训、检查点管理
- 可动态切换 LoRA 适配器、损失函数、优化器等组件

### 3.2 Tinker Client：简洁即用

Tinker 是一个轻量级训练 API。Twinkle 对 Tinker 客户端提供完整支持，几行代码就能拉起训练。已有 Tinker 代码的项目可以直接迎移到 Twinkle 服务端。

```python
import os
from tinker import types
from tqdm import tqdm

from twinkle import init_tinker_client
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor import SelfCognitionProcessor
from twinkle.server.tinker.common import input_feature_to_datum

# 初始化 Tinker 客户端（必须在导入 ServiceClient 之前）
init_tinker_client()

from tinker import ServiceClient

# 基座模型
base_model = 'Qwen/Qwen3.5-4B'
base_url = 'http://www.modelscope.cn/twinkle'


def train():
    # 准备数据集
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))
    dataset.set_template('Template', model_id=f'ms://{base_model}', max_length=256)
    dataset.map(SelfCognitionProcessor('Twinkle模型', 'ModelScope团队'), load_from_cache_file=False)
    dataset.encode(batched=True, load_from_cache_file=False)
    dataloader = DataLoader(dataset=dataset, batch_size=8)

    # 初始化训练客户端
    service_client = ServiceClient(
        base_url=base_url,
        api_key=os.environ.get('MODELSCOPE_TOKEN')
    )
    training_client = service_client.create_lora_training_client(base_model=base_model, rank=16)

    # 训练循环
    for epoch in range(3):
        print(f'Epoch {epoch}')
        for step, batch in tqdm(enumerate(dataloader)):
            # 转换输入格式
            input_datum = [input_feature_to_datum(input_feature) for input_feature in batch]

            # 远端前向 + 反向
            fwdbwd_future = training_client.forward_backward(input_datum, 'cross_entropy')
            # 远端优化器步进
            optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))

            # 等待结果
            fwdbwd_result = fwdbwd_future.result()
            optim_result = optim_future.result()
            print(f'Training Metrics: {optim_result}')

        # 保存检查点
        save_future = training_client.save_state(f'twinkle-lora-{epoch}')
        save_result = save_future.result()
        print(f'Saved checkpoint to {save_result.path}')


if __name__ == '__main__':
    train()
```

**Tinker Client 的特点：**

- 调用方式极简，适合快速上手
- 完全兼容 Tinker 生态，已有代码可无缝迁移
- 支持魔搭官方训练环境（见下文）

### 3.3 魔搭官方训练环境

Twinkle 框架开源的同时，魔搭社区依托自身算力基础设施，提供了托管的模型训练服务（Training as a Service）。开发者无需准备 GPU 资源，通过 API 调用即可免费体验 Twinkle 的训练能力。

**使用方式：**

1. 注册魔搭账号并申请加入 [Twinkle-Explorers](https://modelscope.cn/organization/twinkle-explorers) 组织
2. 在 [Token 管理页面](https://www.modelscope.cn/my/access/token) 获取 API Key
3. 使用上面的 Tinker Client 代码，修改 endpoint：

```python
base_url = 'https://www.modelscope.cn/twinkle'
base_model = 'Qwen/Qwen3-30B-A3B-Instruct-2507'  # 官方环境当前部署的模型
```

---

## 四、如何选择适合你的训练方式

| 场景 | 推荐方案 | 核心优势 |
|------|----------|----------|
| 本地实验调试 | 单卡 / torchrun | 代码即配置，调试效率高 |
| 大规模分布式训练 | torchrun + FSDP2 / Ray | 原生并行性能，生产就绪 |
| 企业内部训推平台 | Twinkle Client + 自托管服务 | 多租户隔离，细粒度控制 |
| 快速验证想法 | Tinker Client + 魔搭官方环境 | 零资源准备，即开即用 |
| 已有 Tinker 生态 | Tinker Client | 无缝迁移，生态兼容 |

**选型建议：**

- 如果你是算法研究员，需要频繁调整训练流程，从 torchrun 模式开始，验证完成后再考虑是否服务化
- 如果你是平台开发者，需要为企业内部提供训练服务，部署 Twinkle Server，根据用户习惯提供 Twinkle Client 或 Tinker Client 两种接入方式
- 如果你只是想快速体验 Twinkle 的能力，直接用魔搭官方环境，5 分钟跑通第一个训练任务

Twinkle 的设计哲学是**不替你做决定，但给你足够的选择空间**。无论是追求极致性能的大规模训练，还是追求极致便捷的 API 调用，都能找到合适的解法。
