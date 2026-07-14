# Tinker 客户端

Tinker Client 适用于已有 Tinker 训练代码的场景。通过 `init_tinker_client` 初始化后，会对 Tinker SDK 进行 patch，使其指向 Twinkle Server，**其余代码可直接复用已有的 Tinker 训练代码**。

## 初始化

```python
# 在导入 ServiceClient 之前，先初始化 Tinker 客户端
from twinkle import init_tinker_client

init_tinker_client()

# 直接使用 tinker 中的 ServiceClient
from tinker import ServiceClient

service_client = ServiceClient(
    base_url='http://localhost:8000',                    # Server 地址
    api_key=os.environ.get('MODELSCOPE_TOKEN')           # 建议设置为 ModelScope Token
)

# 验证连接：列出 Server 上可用的模型
for item in service_client.get_server_capabilities().supported_models:
    print("- " + item.model_name)
```

### init_tinker_client 做了什么？

调用 `init_tinker_client` 时，会自动执行以下操作：

1. **Patch Tinker SDK**：绕过 Tinker 的 `tinker://` 前缀校验，使其可以连接到标准 HTTP 地址
2. **设置请求头**：注入 `X-Ray-Serve-Request-Id` 和 `Authorization` 等必要的认证头

初始化之后，直接导入 `from tinker import ServiceClient` 即可连接到 Twinkle Server，**所有已有的 Tinker 训练代码都可以直接使用**，无需任何修改。

## 完整训练示例

> **注意**：Tinker 兼容模式的 `DataLoader` 和 `Dataset` 只支持从本地 `twinkle` 导入，不支持 `twinkle_client`。

```python
import os
import numpy as np
from tqdm import tqdm
from tinker import types
from twinkle import init_tinker_client
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor import SelfCognitionProcessor
from twinkle.server.common import input_feature_to_datum

# Step 1: 在导入 ServiceClient 之前，先初始化 Tinker 客户端
init_tinker_client()

from tinker import ServiceClient

base_model = 'Qwen/Qwen3.5-4B'
base_url = 'http://localhost:8000'
api_key = 'EMPTY_API_KEY'

# Step 2: 准备数据集
dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))
dataset.set_template('Qwen3_5Template', model_id=f'ms://{base_model}', max_length=256)
dataset.map(SelfCognitionProcessor('twinkle模型', 'twinkle团队'), load_from_cache_file=False)
dataset.encode(batched=True, load_from_cache_file=False)
dataloader = DataLoader(dataset=dataset, batch_size=8)

# Step 3: 初始化训练客户端
service_client = ServiceClient(base_url=base_url, api_key=api_key)

# 创建 LoRA 训练客户端（rank=16 指定 LoRA 适配器秩）
training_client = service_client.create_lora_training_client(base_model=base_model, rank=16)

# Step 4: 训练循环
for epoch in range(3):
    print(f'Epoch {epoch}')
    for step, batch in tqdm(enumerate(dataloader)):
        # 将 Twinkle 的 InputFeature 转换为 Tinker 的 Datum 格式
        input_datum = [input_feature_to_datum(input_feature) for input_feature in batch]

        # 发送数据到 Server：前向 + 反向传播
        fwdbwd_future = training_client.forward_backward(input_datum, 'cross_entropy')

        # 优化器更新：Adam 更新模型权重
        optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))

        # 等待两个操作完成
        fwdbwd_result = fwdbwd_future.result()
        optim_result = optim_future.result()

        # 计算每 token 加权平均 log-loss 用于监控
        logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
        weights = np.concatenate([example.loss_fn_inputs['weights'].tolist() for example in input_datum])
        print(f'Loss per token: {-np.dot(logprobs, weights) / weights.sum():.4f}')
        print(f'Training Metrics: {optim_result}')

    # 每个 epoch 保存检查点
    save_future = training_client.save_state(f'twinkle-lora-{epoch}')
    save_result = save_future.result()
    print(f'Saved checkpoint to {save_result.path}')
```

## 推理采样

Tinker 兼容模式支持推理采样功能（需要 Server 配置了 Sampler 服务）。

### 从训练中采样

在训练完成后，可以直接从训练客户端创建采样客户端：

```python
# 保存当前权重并创建采样客户端
sampling_client = training_client.save_weights_and_get_sampling_client(name='my-model')

# 准备推理输入
prompt = types.ModelInput.from_ints(tokenizer.encode("English: coffee break\nPig Latin:"))
params = types.SamplingParams(
    max_tokens=20,       # 最大生成 token 数
    temperature=0.0,     # 贪心采样（确定性输出）
    stop=["\n"]          # 遇到换行停止
)

# 生成多条补全
result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=8).result()

for i, seq in enumerate(result.sequences):
    print(f"{i}: {tokenizer.decode(seq.tokens)}")
```

### 从检查点采样

也可以加载已保存的检查点进行推理：

```python
import os
from tinker import types
from twinkle import init_tinker_client
from twinkle.data_format import Message, Trajectory
from twinkle.template import Template

# 在导入 ServiceClient 之前，先初始化 Tinker 客户端
init_tinker_client()

from tinker import ServiceClient

base_model = 'Qwen/Qwen3.5-4B'
base_url = 'http://localhost:8000'
api_key = 'EMPTY_API_KEY'

service_client = ServiceClient(base_url=base_url, api_key=api_key)

# 从已保存的检查点创建采样客户端
sampling_client = service_client.create_sampling_client(
    model_path='twinkle://run_id/weights/checkpoint_name',  # 检查点的 twinkle:// 路径
    base_model=base_model
)

# 使用 Twinkle 的 Template 构建多轮对话输入
template = Template(model_id=f'ms://{base_model}')

trajectory = Trajectory(
    messages=[
        Message(role='system', content='You are a helpful assistant'),
        Message(role='user', content='你是谁？'),
    ]
)

input_feature = template.batch_encode([trajectory], add_generation_prompt=True)[0]
input_ids = input_feature['input_ids'].tolist()

prompt = types.ModelInput.from_ints(input_ids)
params = types.SamplingParams(
    max_tokens=50,       # 最大生成 token 数
    temperature=0.2,     # 低温度，更聚焦的回答
)

# 生成多条补全
print('Sampling...')
future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=8)
result = future.result()

# 解码并打印每条响应
print('Responses:')
for i, seq in enumerate(result.sequences):
    print(f'{i}: {repr(template.decode(seq.tokens))}')
```

### 发布检查点到 ModelScope Hub

训练完成后，可以通过 REST client 将检查点发布到 ModelScope Hub：

```python
rest_client = service_client.create_rest_client()

# 从 tinker 路径发布检查点
# 需要在初始化客户端时设置有效的 ModelScope token 作为 api_key
rest_client.publish_checkpoint_from_tinker_path(save_result.path).result()
print("Published checkpoint to ModelScope Hub")
```

## 多轮工具调用（间接方案）

Tinker 兼容客户端本身不提供多轮 agentic rollout 编排能力。如果你在用 Tinker Client 训练/管理 checkpoint，同时又需要多轮工具调用能力（**仅用于生成或评测**），可以在完全不改动 Tinker SDK 代码的前提下，额外使用一个标准的 OpenAI 兼容客户端指向 Gateway 的 `/chat/completions` 端点，并复用 `twinkle_agentic.rollout.APIMultiTurnRollout` 与 `ToolManager`。

这是一条纯增量的使用路径：它复用的是 Twinkle Server 上一个独立于 Tinker 协议命名空间的、已经存在并跑通的 OpenAI 兼容端点，因此不涉及对 Tinker SDK 源码或 `patch_tinker.py` 中现有 monkeypatch 范围的任何修改。

```python
import os
from twinkle.data_format.sampling import SamplingParams
from twinkle_agentic.protocol.openai import OpenAI
from twinkle_agentic.rollout import APIMultiTurnRollout
from twinkle_agentic.tools.tool_manager import ToolManager
# from your_project.tools import CalculatorTool, SearchTool

# An OpenAI-compatible client pointing at the Gateway route,
# NOT at tinker's base_url. This is a plain additive usage of an
# already-existing endpoint; the tinker SDK is untouched.
api = OpenAI(
    model='Qwen/Qwen3.5-4B',
    api_key=os.environ.get('MODELSCOPE_TOKEN', 'EMPTY_API_KEY'),
    base_url='http://localhost:8000/api/v1',  # Gateway /chat/completions route
)

# Register your tools; ToolManager is reused directly from twinkle_agentic.
tool_manager = ToolManager([
    # CalculatorTool(),
    # SearchTool(),
])

rollout = APIMultiTurnRollout(
    api=api,
    tool_manager=tool_manager,
    sampling_params=SamplingParams(temperature=0.7, max_tokens=512),
    max_turns=6,
    concurrency=8,
)

trajectories_in = [
    {'messages': [{'role': 'user', 'content': 'What is 12 * 34?'}]},
]
trajectories_out = rollout(trajectories_in)

# trajectories_out[i]['messages'] contains the full conversation, including
# assistant tool_calls and tool-response turns.
# trajectories_out[i]['stop_reason'] is one of 'stop' | 'length' | 'max_turns' | 'api_error'.
for traj in trajectories_out:
    for message in traj['messages']:
        print(message['role'], ':', message.get('content'))
```

### 能力边界：不可用于 RL 训练

该间接方案返回的 Trajectory **不包含** `logprobs` 字段（端到端验证已确认走 Gateway `/chat/completions` 的 `APIMultiTurnRollout` 产出的 trajectory 中不存在 `logprobs`，也没有 `input_ids` 等训练所需字段）。因此该路径**仅适合生成数据、离线评测等场景，不能直接喂给依赖 token 级对齐信息的 RL 训练（如 GRPO）**。

如果多轮 rollout 的产出需要直接用于 RL 训练（需要 token 级 `logprobs` 与 `labels` 对齐），请使用 Twinkle 客户端侧的 `ClientMultiTurnRollout`（通过 `/twinkle/sample` 返回 `new_input_feature`/`logprobs`），详见《Twinkle 客户端》文档。

### 为什么 Tinker 客户端不支持 embedding 训练与可训练多轮 rollout

这两项能力无法通过 Tinker 兼容客户端提供，根本原因在于 Tinker 协议的数据类型层面缺少所需结构：

- **Embedding 训练**：`tinker.types.Datum` 不具备对比学习所需的样本分组结构（anchor/positive 成对分组），无法承载 embedding 训练的输入语义。
- **可训练多轮 rollout**：`tinker.types.SampledSequence` 不具备 `new_input_feature` 字段，无法在多轮之间传递并累积 token 级对齐信息，因此产出无法用于训练。

由于 Tinker 是第三方 SDK、不可修改其源码，即使想补齐这些字段也需要改动 SDK 本身（不允许）。因此上述能力仅在 Twinkle 客户端侧支持，Tinker 兼容客户端只提供上文所述的"仅生成/评测"间接多轮方案。
