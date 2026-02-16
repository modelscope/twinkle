# Tinker 兼容客户端

Tinker 兼容 Client 适用于已有 Tinker 训练代码的场景。通过 `init_tinker_compat_client` 初始化后，会对 Tinker SDK 进行 patch，使其指向 Twinkle Server，**其余代码可直接复用已有的 Tinker 训练代码**。

## 初始化

```python
from twinkle_client import init_tinker_compat_client

# 初始化 Tinker 兼容客户端
# init_tinker_compat_client 会自动 patch Tinker SDK，
# 使其可以连接到 Twinkle Server 而非 Tinker Server
service_client = init_tinker_compat_client(
    base_url='http://localhost:8000',    # Server 地址
    api_key='your-api-key'               # 认证令牌
)

# 验证连接：列出 Server 上可用的模型
for item in service_client.get_server_capabilities().supported_models:
    print("- " + item.model_name)
```

### init_tinker_compat_client 做了什么？

调用 `init_tinker_compat_client` 时，会自动执行以下操作：

1. **Patch Tinker SDK**：绕过 Tinker 的 `tinker://` 前缀校验，使其可以连接到标准 HTTP 地址
2. **设置请求头**：注入 `serve_multiplexed_model_id` 和 `Authorization` 等必要的认证头
3. **返回 `ServiceClient`**：返回一个标准的 Tinker `ServiceClient` 对象，后续操作与原生 Tinker 完全一致

这意味着在初始化之后，**所有已有的 Tinker 训练代码都可以直接使用**，无需任何修改。

## 完整训练示例

```python
import os
import numpy as np
import dotenv
dotenv.load_dotenv('.env')

from tinker import types
from modelscope import AutoTokenizer
from twinkle_client import init_tinker_compat_client

# Step 1: 初始化客户端（会自动 patch Tinker SDK）
service_client = init_tinker_compat_client(
    base_url='http://localhost:8000',
    api_key=os.environ.get('MODELSCOPE_TOKEN')
)

# Step 2: 查询已有训练运行（可选）
rest_client = service_client.create_rest_client()
response = rest_client.list_training_runs(limit=50).result()
print(f"Found {len(response.training_runs)} training runs")

# Step 3: 创建训练客户端
base_model = "Qwen/Qwen2.5-0.5B-Instruct"

# 新建训练会话
training_client = service_client.create_lora_training_client(
    base_model=base_model
)

# 或从检查点恢复
# resume_path = "twinkle://run_id/weights/checkpoint_name"
# training_client = service_client.create_training_client_from_state_with_optimizer(path=resume_path)

# Step 4: 准备训练数据
examples = [
    {"input": "banana split", "output": "anana-bay plit-say"},
    {"input": "quantum physics", "output": "uantum-qay ysics-phay"},
    {"input": "donut shop", "output": "onut-day op-shay"},
]

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

def process_example(example: dict, tokenizer) -> types.Datum:
    """将原始样本转为 Tinker API 所需的 Datum 格式。

    Datum 包含：
      - model_input: 输入 token IDs
      - loss_fn_inputs: 目标 token 和逐 token 权重（0=忽略, 1=计算损失）
    """
    prompt = f"English: {example['input']}\nPig Latin:"

    # 提示部分：weight=0，不参与损失计算
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)

    # 补全部分：weight=1，参与损失计算
    completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)

    # 拼接并构建 next-token prediction 格式
    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )

processed_examples = [process_example(ex, tokenizer) for ex in examples]

# Step 5: 训练循环
for epoch in range(2):
    for batch in range(5):
        # 发送训练数据到 Server：前向 + 反向传播
        fwdbwd_future = training_client.forward_backward(processed_examples, "cross_entropy")
        # 优化器更新
        optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))

        # 等待结果
        fwdbwd_result = fwdbwd_future.result()
        optim_result = optim_future.result()

        # 计算加权平均 log-loss
        logprobs = np.concatenate([o['logprobs'].tolist() for o in fwdbwd_result.loss_fn_outputs])
        weights = np.concatenate([e.loss_fn_inputs['weights'].tolist() for e in processed_examples])
        print(f"Epoch {epoch}, Batch {batch}: Loss = {-np.dot(logprobs, weights) / weights.sum():.4f}")

    # 每个 epoch 保存检查点
    save_result = training_client.save_state(f"lora-epoch-{epoch}").result()
    print(f"Saved checkpoint to {save_result.path}")
```

## 使用 Twinkle 数据集组件

Tinker 兼容模式也可以利用 Twinkle 的数据集组件来简化数据准备，而不是手动构建 `Datum`：

```python
from tqdm import tqdm
from tinker import types
from twinkle_client import init_tinker_compat_client
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor import SelfCognitionProcessor
from twinkle.server.tinker.common import input_feature_to_datum

base_model = "Qwen/Qwen2.5-0.5B-Instruct"

# 使用 Twinkle 的 Dataset 组件加载和预处理数据
dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))
dataset.set_template('Template', model_id=f'ms://{base_model}', max_length=256)
dataset.map(SelfCognitionProcessor('twinkle模型', 'twinkle团队'), load_from_cache_file=False)
dataset.encode(batched=True, load_from_cache_file=False)
dataloader = DataLoader(dataset=dataset, batch_size=8)

# 初始化 Tinker 兼容客户端
service_client = init_tinker_compat_client(base_url='http://localhost:8000')
training_client = service_client.create_lora_training_client(base_model=base_model, rank=16)

# 训练循环：使用 input_feature_to_datum 转换数据格式
for epoch in range(3):
    for step, batch in tqdm(enumerate(dataloader)):
        # 将 Twinkle 的 InputFeature 转换为 Tinker 的 Datum
        input_datum = [input_feature_to_datum(input_feature) for input_feature in batch]

        fwdbwd_future = training_client.forward_backward(input_datum, "cross_entropy")
        optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))

        fwdbwd_result = fwdbwd_future.result()
        optim_result = optim_future.result()

    training_client.save_state(f"twinkle-lora-{epoch}").result()
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
from tinker import types
from modelscope import AutoTokenizer
from twinkle_client import init_tinker_compat_client

base_model = "Qwen/Qwen2.5-0.5B-Instruct"

# 初始化客户端
service_client = init_tinker_compat_client(base_url='http://localhost:8000')

# 从已保存的检查点创建采样客户端
sampling_client = service_client.create_sampling_client(
    model_path="twinkle://run_id/weights/checkpoint_name",  # 检查点的 twinkle:// 路径
    base_model=base_model
)

# 准备推理输入
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

# 构建多轮对话输入
inputs = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': 'what is your name?'}
]
input_ids = tokenizer.apply_chat_template(inputs, tokenize=True, add_generation_prompt=True)

prompt = types.ModelInput.from_ints(input_ids)
params = types.SamplingParams(
    max_tokens=50,       # 最大生成 token 数
    temperature=0.2,     # 低温度，更聚焦的回答
    stop=["\n"]          # 遇到换行停止
)

# 生成多条补全
result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=8).result()

for i, seq in enumerate(result.sequences):
    print(f"{i}: {tokenizer.decode(seq.tokens)}")
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
