# 协议（Protocol）

Protocol 模块提供了抽象的 LLM API 客户端接口及其 OpenAI 兼容实现。它将 Twinkle 的 `Trajectory` / `SamplingParams` 数据类型与外部 LLM 推理服务连接起来。

## API 基类

```python
from abc import ABC, abstractmethod
from twinkle.data_format import Trajectory
from twinkle.data_format.message import Message
from twinkle.data_format.sampling import SamplingParams

class API(ABC):
    """抽象 LLM API 客户端：Trajectory + SamplingParams -> 助手 Message"""

    @abstractmethod
    def __call__(
        self,
        trajectory: Trajectory,
        sampling_params: SamplingParams,
        **kwargs,
    ) -> Union[Message, List[Message]]:
        raise NotImplementedError()
```

`API` 类定义了一个简单的契约：给定对话轨迹和采样参数，返回一条或多条助手消息。

## OpenAI

`OpenAI` 是内置实现，兼容任何支持 `/v1/chat/completions` 协议的端点（OpenAI、Azure OpenAI、vLLM、SGLang、Ollama 等）。

```python
from twinkle_agentic.protocol.openai import OpenAI

api = OpenAI(
    model='qwen3.5-32b',
    base_url='http://localhost:8000/v1',
    api_key='EMPTY',
)
```

### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | `str` | API 请求中传递的模型名称。 |
| `api_key` | `str` | API 密钥。默认使用 `OPENAI_API_KEY` 环境变量。 |
| `base_url` | `str` | API 端点的基础 URL（如 `http://localhost:8000/v1`）。 |
| `client_kwargs` | `Dict` | 转发给 `openai.OpenAI` 客户端构造函数的额外关键字参数。 |

### 使用方法

```python
from twinkle.data_format import Trajectory
from twinkle.data_format.sampling import SamplingParams

trajectory = {
    'messages': [
        {'role': 'user', 'content': '法国的首都是什么？'},
    ]
}

sp = SamplingParams(temperature=0.7, max_tokens=512)
reply = api(trajectory, sp)
# reply 是一个 Message 字典：{'role': 'assistant', 'content': '...'}
```

### 特性

- **工具调用**：自动将 `trajectory['tools']` 映射到 API 请求，并解析响应中的结构化 `tool_calls`。
- **推理内容**：保留支持推理的模型返回的 `reasoning_content`（如 o1 风格推理）。
- **完成原因**：在返回消息中暴露 `finish_reason`，供多轮驱动器检测长度截断。
- **多样本**：当 `sampling_params.num_samples > 1` 时，返回消息列表（每个 choice 一条）。

### 自定义 API 客户端

要集成非 OpenAI API，请继承 `API`：

```python
from twinkle_agentic.protocol.base import API

class MyCustomAPI(API):

    def __call__(self, trajectory, sampling_params, **kwargs):
        # 调用自定义端点
        response = my_llm_client.chat(
            messages=trajectory['messages'],
            temperature=sampling_params.temperature,
        )
        return {'role': 'assistant', 'content': response.text}
```
