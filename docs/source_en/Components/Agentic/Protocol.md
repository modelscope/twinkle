# Protocol

The Protocol module provides an abstract LLM API client interface and its OpenAI-compatible implementation. It bridges Twinkle's `Trajectory` / `SamplingParams` data types with external LLM inference services.

## API Base Class

```python
from abc import ABC, abstractmethod
from twinkle.data_format import Trajectory
from twinkle.data_format.message import Message
from twinkle.data_format.sampling import SamplingParams

class API(ABC):
    """Abstract LLM API client: Trajectory + SamplingParams -> assistant Message(s)."""

    @abstractmethod
    def __call__(
        self,
        trajectory: Trajectory,
        sampling_params: SamplingParams,
        **kwargs,
    ) -> Union[Message, List[Message]]:
        raise NotImplementedError()
```

The `API` class defines a simple contract: given a conversation trajectory and sampling parameters, return one or more assistant messages.

## OpenAI

`OpenAI` is the built-in implementation that works with any endpoint speaking the `/v1/chat/completions` protocol (OpenAI, Azure OpenAI, vLLM, SGLang, Ollama, etc.).

```python
from twinkle_agentic.protocol.openai import OpenAI

api = OpenAI(
    model='qwen3.5-32b',
    base_url='http://localhost:8000/v1',
    api_key='EMPTY',
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str` | Model name to pass in the API request. |
| `api_key` | `str` | API key. Defaults to the `OPENAI_API_KEY` environment variable. |
| `base_url` | `str` | Base URL of the API endpoint (e.g. `http://localhost:8000/v1`). |
| `client_kwargs` | `Dict` | Extra keyword arguments forwarded to the `openai.OpenAI` client constructor. |

### Usage

```python
from twinkle.data_format import Trajectory
from twinkle.data_format.sampling import SamplingParams

trajectory = {
    'messages': [
        {'role': 'user', 'content': 'What is the capital of France?'},
    ]
}

sp = SamplingParams(temperature=0.7, max_tokens=512)
reply = api(trajectory, sp)
# reply is a Message dict: {'role': 'assistant', 'content': '...'}
```

### Features

- **Tool calls**: Automatically maps `trajectory['tools']` to the API request and parses structured `tool_calls` from the response.
- **Reasoning content**: Preserves `reasoning_content` from models that support it (e.g., o1-style reasoning).
- **Finish reason**: Surfaces `finish_reason` on the returned message so multi-turn drivers can detect length-cap truncation.
- **Multi-sample**: When `sampling_params.num_samples > 1`, returns a list of messages (one per choice).

### Custom API Client

To integrate a non-OpenAI API, subclass `API`:

```python
from twinkle_agentic.protocol.base import API

class MyCustomAPI(API):

    def __call__(self, trajectory, sampling_params, **kwargs):
        # Call your custom endpoint
        response = my_llm_client.chat(
            messages=trajectory['messages'],
            temperature=sampling_params.temperature,
        )
        return {'role': 'assistant', 'content': response.text}
```
