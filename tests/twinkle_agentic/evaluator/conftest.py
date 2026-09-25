from dataclasses import dataclass
from typing import Any

import pytest

from twinkle.data_format import SampleResponse, SampledSequence
from twinkle_agentic.protocol.base import API


class RecordingAPI(API):
    model = 'recording-api'

    def __init__(self, response: Any | None = None):
        self.calls = []
        self.response = response or {'role': 'assistant', 'content': 'ok', 'finish_reason': 'stop'}

    def __call__(self, trajectory, sampling_params, **kwargs):
        self.calls.append((trajectory, sampling_params, kwargs))
        return self.response


class RecordingSampler:
    model_id = 'recording-sampler'

    def __init__(self, dp_world_size: int = 1):
        self.calls = []
        self.device_mesh = type('Mesh', (), {'dp_world_size': dp_world_size})()

    def sample(self, inputs, sampling_params, **kwargs):
        self.calls.append((list(inputs), sampling_params, kwargs))
        return [SampleResponse(sequences=[SampledSequence(stop_reason='stop', tokens=[1], decoded='ok')]) for _ in inputs]


class ToolTemplate:
    def decode(self, tokens):
        return ''.join(str(token) for token in tokens)

    def parse_tool_call(self, decoded):
        if decoded == 'tool':
            return [{'id': 'parsed', 'type': 'function', 'function': {'name': 'lookup', 'arguments': {'q': 'x'}}}]
        return []


@pytest.fixture
def recording_api():
    return RecordingAPI()


@pytest.fixture
def recording_sampler():
    return RecordingSampler()
