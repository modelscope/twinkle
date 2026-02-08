# Copyright (c) ModelScope Contributors. All rights reserved.
from datasets import Features, Value, Sequence

from .message import Tool, ToolCall, Message
from .trajectory import Trajectory
from .input_feature import InputFeature
from .output import ModelOutput
from .sampling import SamplingParams, SampleResponse, SampledSequence

from datasets import Features, Value, Sequence


full_features = Features({
    'input_ids': Sequence(Value('int32')),
    'attention_mask': Sequence(Value('int32')),
    'position_ids': Sequence(Value('int32')),
    'labels': Sequence(Value('int32')),
    'completion_mask': Sequence(Value('bool')),
    'length': Value('int32'),

    'messages': [{
        'role': Value('string'),
        'type': Value('string'),
        'content': Value('string'),
        'tool_calls': [{
            'tool_name': Value('string'),
            'arguments': Value('string'),
        }],
        'reasoning_content': Value('string'),
        'images': Sequence(Value('string')),
        'videos': Sequence(Value('string')),
        'audios': Sequence(Value('string')),
    }],
    'extend_message': Sequence({
        'key': Value('string'),
        'messages': [{
            'role': Value('string'),
            'type': Value('string'),
            'content': Value('string'),
            'tool_calls': [{
                'tool_name': Value('string'),
                'arguments': Value('string'),
            }],
            'reasoning_content': Value('string'),
            'images': Sequence(Value('string')),
            'videos': Sequence(Value('string')),
            'audios': Sequence(Value('string')),
        }],
    }),
    'tools': [{
        'tool_name': Value('string'),
        'description': Value('string'),
        'parameters': Value('string'),
    }],
    'advantages': Value('float32'),
})
