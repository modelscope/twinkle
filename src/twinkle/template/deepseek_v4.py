# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import json
import re
import torch
from transformers import AutoConfig, PreTrainedTokenizerFast
from typing import Any, Dict, List, Literal, Optional

from twinkle.hub import HubOperation
from .base import Template
from .deepseek_v4_encoding import (dsml_token, encode_messages, eos_token, parse_message_from_completion_text,
                                   tool_calls_block_name)


def get_deepseek_v4_tokenizer(tokenizer):
    """Wrap a HF tokenizer with DeepSeek V4's custom chat-template encoder."""
    dsv4_tokenizer = copy.copy(tokenizer)

    added_vocab = tokenizer.get_added_vocab()
    added_vocab_size = len(added_vocab)
    tokenizer_vocab_size = tokenizer.vocab_size

    class _DeepseekV4Tokenizer(tokenizer.__class__):  # type: ignore[misc, valid-type]

        def apply_chat_template(
            self,
            messages,
            tools: Optional[List[Dict[str, Any]]] = None,
            **kwargs,
        ):
            thinking = kwargs.get('thinking', False)
            enable_thinking = kwargs.get('enable_thinking', False)
            thinking = thinking or enable_thinking
            thinking_mode = 'thinking' if thinking else 'chat'

            conversation = kwargs.get('conversation', messages)
            messages = conversation.copy()
            if tools:
                messages.insert(0, {'role': 'system'})
                messages[0]['tools'] = tools

            reasoning_effort = kwargs.get('reasoning_effort')
            if reasoning_effort not in ('max', 'high'):
                reasoning_effort = None

            prompt_str = encode_messages(
                messages,
                thinking_mode=thinking_mode,
                drop_thinking=kwargs.get('drop_thinking', True),
                reasoning_effort=reasoning_effort,
            )

            tokenize = kwargs.get('tokenize', True)
            return_dict = kwargs.get('return_dict', False)
            return_tensors = kwargs.get('return_tensors')

            if not tokenize:
                return {'prompt': prompt_str} if return_dict else prompt_str

            tokenizer_kwargs = {key: kwargs[key] for key in ('truncation', 'max_length') if key in kwargs}
            input_ids = self.encode(
                prompt_str,
                add_special_tokens=False,
                **tokenizer_kwargs,
            )

            if not return_dict and return_tensors is None:
                return input_ids

            attention_mask = [1] * len(input_ids)
            if return_tensors == 'pt':
                input_ids = torch.tensor([input_ids], dtype=torch.long)
                attention_mask = torch.tensor([attention_mask], dtype=torch.long)
            elif return_tensors is not None:
                raise ValueError(f'Unsupported return_tensors: {return_tensors}')

            encoded = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
            if kwargs.get('return_assistant_tokens_mask', False):
                # Fall back to round-by-round labeling in Template by omitting
                # assistant_masks support from this custom tokenizer wrapper.
                pass
            if return_dict:
                return encoded
            return encoded['input_ids']

        def num_special_tokens_to_add(self) -> int:
            return len(self.encode(''))

        def __len__(self) -> int:
            return tokenizer_vocab_size + added_vocab_size

        def get_added_vocab(self) -> dict[str, int]:
            return added_vocab.copy()

    _DeepseekV4Tokenizer.__name__ = f'DSV4{tokenizer.__class__.__name__}'
    dsv4_tokenizer.__class__ = _DeepseekV4Tokenizer
    return dsv4_tokenizer


class DeepseekV4Template(Template):

    _TOOL_CALLS_START = f'<{dsml_token}{tool_calls_block_name}>'
    _TOOL_CALLS_END = f'</{dsml_token}{tool_calls_block_name}>'

    def __init__(
        self,
        model_id: str,
        use_chat_template: bool = True,
        max_length: Optional[int] = 8192,
        truncation_strategy: Literal['raise', 'left', 'right', 'split', 'delete'] = 'raise',
        default_system: Optional[str] = None,
        enable_thinking: bool = True,
        **kwargs,
    ):
        self.model_id = model_id
        model_id = HubOperation.download_model(model_id, ignore_model=True)
        base_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id, **kwargs)
        self.processor = get_deepseek_v4_tokenizer(base_tokenizer)
        self.config = AutoConfig.from_pretrained(model_id, **kwargs)

        self.use_chat_template = use_chat_template
        self.max_length = max_length
        self.enable_thinking = enable_thinking
        self.truncation_strategy = truncation_strategy
        self.default_system = default_system
        self._test_support_assistant_tokens_mask()
        # pre_pipeline_names / post_pipeline_names inherit from Template (class-level).

    def parse(self, decoded: str) -> List[Dict[str, Any]]:
        text = decoded or ''
        if DeepseekV4Template._TOOL_CALLS_START not in text:
            return []

        parse_text = re.sub(
            r'\s*' + re.escape(DeepseekV4Template._TOOL_CALLS_START),
            '\n\n' + DeepseekV4Template._TOOL_CALLS_START,
            text,
            count=1,
        )
        if eos_token not in parse_text:
            parse_text += eos_token

        for thinking_mode in ('chat', 'thinking'):
            try:
                message = parse_message_from_completion_text(parse_text, thinking_mode=thinking_mode)
            except ValueError:
                continue
            tool_calls = message.get('tool_calls', []) or []
            for tool_call in tool_calls:
                function = tool_call.get('function', {})
                arguments = function.get('arguments')
                if isinstance(arguments, str):
                    try:
                        function['arguments'] = json.loads(arguments) if arguments.strip() else {}
                    except json.JSONDecodeError:
                        function['arguments'] = {}
            return tool_calls

        return []

    def clean(self, decoded: str) -> str:
        text = decoded or ''
        while True:
            start = text.find(DeepseekV4Template._TOOL_CALLS_START)
            if start < 0:
                return text.rstrip()

            end = text.find(
                DeepseekV4Template._TOOL_CALLS_END,
                start + len(DeepseekV4Template._TOOL_CALLS_START),
            )
            if end < 0:
                text = text[:start]
                continue

            end += len(DeepseekV4Template._TOOL_CALLS_END)
            text = text[:start] + text[end:]

    def parse_tool_call(self, decoded: str) -> List[Dict[str, Any]]:
        """Prefer DeepSeek's DSML tool-call format; fall back to ToolCallRegistry parsers."""
        text = decoded or ''
        if DeepseekV4Template._TOOL_CALLS_START in text:
            result = self.parse(text)
            if result:
                return result
        return super().parse_tool_call(decoded)

    def clean_tool_call(self, decoded: str) -> str:
        """Prefer DeepSeek's DSML tool-call format; fall back to ToolCallRegistry parsers."""
        text = decoded or ''
        if DeepseekV4Template._TOOL_CALLS_START in text:
            return self.clean(text)
        return super().clean_tool_call(decoded)
