# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import torch
from transformers import AutoConfig, PreTrainedTokenizerFast
from typing import Any, Dict, List, Literal, Optional

from twinkle.hub import HubOperation
from .base import Template
from .deepseek_v4_encoding import encode_messages


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

    def __init__(
        self,
        model_id: str,
        use_chat_template: bool = True,
        max_length: Optional[int] = 8192,
        truncation_strategy: Literal['raise', 'left', 'right', 'split'] = 'raise',
        default_system: Optional[str] = None,
        enable_thinking: bool = True,
        **kwargs,
    ):
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
        self.pre_pipeline = [
            self._add_default_system,
            self._to_standard_reasoning_content,
            self._build_standard_messages,
        ]
        self.post_pipeline = [
            self._check_max_length,
            self._add_attention_fields,
            self._roll_labels,
        ]
