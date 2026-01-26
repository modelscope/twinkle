# Copyright (c) ModelScope Contributors. All rights reserved.
# Reference: swift/swift/megatron/model/mm_gpts/utils.py
from abc import ABC, abstractmethod
from contextlib import contextmanager

import torch
from megatron.core.models.huggingface import HuggingFaceModule as _HuggingFaceModule
from twinkle.model.megatron.args import get_args
from twinkle.utils import deep_getattr
from transformers import PreTrainedModel
from transformers.utils import ContextManagers


@contextmanager
def patch_hf_initialize_weight():

    _origin_initialize_weight = PreTrainedModel._initialize_weights

    def _initialize_weight(self, *args, **kwargs):
        return

    PreTrainedModel._initialize_weights = _initialize_weight
    try:
        yield
    finally:
        PreTrainedModel._initialize_weights = _origin_initialize_weight


@contextmanager
def patch_device_map_meta(model_cls):
    __origin_init__ = model_cls.__init__

    def __init__(self, *args, **kwargs):
        with torch.device('meta'):
            __origin_init__(self, *args, **kwargs)

    model_cls.__init__ = __init__

    try:
        yield
    finally:
        model_cls.__init__ = __origin_init__


class HuggingFaceModule(_HuggingFaceModule, ABC):
    module_mapping = {}  # hf -> mcore

    def __init__(self, config, ignore_init_model_cls=None):
        super().__init__(config)
        args = get_args()
        attn_impl = getattr(args, 'attn_impl', None) or 'flash_attn'
        # Handle both enum and string attention_backend
        attn_backend = args.attention_backend
        is_flash = (getattr(attn_backend, 'name', attn_backend) == 'flash'
                    if attn_backend else False)
        kwargs = {'attn_impl': attn_impl} if is_flash else {}
        ignore_init_model_cls = ignore_init_model_cls or []
        if not isinstance(ignore_init_model_cls, list):
            ignore_init_model_cls = [ignore_init_model_cls]
        context_list = [patch_device_map_meta(model_cls) for model_cls in ignore_init_model_cls]
        context_list.append(patch_hf_initialize_weight())
        kwargs['model_type'] = args.hf_model_type
        from ..register import get_megatron_model_meta
        from transformers import AutoModel, AutoProcessor
        megatron_model_meta = get_megatron_model_meta(args.hf_model_type)
        auto_model_cls = megatron_model_meta.auto_model_cls if megatron_model_meta else AutoModel
        with ContextManagers(context_list):
            model = auto_model_cls.from_pretrained(
                args.model_dir, torch_dtype=args.torch_dtype, trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)

        self.model_config = model.config
        for hf_prefix, mg_prefix in self.module_mapping.items():
            setattr(self, mg_prefix, deep_getattr(model, hf_prefix))
        self._hf_model = [model]
        self.prepare_model(model)
        self.to('cuda')

    def prepare_model(self, hf_model):
        pass

    @abstractmethod
    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        pass
