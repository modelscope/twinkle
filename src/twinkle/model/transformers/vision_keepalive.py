# Copyright (c) ModelScope Contributors. All rights reserved.
"""Keep placeholder-media forwards connected while zeroing their aligned features."""
import torch
from torch import nn

from twinkle.data_format import VISION_DUMMY_FLAG


def _zero(output):
    if torch.is_tensor(output):
        return output * 0
    if isinstance(output, (tuple, list)):
        return type(output)(_zero(value) for value in output)
    if isinstance(output, dict):
        return {key: _zero(value) for key, value in output.items()}
    return output


class VisionKeepAlive:
    """Read the batch marker at the outer forward; zero only placeholder aligner outputs.

    Aligner paths are relative to the base model. The marker hooks belong to the
    outer callable model, since PEFT can call the base model's forward directly.
    """

    def __init__(self, base_model, forward_model, aligners):
        self.active = False
        self._handles = []
        modules = []
        for path in aligners:
            module = base_model.get_submodule(path)
            modules.extend(module if isinstance(module, (nn.ModuleList, nn.Sequential)) else [module])
        for module in dict.fromkeys(modules):
            self._handles.append(module.register_forward_hook(self._hook))
        self._handles.append(forward_model.register_forward_pre_hook(
            self._pre_hook, with_kwargs=True, prepend=True))
        self._handles.append(forward_model.register_forward_hook(
            self._post_hook, with_kwargs=True, always_call=True))

    def _hook(self, module, args, output):
        return _zero(output) if self.active else output

    def _pre_hook(self, module, args, kwargs):
        self.active = bool(kwargs.pop(VISION_DUMMY_FLAG, False))
        return args, kwargs

    def _post_hook(self, module, args, kwargs, output):
        self.active = False
        return output

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
