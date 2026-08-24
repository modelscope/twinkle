# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Mapping


def _remap_names(values, name_mapping: Mapping[str, str]):
    """只转换参数名称，数字参数 ID 保持不变。"""
    return [name_mapping.get(value, value) if isinstance(value, str) else value for value in values]


def remap_optimizer_state_names(state_dict: dict[str, Any], name_mapping: Mapping[str, str]) -> None:
    """原地转换 optimizer state 中作为参数身份标识的名称。

    普通 optimizer 的 ``state`` 使用数字 ID；FSDP 的完整 state 使用参数 FQN。
    这里仅转换字符串，因此两种格式可以共用同一个入口。
    """
    if not name_mapping:
        return

    state = state_dict.get('state')
    if isinstance(state, dict):
        state_dict['state'] = {
            name_mapping.get(name, name) if isinstance(name, str) else name: value
            for name, value in state.items()
        }

    for group in state_dict.get('param_groups', []):
        for field in ('params', 'param_names'):
            values = group.get(field)
            if isinstance(values, (list, tuple)):
                group[field] = _remap_names(values, name_mapping)
