# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tagged serialization for non-JSON domain objects used by the wire contract.

Heavy domain types are resolved lazily so importing :mod:`twinkle.protocol` does
not load PEFT or the dataset implementation.
"""
import json
from dataclasses import fields
from functools import lru_cache
from numbers import Number
from pydantic import BaseModel
from typing import Any, Mapping

primitive_types = (str, Number, bool, type(None))
container_types = (Mapping, list, tuple, set, frozenset)
basic_types = (*primitive_types, *container_types)


@lru_cache(maxsize=1)
def _dataset_meta_cls():
    from twinkle.dataset import DatasetMeta
    return DatasetMeta


@lru_cache(maxsize=1)
def _dataset_meta_fields() -> frozenset[str]:
    return frozenset(field.name for field in fields(_dataset_meta_cls()))


@lru_cache(maxsize=1)
def _lora_config_cls():
    from peft import LoraConfig
    return LoraConfig


def _serialize_data_slice(data_slice):
    """Serialize data_slice (Iterable) into a JSON-compatible dict."""
    if data_slice is None:
        return None
    if isinstance(data_slice, range):
        return {'_slice_type_': 'range', 'start': data_slice.start, 'stop': data_slice.stop, 'step': data_slice.step}
    if isinstance(data_slice, (list, tuple)):
        return {'_slice_type_': 'list', 'values': list(data_slice)}
    raise ValueError(f'Http mode does not support data_slice of type {type(data_slice).__name__}. '
                     'Supported types: range, list, tuple.')


def _deserialize_data_slice(data_slice):
    """Deserialize a dict back into the original data_slice object."""
    if data_slice is None:
        return None
    if not isinstance(data_slice, dict) or '_slice_type_' not in data_slice:
        return data_slice
    slice_type = data_slice['_slice_type_']
    if slice_type == 'range':
        return range(data_slice['start'], data_slice['stop'], data_slice['step'])
    if slice_type == 'list':
        return data_slice['values']
    raise ValueError(f'Unsupported data_slice type: {slice_type}')


def serialize_object(obj) -> Any:
    if isinstance(obj, (bytes, bytearray, memoryview)):
        raise TypeError(f'Unsupported binary object: {type(obj).__name__}')
    if isinstance(obj, _dataset_meta_cls()):
        data = {name: getattr(obj, name) for name in _dataset_meta_fields()}
        data['data_slice'] = _serialize_data_slice(data.get('data_slice'))
        data['_TWINKLE_TYPE_'] = 'DatasetMeta'
        return json.dumps(data, ensure_ascii=False)
    elif isinstance(obj, _lora_config_cls()):
        filtered_dict = {}
        for _subkey, _subvalue in obj.__dict__.items():
            if isinstance(_subvalue, basic_types) and not _subkey.startswith('_'):
                # Convert set/frozenset to list for JSON serialization
                if isinstance(_subvalue, (set, frozenset)):
                    filtered_dict[_subkey] = list(_subvalue)
                else:
                    filtered_dict[_subkey] = _subvalue
        filtered_dict['_TWINKLE_TYPE_'] = 'LoraConfig'
        return json.dumps(filtered_dict, ensure_ascii=False)
    elif isinstance(obj, BaseModel):
        # Pydantic models: convert to dict for JSON serialization by requests
        return obj.model_dump(mode='json')
    elif isinstance(obj, Mapping):
        return json.dumps(obj, ensure_ascii=False)
    elif isinstance(obj, basic_types):
        return obj
    else:
        raise ValueError(f'Unsupported object: {obj}')


def deserialize_object(data: str) -> Any:
    try:
        data = json.loads(data)
    except Exception:  # noqa
        return data

    if '_TWINKLE_TYPE_' in data:
        _type = data.pop('_TWINKLE_TYPE_')
        if _type == 'DatasetMeta':
            fields_set = _dataset_meta_fields()
            data = {key: value for key, value in data.items() if key in fields_set}
            data['data_slice'] = _deserialize_data_slice(data.get('data_slice'))
            return _dataset_meta_cls()(**data)
        elif _type == 'LoraConfig':
            return _lora_config_cls()(**data)
        else:
            raise ValueError(f'Unsupported type: {_type}')
    else:
        return data
