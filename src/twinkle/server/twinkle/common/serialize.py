# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import sys
from numbers import Number
from typing import Mapping, Any

if sys.version_info[:2] <= (3, 11):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

from peft import LoraConfig

from twinkle.dataset import DatasetMeta

supported_types = {
    DatasetMeta,
    LoraConfig,
}

primitive_types = (str, Number, bool, bytes, type(None))
container_types = (Mapping, list, tuple, set, frozenset)
basic_types = (*primitive_types, *container_types)


def serialize_object(obj) -> str:
    if isinstance(obj, DatasetMeta):
        assert obj.data_slice is None, 'Http mode does not support data_slice'
        data = obj.__dict__
        data['_TWINKLE_TYPE_'] = 'DatasetMeta'
        return json.dumps(data, ensure_ascii=False)
    elif isinstance(obj, LoraConfig):
        filtered_dict = {
            _subkey: _subvalue
            for _subkey, _subvalue in obj.__dict__.items()
            if isinstance(_subvalue, basic_types) and not _subkey.startswith('_')
        }
        filtered_dict['_TWINKLE_TYPE_'] = 'LoraConfig'
        return json.dumps(filtered_dict, ensure_ascii=False)
    elif isinstance(obj, (Mapping, TypedDict)):
        return json.dumps(obj, ensure_ascii=False)
    elif isinstance(obj, basic_types):
        return obj
    else:
        raise ValueError(f'Unsupported object: {obj}')


def deserialize_object(data: str) -> Any:
    try:
        data = json.loads(data)
    except Exception: # noqa
        return data

    if '_TWINKLE_TYPE_' in data:
        _type = data.pop('_TWINKLE_TYPE_')
        if _type == 'DatasetMeta':
            return DatasetMeta(**data)
        elif _type == 'LoraConfig':
            return LoraConfig(**data)
        else:
            raise ValueError(f'Unsupported type: {_type}')
    else:
        return data



