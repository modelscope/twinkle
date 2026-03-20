# Copyright (c) ModelScope Contributors. All rights reserved.
import json
from numbers import Number
from peft import LoraConfig
from typing import Mapping

from twinkle.dataset import DatasetMeta

primitive_types = (str, Number, bool, bytes, type(None))
container_types = (Mapping, list, tuple, set, frozenset)
basic_types = (*primitive_types, *container_types)


def _serialize_data_slice(data_slice):
    if data_slice is None:
        return None
    if isinstance(data_slice, range):
        return {'_slice_type_': 'range', 'start': data_slice.start, 'stop': data_slice.stop, 'step': data_slice.step}
    if isinstance(data_slice, (list, tuple)):
        return {'_slice_type_': 'list', 'values': list(data_slice)}
    raise ValueError(f'Http mode does not support data_slice of type {type(data_slice).__name__}. '
                     'Supported types: range, list, tuple.')

def serialize_object(obj) -> str:
    if isinstance(obj, DatasetMeta):
        data = obj.__dict__.copy()
        data['data_slice'] = _serialize_data_slice(data.get('data_slice'))
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
    elif isinstance(obj, Mapping):
        return json.dumps(obj, ensure_ascii=False)
    elif isinstance(obj, basic_types):
        return obj
    else:
        raise ValueError(f'Unsupported object: {obj}')