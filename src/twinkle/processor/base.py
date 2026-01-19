# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Union, List, Optional, Dict, Any
import numpy as np
from twinkle import Platform, DeviceMesh, remote_class, remote_function
from twinkle.data_format import InputFeature, to_transformers_dict


@remote_class()
class InputProcessor:

    padding_map = {
        'input_ids': 0,
        'inputs_embeds': 0.0,
        'attention_mask': 0,
        'labels': -100,
        'loss_scale': 0.0,
        'position_ids': -1,
        'length': -1,
    }

    def __init__(self, device_mesh: Optional[DeviceMesh] = None, padding_free: bool = False, **kwargs):
        self.device_mesh = device_mesh
        self.padding_side = kwargs.get('padding_side', 'right')
        self.padding_free = padding_free

    @remote_function()
    def __call__(self, inputs: Union[InputFeature, List[InputFeature]]):
        if isinstance(inputs, list):
            inputs = self.collate_fn(inputs)
        return self.prepare_inputs(inputs)

    @remote_function()
    def prepare_inputs(self, inputs: InputFeature) -> InputFeature:
        import torch
        for key in list(inputs.keys()):
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(Platform.get_local_device())
        return inputs

    @staticmethod
    def _pad_sequence(sequences, padding_value, padding_side):
        if padding_side == 'right':
            from torch.nn.utils.rnn import pad_sequence
            return pad_sequence(sequences, batch_first=True, padding_value=padding_value)

        import torch
        max_len = max([s.shape[0] for s in sequences])

        padded_sequences = []
        for seq in sequences:
            pad_length = max_len - seq.shape[0]
            pad_tuple = [0] * ((seq.dim() - 1) * 2) + [pad_length, 0]
            padded_seq = torch.nn.functional.pad(seq, tuple(pad_tuple), 'constant', padding_value)
            padded_sequences.append(padded_seq)
        return torch.stack(padded_sequences)

    @remote_function()
    def collate_fn(self, inputs: List[InputFeature]) -> Dict[str, Any]:
        import torch
        keys = inputs[0].keys()
        result = {}
        if self.padding_free:
            for key in keys:
                values = [item[key] for item in inputs]
                if isinstance(values[0], np.ndarray):
                    value = np.concatenate(values, axis=-1)
                    value = torch.from_numpy(value)
                elif isinstance(values[0], list) and isinstance(values[0][0], (int, float, np.number)):
                    values = [v for lst in values for v in lst]
                    value = torch.tensor(values)
                elif isinstance(values[0], torch.Tensor):
                    value = torch.cat(values, dim=-1)
                else:
                    value = values
                result[key] = value
            result = InputFeature(**result)
        else:
            for key in keys:
                values = [item[key] for item in inputs]

                if isinstance(values[0], np.ndarray):
                    values = [torch.from_numpy(v) for v in values]
                    result[key] = InputProcessor._pad_sequence(values, self.padding_map[key], self.padding_side)
                elif isinstance(values[0], list) and isinstance(values[0][0], (int, float, np.number)):
                    values = [torch.tensor(v) for v in values]
                    result[key] = InputProcessor._pad_sequence(values, self.padding_map[key], self.padding_side)
                elif isinstance(values[0], torch.Tensor):
                    result[key] = InputProcessor._pad_sequence(values, self.padding_map[key], self.padding_side)
                else:
                    result[key] = values
            result = InputFeature(**result)
        return to_transformers_dict(result)
