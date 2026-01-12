# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Union, List, Optional
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
    }

    def __init__(self, device_mesh: Optional[DeviceMesh] = None, **kwargs):
        self.device_mesh = device_mesh
        self.padding_side = kwargs.get('padding_side', 'right')

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

    def _inner_collate_fn(self, batch):
        import torch
        result = {}
        keys = batch[0].keys()

        for key in keys:
            values = [item[key] for item in batch]

            if isinstance(values[0], np.ndarray):
                values = [torch.from_numpy(v) for v in values]
                result[key] = InputProcessor._pad_sequence(values, self.padding_map[key], self.padding_side)
            elif isinstance(values[0], torch.Tensor):
                result[key] = InputProcessor._pad_sequence(values, self.padding_map[key], self.padding_side)
            else:
                result[key] = values

        return result

    @remote_function()
    def collate_fn(self, inputs: List[InputFeature]) -> InputFeature:
        batch_encoded = self._inner_collate_fn([to_transformers_dict(_input) for _input in inputs])
        return InputFeature(**batch_encoded)
