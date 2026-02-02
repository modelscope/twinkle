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
        'pixel_values': 0.0,
        'image_grid_thw': 0,
        'pixel_values_videos': 0.0,
        'video_grid_thw': 0,
        'input_features': 0.0,
        'feature_attention_mask': 0,
    }

    # VLM fields to concatenate (not pad) in batch
    VLM_CONCAT_FIELDS = {
        'pixel_values', 'image_grid_thw',
        'pixel_values_videos', 'video_grid_thw',
        'input_features', 'feature_attention_mask',
        'grid_thws',
    }

    def __init__(self, device_mesh: Optional[DeviceMesh] = None,
                 padding_free: bool = False,
                 **kwargs):
        self.device_mesh = device_mesh
        self.padding_side = kwargs.get('padding_side', 'right')
        self.padding_free = padding_free
        self.return_4d_attention_mask = kwargs.get('return_4d_attention_mask', False)

    @remote_function()
    def __call__(self, inputs: Union[InputFeature, List[InputFeature]], **kwargs):
        _collated = inputs
        if isinstance(inputs, list):
            _collated = self.collate_fn(inputs, **kwargs)
        if not isinstance(_collated, list):
            # macro_batch_size is None, so it's a dict
            return self.prepare_inputs(_collated)
        else:
            # a list of macro batches
            return [self.prepare_inputs(_macro_batch) for _macro_batch in _collated]
        

    @remote_function()
    def prepare_inputs(self, inputs: InputFeature) -> InputFeature:
        import torch
        for key in list(inputs.keys()):
            value = inputs[key]
            # Ray/pyarrow can return numpy or list scalars; normalize to tensors.
            # After distributed/datasets.map, labels/completion_mask may become numpy arrays or lists,
            # so tensor ops like labels != ignore_index or .to(device) would fail without this.
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
                value = torch.tensor(value)
            if isinstance(value, torch.Tensor):
                value = value.to(Platform.get_local_device())
            inputs[key] = value
        return inputs

    @staticmethod
    def _pad_sequence(sequences, padding_value, padding_side):
        if padding_side == 'right':
            from torch.nn.utils.rnn import pad_sequence
            return pad_sequence(sequences, batch_first=True, padding_value=padding_value)
        else:
            # left padding
            import torch
            max_len = max([s.shape[0] for s in sequences])

            padded_sequences = []
            for seq in sequences:
                pad_length = max_len - seq.shape[0]
                pad_tuple = [0] * ((seq.dim() - 1) * 2) + [pad_length, 0]
                padded_seq = torch.nn.functional.pad(seq, tuple(pad_tuple), 'constant', padding_value)
                padded_sequences.append(padded_seq)
            return torch.stack(padded_sequences)

    def _create_4d_attention_mask(self, attention_mask):
        import torch
        seq_lens = [s.shape[0] for s in attention_mask]
        max_len = max(seq_lens)
        attention_mask = torch.tril(torch.ones(
            (len(seq_lens), max_len, max_len), dtype=torch.bool)).view(len(seq_lens), 1, max_len, max_len)
        assert attention_mask.dtype is torch.bool, f'attention_mask.dtype: {attention_mask.dtype}'
        for i, seq_len in enumerate(seq_lens):
            attention_mask[i, :, :, seq_len:] = 0
        attention_mask = ~attention_mask
        return attention_mask

    def _collate_macro_batch(self, inputs: List[InputFeature]) -> Dict[str, Any]:
        import torch

        vlm_fields = {k: [] for k in self.VLM_CONCAT_FIELDS}
        text_inputs = []
        for inp in inputs:
            inp = dict(inp)
            for field in self.VLM_CONCAT_FIELDS:
                if field in inp:
                    vlm_fields[field].append(inp.pop(field))
            text_inputs.append(inp)

        # Collect text field keys preserving first-seen order (dict.fromkeys deduplicates while keeping order).
        # This avoids treating VLM fields as text and fixes KeyError on pure-text batches.
        text_keys = list(dict.fromkeys(key for inp in text_inputs for key in inp.keys()))

        result = {}

        if self.padding_free:
            for key in text_keys:
                values = [item[key] for item in text_inputs]
                if self.return_4d_attention_mask and key == 'attention_mask':
                    # padding_free & 4D attention_mask is not needed
                    continue
                if isinstance(values[0], np.ndarray):
                    value = np.expand_dims(np.concatenate(values, axis=0), axis=0)
                    value = torch.from_numpy(value)
                elif isinstance(values[0], list) and isinstance(values[0][0], (int, float, np.number)):
                    values = [[v for lst in values for v in lst]]
                    value = torch.tensor(values)
                elif isinstance(values[0], torch.Tensor):
                    value = torch.cat(values, dim=0).unsqueeze(0)
                else:
                    value = values
                result[key] = value
            result = InputFeature(**result)
        else:
            for key in text_keys:
                values = [item[key] for item in text_inputs]
                if self.return_4d_attention_mask and key == 'attention_mask':
                    values = [torch.tensor(v) for v in values]
                    result[key] = self._create_4d_attention_mask(values)
                elif isinstance(values[0], np.ndarray):
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

        for field, values in vlm_fields.items():
            if values:
                if isinstance(values[0], np.ndarray):
                    result[field] = torch.from_numpy(np.concatenate(values, axis=0))
                elif isinstance(values[0], torch.Tensor):
                    result[field] = torch.cat(values, dim=0)

        return to_transformers_dict(result)

    @remote_function()
    def collate_fn(self, inputs: List[InputFeature], micro_batch_size: Optional[int] = None, variable_seq_lengths=False) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        if micro_batch_size is None:
            # normal collate
            return self._collate_macro_batch(inputs)
        elif variable_seq_lengths:
            # each macro batch has its own length
            assert len(inputs) >= micro_batch_size
            outputs = []
            for i in range(0, len(inputs), micro_batch_size):
                outputs.append(self._collate_macro_batch(inputs[i:i + micro_batch_size]))
            return outputs
        else:
            # each macro batch shares the same length
            res = self._collate_macro_batch(inputs)
            keys = list(res.keys())
            outputs = []
            for i in range(0, len(inputs), micro_batch_size):
                output = {}
                for key in keys:
                    output[key] = res[key][i:i + micro_batch_size]
                outputs.append(output)
            return outputs
