# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass
from typing import Union, List, Optional, Dict, Any, Literal
import numpy as np
import torch
from twinkle import torch_util
from twinkle import Platform, DeviceMesh, remote_class, remote_function
from twinkle.data_format import InputFeature


@dataclass
class PackedSeqParams:
    qkv_format: str = None
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_kv: torch.Tensor = None
    cu_seqlens_q_padded: torch.Tensor = None
    cu_seqlens_kv_padded: torch.Tensor = None
    max_seqlen_q: int = None
    max_seqlen_kv: int = None


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
                 framework: Literal['transformers', 'megatron'] = 'transformers',
                 **kwargs):
        self.device_mesh = device_mesh
        self.padding_side = kwargs.get('padding_side', 'right')
        self.padding_free = padding_free
        self.framework = framework
        self.return_4d_attention_mask = self.framework == 'megatron'

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
        if self.framework == 'megatron':
            inputs = self._to_megatron_dict(inputs)
        elif self.framework == 'transformers':
            inputs = self._to_transformers_dict(inputs)

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

    @staticmethod
    def _create_4d_attention_mask(attention_mask):
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

    @staticmethod
    def _get_packed_seq_params(position_ids):
        assert position_ids.shape[0] == 1, f'position_ids.shape: {position_ids.shape}'
        position_ids_f = position_ids.flatten()
        indices_q = torch.arange(position_ids_f.shape[0], device=position_ids_f.device, dtype=torch.int32)

        cu_seqlens = torch.cat([
            indices_q[position_ids_f == 0],
            torch.tensor(position_ids_f.shape, device=position_ids_f.device, dtype=torch.int32),
        ])

        max_length = cu_seqlens.diff().max()  # position_ids_f.max() + 1
        packed = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_length,
            max_seqlen_kv=max_length,
            qkv_format='thd')

        if torch_util.is_torch_npu_available():
            packed.cu_seqlens_q_padded = cu_seqlens
            packed.cu_seqlens_kv_padded = cu_seqlens

        return packed

    @staticmethod
    def _any_packing_free(inputs):
        is_padding_free = False
        for _input in inputs:
            position_ids = np.array(_input['position_ids'])
            # multiple 0/1, multiple sequences
            zero_count = np.sum(position_ids == 0)
            one_count = np.sum(position_ids == 1)
            is_padding_free = is_padding_free or (zero_count > 1 and one_count > 1)
        return is_padding_free

    @staticmethod
    def _to_transformers_dict(feature: InputFeature) -> dict:
        """Transfer the InputFeature object to a dict needed by `transformers` models."""
        import torch
        output = {}
        _keys = ['input_ids', 'input_embeddings', 'attention_mask', 'position_ids', 'labels', 'completion_mask',
                 'logits_to_keep', 'num_items_in_batch']
        for key in list(feature.keys()):
            if key in _keys:
                output[key] = np.array(feature[key]) if not isinstance(feature[key], torch.Tensor) else feature[key]
        return output

    @staticmethod
    def _to_megatron_dict(feature: InputFeature) -> dict:
        """Transfer the InputFeature object to a dict needed by `transformers` models."""
        import torch
        output = {}
        _keys = ['input_ids', 'input_embeddings', 'attention_mask', 'position_ids', 'labels', 'completion_mask',
                 'logits_to_keep', 'num_items_in_batch', 'packed_seq_params']
        for key in list(feature.keys()):
            if key in _keys:
                if key != 'packed_seq_params':
                    output[key] = np.array(feature[key]) if not isinstance(feature[key], torch.Tensor) else feature[key]
                else:
                    output[key] = feature[key]
        return output

    def _split_cp(self, inputs: Dict[str, Any]) -> Dict[str, Any]:

        cp_size = self.device_mesh.cp_world_size
        cp_rank = self.device_mesh.cp_rank
        input_ids = inputs.get('input_ids')
        position_ids = inputs.get('position_ids')
        attention_mask = inputs.get('attention_mask')
        batch_labels = inputs.get('labels')

        def split_tensor_for_cp(tensor, dim=-1):
            """
            Split tensor along sequence dimension for Context Parallel.

            With causal masking, split into 2*CP chunks and assign alternating
            chunks to balance workload across CP ranks.
            For CP rank i: chunks [i, 2*CP-1-i]
            """
            if tensor is None or cp_size <= 1:
                return tensor

            if dim < 0:
                dim = (dim + tensor.ndim) % tensor.ndim

            seq_len = tensor.shape[dim]

            # Reshape to [batch, 2*cp_size, seq_per_chunk, ...]
            view_shape = list(tensor.shape)
            view_shape[dim:dim + 1] = [2 * cp_size, seq_len // (2 * cp_size)]
            reshaped = tensor.view(*view_shape)

            # Select chunks [cp_rank, 2*cp_size-1-cp_rank]
            index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)],
                                 device='cpu',
                                 pin_memory=True).cuda(non_blocking=True)
            selected = reshaped.index_select(dim, index)

            # Reshape back: [batch, 2*seq_per_chunk, ...]
            out_shape = list(tensor.shape)
            out_shape[dim] = seq_len // cp_size
            return selected.reshape(*out_shape)

        if cp_size > 1:
            input_ids = split_tensor_for_cp(input_ids, dim=-1)
            position_ids = split_tensor_for_cp(position_ids, dim=-1)
            attention_mask = split_tensor_for_cp(attention_mask, dim=-1)
            batch_labels = split_tensor_for_cp(batch_labels, dim=-1)

        inputs['input_ids'] = input_ids
        inputs['position_ids'] = position_ids
        inputs['attention_mask'] = attention_mask
        inputs['labels'] = batch_labels
        return inputs

    def _pad_cp(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Pad sequence for parallel compatibility
        # 1. For CP > 1: Megatron's RoPE requires seq_len % (2 * cp_size) == 0
        # 2. For sequence_parallel with TP > 1: seq_len must be divisible by TP size
        cp_size = self.device_mesh.cp_world_size
        tp_size = self.device_mesh.tp_world_size
        input_ids = inputs.get('input_ids')
        position_ids = inputs.get('position_ids')
        attention_mask = inputs.get('attention_mask')
        batch_labels = inputs.get('labels')

        if input_ids is not None:
            seq_len = input_ids.shape[1]

            # Calculate required divisor based on parallelism settings
            if cp_size > 1:
                divisor = 2 * cp_size
            elif self.device_mesh.sequence_parallel and tp_size > 1:
                divisor = tp_size
            else:
                divisor = 1

            if divisor > 1 and seq_len % divisor != 0:
                pad_len = divisor - (seq_len % divisor)
                # Pad input_ids
                input_ids = torch.nn.functional.pad(input_ids,
                                                    (0, pad_len),
                                                    value=0)
                # Pad labels if present
                if batch_labels is not None:
                    batch_labels = torch.nn.functional.pad(batch_labels,
                                                           (0, pad_len),
                                                           value=-100)
                # Pad attention_mask if present
                if attention_mask is not None:
                    attention_mask = torch.nn.functional.pad(
                        attention_mask, (0, pad_len), value=0)
                # Pad position_ids if present
                if position_ids is not None:
                    position_ids = torch.nn.functional.pad(position_ids,
                                                           (0, pad_len),
                                                           value=0)
        inputs['input_ids'] = input_ids
        inputs['position_ids'] = position_ids
        inputs['attention_mask'] = attention_mask
        inputs['labels'] = batch_labels
        return inputs

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

        padding_free = self.padding_free or InputProcessor._any_packing_free(inputs)
        if padding_free:
            for key in text_keys:
                values = [item[key] for item in text_inputs]
                if key == 'attention_mask':
                    # attention_mask is not needed
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

        result = self._pad_cp(result)
        if padding_free:
            result['packed_seq_params'] = InputProcessor._get_packed_seq_params(result['position_ids'])
        result = self._split_cp(result)
        return result

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
