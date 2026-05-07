# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

from twinkle import DeviceMesh, Platform, remote_class, remote_function, torch_util
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
    # Fields required by newer megatron-core TE attention (dynamic CP)
    cp_group: object = None
    local_cp_size: int = None


@remote_class()
class InputProcessor:
    padding_map = {
        'input_ids': 0,
        'mm_token_type_ids': 0,
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
        'mm_token_type_ids': 0,
    }

    # VLM fields to concatenate (not pad) in batch
    VLM_CONCAT_FIELDS = {
        'pixel_values',
        'image_grid_thw',
        'pixel_values_videos',
        'video_grid_thw',
        'input_features',
        'feature_attention_mask',
        'grid_thws',
    }

    def __init__(self,
                 device_mesh: Optional[DeviceMesh] = None,
                 padding_free: bool = False,
                 framework: Literal['transformers', 'megatron'] = 'transformers',
                 **kwargs):
        self.device_mesh = device_mesh
        # right is always used in training, and is fit for megatron
        self.padding_side = kwargs.get('padding_side', 'right')
        self.padding_free = padding_free
        self.framework = framework
        self.process_pipeline = [
            self.prepare_inputs,
            self.pad_cp,
            self.collate_fn,
            self.to_transformers_dict,
            self.add_extra_padding_free_args,
            self.prepare_transformers_padding_free_patch,
            self.drop_causal_4d_mask,
            self.split_cp,
            self.apply_transformers_sp,
            self.prepare_outputs,
        ]

    @remote_function()
    def __call__(self, inputs: Union[InputFeature, List[InputFeature]],
                 **kwargs) -> Union[InputFeature, List[InputFeature]]:
        for pipe in self.process_pipeline:
            inputs = pipe(inputs, **kwargs)
        return inputs

    def prepare_outputs(self, inputs: List[InputFeature], **kwargs) -> Union[List[InputFeature], InputFeature]:
        if self.framework == 'transformers':
            return inputs[0]
        else:
            for _input in inputs:
                if 'position_ids' in _input and _input['position_ids'].dim() > 2:
                    # megatron needs 3, 1, N
                    _input['position_ids'] = _input['position_ids'][1:]
            return inputs

    def prepare_inputs(self, inputs: Union[List[InputFeature], InputFeature], **kwargs) -> List[InputFeature]:

        def to_tensor(_input):
            import torch
            for key in list(_input.keys()):
                value = _input[key]
                # Ray/pyarrow can return numpy or list scalars; normalize to tensors.
                # After distributed/datasets.map, labels/completion_mask may become numpy arrays or lists,
                # so tensor ops like labels != ignore_index or .to(device) would fail without this.
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value)
                elif (isinstance(value, list) and isinstance(value[0],
                                                             (int, float, np.number))) or key == 'position_ids':
                    value = torch.tensor(value)
                elif (isinstance(value, list)) and key in ('completion_mask', 'mm_token_type_ids'):
                    value = torch.tensor(value)
                elif key in self.VLM_CONCAT_FIELDS:
                    if not isinstance(value[0], torch.Tensor):
                        value = [torch.tensor(v) for v in value]
                        value = torch.cat(value, dim=0)
                if isinstance(value, torch.Tensor):
                    value = value.to(Platform.get_local_device())
                    if value.dim() == 1:
                        value = value.unsqueeze(0)
                _input[key] = value
            return _input

        return [to_tensor(_input) for _input in inputs]

    def apply_transformers_sp(self, inputs: List[InputFeature], **kwargs) -> List[InputFeature]:
        sp_strategy = kwargs.get('sp_strategy')
        if self.framework != 'transformers' or sp_strategy is None:
            return inputs
        padding_free = bool(self.padding_free or self._any_packing(inputs))
        results = []
        for _input in inputs:
            payload = dict(_input)
            payload['padding_free'] = padding_free
            results.append(InputFeature(**sp_strategy.preprocess_inputs(payload)))
        return results

    def postprocess_tensor_sp(self, inputs: Dict[str, Any], outputs: Dict[str, Any],
                              **kwargs) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Adjust SP tensors after forward and before loss computation.

        Pipeline: SP gather → packed-sequence unpack.
        After this call, logps and labels are in per-sequence batch format
        ``[num_sequences, max_seq_len]`` when the input was packed, or left
        unchanged for normal (non-packed) batches.
        """
        sp_strategy = kwargs.get('sp_strategy')
        if self.framework == 'transformers' and sp_strategy is not None:
            return sp_strategy.gather_loss_tensors(inputs, outputs)
        return inputs, outputs

    def pad_cp(self, inputs: List[InputFeature], **kwargs) -> List[InputFeature]:

        if self.device_mesh is None:
            return inputs
        if self.framework == 'transformers':
            return inputs

        def _pad_cp(_input: InputFeature) -> InputFeature:
            # Pad sequence for parallel compatibility
            # 1. For CP > 1: Megatron's RoPE requires seq_len % (2 * cp_size) == 0
            # 2. For sequence_parallel with TP > 1: seq_len must be divisible by TP size
            cp_size = self.device_mesh.cp_world_size
            tp_size = self.device_mesh.tp_world_size
            position_ids = _input.get('position_ids')

            def pad_cp_inputs(input_tensor: torch.Tensor, padding_value: int) -> torch.Tensor:
                if input_tensor is None:
                    return input_tensor

                seq_len = input_tensor.shape[-1]

                # Calculate required divisor based on parallelism settings
                if cp_size > 1:
                    divisor = 2 * cp_size
                elif self.device_mesh.sequence_parallel and tp_size > 1:
                    divisor = tp_size
                else:
                    divisor = 1

                if divisor > 1 and seq_len % divisor != 0:
                    pad_len = divisor - (seq_len % divisor)
                    input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_len), value=padding_value)
                return input_tensor

            if cp_size > 1:
                position_ids_f = position_ids.flatten()
                indices_q = torch.arange(position_ids_f.shape[0], device=position_ids_f.device, dtype=torch.int32)
                cu_seqlens = torch.cat([
                    indices_q[position_ids_f == 0],
                    torch.tensor(position_ids_f.shape, device=position_ids_f.device, dtype=torch.int32),
                ])

                for key in [
                        'input_ids', 'position_ids', 'attention_mask', 'labels', 'completion_mask', 'mm_token_type_ids'
                ]:
                    value = _input.get(key)
                    if value is None:
                        continue
                    result = []
                    for i in range(cu_seqlens.shape[0]):
                        if i == cu_seqlens.shape[0] - 1:
                            break
                        _value_slice = value[..., cu_seqlens[i]:cu_seqlens[i + 1]]
                        result.append(pad_cp_inputs(_value_slice, padding_value=self.padding_map.get(key, 0)))
                    value = torch.cat(result, dim=-1)
                    _input[key] = value
            elif self.device_mesh.sequence_parallel and tp_size > 1:
                # Sequence parallel without CP still requires seq_len % TP == 0
                for key in [
                        'input_ids', 'position_ids', 'attention_mask', 'labels', 'completion_mask', 'mm_token_type_ids'
                ]:
                    value = _input.get(key)
                    if value is not None:
                        _input[key] = pad_cp_inputs(value, padding_value=self.padding_map.get(key, 0))
            return _input

        return [_pad_cp(_inp) for _inp in inputs]

    def split_cp(self, inputs: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:

        if self.device_mesh is None:
            return inputs
        if self.framework == 'transformers':
            return inputs

        def _split_cp(inputs: Dict[str, Any]) -> Dict[str, Any]:

            cp_size = self.device_mesh.cp_world_size
            cp_rank = self.device_mesh.cp_rank
            input_ids = inputs.get('input_ids')
            position_ids = inputs.get('position_ids')
            attention_mask = inputs.get('attention_mask')
            batch_labels = inputs.get('labels')
            packed_seq_params: PackedSeqParams = inputs.get('packed_seq_params')
            if packed_seq_params is not None:
                cu_seqlens_q = getattr(packed_seq_params, 'cu_seqlens_q', None)
            else:
                cu_seqlens_q = None

            def split_cp_inputs(inputs: torch.Tensor, cu_seqlens: Optional[torch.Tensor], dim: int):
                if inputs is None:
                    return inputs
                if dim < 0:
                    dim = (dim + inputs.ndim) % inputs.ndim
                new_inputs = []
                for i in range(1 if cu_seqlens is None else (cu_seqlens.shape[0] - 1)):
                    if cu_seqlens is None:
                        val = inputs
                    else:
                        slices = [slice(None)] * inputs.ndim
                        slices[dim] = slice(cu_seqlens[i], cu_seqlens[i + 1])
                        val = inputs[tuple(slices)]
                    view_shape = (*inputs.shape[:dim], 2 * cp_size, val.shape[dim] //
                                  (2 * cp_size), *inputs.shape[dim + 1:])
                    val = val.view(view_shape)
                    index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device='cpu',
                                         pin_memory=True).cuda(non_blocking=True)
                    val = val.index_select(dim, index)
                    view_shape = (*inputs.shape[:dim], -1, *inputs.shape[dim + 1:])
                    new_inputs.append(val.view(view_shape))
                return torch.cat(new_inputs, dim=dim)

            if cp_size > 1:
                input_ids = split_cp_inputs(input_ids, cu_seqlens_q, dim=1)
                position_ids = split_cp_inputs(position_ids, cu_seqlens_q, dim=1)
                # attention_mask = split_cp_inputs(attention_mask, cu_seqlens_q, dim=1)
                batch_labels = split_cp_inputs(batch_labels, cu_seqlens_q, dim=1)

                completion_mask = inputs.get('completion_mask')
                if completion_mask is not None:
                    inputs['completion_mask'] = split_cp_inputs(completion_mask, cu_seqlens_q, dim=-1)

                mm_token_type_ids = inputs.get('mm_token_type_ids')
                if mm_token_type_ids is not None:
                    inputs['mm_token_type_ids'] = split_cp_inputs(mm_token_type_ids, cu_seqlens_q, dim=-1)

            inputs['input_ids'] = input_ids
            inputs['position_ids'] = position_ids
            inputs['attention_mask'] = attention_mask
            inputs['labels'] = batch_labels
            return inputs

        return [_split_cp(input) for input in inputs]

    def add_extra_padding_free_args(self, inputs: List[InputFeature], **kwargs) -> List[InputFeature]:
        for _inp in inputs:
            padding_free = bool(self.padding_free or self._any_packing([_inp]))
            if padding_free and self.framework == 'megatron':
                _inp['packed_seq_params'] = self._get_packed_seq_params(_inp['position_ids'])
        return inputs

    def prepare_transformers_padding_free_patch(self, inputs: List[InputFeature], **kwargs) -> List[InputFeature]:
        if self.framework != 'transformers':
            return inputs
        model = kwargs.get('model')
        if model is None:
            return inputs
        padding_free = bool(self.padding_free or self._any_packing(inputs))
        if not padding_free or bool(kwargs.get('enable_sp', False)):
            return inputs

        from twinkle.patch import apply_patch
        from twinkle.patch.gdn_padding_free import GatedDeltaNetPaddingFreePatch

        apply_patch(
            model,
            GatedDeltaNetPaddingFreePatch,
            hf_config=kwargs.get('hf_config'),
            enable_sp=False,
        )
        if not getattr(model, '_twinkle_gdn_padding_free_patched', False):
            return inputs

        for _inp in inputs:
            position_ids = _inp.get('position_ids')
            if position_ids is None or not torch.is_tensor(position_ids):
                continue
            packed_seq_params = self._get_packed_seq_params(position_ids)
            _inp['cu_seq_lens_q'] = packed_seq_params.cu_seqlens_q.to(dtype=torch.int32, device=position_ids.device)
            _inp['cu_seq_lens_k'] = packed_seq_params.cu_seqlens_kv.to(dtype=torch.int32, device=position_ids.device)
            _inp['max_length_q'] = int(packed_seq_params.max_seqlen_q)
            _inp['max_length_k'] = int(packed_seq_params.max_seqlen_kv)
        return inputs

    def drop_causal_4d_mask(self, inputs: List[InputFeature], **kwargs) -> List[InputFeature]:
        """On NPU, drop the generic 4D dense mask so MindSpeed can build
        its own compressed causal mask for FlashAttention."""
        if Platform.device_prefix() != 'npu':
            return inputs
        attention_mask_type = kwargs.get('attention_mask_type')
        if attention_mask_type != 'causal':
            return inputs
        for _inp in inputs:
            attention_mask = _inp.get('attention_mask')
            if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 4:
                _inp['attention_mask'] = None
        return inputs

    @staticmethod
    def _pad_sequence(sequences, padding_value, padding_side):
        if padding_side == 'right':
            from twinkle.utils import pad_and_stack_tensors
            return pad_and_stack_tensors(sequences, pad_value=padding_value, concat=sequences[0].dim() >= 2)
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
        device = attention_mask[0].device
        attention_mask = torch.tril(torch.ones((len(seq_lens), max_len, max_len), dtype=torch.bool,
                                               device=device)).view(len(seq_lens), 1, max_len, max_len)
        assert attention_mask.dtype is torch.bool, f'attention_mask.dtype: {attention_mask.dtype}'
        for i, seq_len in enumerate(seq_lens):
            attention_mask[i, :, :, seq_len:] = 0
        attention_mask = ~attention_mask
        return attention_mask

    @staticmethod
    def _get_packed_seq_params(position_ids):
        if position_ids.shape[0] > 1:
            position_ids = position_ids[:1]
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
    def _is_packed_position_ids(position_ids: 'torch.Tensor') -> bool:
        """Detect packed sequences by multiple (0, 1, ...) resets in position_ids."""
        if position_ids is None or not isinstance(position_ids, torch.Tensor):
            return False
        pos = position_ids
        if pos.dim() == 3:
            pos = pos[0]  # mrope: [3, batch, seq]
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)
        if pos.dim() != 2:
            return False
        for i in range(pos.shape[0]):
            row = pos[i]
            if int((row == 0).sum()) > 1 and int((row == 1).sum()) > 1:
                return True
        return False

    @staticmethod
    def _any_packing(inputs: List[InputFeature]):
        for _input in inputs:
            if InputProcessor._is_packed_position_ids(_input.get('position_ids')):
                return True
        return False

    @staticmethod
    def _unpack_by_position_ids(
        position_ids: 'torch.Tensor',
        *tensors: 'torch.Tensor',
        padding_values: Optional[List] = None,
    ) -> 'List[torch.Tensor]':
        """Split packed tensors into ``[num_seqs, max_seq_len, ...]``.

        Sequence boundaries are detected where ``position_ids`` resets to 0.
        Each tensor may have arbitrary trailing dimensions (e.g. ``[1, T]``
        for labels/logps or ``[1, T, V]`` for logits).

        Args:
            position_ids: ``[1, T]`` or ``[3, 1, T]`` (mrope) packed position ids.
            *tensors: Tensors to unpack; leading dims are squeezed to ``[T, ...]``.
            padding_values: Per-tensor fill value for right-padding (default 0).

        Returns:
            List of unpacked tensors, each ``[num_seqs, max_seq_len, ...]``.
        """
        pos = position_ids
        if pos.dim() == 3:
            pos = pos[0]  # mrope
        pos_flat = pos.view(-1)

        boundaries = (pos_flat == 0).nonzero(as_tuple=True)[0]
        total_len = pos_flat.shape[0]
        boundaries = torch.cat([boundaries, pos_flat.new_tensor([total_len])])
        n_seqs = boundaries.shape[0] - 1

        if padding_values is None:
            padding_values = [0] * len(tensors)

        results = []
        for tensor, pad_val in zip(tensors, padding_values):
            # Normalize to [T, ...] (squeeze batch-1 dim if present)
            t = tensor.squeeze(0) if tensor.dim() >= 2 and tensor.shape[0] == 1 else tensor
            trailing = t.shape[1:]
            seqs = [t[boundaries[i]:boundaries[i + 1]] for i in range(n_seqs)]
            max_len = max(s.shape[0] for s in seqs)
            out = t.new_full((n_seqs, max_len, *trailing), pad_val)
            for i, s in enumerate(seqs):
                out[i, :s.shape[0]] = s
            results.append(out)
        return results

    def unpack_packed_sequences(
        self,
        inputs: Dict[str, Any],
        outputs: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Unpack packed (padding_free) sequences into per-sequence batch format.

        Called after SP gather / CP gather, before loss computation.
        Unpacks ``labels`` and any present output keys (``logps``, ``logits``)
        from ``[1, total_tokens, ...]`` to ``[num_sequences, max_seq_len, ...]``.
        Keys that are ``None`` are silently skipped.
        """
        labels = inputs.get('labels')
        position_ids = inputs.get('position_ids')

        if labels is None or position_ids is None:
            return inputs, outputs
        if not self._is_packed_position_ids(position_ids):
            return inputs, outputs

        from copy import copy

        # Collect output keys to unpack: (key, pad_value)
        output_keys = []
        for key, pad_val in [('logps', 0), ('logits', 0)]:
            if outputs and outputs.get(key) is not None:
                output_keys.append((key, pad_val))

        all_tensors = [labels] + [outputs[k] for k, _ in output_keys]
        all_pads = [-100] + [p for _, p in output_keys]
        unpacked = self._unpack_by_position_ids(position_ids, *all_tensors, padding_values=all_pads)

        inputs = copy(inputs)
        inputs['labels'] = unpacked[0]

        if output_keys:
            outputs = copy(outputs)
            for i, (key, _) in enumerate(output_keys):
                outputs[key] = unpacked[i + 1]

        return inputs, outputs

    def unpack_inputs(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Unpack a list of packed microbatch inputs into per-sequence format."""
        return [self.unpack_packed_sequences(inp)[0] for inp in inputs]

    @staticmethod
    def to_transformers_dict(inputs: List[InputFeature], **kwargs) -> List[InputFeature]:
        import torch
        results = []
        for _input in inputs:
            output = {}
            _keys = [
                'input_ids',
                'input_embeddings',
                'attention_mask',
                'position_ids',
                'labels',
                'completion_mask',
            ] + list(InputProcessor.VLM_CONCAT_FIELDS)
            for key in list(_input.keys()):
                if key in _keys:
                    output[key] = np.array(_input[key]) if not isinstance(_input[key], torch.Tensor) else _input[key]
            results.append(InputFeature(**output))
        return results

    def _collate_macro_batch(self, inputs: List[InputFeature]) -> InputFeature:
        import torch

        for _input in inputs:
            for key in list(_input.keys()):
                if isinstance(_input[key], torch.Tensor):
                    _input[key] = _input[key].squeeze()

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

        def is_mm_position_ids(position_ids):
            if position_ids is None:
                return False
            return position_ids.dim() > 1 and position_ids.shape[0] > 1

        padding_free = self.padding_free or self._any_packing(inputs)
        if padding_free:
            for key in text_keys:
                values = [item[key] for item in text_inputs]
                if key == 'attention_mask':
                    # attention_mask is not needed
                    continue
                if key == 'position_ids' and is_mm_position_ids(values[0]):
                    # mrope needs to cat the sequence and unsequeeze the middle dim
                    value = torch.cat(values, dim=-1).unsqueeze(1)
                elif isinstance(values[0], torch.Tensor):
                    value = torch.cat(values, dim=0).unsqueeze(0)
                else:
                    value = values
                result[key] = value
            result = InputFeature(**result)
        else:
            for key in text_keys:
                values = [item[key] for item in text_inputs]
                if self.framework == 'megatron' and key == 'attention_mask':
                    result[key] = self._create_4d_attention_mask(values)
                elif key == 'position_ids' and is_mm_position_ids(values[0]):
                    result[key] = InputProcessor._pad_sequence(values, self.padding_map[key], self.padding_side)
                    result[key] = result[key].reshape(values[0].shape[0], len(values), -1)
                elif isinstance(values[0], torch.Tensor):
                    result[key] = InputProcessor._pad_sequence(values, self.padding_map[key], self.padding_side)
                    if result[key].dim() == 1:
                        result[key] = result[key].unsqueeze(0)
                else:
                    result[key] = values
            result = InputFeature(**result)
        for field, values in vlm_fields.items():
            if values:
                _values = []
                for value in values:
                    if value.dim() == 1:
                        # image_thw may be squeezed
                        value = value.unsqueeze(0)
                    _values.append(value)
                result[field] = _values
        return result

    def collate_fn(self,
                   inputs: List[InputFeature],
                   micro_batch_size: Optional[int] = None,
                   variable_seq_lengths=False,
                   **kwargs) -> List[InputFeature]:
        if len(inputs) == 1 and self.framework != 'megatron':
            return inputs
        if micro_batch_size is None:
            # normal collate
            outputs = self._collate_macro_batch(inputs)
            for key in outputs:
                if key in self.VLM_CONCAT_FIELDS:
                    outputs[key] = torch.cat(outputs[key], dim=0)
            return [outputs]
        padding_free = self.padding_free or self._any_packing(inputs)
        if variable_seq_lengths or padding_free:
            # each micro batch has its own packed length
            assert len(inputs) >= micro_batch_size
            outputs = []
            for i in range(0, len(inputs), micro_batch_size):
                _output = self._collate_macro_batch(inputs[i:i + micro_batch_size])
                for key in _output:
                    if key in self.VLM_CONCAT_FIELDS:
                        _output[key] = torch.cat(_output[key], dim=0)
                outputs.append(_output)
            return outputs
        else:
            # each macro batch shares the same length
            res = self._collate_macro_batch(inputs)
            keys = list(res.keys())
            outputs = []
            for i in range(0, len(inputs), micro_batch_size):
                end = i + micro_batch_size
                output = {}
                for key in keys:
                    if key == 'position_ids' and res[key].dim() > 2:
                        output[key] = res[key][:, i:end, :]
                    elif key in self.VLM_CONCAT_FIELDS:
                        output[key] = torch.cat(res[key][i:end], dim=0)
                    else:
                        output[key] = res[key][i:end]
                outputs.append(output)
            return outputs

    def postprocess_tensor_cp(self, tensor):
        """All-gather and reconstruct full sequence from CP-split tensor.

        Uses load-balanced split pattern: each CP rank holds chunks [rank] and
        [2*cp_size - rank - 1] from the original 2*cp_size chunks.

        Only the current rank's slice retains the original tensor (and its
        gradient graph); other ranks' slices are plain copies.  This means
        backward through the reconstructed tensor only produces gradients for
        the local chunk, naturally distributing the gradient across CP ranks
        without extra scaling.

        Args:
            tensor: [batch_size, seq_len/cp_size] CP-split tensor

        Returns:
            [batch_size, full_seq_len] reconstructed full tensor
        """
        if self.device_mesh.cp_world_size <= 1:
            return tensor

        from megatron.core import parallel_state as mpu
        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()
        cp_group = mpu.get_context_parallel_group()

        gathered = [torch.empty_like(tensor) for _ in range(cp_size)]
        torch.distributed.all_gather(gathered, tensor.contiguous(), group=cp_group)
        gathered[cp_rank] = tensor

        batch_size = tensor.shape[0]
        seq_len_per_cp = tensor.shape[1]
        full_seq_len = seq_len_per_cp * cp_size
        chunk_len = full_seq_len // (2 * cp_size)
        half_len = seq_len_per_cp // 2

        output = tensor.new_zeros(batch_size, full_seq_len)
        for j in range(cp_size):
            o = gathered[j]
            output[:, j * chunk_len:(j + 1) * chunk_len] = o[:, :half_len]
            reverse_idx = 2 * cp_size - j - 1
            output[:, reverse_idx * chunk_len:(reverse_idx + 1) * chunk_len] = o[:, half_len:]

        return output
