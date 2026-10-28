# Copyright (c) ModelScope Contributors. All rights reserved.
from copy import copy
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import numpy as np
import torch
from typing import Dict, List

from twinkle import DeviceMesh, Platform, remote_class, remote_function, torch_util
from twinkle.data_format import InputFeature
from twinkle.perf import timed
from .fields import PADDING_VALUES, VLM_CONCAT_FIELDS, TRANSFORMERS_INPUT_KEYS
from twinkle.utils.transformers_utils import is_flash_attention_implementation


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
    padding_map = PADDING_VALUES

    def __init__(
        self,
        device_mesh: DeviceMesh | None = None,
        padding_free: bool = False,
        framework: Literal["transformers", "megatron"] = "transformers",
        **kwargs,
    ):
        self.device_mesh = device_mesh
        # right is always used in training, and is fit for megatron
        self.padding_side = kwargs.get("padding_side", "right")
        self.padding_free = padding_free
        self.framework = framework
        self.process_pipeline = [
            self.prepare_inputs,
            self.pad_cp,
            self.align_routed_experts,
            self.collate_fn,
            self.to_transformers_dict,
            self.add_extra_padding_free_args,
            self.prepare_gdn_padding_free_args,
            self.drop_causal_4d_mask,
            self.split_cp,
            self.prepare_outputs,
        ]

    @remote_function()
    @timed
    def __call__(self, inputs: InputFeature | list[InputFeature], **kwargs) -> InputFeature | list[InputFeature]:
        for pipe in self.process_pipeline:
            inputs = pipe(inputs, **kwargs)
        return inputs

    def prepare_outputs(self, inputs: list[InputFeature], **kwargs) -> list[InputFeature] | InputFeature:
        if self.framework == "transformers":
            return inputs[0]
        else:
            for _input in inputs:
                if "position_ids" in _input and _input["position_ids"] is not None and _input["position_ids"].dim() > 2:
                    # megatron needs 3, 1, N
                    _input["position_ids"] = _input["position_ids"][1:]
            return inputs

    def prepare_inputs(self, inputs: Union[List[InputFeature], InputFeature], input_key_specs=None,
                       **kwargs) -> List[InputFeature]:
        from .fields import COMMON_INPUT_SPECS

        specs = COMMON_INPUT_SPECS if input_key_specs is None else input_key_specs
        registered_keys = {spec.name for spec in specs}

        def to_tensor(_input):
            for key in list(_input.keys()):
                value = _input[key]
                # Ray/pyarrow can return numpy or list scalars; normalize to tensors.
                # After distributed/datasets.map, labels/completion_mask may become numpy arrays or lists,
                # so tensor ops like labels != ignore_index or .to(device) would fail without this.
                if key in registered_keys and isinstance(value, (list, np.ndarray)):
                    value = torch.as_tensor(value)
                elif isinstance(value, np.ndarray):
                    value = torch.from_numpy(value)
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float, np.number)):
                    value = torch.tensor(value)
                elif (
                    key == "labels"
                    and isinstance(value, (int, float, np.number))
                    and not isinstance(value, (bool, np.bool_))
                ):
                    # seq_cls / regression / reranker emit ONE scalar label per sequence (a class index or
                    # a regression target), not a per-token sequence. The branches above only catch ndarray
                    # and numeric LIST values, so a bare scalar would fall through, get collated into a
                    # plain list/numpy [B] object, be skipped by to_device, and then blow up in the loss on
                    # `labels.to(device)` (numpy has no `.to`). Wrap it as a 1-element tensor so it rides the
                    # normal tensor path to a batched [B, 1] device tensor; SeqClsLoss reduces it back with
                    # view(-1) / squeeze. bool is excluded because it subclasses int.
                    value = torch.tensor([value])
                elif key == "position_ids" and not isinstance(value, torch.Tensor):
                    if value is None:
                        continue
                    value = torch.tensor(value)
                elif (isinstance(value, list)) and key in ("completion_mask", "mm_token_type_ids", "routed_experts"):
                    value = torch.tensor(value)
                elif key in VLM_CONCAT_FIELDS:
                    if not isinstance(value[0], torch.Tensor):
                        value = [torch.tensor(v) for v in value]
                        value = torch.cat(value, dim=0)
                if isinstance(value, torch.Tensor):
                    value = value.to(Platform.get_local_device())
                    if key == 'labels' and value.dim() == 0:
                        value = value.reshape(1)
                    if value.dim() == 1:
                        value = value.unsqueeze(0)
                _input[key] = value
            return _input

        return [to_tensor(_input) for _input in inputs]

    def pad_cp(self, inputs: List[InputFeature], **kwargs) -> List[InputFeature]:

        if self.device_mesh is None:
            return inputs
        if self.framework == "transformers":
            return inputs

        def _pad_cp(_input: InputFeature) -> InputFeature:
            # Pad sequence for parallel compatibility
            # 1. For CP > 1: Megatron's RoPE requires seq_len % (2 * cp_size) == 0
            # 2. For sequence_parallel with TP > 1: seq_len must be divisible by TP size
            cp_size = self.device_mesh.cp_world_size
            tp_size = self.device_mesh.tp_world_size
            position_ids = _input.get("position_ids")

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
                pos_for_cu = position_ids[:1] if position_ids.dim() >= 2 and position_ids.shape[0] > 1 else position_ids
                position_ids_f = pos_for_cu.flatten()
                indices_q = torch.arange(position_ids_f.shape[0], device=position_ids_f.device, dtype=torch.int32)
                cu_seqlens = torch.cat(
                    [
                        indices_q[position_ids_f == 0],
                        torch.tensor(position_ids_f.shape, device=position_ids_f.device, dtype=torch.int32),
                    ]
                )

                for key in [
                    "input_ids",
                    "position_ids",
                    "attention_mask",
                    "labels",
                    "completion_mask",
                    "loss_scale",
                    "mm_token_type_ids",
                ]:
                    value = _input.get(key)
                    if value is None:
                        continue
                    result = []
                    for i in range(cu_seqlens.shape[0]):
                        if i == cu_seqlens.shape[0] - 1:
                            break
                        _value_slice = value[..., cu_seqlens[i] : cu_seqlens[i + 1]]
                        result.append(pad_cp_inputs(_value_slice, padding_value=self.padding_map.get(key, 0)))
                    value = torch.cat(result, dim=-1)
                    _input[key] = value
            elif self.device_mesh.sequence_parallel and tp_size > 1:
                # Sequence parallel without CP still requires seq_len % TP == 0
                for key in [
                    "input_ids",
                    "position_ids",
                    "attention_mask",
                    "labels",
                    "completion_mask",
                    "loss_scale",
                    "mm_token_type_ids",
                ]:
                    value = _input.get(key)
                    if value is not None:
                        _input[key] = pad_cp_inputs(value, padding_value=self.padding_map.get(key, 0))
            return _input

        return [_pad_cp(_inp) for _inp in inputs]

    def split_cp(self, inputs: list[dict[str, Any]], **kwargs) -> list[dict[str, Any]]:
        if self.device_mesh is None:
            return inputs
        if self.framework == "transformers":
            return inputs

        def _split_cp(inputs: dict[str, Any]) -> dict[str, Any]:
            cp_size = self.device_mesh.cp_world_size
            cp_rank = self.device_mesh.cp_rank
            input_ids = inputs.get("input_ids")
            position_ids = inputs.get("position_ids")
            attention_mask = inputs.get("attention_mask")
            batch_labels = inputs.get("labels")
            packed_seq_params: PackedSeqParams = inputs.get("packed_seq_params")
            if packed_seq_params is not None:
                cu_seqlens_q = getattr(packed_seq_params, "cu_seqlens_q", None)
            else:
                cu_seqlens_q = None

            def split_cp_inputs(inputs: torch.Tensor, cu_seqlens: torch.Tensor | None, dim: int):
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
                    view_shape = (
                        *inputs.shape[:dim],
                        2 * cp_size,
                        val.shape[dim] // (2 * cp_size),
                        *inputs.shape[dim + 1 :],
                    )
                    val = val.view(view_shape)
                    index = torch.tensor(
                        [cp_rank, (2 * cp_size - cp_rank - 1)],
                        device=inputs.device,
                        dtype=torch.long,
                    )
                    val = val.index_select(dim, index)
                    view_shape = (*inputs.shape[:dim], -1, *inputs.shape[dim + 1 :])
                    new_inputs.append(val.view(view_shape))
                return torch.cat(new_inputs, dim=dim)

            if cp_size > 1:
                input_ids = split_cp_inputs(input_ids, cu_seqlens_q, dim=1)
                position_ids = split_cp_inputs(position_ids, cu_seqlens_q, dim=-1)
                # attention_mask = split_cp_inputs(attention_mask, cu_seqlens_q, dim=1)
                batch_labels = split_cp_inputs(batch_labels, cu_seqlens_q, dim=1)

                completion_mask = inputs.get("completion_mask")
                if completion_mask is not None:
                    inputs["completion_mask"] = split_cp_inputs(completion_mask, cu_seqlens_q, dim=-1)

                loss_scale = inputs.get("loss_scale")
                if loss_scale is not None:
                    inputs["loss_scale"] = split_cp_inputs(loss_scale, cu_seqlens_q, dim=-1)

                mm_token_type_ids = inputs.get("mm_token_type_ids")
                if mm_token_type_ids is not None:
                    inputs["mm_token_type_ids"] = split_cp_inputs(mm_token_type_ids, cu_seqlens_q, dim=-1)

            inputs["input_ids"] = input_ids
            inputs["position_ids"] = position_ids
            inputs["attention_mask"] = attention_mask
            inputs["labels"] = batch_labels
            return inputs

        return [_split_cp(input) for input in inputs]

    def add_extra_padding_free_args(self, inputs: list[InputFeature], **kwargs) -> list[InputFeature]:
        for _inp in inputs:
            padding_free = bool(self.padding_free or self._any_packing([_inp]))
            if padding_free and self.framework == "megatron":
                _inp["packed_seq_params"] = self._get_packed_seq_params(_inp["position_ids"])
        return inputs

    def prepare_gdn_padding_free_args(self, inputs: list[InputFeature], **kwargs) -> list[InputFeature]:
        if self.framework != "transformers" or not kwargs.get("gdn_padding_free_enabled", False):
            return inputs
        padding_free = bool(self.is_padding_free(inputs))
        if not padding_free or bool(kwargs.get("enable_sp", False)):
            return inputs

        hf_config = kwargs.get("hf_config")
        attn_implementation = getattr(hf_config, "_attn_implementation", None)
        if not is_flash_attention_implementation(attn_implementation):
            raise RuntimeError(
                "Transformers padding_free/packed batches require a FlashAttention backend; "
                f"got attn_implementation={attn_implementation!r}. SDPA/eager attention cannot isolate "
                "logical sequences after they are concatenated into one physical row."
            )

        for _inp in inputs:
            position_ids = _inp.get("position_ids")
            if position_ids is None or not torch.is_tensor(position_ids):
                continue
            packed_seq_params = self._get_packed_seq_params(position_ids)
            _inp["cu_seq_lens_q"] = packed_seq_params.cu_seqlens_q.to(dtype=torch.int32, device=position_ids.device)
            _inp["cu_seq_lens_k"] = packed_seq_params.cu_seqlens_kv.to(dtype=torch.int32, device=position_ids.device)
            _inp["max_length_q"] = int(packed_seq_params.max_seqlen_q)
            _inp["max_length_k"] = int(packed_seq_params.max_seqlen_kv)
        return inputs

    def drop_causal_4d_mask(self, inputs: list[InputFeature], **kwargs) -> list[InputFeature]:
        """On NPU, drop the generic 4D dense mask so MindSpeed can build
        its own compressed causal mask for FlashAttention."""
        if Platform.device_prefix() != "npu":
            return inputs
        attention_mask_type = kwargs.get("attention_mask_type")
        if attention_mask_type != "causal":
            return inputs
        for _inp in inputs:
            attention_mask = _inp.get("attention_mask")
            if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 4:
                _inp["attention_mask"] = None
        return inputs

    @staticmethod
    def _pad_sequence(sequences, padding_value, padding_side, concat=None):
        if padding_side == "right":
            from twinkle.utils import pad_and_stack_tensors

            return pad_and_stack_tensors(
                sequences, pad_value=padding_value, concat=concat if concat is not None else (sequences[0].dim() >= 2)
            )
        else:
            # left padding
            max_len = max([s.shape[0] for s in sequences])

            padded_sequences = []
            for seq in sequences:
                pad_length = max_len - seq.shape[0]
                pad_tuple = [0] * ((seq.dim() - 1) * 2) + [pad_length, 0]
                padded_seq = torch.nn.functional.pad(seq, tuple(pad_tuple), "constant", padding_value)
                padded_sequences.append(padded_seq)
            return torch.stack(padded_sequences)

    @staticmethod
    def _create_4d_attention_mask(attention_mask):
        seq_lens = [s.shape[0] for s in attention_mask]
        max_len = max(seq_lens)
        device = attention_mask[0].device
        attention_mask = torch.tril(
            torch.ones((len(seq_lens), max_len, max_len), dtype=torch.bool, device=device)
        ).view(len(seq_lens), 1, max_len, max_len)
        assert attention_mask.dtype is torch.bool, f"attention_mask.dtype: {attention_mask.dtype}"
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

        cu_seqlens = torch.cat(
            [
                indices_q[position_ids_f == 0],
                torch.tensor(position_ids_f.shape, device=position_ids_f.device, dtype=torch.int32),
            ]
        )

        max_length = cu_seqlens.diff().max()  # position_ids_f.max() + 1
        packed = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_length,
            max_seqlen_kv=max_length,
            qkv_format="thd",
        )
        if torch_util.is_torch_npu_available():
            packed.cu_seqlens_q_padded = cu_seqlens
            packed.cu_seqlens_kv_padded = cu_seqlens

        return packed

    @staticmethod
    def _is_packed_position_ids(position_ids: "torch.Tensor") -> bool:
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
    def _any_packing(inputs: list[InputFeature]):
        for _input in inputs:
            if InputProcessor._is_packed_position_ids(_input.get("position_ids")):
                return True
        return False

    def is_padding_free(self, inputs: InputFeature | list[InputFeature]) -> bool:
        """Return the effective packed/padding-free layout without applying parallelism."""
        rows = inputs if isinstance(inputs, list) else [inputs]
        return bool(self.padding_free or self._any_packing(rows))

    @staticmethod
    def _unpack_by_position_ids(
        position_ids: "torch.Tensor",
        *tensors: "torch.Tensor",
        padding_values: list | None = None,
    ) -> "list[torch.Tensor]":
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
        for tensor, pad_val in zip(tensors, padding_values, strict=False):
            # Normalize to [T, ...] (squeeze batch-1 dim if present)
            t = tensor.squeeze(0) if tensor.dim() >= 2 and tensor.shape[0] == 1 else tensor
            trailing = t.shape[1:]
            seqs = [t[boundaries[i] : boundaries[i + 1]] for i in range(n_seqs)]
            max_len = max(s.shape[0] for s in seqs)
            out = t.new_full((n_seqs, max_len, *trailing), pad_val)
            for i, s in enumerate(seqs):
                out[i, : s.shape[0]] = s
            results.append(out)
        return results

    def unpack_packed_sequences(
        self,
        inputs: Dict[str, Any],
        outputs: Optional[Dict[str, Any]] = None,
        task: str = 'causal_lm',
        input_key_specs=None,
        output_key_specs=None,
    ) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Unpack packed (padding_free) sequences into per-sequence batch format.

        Called after SP gather / CP gather, before loss computation.
        Unpacks ``labels``, ``completion_mask`` and any present output keys
        (``logps``, ``logits``) from ``[1, total_tokens, ...]`` to
        ``[num_sequences, max_seq_len, ...]``. Keys that are ``None`` are silently
        skipped.

        Input/output key specs describe the tensor layout and padding values;
        this restoration step does not require a parallel strategy.

        For ``task='embedding'`` the outputs are already pooled to ``[n_seqs, H]``
        by model-side task postprocessing, so this is a no-op. Same for
        ``task='generative_reranker'``, whose logits are already reduced to ``[n_seqs, 1]``.
        """
        if task in ("embedding", "generative_reranker", "seq_cls", "reranker"):
            return inputs, outputs
        position_ids = inputs.get("position_ids")

        if position_ids is None:
            return inputs, outputs
        if not self._is_packed_position_ids(position_ids):
            return inputs, outputs

        from copy import copy

        from .fields import (
            COMMON_INPUT_SPECS, COMMON_OUTPUT_SPECS,
        )

        input_specs = COMMON_INPUT_SPECS if input_key_specs is None else input_key_specs
        output_specs = COMMON_OUTPUT_SPECS if output_key_specs is None else output_key_specs
        input_keys = [spec for spec in input_specs
                      if spec.stage == 'auxiliary' and spec.unpack_packed and inputs.get(spec.name) is not None]
        output_keys = [spec for spec in output_specs
                       if spec.unpack_packed and outputs and outputs.get(spec.name) is not None]
        all_specs = input_keys + output_keys
        all_tensors = [inputs[spec.name] for spec in input_keys] + [outputs[spec.name] for spec in output_keys]
        # The unpack primitive expects [1, T, ...]; restore the declared axis
        # afterwards so tensors like [1, H, T] retain their public layout.
        axes = [spec.sequence_dim(tensor.dim()) for spec, tensor in zip(all_specs, all_tensors)]
        all_tensors = [tensor.movedim(axis, 1) for tensor, axis in zip(all_tensors, axes)]
        all_pads = [spec.pad_value for spec in all_specs]
        unpacked = self._unpack_by_position_ids(position_ids, *all_tensors, padding_values=all_pads)
        unpacked = [tensor.movedim(1, axis) for tensor, axis in zip(unpacked, axes)]

        inputs = copy(inputs)
        for i, spec in enumerate(input_keys):
            inputs[spec.name] = unpacked[i]

        if output_keys:
            outputs = copy(outputs)
            offset = len(input_keys)
            for i, spec in enumerate(output_keys):
                outputs[spec.name] = unpacked[offset + i]

        return inputs, outputs

    def unpack_inputs(self, inputs: list[dict[str, Any]], task: str = "causal_lm") -> list[dict[str, Any]]:
        """Unpack a list of packed microbatch inputs into per-sequence format.

        For ``task='embedding'`` this is a no-op since embedding outputs are
        already pooled to ``[n_seqs, H]`` upstream.
        """
        return [self.unpack_packed_sequences(inp, task=task)[0] for inp in inputs]

    @staticmethod
    def to_transformers_dict(inputs: List[InputFeature], input_key_specs=None, **kwargs) -> List[InputFeature]:
        import torch

        registered_keys = [spec.name for spec in input_key_specs] if input_key_specs is not None else []
        results = []
        for _input in inputs:
            output = {}
            _keys = TRANSFORMERS_INPUT_KEYS | set(registered_keys)
            for key in list(_input.keys()):
                if key not in _keys:
                    continue
                value = _input[key]
                if key == "channel" or isinstance(value, torch.Tensor) or not isinstance(value, (list, np.ndarray)):
                    output[key] = value
                else:
                    output[key] = np.array(value)
            results.append(InputFeature(**output))
        return results

    def _collate_sequence_key(self, values, spec, padding_free):
        samples = []
        seq_dim = None
        for value in values:
            if not torch.is_tensor(value):
                raise TypeError(f'Registered sequence input {spec.name!r} must be a tensor before collation')
            if value.dim() == 1:
                value = value.unsqueeze(0)
            dim = spec.sequence_dim(value.dim())
            if dim == 0 or value.shape[0] != 1:
                raise ValueError(f'Sequence input {spec.name!r} requires one sample with a leading batch dimension')
            if seq_dim is not None and seq_dim != dim - 1:
                raise ValueError(f'Inconsistent sequence axis for input {spec.name!r}')
            seq_dim = dim - 1
            samples.append(value.squeeze(0))
        if padding_free:
            return torch.cat(samples, dim=seq_dim).unsqueeze(0)
        length = max(value.shape[seq_dim] for value in samples)
        padded = []
        for value in samples:
            shape = list(value.shape)
            shape[seq_dim] = length
            result = value.new_full(shape, spec.pad_value)
            slices = [slice(None)] * value.dim()
            start = length - value.shape[seq_dim] if self.padding_side == 'left' else 0
            slices[seq_dim] = slice(start, start + value.shape[seq_dim])
            result[tuple(slices)] = value
            padded.append(result)
        return torch.stack(padded)

    # when training on mixed text + multimodal data, some fields (e.g. mm_token_type_ids)
    # only exist on multimodal samples. We pad those fields for the missing samples instead of
    # letting collate blow up with a KeyError.
    def _fill_optional_sequence_fields(self, batch: list[InputFeature]) -> None:
        if len(batch) < 2:
            return
        seq_fields = set(self.padding_map) - set(VLM_CONCAT_FIELDS)
        present = {k for feat in batch for k in feat if k in seq_fields}
        for key in present:
            missing = [feat for feat in batch if feat.get(key) is None]
            if not missing or len(missing) == len(batch):
                continue  # all-present (no gap) or all-absent (nothing to align to) -> leave as is
            pad_value = self.padding_map[key]
            for feat in missing:
                input_ids = feat.get("input_ids")
                if input_ids is None:
                    continue
                # Match the row's device/dtype: prepare_inputs may already have moved rows to the
                # accelerator, and a CPU fill would break the cat inside the collate below.
                reference = input_ids if isinstance(input_ids, torch.Tensor) else None
                length = reference.shape[-1] if reference is not None else len(input_ids)
                device = reference.device if reference is not None else None
                shape = reference.shape if reference is not None else (length,)
                # completion_mask is the one field with a derivation rule: every other layer
                # (template, grpo, metric, ledger) reads "absent" as "whatever is scored is the
                # policy's own completion", so a zero fill would silently drop the row from the
                # loss instead of aligning it with the rest of the batch.
                labels = feat.get("labels") if key == "completion_mask" else None
                if labels is not None:
                    labels = labels if isinstance(labels, torch.Tensor) else torch.as_tensor(np.asarray(labels))
                    if labels.shape[-1] == length:
                        feat[key] = (labels.reshape(shape) != -100).to(dtype=torch.long, device=device)
                        continue
                feat[key] = torch.full(shape, pad_value, dtype=torch.long, device=device)

    #: Audio fields whose leading dim is the audio count and must survive collate under the batched
    #: Qwen-audio contract (see _collate_macro_batch). The gemma4 channels-last contract instead
    #: relies on the blanket squeeze folding a single-audio leading dim away, so the squeeze is only
    #: skipped for these fields when the batch is on the Qwen-audio contract.
    _PER_AUDIO_BATCHED_FIELDS = ('input_features', 'feature_attention_mask')

    def _collate_macro_batch(self, inputs: List[InputFeature], input_key_specs=None) -> InputFeature:
        import torch
        from .fields import COMMON_INPUT_SPECS

        builtin_names = {spec.name for spec in COMMON_INPUT_SPECS}
        specs = COMMON_INPUT_SPECS if input_key_specs is None else input_key_specs
        key_specs = {spec.name: spec for spec in specs
                     if spec.stage == 'auxiliary' or (spec.stage == 'forward' and spec.name not in builtin_names)}
        # Two mutually-exclusive audio contracts share the `input_features` field name and are told
        # apart by the mask field that rides alongside it -- the producer (the model's template)
        # emits one pair or the other, so the field name IS the contract marker:
        #   * plane-masked pairs `input_features` with `input_features_mask`: per-sample features
        #     are 2-D [plane0, plane1], the 1-D [plane0] mask is expanded to the feature plane, the
        #     features get a trailing singleton, and the batch folds into dim 0 (channels-last).
        #   * per-audio-batched pairs `input_features` with `feature_attention_mask`: per-sample
        #     features are [n_audios, freq, time] with a [n_audios, time] mask; the leading
        #     audio-count dim stays intact and the concat below stacks both on dim 0.
        # Detect the contract before the squeeze: the squeeze would strip the audio-count dim
        # ([1, freq, time] -> [freq, time]), the concat would then fold the audios into dim 0
        # ([freq*n, time, 1]), and the [n, time] mask could no longer describe the result.
        # TODO: this probe is a model-family check in disguise -- see _PER_AUDIO_BATCHED_FIELDS
        # for the cleanup plan.
        per_audio_batched = any("feature_attention_mask" in _input for _input in inputs)
        # Work on local copies so squeezing doesn't mutate the caller's original samples.
        squeezed = []
        for _input in inputs:
            _input = dict(_input)
            for key in list(_input.keys()):
                if isinstance(_input[key], torch.Tensor):
                    if per_audio_batched and key in self._PER_AUDIO_BATCHED_FIELDS:
                        continue  # keep the audio-count leading dim intact for get_audio_features
                    if key not in key_specs:
                        _input[key] = _input[key].squeeze()
            squeezed.append(_input)
        inputs = squeezed
        self._fill_optional_sequence_fields(inputs)

        vlm_fields = {k: [] for k in VLM_CONCAT_FIELDS}
        text_inputs = []
        for inp in inputs:
            inp = dict(inp)
            for field in VLM_CONCAT_FIELDS:
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

        padding_free = self.is_padding_free(inputs)
        if padding_free:
            for key in text_keys:
                values = [item[key] for item in text_inputs]
                if key in key_specs:
                    result[key] = self._collate_sequence_key(values, key_specs[key], True)
                    continue
                elif key == 'attention_mask':
                    # attention_mask is not needed
                    continue
                if key == "position_ids" and is_mm_position_ids(values[0]):
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
                if key in key_specs:
                    result[key] = self._collate_sequence_key(values, key_specs[key], False)
                elif self.framework == 'megatron' and key == 'attention_mask':
                    result[key] = self._create_4d_attention_mask(values)
                elif key == "position_ids" and is_mm_position_ids(values[0]):
                    result[key] = InputProcessor._pad_sequence(values, self.padding_map[key], self.padding_side)
                    num_axes = values[0].shape[0]
                    result[key] = result[key].reshape(len(values), num_axes, -1).permute(1, 0, 2).contiguous()
                elif isinstance(values[0], torch.Tensor):
                    concat = False if (key == "routed_experts") else None
                    result[key] = InputProcessor._pad_sequence(
                        values, self.padding_map.get(key, 0), self.padding_side, concat
                    )
                    if result[key].dim() == 1:
                        result[key] = result[key].unsqueeze(0)
                else:
                    result[key] = values
            result = InputFeature(**result)
        for field, values in vlm_fields.items():
            if values:
                _values = []
                for i, value in enumerate(values):
                    if not per_audio_batched:
                        # Plane-masked audio contract: per-sample features are 2-D and get a
                        # trailing singleton; the 1-D mask is expanded to the feature plane. Batch
                        # is folded into dim 0 by the concat below.
                        if (
                            field == "input_features"
                        ):  # [freq_bins, time_steps] -> [freq_bins, time_steps, num_features]
                            assert len(value.shape) == 2
                            value = value.unsqueeze(-1)
                        if field == "input_features_mask":  # [freq_bins,] -> [freq_bins, time_steps]
                            assert len(value.shape) == 1
                            input_features_shape = vlm_fields["input_features"][i].shape
                            assert value.shape[0] == input_features_shape[0]
                            value = value.unsqueeze(1).expand(input_features_shape[:2])
                    # Per-audio-batched contract: `input_features` is already [n, freq, time] and
                    # `feature_attention_mask` is [n, time] (both kept un-squeezed above); the concat
                    # below stacks them on dim 0 into the [sum(n), freq, time] / [sum(n), time]
                    # layout the audio encoder consumes. No reshape here.
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
                   input_key_specs=None,
                   **kwargs) -> List[InputFeature]:
        if len(inputs) == 1 and self.framework != 'megatron':
            return inputs
        if micro_batch_size is None:
            # normal collate
            outputs = self._collate_macro_batch(inputs, input_key_specs=input_key_specs)
            for key in outputs:
                if key in VLM_CONCAT_FIELDS:
                    outputs[key] = torch.cat(outputs[key], dim=0)
            return [outputs]
        padding_free = self.is_padding_free(inputs)
        if variable_seq_lengths or padding_free:
            # each micro batch has its own packed length
            assert len(inputs) >= micro_batch_size
            outputs = []
            for i in range(0, len(inputs), micro_batch_size):
                _output = self._collate_macro_batch(inputs[i:i + micro_batch_size],
                                                    input_key_specs=input_key_specs)
                for key in _output:
                    if key in VLM_CONCAT_FIELDS:
                        _output[key] = torch.cat(_output[key], dim=0)
                outputs.append(_output)
            return outputs
        else:
            # each macro batch shares the same length
            res = self._collate_macro_batch(inputs, input_key_specs=input_key_specs)
            keys = list(res.keys())
            outputs = []
            for i in range(0, len(inputs), micro_batch_size):
                end = i + micro_batch_size
                output = {}
                for key in keys:
                    if key == "position_ids" and res[key].dim() > 2:
                        output[key] = res[key][:, i:end, :]
                    elif key in VLM_CONCAT_FIELDS:
                        output[key] = torch.cat(res[key][i:end], dim=0)
                    else:
                        output[key] = res[key][i:end]
                outputs.append(output)
            return outputs

    def postprocess_tensor_cp(self, tensor, cu_seqlens=None):
        """All-gather and reconstruct full sequence from a CP load-balanced shard.

        Thin wrapper over :func:`twinkle.utils.torch_utils.gather_cp_load_balanced`
        that resolves the CP group via Megatron's ``parallel_state``. Pass
        ``cu_seqlens`` (full-sequence pre-split boundaries, e.g. from
        ``packed_seq_params.cu_seqlens_q``) for multi-segment packed batches.
        """
        if self.device_mesh.cp_world_size <= 1:
            return tensor
        from megatron.core import parallel_state as mpu

        from twinkle.utils.torch_utils import gather_cp_load_balanced

        return gather_cp_load_balanced(tensor, mpu.get_context_parallel_group(), seq_dim=1, cu_seqlens=cu_seqlens)

    @staticmethod
    def _last_valid_indices(attention_mask: torch.Tensor) -> torch.Tensor:
        """Last valid (non-pad) index per row, direction-agnostic. Ports swift.get_last_valid_indices.

        flip + argmax finds the first 1 from the right, i.e. the last valid token, so left/right/no
        padding all resolve correctly within one batch.
        """
        seq_len = attention_mask.shape[1]
        return seq_len - 1 - torch.fliplr(attention_mask).argmax(dim=1)

    def align_routed_experts(self, inputs: list[InputFeature] | InputFeature, **kwargs) -> list[InputFeature]:
        def align_to(_input):
            routed_experts = _input.get("routed_experts", None)
            input_ids = _input.get("input_ids", None)
            input_seq_len = input_ids.shape[-1] if input_ids is not None else 0
            if routed_experts is None:
                # No reference routing to replay, so a stray mask is meaningless (and would reach the model
                # as an unexpected column) -- drop it.
                _input.pop("replay_mask", None)
                return _input
            # The number of experts in the output can be 1 less than (prompt_length + response_token_count)
            # This gap of 1 is expected
            # For more details, please refer PR https://github.com/vllm-project/vllm/pull/28284
            experts_seq_len = routed_experts.shape[0]
            # Padding routed_experts(seq_len, layer_num, topk) seq_len to match the seq_len of the input_ids
            padding_routed_experts = routed_experts
            padding_len = input_seq_len - experts_seq_len
            if padding_len > 0:
                padding_routed_experts = torch.nn.functional.pad(
                    routed_experts, (0, 0, 0, 0, 0, padding_len), "constant", self.padding_map.get("routed_experts", 0)
                )
            _input["routed_experts"] = padding_routed_experts.unsqueeze(0)

            # The per-token replay_mask rides alongside routed_experts and must stay token-aligned with it:
            # pad it to the SAME input_seq_len, and clamp away every row that has no reference routing -- the
            # sampler's short-by-one tail (PR 28284) plus the CP padding -- so those tokens recompute their
            # routing natively instead of replaying zero-filled expert ids. prepare_inputs already gave the
            # mask a leading batch dim ([1, seq]); restore that shape after the seq-axis pad. Kept as long
            # (0/1) to match routed_experts' int64 through the CP/SP scatter; consumers call ``.bool()``.
            replay_mask = _input.get("replay_mask", None)
            if replay_mask is not None:
                mask = replay_mask.reshape(-1)[:experts_seq_len]
                mask_pad_len = input_seq_len - mask.shape[0]
                if mask_pad_len > 0:
                    mask = torch.nn.functional.pad(
                        mask, (0, mask_pad_len), "constant", self.padding_map.get("replay_mask", 0)
                    )
                _input["replay_mask"] = mask.to(torch.long).unsqueeze(0)

            return _input

        return [align_to(_input) for _input in inputs]
