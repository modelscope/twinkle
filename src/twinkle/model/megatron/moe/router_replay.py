# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import Any, List, Optional

import torch
from twinkle import Platform
from twinkle.utils import get_logger

logger = get_logger()

try:
    from megatron.core.transformer.moe.router_replay import RouterReplay as McoreRouterReplay, RouterReplayAction
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
    ROUTER_REPLAY_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the installed megatron-core version
    logger.warning('RouterReplay not available in current megatron-core version')
    McoreRouterReplay = None
    RouterReplayAction = None
    MoEAlltoAllTokenDispatcher = None
    ROUTER_REPLAY_AVAILABLE = False


class RouterReplay:
    """Megatron routing replay: per-``forward_backward`` orchestration plus the backend
    capability surface (mode resolution, cp/sp slicing, whole-model injection, RECORD
    assembly, dispatcher patch). A null object when replay is disabled.
    """

    def __init__(self, *, enabled: bool, mode: Any, forward_only: bool, cp_size: Optional[int],
                 apply_patch) -> None:
        self.recorded: List[Optional[torch.Tensor]] = []
        self.action = None
        self._cp_size = int(cp_size or 1)
        if enabled and mode is not None:
            self.action = RouterReplay.resolve_router_replay_mode(mode, allow_record=forward_only)
            apply_patch()

    @property
    def active(self) -> bool:
        return self.action is not None

    def arm(self) -> None:
        """Set the action on every local router; call right before the schedule (mirrors legacy train_step)."""
        if self.action is not None:
            RouterReplay.set_global_router_replay_action(self.action)

    def pre_mb(self, model: Any, batch: dict, routed_experts: Optional[torch.Tensor] = None,
               replay_mask: Optional[torch.Tensor] = None) -> None:
        if self.action is None:
            return
        RouterReplay.prepare_replay_forward(self.action, model, routed_experts, batch.get('packed_seq_params'), replay_mask)

    def post_mb(self, model: Any, batch: dict) -> None:
        if self.action is None:
            return
        # Full (pre-CP) seq length sizes a zero-MoE chunk's placeholder: the processor's
        # split_cp left input_ids CP-local, and full = local * cp matches the MoE chunks' rows.
        batch_size = batch['input_ids'].shape[0] if 'input_ids' in batch else 1
        full_seq = batch['input_ids'].shape[1] * self._cp_size if 'input_ids' in batch else None
        RouterReplay.finish_replay_forward(self.action, model, batch.get('packed_seq_params'), self.recorded,
                              batch_size, full_seq)

    def finish(self, model_output: dict, tf_config: Any, num_microbatches: int, vpp_size: int,
               manual_cleanup: bool) -> None:
        if self.action is None:
            return
        if self.action == RouterReplayAction.RECORD and self.recorded:
            # Regroup per-(micro-batch x chunk) recordings into the whole-model tensor: SUM each
            # micro-batch's chunks, concat micro-batches, all-reduce(SUM) across the PP group.
            assembled = RouterReplay.assemble_recorded_routing(self.recorded, tf_config, num_microbatches, vpp_size)
            if assembled is not None:
                model_output['routed_experts'] = assembled
        if not manual_cleanup:
            RouterReplay.clear_global_router_replay_action()
            RouterReplay.clear_global_indices()

    @staticmethod
    def _require_available() -> None:
        """Fail loudly when mcore lacks ``RouterReplay``: silently recomputing routing would
corrupt an MoE policy's importance ratio."""
        if not ROUTER_REPLAY_AVAILABLE:
            raise RuntimeError('Megatron routing replay is not supported: upgrade megatron-core to 0.16.0 '
                               'or higher (mcore.transformer.moe.router_replay.RouterReplay is missing).')

    @staticmethod
    def _tf_config(model: Any) -> Any:
        """Return the mcore ``TransformerConfig`` of an (unwrapped) Megatron model chunk."""
        config = getattr(model, 'config', None)
        if config is None:
            raise ValueError('Megatron routing replay needs a model exposing `.config` (the mcore '
                             f'TransformerConfig); got {type(model).__name__} without one.')
        return config

    @staticmethod
    def resolve_router_replay_action(action: Any) -> Any:
        """Resolve a backend-neutral action token to this backend's enum.

        ``None`` and enum values pass through; strings ('record' / 'replay_forward' /
        'replay_backward') are looked up by value.
        """
        RouterReplay._require_available()
        if action is None or isinstance(action, RouterReplayAction):
            return action
        if isinstance(action, str):
            try:
                return RouterReplayAction(action)
            except ValueError:
                raise ValueError(f'Unknown router_replay_action {action!r}; expected one of '
                                 f'{[a.value for a in RouterReplayAction]} or None.') from None
        raise TypeError(f'router_replay_action must be a str, RouterReplayAction or None, got {type(action).__name__}.')

    @staticmethod
    def resolve_router_replay_mode(mode: Any, *, allow_record: bool = False) -> Any:
        """Resolve the dev-side mode scalar ('R2'/'R3') to this backend's action.

        'R3' always REPLAYS (the sampler exported the routing); 'R2' RECORDs on the capture pass
        (*allow_record* = forward_only) and REPLAYS on the training forward. None/'disabled' disable.
        """
        if mode is None or mode == 'disabled':
            return None
        if mode == 'R3':
            return RouterReplayAction.REPLAY_FORWARD
        if mode == 'R2':
            return RouterReplayAction.RECORD if allow_record else RouterReplayAction.REPLAY_FORWARD
        raise ValueError(f'Unknown router_replay_mode {mode!r}; expected one of disabled/R2/R3.')

    @staticmethod
    def iter_model_routers(model: Any):
        """Yield ``(layer_number, router_replay)`` for every replay-enabled router of *model*.

        Walks the forwarded module rather than mcore's process-global registry, which can
        hold stale routers from a rebuilt decoder.
        """
        from megatron.core.transformer.moe.router import TopKRouter

        for chunk in model if isinstance(model, (list, tuple)) else [model]:
            # Scope to the decoder: ``mtp`` is a sibling of ``decoder`` on GPTModel/HybridModel and the replay
            # tensor carries no MTP rows, so walking the decoder skips the MTP routers.
            for module in getattr(chunk, 'decoder', chunk).modules():
                router = getattr(module, 'router_replay', None)
                if (isinstance(module, TopKRouter) and router is not None
                        and getattr(module, 'layer_number', None) is not None):
                    yield module.layer_number, router

    @staticmethod
    def _slice_seq_for_cp_sp(tensor: torch.Tensor, packed_seq_params: Any = None) -> torch.Tensor:
        """Slice the seq dim of a ``[bs, seq, ...]`` tensor to this rank's local (cp, sp) shard.

        Shared by the index and mask paths so they stay token-for-token aligned; the SP
        scatter is a no-op when SP/TP is 1.
        """
        from megatron.core import mpu
        from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region

        if mpu.get_context_parallel_world_size() > 1:
            from mcore_bridge import split_cp_inputs
            tensor = split_cp_inputs(tensor, getattr(packed_seq_params, 'cu_seqlens_q', None), 1)
        return scatter_to_sequence_parallel_region(tensor.transpose(0, 1)).transpose(0, 1)

    @staticmethod
    def get_local_replay_mask_for_current_rank(global_replay_mask: Optional[torch.Tensor],
                                               tf_config: Any,
                                               packed_seq_params: Any = None) -> Optional[torch.Tensor]:
        """Slice a whole-model ``[bs, seq]`` replay mask to this rank's ``[local_seq*bs]`` tokens.

        mcore tokens are seq-major (row = s*bs + b), so the sliced mask is transposed before
        flattening to align with the router's scores; None means whole-sequence replay.
        """
        if global_replay_mask is None:
            return None
        local = RouterReplay._slice_seq_for_cp_sp(global_replay_mask, packed_seq_params)  # [bs, local_seq]
        return local.transpose(0, 1).flatten(0, 1)  # [local_seq*bs], seq-major to match the router's scores

    @staticmethod
    def get_local_topk_idx_for_current_rank(global_topk_idx: Optional[torch.Tensor],
                                            tf_config: Any,
                                            packed_seq_params: Any = None) -> Optional[torch.Tensor]:
        """Slice a whole-model ``[bs, seq, num_layers, topk]`` tensor's seq dim to the local (cp, sp) shard.

        The layer dim stays whole-model wide: PP/VPP ranks hold (non-contiguous) layer
        subsets, and ``set_router_replay_data`` addresses each router by global layer_number.
        """
        if global_topk_idx is None:
            return None
        return RouterReplay._slice_seq_for_cp_sp(global_topk_idx, packed_seq_params)

    @staticmethod
    def _all_gather_seq(tokens: torch.Tensor, group: Any) -> torch.Tensor:
        """All-gather along dim 0 in group-rank order -- the inverse of the contiguous SP
        scatter the slice helper applies (no autograd: indices are integers)."""
        world = torch.distributed.get_world_size(group=group) if torch.distributed.is_initialized() else 1
        if world <= 1:
            return tokens
        gathered = [torch.empty_like(tokens) for _ in range(world)]
        torch.distributed.all_gather(gathered, tokens.contiguous(), group=group)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def get_router_replay_data(model: Any, batch_size: int = 1, packed_seq_params: Any = None,
                               full_seq: Optional[int] = None) -> Optional[torch.Tensor]:
        """Collect one micro-batch's recorded routing as a whole-model all-layer tensor.

        Returns ``[batch_size, full_seq, num_layers, topk]`` with only this chunk's MoE slots
        filled (slot ``layer_number - 1``); disjoint slots turn the VPP/PP merge into a plain
        SUM. A zero-MoE chunk contributes zeros built from *full_seq*; None only when there
        is nothing to record and no *full_seq* to build zeros from.
        """
        RouterReplay._require_available()
        tf_config = RouterReplay._tf_config(model)
        from megatron.core import mpu

        num_layers = tf_config.num_layers
        topk = getattr(tf_config, 'moe_router_topk', None)
        routers = sorted(RouterReplay.iter_model_routers(model), key=lambda item: item[0])
        if not routers:
            # Zero-MoE chunk: contribute zeros so schedule alignment and the PP all-reduce stay intact.
            if full_seq is None or not isinstance(topk, int) or topk <= 0:
                return None
            from twinkle import Platform
            return torch.zeros((batch_size, full_seq, num_layers, topk), dtype=torch.uint8,
                               device=Platform.get_local_device())
        # Distinct layer_numbers required: the scatter writes slots by assignment, so a
        # duplicate would silently overwrite and corrupt the record.
        layer_numbers = [layer_number for layer_number, _ in routers]
        if len(layer_numbers) != len(set(layer_numbers)):
            raise RuntimeError(
                f'router replay RECORD found duplicate layer numbers in the forwarded model: {layer_numbers}.')
        local_layers = []
        for _, router in routers:
            if router.recorded_topk_idx is None:
                raise RuntimeError(
                    'router replay RECORD did not capture every local MoE router (one has no recorded_topk_idx); '
                    'the record forward is incomplete.')
            local_layers.append(router.recorded_topk_idx.to(torch.uint8))
        num_local_moe = len(local_layers)
        local_tokens, topk = local_layers[0].shape[0], local_layers[0].shape[-1]
        assert local_tokens % batch_size == 0, (
            f'recorded routing token count {local_tokens} is not divisible by batch_size {batch_size}')
        local_seq = local_tokens // batch_size
        # mcore tokens are seq-major (row = s*bs + b); reshape accordingly then permute to
        # batch-major [bs, local_seq, local_moe, topk] (a plain reshape(bs, seq) scrambles tokens).
        local = torch.stack(local_layers, dim=0).reshape(num_local_moe, local_seq, batch_size,
                                                         topk).permute(2, 1, 0, 3).contiguous()
        # 1. inverse SP scatter (the router saw dim 0 of [seq, bs, ...] scattered across the TP group).
        if getattr(tf_config, 'sequence_parallel', False) and mpu.get_tensor_model_parallel_world_size() > 1:
            local = RouterReplay._all_gather_seq(local.transpose(0, 1).contiguous(),
                                    mpu.get_tensor_model_parallel_group()).transpose(0, 1).contiguous()
        # 2. inverse CP split (the router saw the load-balanced-split seq dim, i.e. dim 1 here).
        if mpu.get_context_parallel_world_size() > 1:
            from twinkle.utils.torch_utils import gather_cp_load_balanced
            cu = getattr(packed_seq_params, 'cu_seqlens_q', None) if packed_seq_params is not None else None
            local = gather_cp_load_balanced(local, mpu.get_context_parallel_group(), seq_dim=1, cu_seqlens=cu)
        # local now: [bs, full_seq, num_local_moe, topk]
        gathered_full_seq = local.shape[1]
        # 3. scatter each local MoE layer to its global slot (layer_number - 1); VPP chunks
        #    hold non-contiguous global layers, so address by layer_number, never a range.
        all_layer = local.new_zeros((batch_size, gathered_full_seq, num_layers, topk))
        for slot, (layer_number, _) in enumerate(routers):
            idx = layer_number - 1
            if not 0 <= idx < num_layers:
                raise RuntimeError(
                    f'router replay RECORD got an out-of-range layer_number {layer_number} '
                    f'(num_layers={num_layers}).')
            all_layer[:, :, idx, :] = local[:, :, slot, :]
        return all_layer

    @staticmethod
    def set_router_replay_data(routed_experts: Optional[torch.Tensor],
                               model: Any,
                               packed_seq_params: Any = None,
                               replay_mask: Optional[torch.Tensor] = None) -> None:
        """Inject a whole-model ``routed_experts`` tensor as the local routers' replay target.

        ``[bs, seq_len, num_layers, topk]`` across ALL layers; only the seq dim is sliced and
        routers are addressed by global ``layer_number`` (PP/VPP-correct). ``replay_mask`` is
        an optional per-token selector -- masked-out tokens recompute routing natively.
        """
        RouterReplay._require_available()
        if routed_experts is None:
            return
        tf_config = RouterReplay._tf_config(model)
        local_topk_idx = RouterReplay.get_local_topk_idx_for_current_rank(routed_experts, tf_config, packed_seq_params)
        if local_topk_idx is None:
            return
        local_mask = RouterReplay.get_local_replay_mask_for_current_rank(replay_mask, tf_config, packed_seq_params)
        # Permute to mcore's seq-major token order (row = s*bs + b) before flattening so the
        # targets align with the router's scores; the layer dim stays whole-model wide.
        layers_topk_idx_reshape = local_topk_idx.permute(1, 0, 2, 3).flatten(0, 1).transpose(0, 1).to(
            Platform.get_local_device())
        num_layers = tf_config.num_layers
        if layers_topk_idx_reshape.shape[0] != num_layers:
            raise RuntimeError(
                f'router replay expects an all-layer routing tensor with {num_layers} layer rows (the model\'s '
                f'total transformer-layer count) but got {layers_topk_idx_reshape.shape[0]}; the routing tensor '
                'and the model disagree on layer count (check the sampler / R2 RECORD boundary format).')
        # Address each router by its OWN global layer_number, scoping the walk to the forwarded model's decoder
        # (iter_model_routers skips MTP siblings and any stale process-global-registry routers).
        for layer_number, router in RouterReplay.iter_model_routers(model):
            idx = layer_number - 1
            if not 0 <= idx < num_layers:
                raise RuntimeError(
                    f'router replay got an out-of-range layer_number {layer_number} (num_layers={num_layers}); '
                    "the forwarded model's layer numbering disagrees with the routing tensor width.")
            target = layers_topk_idx_reshape[idx].to(torch.int64)
            if local_mask is None:
                # Plain mcore RouterReplay.set_target_indices takes no mask; masked replay needs
                # the MaskedRouterReplay router installed by mcore-bridge.
                router.set_target_indices(target)
            else:
                router.set_target_indices(target, local_mask)

    @staticmethod
    def set_global_router_replay_action(action: Any) -> None:
        """Set *action* on every registered mcore router instance."""
        RouterReplay._require_available()
        McoreRouterReplay.set_global_router_replay_action(action)

    @staticmethod
    def clear_global_router_replay_action() -> None:
        """Reset the action to None on every registered mcore router instance."""
        RouterReplay._require_available()
        McoreRouterReplay.clear_global_router_replay_action()

    @staticmethod
    def clear_global_indices() -> None:
        """Clear recorded / target indices on every registered mcore router instance."""
        RouterReplay._require_available()
        McoreRouterReplay.clear_global_indices()

    @staticmethod
    def prepare_replay_forward(action: Any, model: Any, routed_experts: Optional[torch.Tensor],
                               packed_seq_params: Any = None, replay_mask: Optional[torch.Tensor] = None) -> None:
        """Per-micro-batch pre-forward step: reset routers stuck in ``REPLAY_BACKWARD`` from a
        previous micro-batch back to ``REPLAY_FORWARD``, then load this batch's replay target.
        No-op for None action or a chunk with no MoE router.
        """
        if action is None:
            return
        RouterReplay._require_available()
        routers = [router for _, router in RouterReplay.iter_model_routers(model)]
        if not routers:
            return
        if routers[0].router_replay_action == RouterReplayAction.REPLAY_BACKWARD:
            for router in routers:
                router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
        if routers[0].router_replay_action == RouterReplayAction.REPLAY_FORWARD:
            RouterReplay.set_router_replay_data(routed_experts, model, packed_seq_params, replay_mask)

    @staticmethod
    def finish_replay_forward(action: Any,
                              model: Any,
                              packed_seq_params: Any = None,
                              recorded_sink: Optional[List[torch.Tensor]] = None,
                              batch_size: int = 1,
                              full_seq: Optional[int] = None) -> None:
        """Per-micro-batch post-forward step.

        ``REPLAY_FORWARD`` flips the chunk's routers to ``REPLAY_BACKWARD`` so the backward
        recompute replays the same routing; ``RECORD`` appends one entry per scheduled step
        (zeros for a zero-MoE chunk) so the driver can regroup by micro-batch under VPP.
        """
        if action is None:
            return
        RouterReplay._require_available()
        if action == RouterReplayAction.RECORD:
            if recorded_sink is not None:
                recorded_sink.append(RouterReplay.get_router_replay_data(model, batch_size, packed_seq_params, full_seq))
            return
        if action == RouterReplayAction.REPLAY_FORWARD:
            for _, router in RouterReplay.iter_model_routers(model):
                router.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)

    @staticmethod
    def _pp_sum_gather(local_dense: torch.Tensor, tf_config: Any) -> torch.Tensor:
        """All-reduce(SUM) across the PP group: each rank fills only its own stage's layer
        slots (disjoint), so the sum rebuilds the whole-model routing. No-op when PP=1."""
        from megatron.core import mpu
        if mpu.get_pipeline_model_parallel_world_size() <= 1:
            return local_dense
        gathered = local_dense.clone()
        torch.distributed.all_reduce(gathered, op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_pipeline_model_parallel_group())
        return gathered

    @staticmethod
    def assemble_recorded_routing(recorded_per_step: List[Optional[torch.Tensor]], tf_config: Any,
                                  num_microbatches: int, vpp_size: Optional[int]) -> Optional[torch.Tensor]:
        """Regroup per-(micro-batch x chunk) RECORD outputs into the whole-model routing tensor.

        Under VPP, regroup by micro-batch via mcore's ``get_schedule_table`` and SUM each
        micro-batch's chunks (disjoint slots); then concat micro-batches sharing a seq length
        (else keep the per-micro-batch list, like the variable-seq ``logps`` path) and
        all-reduce across the PP group.
        """
        if vpp_size is not None and vpp_size > 1:
            from megatron.core.pipeline_parallel.schedules import get_schedule_table
            group = getattr(tf_config, 'microbatch_group_size_per_vp_stage', None)
            schedule = get_schedule_table(num_microbatches, vpp_size, group)
            if len(recorded_per_step) != len(schedule):
                raise RuntimeError(
                    f'R2 RECORD captured {len(recorded_per_step)} forward steps but the VPP schedule expects '
                    f'{len(schedule)} (num_microbatches={num_microbatches} x vpp_size={vpp_size}); a scheduled '
                    'chunk did not append its routing (finish_replay_forward must append for every step, '
                    'including zero-MoE chunks).')
            per_mb: List[Optional[torch.Tensor]] = [None] * num_microbatches
            for step, (mb_id, _chunk_id) in enumerate(schedule):
                entry = recorded_per_step[step]
                if entry is None:
                    continue
                per_mb[mb_id] = entry if per_mb[mb_id] is None else per_mb[mb_id] + entry
            recorded = [t for t in per_mb if t is not None]
        else:
            recorded = [t for t in recorded_per_step if t is not None]
        if not recorded:
            return None
        if all(t.shape == recorded[0].shape for t in recorded):
            return RouterReplay._pp_sum_gather(torch.cat(recorded, dim=0), tf_config)
        # Variable seq length: gather each micro-batch separately (all PP ranks see the same per-mb shapes).
        return [RouterReplay._pp_sum_gather(t, tf_config) for t in recorded]

    @staticmethod
    def apply_router_replay_patch(model: Any = None) -> None:
        """Patch ``MoEAlltoAllTokenDispatcher.preprocess`` so all-to-all split sizes survive
        replayed routing (duplicate indices make ``routing_map.sum() < num_tokens * topk``).
        Idempotent and class-global; *model* is signature parity with the transformers side.
        """
        RouterReplay._require_available()
        if MoEAlltoAllTokenDispatcher is None or hasattr(MoEAlltoAllTokenDispatcher, '_preprocess_patched'):
            return
        logger.info('Applying Megatron MoE router replay patch...')
        original_preprocess = MoEAlltoAllTokenDispatcher.preprocess

        def patched_preprocess(self, routing_map):
            result = original_preprocess(self, routing_map)
            # With router replay, duplicate indices can reduce the actual routed token count, so derive it from
            # the routing map instead of assuming num_tokens * topk.
            if (getattr(self.config, 'moe_enable_routing_replay', False) and not self.drop_and_pad
                    and self.config.moe_expert_capacity_factor is None
                    and not (getattr(self.config, 'moe_router_padding_for_quantization', None)
                             or getattr(self.config, 'moe_router_padding_for_fp8', None))):
                self.num_out_tokens = int(routing_map.sum().item())
            return result

        MoEAlltoAllTokenDispatcher.preprocess = patched_preprocess
        MoEAlltoAllTokenDispatcher._preprocess_patched = True
