# Copyright (c) ModelScope Contributors. All rights reserved.
"""Megatron MoE routing-replay utilities.

The Megatron counterpart of ``twinkle.model.transformers.moe.router_replay``. Both backends expose the
same public surface (``RouterReplayAction`` + ``set_global_router_replay_action`` /
``clear_global_router_replay_action`` / ``clear_global_indices`` / ``set_router_replay_data`` /
``get_router_replay_data`` / ``apply_router_replay_patch``) so ``MegatronModel`` can drive routing
replay symmetrically to ``TransformersModel`` -- basic principle 1 (the two training backends are
equivalent except for ``generate``).

Unlike the FSDP/transformers path, which owns its own per-block replay registry, Megatron delegates the
record/replay state to mcore's ``megatron.core.transformer.moe.router_replay.RouterReplay`` (one instance per
MoE router, kept in ``RouterReplay.global_router_replay_instances``). What lives here is the Megatron-only
glue mcore does not provide: mapping a whole-model ``routed_experts`` tensor (as reported across ALL
transformer layers, including dense ones) onto the LOCAL (pp, vp, cp, sp) layer/token slice each rank
actually routes, and the ``MoEAlltoAllTokenDispatcher`` patch that keeps the all-to-all split sizes
consistent when replayed indices contain duplicates.

Requires ``megatron-core >= 0.16`` (the version that ships ``RouterReplay``); when it is absent every
entry point raises rather than silently degrading to recomputed routing.
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch

from twinkle.utils import get_logger

logger = get_logger()

try:
    from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
    ROUTER_REPLAY_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the installed megatron-core version
    logger.warning('RouterReplay not available in current megatron-core version')
    RouterReplay = None
    RouterReplayAction = None
    MoEAlltoAllTokenDispatcher = None
    ROUTER_REPLAY_AVAILABLE = False


def _require_available() -> None:
    """Fail loudly when mcore lacks ``RouterReplay`` instead of silently recomputing routing.

    Recomputed routing makes an MoE policy's importance ratio silently wrong, so an unavailable
    ``megatron-core`` must abort rather than degrade.
    """
    if not ROUTER_REPLAY_AVAILABLE:
        raise RuntimeError('Megatron routing replay is not supported: upgrade megatron-core to 0.16.0 '
                           'or higher (mcore.transformer.moe.router_replay.RouterReplay is missing).')


def _tf_config(model: Any) -> Any:
    """Return the mcore ``TransformerConfig`` of an (unwrapped) Megatron model chunk."""
    config = getattr(model, 'config', None)
    if config is None:
        raise ValueError('Megatron routing replay needs a model exposing `.config` (the mcore '
                         f'TransformerConfig); got {type(model).__name__} without one.')
    return config


def resolve_router_replay_action(action: Any) -> Any:
    """Resolve a backend-neutral action token to this backend's ``RouterReplayAction`` enum.

    Callers (swift/dev) stay backend-agnostic per basic principle 1, so they pass the action as a plain
    string (``'record'`` / ``'replay_forward'`` / ``'replay_backward'``) rather than importing a
    backend-specific enum. ``None`` passes through (replay disabled for this forward); an already-enum
    value passes through; a string is looked up by value.
    """
    _require_available()
    if action is None or isinstance(action, RouterReplayAction):
        return action
    if isinstance(action, str):
        try:
            return RouterReplayAction(action)
        except ValueError:
            raise ValueError(f'Unknown router_replay_action {action!r}; expected one of '
                             f'{[a.value for a in RouterReplayAction]} or None.') from None
    raise TypeError(f'router_replay_action must be a str, RouterReplayAction or None, got {type(action).__name__}.')


def iter_model_routers(model: Any):
    """Yield ``(layer_number, RouterReplay)`` for every replay-enabled router owned by *model*.

    *model* is a single module or the list of virtual-pipeline chunks. The process-global
    ``RouterReplay.global_router_replay_instances`` list can hold MORE routers than this model has local
    layers -- a model that rebuilds its decoder appends a fresh set while the discarded one lingers -- so a
    positional slice of it can address the wrong objects, and MTP heads restart layer numbering (mcore gives
    them ``num_layers + mtp_layer_number`` so they never alias decoder rows). Walking the module that is
    actually forwarded scopes the routers to *this* model. Ported from verl's ``iter_model_routers``
    (``router_replay_utils.py``), retargeted to mcore's ``global_router_replay_instances`` / ``TopKRouter``.
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


# ---------------------------------------------------------------------------
# Sequence-dim (cp, sp) slicing -- shared by the routing-index and replay-mask paths
# ---------------------------------------------------------------------------


def _slice_seq_for_cp_sp(tensor: torch.Tensor, packed_seq_params: Any = None) -> torch.Tensor:
    """Slice dim=1 (sequence) of a ``[bs, seq, ...]`` tensor to this rank's local (cp, sp) shard.

    Shared by the routing-index path (``get_local_topk_idx_for_current_rank``) and the replay-mask path
    (``get_local_replay_mask_for_current_rank``) so a mask and its indices are sliced IDENTICALLY and stay
    token-for-token aligned at the router (``get_replay_topk`` blends them per row, so any mis-slice would
    silently pin the wrong tokens). Mirrors the legacy two-step slice verbatim: a context-parallel split on
    the seq dim, then the sequence-parallel scatter (which splits dim 0 of the seq-major view across the TP
    group); ``scatter_to_sequence_parallel_region`` is a no-op when SP/TP is 1, so it is applied unconditionally
    exactly as the index path always has.
    """
    from megatron.core import mpu
    from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region

    if mpu.get_context_parallel_world_size() > 1:
        from mcore_bridge import split_cp_inputs
        tensor = split_cp_inputs(tensor, getattr(packed_seq_params, 'cu_seqlens_q', None), 1)
    return scatter_to_sequence_parallel_region(tensor.transpose(0, 1)).transpose(0, 1)


def get_local_replay_mask_for_current_rank(global_replay_mask: Optional[torch.Tensor],
                                           tf_config: Any,
                                           packed_seq_params: Any = None) -> Optional[torch.Tensor]:
    """Slice a whole-model per-token ``[bs, seq]`` replay mask to this rank's local ``[local_seq*bs]`` tokens.

    The mask carries no layer dim (a token either influences the trained log-probs or not, identically across
    layers), so unlike the index path there is no pp layer filter -- only the cp/sp seq slice, which is the
    SAME ``_slice_seq_for_cp_sp`` the indices use. mcore tokens are SEQ-major (row = s*bs + b), so the sliced
    ``[bs, local_seq]`` mask is transposed before flattening to stay aligned row-for-row with the router's
    ``scores`` in ``MaskedRouterReplay.get_replay_topk`` (the transformers backend is batch-major and flattens
    directly; the ``[bs, seq]`` boundary shape is identical for both). The result has ``numel == local_seq*bs``,
    matching ``scores.shape[0]``. Returns *None* when no mask is supplied (whole-sequence replay).
    """
    if global_replay_mask is None:
        return None
    local = _slice_seq_for_cp_sp(global_replay_mask, packed_seq_params)  # [bs, local_seq]
    return local.transpose(0, 1).flatten(0, 1)  # [local_seq*bs], seq-major to match the router's scores


def get_local_topk_idx_for_current_rank(global_topk_idx: Optional[torch.Tensor],
                                        tf_config: Any,
                                        packed_seq_params: Any = None) -> Optional[torch.Tensor]:
    """Slice a whole-model ``routed_experts`` tensor's SEQ dim to this rank's local (cp, sp) shard.

    ``global_topk_idx`` is ``[bs, seq_len, num_layers, topk]`` reported across ALL transformer layers (vLLM
    emits routing for dense layers too). The LAYER dim is deliberately NOT filtered here: under PP>1 each
    rank holds a different layer range, and under VPP a rank's chunks hold NON-CONTIGUOUS global layers, so
    a contiguous local-range filter would mis-map rows onto routers. Instead the whole all-layer tensor is
    kept and ``set_router_replay_data`` addresses each local router by its own global ``layer_number``
    (``idx = layer_number - 1``), correct for any (pp, vp) layout. Only the seq dim is sliced for context
    parallel (cp) and sequence parallel (sp), via the SAME helper the replay-mask path uses so indices and
    mask stay token-for-token aligned.
    """
    if global_topk_idx is None:
        return None
    return _slice_seq_for_cp_sp(global_topk_idx, packed_seq_params)


# ---------------------------------------------------------------------------
# Record / replay data exchange (public API mirrors the transformers module)
# ---------------------------------------------------------------------------


def _all_gather_seq(tokens: torch.Tensor, group: Any) -> torch.Tensor:
    """All-gather ``tokens`` along dim 0 in group-rank order (inverse of the contiguous SP scatter).

    Plain ``all_gather`` (not ``gather_from_sequence_parallel_region``) because recorded routing indices
    are integers with no autograd, and the SP scatter ``get_local_topk_idx_for_current_rank`` applies
    splits dim 0 into contiguous per-rank chunks -- concatenating in rank order undoes it exactly.
    """
    world = torch.distributed.get_world_size(group=group) if torch.distributed.is_initialized() else 1
    if world <= 1:
        return tokens
    gathered = [torch.empty_like(tokens) for _ in range(world)]
    torch.distributed.all_gather(gathered, tokens.contiguous(), group=group)
    return torch.cat(gathered, dim=0)


def get_router_replay_data(model: Any, batch_size: int = 1, packed_seq_params: Any = None,
                           full_seq: Optional[int] = None) -> Optional[torch.Tensor]:
    """Collect ONE micro-batch's recorded routing on this (pp, vp) rank as a whole-model all-layer tensor.

    Returns ``[batch_size, full_seq, num_layers, topk]`` where ``num_layers`` is ``tf_config.num_layers``
    (the model's TOTAL decoder-layer count) and ONLY this chunk's MoE-layer slots are filled -- each router
    writes to slot ``layer_number - 1`` -- while every other slot is zero. That all-layer dense layout is the
    shared dev/sampler boundary format for both backends, and it turns the cross-chunk / cross-stage merge
    into a plain SUM: VPP chunks and PP stages fill DISJOINT layer slots, so the driver sums the per-chunk
    tensors of one micro-batch (merging VPP, see ``assemble_recorded_routing``) and all-reduces across the PP
    group (merging stages, see ``_pp_sum_gather``) to rebuild the whole-model routing on every rank. Dense
    (non-MoE) layer slots stay zero, exactly as the sampler reports them.

    A chunk with no MoE router (an all-dense VPP chunk or PP stage) contributes an all-zero tensor of the
    same shape -- built from *full_seq* (the micro-batch's pre-CP sequence length, which the caller derives
    from its own input) -- so it still lines up with mcore's schedule table and takes part in the SUM
    collectives without stalling them. Returns *None* only when there is nothing to record AND no *full_seq*
    to build zeros from (then the driver skips this step).
    """
    _require_available()
    tf_config = _tf_config(model)
    from megatron.core import mpu

    num_layers = tf_config.num_layers
    topk = getattr(tf_config, 'moe_router_topk', None)
    routers = sorted(iter_model_routers(model), key=lambda item: item[0])
    if not routers:
        # Zero-MoE chunk: contribute zeros so schedule alignment and the PP all-reduce stay intact.
        if full_seq is None or not isinstance(topk, int) or topk <= 0:
            return None
        from twinkle import Platform
        return torch.zeros((batch_size, full_seq, num_layers, topk), dtype=torch.uint8,
                           device=Platform.get_local_device())
    # A chunk's routers must have DISTINCT global layer_numbers: the dense scatter below writes slot
    # ``layer_number - 1`` by assignment, so a duplicate would silently overwrite (last wins) and corrupt
    # the record. Fail loudly instead (mirrors verl's merge_router_topk_indices duplicate check).
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
    # mcore flattens the decoder's [seq, bs, hidden] activations to tokens in SEQ-major order (row =
    # s*batch_size + b; see router.py's ``logits.view(-1, num_experts)`` over ``[seq_length, bsz]``), so
    # ``recorded_topk_idx`` rows are ``[local_seq, batch_size]``. Reshape in THAT order and permute to the
    # batch-major ``[bs, local_seq]`` boundary the sampler / transformers backend share -- a naive
    # ``reshape(batch_size, local_seq)`` would scramble tokens whenever bs>1 and seq>1.
    # [local_moe, local_seq*bs, topk] -> [bs, local_seq, local_moe, topk]
    local = torch.stack(local_layers, dim=0).reshape(num_local_moe, local_seq, batch_size,
                                                     topk).permute(2, 1, 0, 3).contiguous()
    # 1. inverse SP scatter (the router saw dim 0 of [seq, bs, ...] scattered across the TP group).
    if getattr(tf_config, 'sequence_parallel', False) and mpu.get_tensor_model_parallel_world_size() > 1:
        local = _all_gather_seq(local.transpose(0, 1).contiguous(),
                                mpu.get_tensor_model_parallel_group()).transpose(0, 1).contiguous()
    # 2. inverse CP split (the router saw the load-balanced-split seq dim, i.e. dim 1 here).
    if mpu.get_context_parallel_world_size() > 1:
        from twinkle.utils.torch_utils import gather_cp_load_balanced
        cu = getattr(packed_seq_params, 'cu_seqlens_q', None) if packed_seq_params is not None else None
        local = gather_cp_load_balanced(local, mpu.get_context_parallel_group(), seq_dim=1, cu_seqlens=cu)
    # local now: [bs, full_seq, num_local_moe, topk]
    gathered_full_seq = local.shape[1]
    # 3. scatter each local MoE layer to its GLOBAL slot (layer_number - 1). VPP chunks hold non-contiguous
    #    global layers, so address by layer_number -- never a contiguous range -- to land each row correctly.
    all_layer = local.new_zeros((batch_size, gathered_full_seq, num_layers, topk))
    for slot, (layer_number, _) in enumerate(routers):
        idx = layer_number - 1
        if not 0 <= idx < num_layers:
            raise RuntimeError(
                f'router replay RECORD got an out-of-range layer_number {layer_number} '
                f'(num_layers={num_layers}).')
        all_layer[:, :, idx, :] = local[:, :, slot, :]
    return all_layer


def set_router_replay_data(routed_experts: Optional[torch.Tensor],
                           model: Any,
                           packed_seq_params: Any = None,
                           replay_mask: Optional[torch.Tensor] = None) -> None:
    """Inject a whole-model ``routed_experts`` tensor as the replay target of this rank's local routers.

    ``routed_experts`` is ``[bs, seq_len, num_layers, topk]`` as reported across ALL transformer layers
    (vLLM emits routing for dense layers too -- see ``get_local_topk_idx_for_current_rank``). Only its seq
    dim is sliced to this (cp, sp) rank's local tokens; the layer dim stays whole-model wide so each local
    router is addressed by its OWN global ``layer_number`` (``idx = layer_number - 1``) and receives that
    row as its ``[total_seq, topk]`` target. Addressing by ``layer_number`` rather than a positional
    ``[i]``-th row is what makes PP>1 and VPP correct: a rank holds a subset of layers, and VPP chunks hold
    NON-CONTIGUOUS global layers, so only ``layer_number`` reliably indexes the whole-model layer dim.

    ``replay_mask`` is an optional whole-model per-token ``[bs, seq_len]`` selector for SELECTIVE replay:
    masked-in tokens replay their recorded routing, masked-out tokens recompute natively under the current
    weights (see ``MaskedRouterReplay.get_replay_topk``). It is sliced with the SAME cp/sp helper as the
    indices so the two align token-for-token, and the one mask is shared across every local layer (the
    replay decision is per token, not per layer). ``None`` degrades to whole-sequence replay.

    Convention note (resolved from legacy swift's two call sites): the *logps* path
    (``swift/megatron/trainers/utils.py``) pre-slices via ``get_local_topk_idx_for_current_rank`` before
    distributing, which is the convention matching the real all-layer vLLM tensor and is what this port
    follows; the legacy *training* path (``grpo_trainer.forward_step``) instead indexed a full-global-layer
    tensor with ``[i + offset]`` and did no cp/sp slice. This port keeps the pre-slice convention so the
    index is ``[i]`` (offset already applied by the router-list slice) and cp/sp are handled, and is
    therefore correct for pp/vp/cp/sp alike. Mirrors the transformers-side
    ``set_router_replay_data(routed_experts, model)`` signature plus the ``packed_seq_params`` the Megatron
    cp slice needs.
    """
    _require_available()
    if routed_experts is None:
        return
    tf_config = _tf_config(model)
    local_topk_idx = get_local_topk_idx_for_current_rank(routed_experts, tf_config, packed_seq_params)
    if local_topk_idx is None:
        return
    local_mask = get_local_replay_mask_for_current_rank(replay_mask, tf_config, packed_seq_params)
    # [bs, local_seq, num_layers, topk] (batch-major boundary) -> [num_layers, local_seq*bs, topk] in
    # mcore's SEQ-major token order (row = s*bs + b), the order the router's scores arrive in during
    # get_replay_topk -- so permute the seq dim ahead of the batch dim BEFORE flattening (a plain
    # flatten(0, 1) would feed batch-major targets to a seq-major router and pin the wrong tokens). The
    # layer dim stays WHOLE-MODEL wide so each router picks its own row by global layer_number
    # (VPP/PP-safe; see the slice helper).
    layers_topk_idx_reshape = local_topk_idx.permute(1, 0, 2, 3).flatten(0, 1).transpose(0, 1).to(
        torch.cuda.current_device() if torch.cuda.is_available() else local_topk_idx.device)
    num_layers = tf_config.num_layers
    if layers_topk_idx_reshape.shape[0] != num_layers:
        raise RuntimeError(
            f'router replay expects an all-layer routing tensor with {num_layers} layer rows (the model\'s '
            f'total transformer-layer count) but got {layers_topk_idx_reshape.shape[0]}; the routing tensor '
            'and the model disagree on layer count (check the sampler / R2 RECORD boundary format).')
    # Address each router by its OWN global layer_number, scoping the walk to the forwarded model's decoder
    # (iter_model_routers skips MTP siblings and any stale process-global-registry routers).
    for layer_number, router in iter_model_routers(model):
        idx = layer_number - 1
        if not 0 <= idx < num_layers:
            raise RuntimeError(
                f'router replay got an out-of-range layer_number {layer_number} (num_layers={num_layers}); '
                "the forwarded model's layer numbering disagrees with the routing tensor width.")
        router.set_target_indices(layers_topk_idx_reshape[idx].to(torch.int64), local_mask)


def set_global_router_replay_action(action: Any) -> None:
    """Set *action* on every registered mcore router instance."""
    _require_available()
    RouterReplay.set_global_router_replay_action(action)


def clear_global_router_replay_action() -> None:
    """Reset the action to None on every registered mcore router instance."""
    _require_available()
    RouterReplay.clear_global_router_replay_action()


def clear_global_indices() -> None:
    """Clear recorded / target indices on every registered mcore router instance."""
    _require_available()
    RouterReplay.clear_global_indices()


def prepare_replay_forward(action: Any, model: Any, routed_experts: Optional[torch.Tensor],
                           packed_seq_params: Any = None, replay_mask: Optional[torch.Tensor] = None) -> None:
    """Per-micro-batch pre-forward routing-replay step (mirrors legacy ``grpo_trainer.forward_step``).

    Under 1F1B/interleaved scheduling a previous micro-batch leaves its routers in ``REPLAY_BACKWARD``
    (set after its forward, consumed by its backward recompute), so reset them to ``REPLAY_FORWARD``
    first, then load *this* micro-batch's replay target (and its optional per-token ``replay_mask`` for
    selective replay). No-op when *action* is None (replay disabled) or for a chunk with no MoE router
    (an all-dense VPP chunk / PP stage has nothing to replay).
    """
    if action is None:
        return
    _require_available()
    routers = [router for _, router in iter_model_routers(model)]
    if not routers:
        return
    if routers[0].router_replay_action == RouterReplayAction.REPLAY_BACKWARD:
        for router in routers:
            router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    if routers[0].router_replay_action == RouterReplayAction.REPLAY_FORWARD:
        set_router_replay_data(routed_experts, model, packed_seq_params, replay_mask)


def finish_replay_forward(action: Any,
                          model: Any,
                          packed_seq_params: Any = None,
                          recorded_sink: Optional[List[torch.Tensor]] = None,
                          batch_size: int = 1,
                          full_seq: Optional[int] = None) -> None:
    """Per-micro-batch post-forward routing-replay step.

    ``REPLAY_FORWARD`` -> flip this chunk's routers to ``REPLAY_BACKWARD`` so the backward activation
    recompute replays the SAME routing (mcore pops the target that ``set_target_indices`` pushed onto
    ``replay_backward_list``). ``RECORD`` -> collect this micro-batch/chunk's recorded routing into
    *recorded_sink*, ALWAYS appending one entry per scheduled forward step (an all-zero tensor for a
    zero-MoE chunk, built from *full_seq*) so the driver can regroup by micro-batch against mcore's
    schedule table under VPP (see ``assemble_recorded_routing``). The mode is taken from *action* rather
    than the routers' state so a zero-MoE chunk -- which has no routers to inspect -- still records its
    placeholder. No-op when *action* is None.
    """
    if action is None:
        return
    _require_available()
    if action == RouterReplayAction.RECORD:
        if recorded_sink is not None:
            recorded_sink.append(get_router_replay_data(model, batch_size, packed_seq_params, full_seq))
        return
    if action == RouterReplayAction.REPLAY_FORWARD:
        for _, router in iter_model_routers(model):
            router.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)


def _pp_sum_gather(local_dense: torch.Tensor, tf_config: Any) -> torch.Tensor:
    """All-reduce(SUM) a whole-model all-layer dense routing tensor across the PP group.

    Every PP rank holds the SAME-shaped ``[bs, full_seq, num_layers, topk]`` tensor with only its own
    stage's layer slots filled (others zero), so a single SUM all-reduce reconstructs the whole-model
    routing on every rank -- simpler than verl's compact all_gather+concat, which must exchange per-stage
    layer counts and pad the layer dim for NCCL. The slots are disjoint across stages, so the sum never
    double-counts (and never overflows the uint8 payload). No-op when PP=1.
    """
    from megatron.core import mpu
    if mpu.get_pipeline_model_parallel_world_size() <= 1:
        return local_dense
    gathered = local_dense.clone()
    torch.distributed.all_reduce(gathered, op=torch.distributed.ReduceOp.SUM,
                                 group=mpu.get_pipeline_model_parallel_group())
    return gathered


def assemble_recorded_routing(recorded_per_step: List[Optional[torch.Tensor]], tf_config: Any,
                              num_microbatches: int, vpp_size: Optional[int]) -> Optional[torch.Tensor]:
    """Regroup per-(micro-batch x chunk) RECORD outputs into the whole-model routing tensor.

    ``recorded_per_step`` holds one all-layer dense ``[bs_mb, full_seq, num_layers, topk]`` tensor per
    scheduled forward step, in schedule order (a zero-MoE chunk contributes zeros; a step that recorded
    nothing contributes None). Under VPP the steps of one micro-batch are interleaved with other
    micro-batches', so regroup by micro-batch via mcore's ``get_schedule_table`` and SUM each micro-batch's
    chunk tensors (disjoint layer slots -> exact merge). Then concatenate micro-batches along the batch dim
    when they share a seq length (else keep the per-micro-batch list, exactly like the variable-seq ``logps``
    return path) and all-reduce(SUM) across the PP group to rebuild the whole-model routing on every rank.
    The twinkle-native counterpart of verl's ``reorder_and_merge_vpp_layers`` + ``pp_gather``, adapted to the
    all-layer dense boundary (SUM-merge by layer slot instead of compact concat + layer-count padding).
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
        return _pp_sum_gather(torch.cat(recorded, dim=0), tf_config)
    # Variable seq length: gather each micro-batch separately (all PP ranks see the same per-mb shapes).
    return [_pp_sum_gather(t, tf_config) for t in recorded]


def apply_router_replay_patch(model: Any = None) -> None:
    """Patch ``MoEAlltoAllTokenDispatcher.preprocess`` so all-to-all split sizes survive replayed routing.

    Idempotent and global (the patch is on the dispatcher class, so *model* is accepted only for signature
    parity with the transformers-side ``apply_router_replay_patch(model)``). With routing replay, duplicate
    indices in ``top_indices`` can make ``routing_map.sum() < num_tokens * topk``, which would otherwise
    desync the all-to-all split sizes.
    """
    _require_available()
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
