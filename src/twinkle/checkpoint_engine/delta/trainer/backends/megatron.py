# Copyright (c) ModelScope Contributors. All rights reserved.
"""Megatron implementation of the delta trainer backend.

The HF coordinate mapping is DERIVED, not declared: the probe pass overwrites every
mcore parameter this rank holds with unique fp64 position codes -- global
rank/parameter/element packed into the mantissa bits, exponent set to a subnormal
range no real weight can reach -- runs one bridge export, and reads back where each
code lands in the exported HF tensors. Contiguous segments fitted from those
observations carry the mapping, so every per-model transform the bridge performs
(QKV split, gate/up split, TP gather, expert restack, padded-vocab trim) is captured
from the export's own behaviour: there is no hand-written table that could drift
from it. The mapping is per PARAMETER (not per digit-normalised pattern): a
pattern's parameters share a shape but not an HF target, so one representative's
segments cannot serve its same-shaped siblings.

Sharding axes: the steady gather runs on the trainer world group with a global
parameter/slot order, so every axis the trainer side can take -- TP, ETP, EP, PP, VPP,
CP/SP (sequence-side, never touches parameter placement) and DP (replicas keep
lockstep with empty deltas) -- is supported. The export side needs no changes: the
bridge export already gathers TP/ETP shards and broadcasts across PP/EP holders,
which is also what makes the seed and the probe observations model-complete on every
rank. Adapters and quantized parameters are rejected up front (probe codes would not
survive the PEFT/dequantize paths).
"""
from __future__ import annotations

import torch
import torch.distributed as dist

from ..spec import ShardSpec
from .base import DeltaTrainerBackend, TensorShards, TensorWeights

# fp64 bit layout of one probe code: [exponent 0x001][param 11b][rank 9b][element 32b].
# Everything except the exponent bits lives in the mantissa, so the code stays in the
# ~1e-308 subnormal band no real weight (smallest bf16 denormal ~1e-38) can enter:
# a simple range test identifies codes on the export side with zero false positives.
# The rank field is the GLOBAL rank: PP stages, expert holders and DP replicas are all
# told apart by it, which is what lets one probe pass serve every sharding axis. The
# param field is a PER-RANK ordinal over that rank's own parameters (each rank decodes
# only the codes its own rank stamped, so the numbering needs no global agreement).
_PROBE_EXPONENT = 0x001
_PROBE_BASE = _PROBE_EXPONENT << 52
_PROBE_RANK_SHIFT = 32
_PROBE_PARAM_SHIFT = 41
_PROBE_PARAM_BITS = 11
_PROBE_RANK_BITS = 9
_PROBE_ELEMENT_LIMIT = 1 << _PROBE_RANK_SHIFT


def _fit_segments(i: torch.Tensor, j: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit (starts, dsts, lens): maximal runs where BOTH i and j advance by one.

    The export's block structure (reshape + slice + concat) makes the i->j relation
    piecewise-translation, and the piece boundaries are exactly where a run breaks.
    ``starts`` carries shard element indices (sorted i has gaps between segments, so
    array positions would be wrong); ``lens`` is the run length in elements.
    """
    keep = torch.ones(i.numel(), dtype=torch.bool, device=i.device)
    keep[:-1] = (i[1:] == i[:-1] + 1) & (j[1:] == j[:-1] + 1)
    breaks = (~keep).nonzero().view(-1)
    pos_starts = torch.cat([torch.zeros(1, dtype=torch.int64, device=i.device), breaks + 1])
    ends = torch.cat([breaks, torch.full((1,), i.numel(), dtype=torch.int64, device=i.device)])
    return i[pos_starts], j[pos_starts], ends - pos_starts


class MegatronDeltaBackend(DeltaTrainerBackend):
    """Delta backend for MegatronModel (probe-derived HF coordinates).

    Works on the full trainer world group with a global parameter order, so any
    sharding of the parameter set (TP/ETP/EP/PP/VPP) and replicas (DP) keep the
    count-matrix gather in lockstep: holders emit their shard delta, everyone else
    emits the same slot sequence with zero counts.
    """

    #: Probe-fitted mapping, {param: {hf_key: (starts, dsts, lens)}} -- tensors stay
    #: on the trainer device; each entry maps a run of this rank's shard elements to a
    #: flat run of the HF tensor named by hf_key (the parameter's OWN tensors: the
    #: codes stamped on it carry its identity, so same-shaped siblings never share).
    _delta_segments: dict = None
    _delta_hf_shapes: dict = None
    #: Global tables aggregated from every rank's probe view (see _derive_delta_segments):
    #: the steady emit order over ALL trainer parameters (PP/EP/DP ranks hold subsets),
    #: the emit dtype each empty stand-in must carry, the global slot key order per
    #: parameter (an EP expert's keys are the union over expert holders), the holder
    #: set per parameter, and which parameters are fully replicated on every holder
    #: (only the smallest holder rank contributes, so the value ships exactly once).
    _delta_param_order: list = None
    _delta_param_meta: dict = None
    _delta_slot_keys: dict = None
    _delta_holders: dict = None
    _delta_replicated: dict = None

    def _lazy_wrap_model(self) -> None:
        """Transformers-only hook invoked by :meth:`DeltaTrainerMixin.prepare_delta_sync`;
        Megatron needs no lazy FSDP wrapping, so this is a deliberate no-op."""

    def validate_delta_backend(self) -> None:
        if dist.is_initialized() and dist.get_world_size() > (1 << _PROBE_RANK_BITS):
            raise ValueError(
                f'megatron delta probe codes address at most {1 << _PROBE_RANK_BITS} trainer ranks, '
                f'got world size {dist.get_world_size()}')
        adapter_config = None
        group = self._get_default_group()
        optimizer_config = self.optimizer_group.get(group) if group else None
        if optimizer_config is not None:
            adapter_config = optimizer_config.adapter_config
        if adapter_config is not None:
            raise ValueError('megatron delta sync does not support LoRA adapters')
        models = self.strategy.unwrap_model(self.model)
        for chunk in models:
            for param_name, param in chunk.named_parameters():
                if param.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                    raise ValueError(
                        f'megatron delta sync requires bf16/fp16/fp32 parameters, got {param.dtype} '
                        '(quantized weights take the dequantize path in export and their probe '
                        'codes would not survive it)')
                if param.numel() >= _PROBE_ELEMENT_LIMIT:
                    raise ValueError(
                        f'megatron delta probe codes address at most {_PROBE_ELEMENT_LIMIT} elements '
                        f'per parameter, got {param.numel()} on {param_name}')

    def get_per_tensor_param(self) -> tuple[TensorWeights, None]:
        # The bridge export IS the seed contract: full HF keys, every rank iterates
        # (collective), tensors stay on the trainer device for the wire.
        return self.get_hf_state_dict(adapter_name=''), None

    def get_per_tensor_param_shard(self) -> tuple[TensorShards, None]:
        from megatron.core import mpu

        # The world group spans every sharding axis, so its rank 0 IS the global
        # rank 0 the count-matrix gather converges on. A single-process trainer
        # never initializes the default group; the None group keeps the unsharded
        # local path (rank 0 consumes its own delta directly).
        world_group = dist.group.WORLD if dist.is_initialized() else None
        dp_rank = mpu.get_data_parallel_rank()
        global_rank = dist.get_rank() if dist.is_initialized() else 0

        def generate():
            models = self.strategy.unwrap_model(self.model)
            params = {f'chunk{ci}.{n}': p
                      for ci, chunk in enumerate(models) for n, p in chunk.named_parameters()}
            for name in self._delta_param_order:
                param = params.get(name)
                # Replicated parameters (dense params the trainer does not shard: MoE
                # routers, layernorms) carry identical values on every holder.
                if self._delta_replicated[name]:
                    # Replicated parameters carry identical values on every holder, so
                    # the smallest holder contributes -- a replicated param may live
                    # on a PP subset (e.g. a MoE router on its stage only) that the
                    # global rank 0 does not even hold.
                    contributes = global_rank == self._delta_holders[name][0]
                else:
                    # Sharded parameters contribute from the dp_rank-0 chain only:
                    # DP replicas of a shard are byte-identical.
                    contributes = param is not None and dp_rank == 0
                if param is not None and contributes:
                    # place=0 keeps the delta in shard-local coordinates: the probe
                    # segments do the shard->HF translation in _hf_delta_entry, so the
                    # generic DTensor placement machinery is bypassed on purpose.
                    spec = ShardSpec(
                        full_shape=tuple(param.data.shape), place=0,
                        gather_group=world_group, contributes=True)
                    local = param.data.reshape(-1)
                    if local.is_floating_point() and local.dtype != torch.bfloat16:
                        # Match the rollout's serving dtype: the bf16 rounding IS the
                        # sparsity source (unchanged bf16 bits need no patch).
                        local = local.to(torch.bfloat16)
                else:
                    # Non-holder (PP/EP) or replica (DP / non-rank-0 of a replicated
                    # pattern): an empty stand-in in the emit dtype keeps the
                    # count-matrix gather in lockstep and pins no snapshot memory.
                    dtype = getattr(torch, self._delta_param_meta[name])
                    local = torch.empty(0, dtype=dtype, device=self._delta_device)
                    spec = ShardSpec(full_shape=(), place=0,
                                     gather_group=world_group, contributes=False)
                yield name, local, spec

        return generate(), None

    # ------------------------------------------------------------ probe pass

    def prime_delta_snapshots(self) -> None:
        # Probe first: it overwrites parameter data in place (restored in its finally),
        # so it must complete before the snapshot pins the real values.
        self._derive_delta_segments()
        super().prime_delta_snapshots()

    def _derive_delta_segments(self) -> None:
        """Probe pass, then aggregate the global tables the steady lockstep needs.

        Probe: overwrite EVERY parameter this rank holds with fp64 codes carrying the
        GLOBAL rank and a per-rank parameter ordinal, run one bridge export, decode
        this rank's own landings into contiguous segments. The export gathers TP/ETP
        shards and broadcasts across PP/EP holders, so every rank's export already
        carries the whole model -- the observations are model-complete without any
        extra collective.

        Aggregation: PP/EP ranks hold disjoint parameter subsets and EP holders map
        into disjoint expert keys, but the steady gather aligns on a count matrix
        that every rank must fill identically. The global parameter order, the emit
        dtype of each empty stand-in, the per-parameter slot key order and the
        replicated-parameter verdict are therefore all-gathered from the ranks' own
        probe views and merged deterministically -- still an observation of the
        export, not a hand-written second source of truth.
        """
        models = self.strategy.unwrap_model(self.model)
        global_rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        world_group = dist.group.WORLD
        entries = [(f'chunk{ci}.{n}', p)
                   for ci, chunk in enumerate(models) for n, p in chunk.named_parameters()]
        param_meta: dict[str, str] = {}
        for key, param in entries:
            # The steady emit converts floating shards to bf16 (the rollout serving
            # dtype), so non-holders must build their empty stand-ins in that dtype too.
            emit_dtype = torch.bfloat16 if param.is_floating_point() else param.dtype
            param_meta[key] = str(emit_dtype).replace('torch.', '')
        if not entries:
            self._delta_segments = {}
            self._delta_param_order = []
            self._delta_param_meta = {}
            self._delta_slot_keys = {}
            self._delta_holders = {}
            self._delta_replicated = {}
            return
        if len(entries) > (1 << _PROBE_PARAM_BITS):
            raise ValueError(
                f'megatron delta probe codes address at most {1 << _PROBE_PARAM_BITS} parameters '
                f'per rank, this rank holds {len(entries)}')

        backups: list[tuple[torch.Tensor, torch.Tensor]] = []
        # The export reads parameters across TP/ETP/PP/EP groups (all-reduce src
        # selection, gathers, broadcasts), so the overwrite -- and the restore in
        # the finally below -- must be synchronized over the whole trainer world,
        # not just this rank's TP group.
        if world > 1:
            dist.barrier(group=world_group)
        for param_id, (_key, param) in enumerate(entries):
            backups.append((param, param.detach().clone()))
            codes = (torch.arange(param.numel(), dtype=torch.int64, device=param.device)
                     | (global_rank << _PROBE_RANK_SHIFT)
                     | (param_id << _PROBE_PARAM_SHIFT)
                     | _PROBE_BASE)
            param.data = codes.view(torch.float64).reshape(param.shape)
        if world > 1:
            dist.barrier(group=world_group)

        # {param: {hf_key: [i tensor, j tensor]}} -- one element may land in several
        # HF keys (tied embeddings are exported under both names), so i->key is 1..N.
        landings: dict[str, dict[str, list]] = {key: {} for key, _ in entries}
        hf_shapes: dict[str, tuple] = {}
        try:
            with torch.no_grad():
                for hf_key, tensor in self.get_hf_state_dict(adapter_name=''):
                    if tensor is None or not tensor.is_floating_point():
                        continue
                    hf_shapes[hf_key] = tuple(tensor.shape)
                    flat = tensor.reshape(-1)
                    mask = (flat > 0) & (flat < 1e-300)
                    if not mask.any():
                        continue
                    j = mask.nonzero().view(-1)
                    codes = flat[j].contiguous().view(torch.int64)
                    pid = (codes >> _PROBE_PARAM_SHIFT) & ((1 << _PROBE_PARAM_BITS) - 1)
                    rank = (codes >> _PROBE_RANK_SHIFT) & ((1 << _PROBE_RANK_BITS) - 1)
                    i = codes & (_PROBE_ELEMENT_LIMIT - 1)
                    mine = rank == global_rank
                    for param_id in pid[mine].unique().tolist():
                        key = entries[param_id][0]
                        sel = mine & (pid == param_id)
                        landings[key].setdefault(hf_key, []).append((i[sel], j[sel]))
        finally:
            for param, backup in backups:
                param.data = backup
            if world > 1:
                dist.barrier(group=world_group)

        segments: dict[str, dict] = {}
        covered: dict[str, int] = {}
        slot_keys_local: dict[str, list] = {}
        for key, _param in entries:
            table = {}
            for hf_key, pairs in landings[key].items():
                i_t = torch.cat([pair[0] for pair in pairs])
                j_t = torch.cat([pair[1] for pair in pairs])
                numel = 1
                for dim in hf_shapes[hf_key]:
                    numel *= dim
                if numel >= (1 << 31):
                    raise ValueError(
                        f'{hf_key}: {numel} elements exceeds the int32 position encoding')
                order = i_t.argsort()
                starts, dsts, lens = _fit_segments(i_t[order], j_t[order])
                table[hf_key] = (starts, dsts, lens)
            if not table:
                # This rank's export is model-complete (the bridge gathers/broadcasts
                # across TP/ETP/PP/EP): a held parameter observing zero landings means
                # its shard never reached any HF tensor -- a genuine probe failure,
                # not a sharding-axis artifact (ranks that hold none of the parameter
                # never list it here in the first place).
                raise RuntimeError(
                    f'megatron delta probe found no export mapping for parameter {key!r}; '
                    'the bridge export did not carry this rank\'s shard into any HF tensor')
            segments[key] = table
            covered[key] = int(sum(int(seg[2].sum()) for seg in table.values()))
            slot_keys_local[key] = sorted(table)
        self._delta_segments = segments
        self._delta_hf_shapes = hf_shapes

        # ---- global aggregation over the trainer world ----
        held = [key for key, _ in entries]
        if world > 1:
            views = [None] * world
            dist.all_gather_object(
                views,
                {'params': param_meta, 'slots': slot_keys_local, 'covered': covered, 'held': held},
                group=world_group)
        else:
            views = [{'params': param_meta, 'slots': slot_keys_local, 'covered': covered, 'held': held}]
        self._delta_param_order = sorted({name for v in views for name in v['params']})
        self._delta_param_meta = {name: dtype for v in views for name, dtype in v['params'].items()}
        # Each rank needs EVERY parameter's slot order to fill its own zero-count slots
        # for parameters it does not hold (PP/EP/DP ranks hold subsets).
        self._delta_slot_keys = {
            name: sorted({key for v in views for key in v['slots'].get(name, ())})
            for name in self._delta_param_order}
        self._delta_holders = {
            name: [r for r, v in enumerate(views) if name in v['held']]
            for name in self._delta_param_order}
        self._delta_replicated = {}
        for name in self._delta_param_order:
            total = 0
            for hf_key in self._delta_slot_keys[name]:
                numel = 1
                for dim in hf_shapes[hf_key]:
                    numel *= dim
                total += numel
            # max(covered) == total: some rank maps the WHOLE parameter -> the values
            # are replicated on every holder and only the smallest holder contributes.
            # The strict equality keeps the verdict conservative: a false "sharded"
            # merely duplicates contributions (harmless), a false "replicated"
            # would drop them (corrupting).
            self._delta_replicated[name] = bool(total) and (
                max(v['covered'].get(name, 0) for v in views) == total)

    # ------------------------------------------------------------ entry builder

    def _hf_delta_entry(self, name, spec, place, lidx, lval):
        table = self._delta_segments.get(name) or {}
        slot_keys = self._delta_slot_keys.get(name)
        if slot_keys is None:
            raise RuntimeError(
                f'{name}: no probe-derived delta mapping; the probe pass did not cover this parameter')
        empty_i = torch.empty(0, dtype=torch.int32, device=lval.device)
        empty_v = torch.empty(0, dtype=lval.dtype, device=lval.device)
        slots: list[tuple[str, tuple]] = []
        idx_parts: list[torch.Tensor] = []
        val_parts: list[torch.Tensor] = []
        # The GLOBAL key order (identical on every rank -- holders, non-holders and
        # replicas alike), with every slot always present: that is what keeps the
        # count-matrix gather aligned when PP/EP split the parameter set and when a
        # rank's own delta happens to fall entirely outside its segments.
        for hf_key in slot_keys:
            shape = self._delta_hf_shapes[hf_key]
            slots.append((hf_key, shape))
            seg = table.get(hf_key)
            if seg is None or not lidx.numel():
                idx_parts.append(empty_i)
                val_parts.append(empty_v)
                continue
            starts, dsts, lens = seg
            pos = torch.searchsorted(starts, lidx, right=True) - 1
            p = pos.clamp(min=0)
            # searchsorted guarantees lidx >= starts[pos] for pos >= 0, so the lower
            # bound is the pos >= 0 test itself; pos < 0 marks a shard element no
            # segment covers (e.g. a padded-vocab row trimmed out of the export).
            within = (pos >= 0) & (lidx < starts[p] + lens[p])
            if not int(within.sum()):
                idx_parts.append(empty_i)
                val_parts.append(empty_v)
                continue
            sel = within.nonzero().view(-1)
            p = pos[sel]
            gidx = dsts[p] + (lidx[sel] - starts[p])
            idx_parts.append(gidx.to(torch.int32))
            val_parts.append(lval[sel])
        counts = torch.tensor([part.numel() for part in idx_parts], dtype=torch.int64)
        gidx = torch.cat(idx_parts) if idx_parts else empty_i
        gval = torch.cat(val_parts) if val_parts else empty_v
        return slots, str(gval.dtype).replace('torch.', ''), counts, gidx, gval
