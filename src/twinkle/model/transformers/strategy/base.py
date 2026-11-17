# Copyright (c) ModelScope Contributors. All rights reserved.
"""The contract and shared defaults of the transformers-side training strategies.

A strategy is what ``TrainableModel`` drives a backend's parallelism machinery
through. This base class owns the three layers of that contract:

1. **Required, engine-specific** (``@abstractmethod``): wrapping, state-dict
   collection, optimizer checkpoints -- every strategy implements these against
   its own engine, and a concrete strategy missing one of them fails at
   INSTANTIATION with the missing-method list, not as a bare ``AttributeError``
   deep inside the model layer.
2. **Required, with a universal default**: methods every strategy can answer
   with a trivial default (``init_kwargs`` -> no extra kwargs, EP hooks ->
   pass-through, ``pretrained_load_context`` -> nullcontext) live here once
   instead of being re-declared per strategy.
3. **Optional capabilities**: what the model layer probes before use and what a
   strategy without it must refuse loudly (``offload_to_cpu`` /
   ``reload_to_gpu``, required by colocation).

``MegatronStrategy`` (model/megatron/strategy/megatron.py) does NOT inherit this
base and does not implement most of this face -- and does not need to: its model
is a chunk list whose state-dict collection and optimizer persistence go through
the bridge backend and mcore's dist-checkpoint (MegatronModel never calls
``get_full_state_dict`` / ``save_optimizer_checkpoint`` / ``pretrained_load_context``
on its strategy). What the two families DO share is the smaller contact surface
``MegatronModel`` actually calls (wrap/unwrap, offload/reload, reduce_loss,
the mcore-specific hooks). A second megatron-side strategy, if one ever appears,
would take ``MegatronStrategy`` itself as the reference face.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
import os
from typing import Any, Dict, Optional

# Env flags transformers/accelerate read to pick their FSDP-aware loading path (rank0 load,
# meta-device placement, no full CPU materialization per rank).
_FSDP_EFFICIENT_LOADING_ENV = {
    'ACCELERATE_USE_FSDP': 'true',
    'FSDP_CPU_RAM_EFFICIENT_LOADING': 'true',
}


@contextmanager
def fsdp_pretrained_load_context(enabled: bool):
    """Enable the env flags required for transformers FSDP-aware loading when needed.

    A load-scoped transaction: the flags are restored (or popped) even when loading raises,
    so later imports in the same process never mistake themselves for FSDP-aware loaders.
    """
    if not enabled:
        yield
        return

    saved_env = {key: os.environ.get(key) for key in _FSDP_EFFICIENT_LOADING_ENV}
    os.environ.update(_FSDP_EFFICIENT_LOADING_ENV)
    try:
        yield
    finally:
        for key, old_val in saved_env.items():
            if old_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_val


class TransformerStrategyBase(ABC):
    """Required face + shared defaults for accelerate/deepspeed/native_fsdp strategies."""

    # ── layer 1: required, engine-specific ─────────────────────────────────

    @abstractmethod
    def wrap_model(self, model, *args):
        """Attach the parallelism machinery (FSDP/ZeRO/...), returning the wrapped model."""
        ...

    @abstractmethod
    def unwrap_model(self, model):
        """Peel the parallelism wrappers off, returning the underlying module."""
        ...

    @abstractmethod
    def get_full_state_dict(self, model) -> Dict[str, Any]:
        """Collect the full state dict (parameters and persistent buffers)."""
        ...

    @abstractmethod
    def load_full_state_dict(self, model, state_dict) -> None:
        """Load a full (non-sharded) state dict into the model in-place."""
        ...

    def _probe_load_precision(self, model, state_dict, *, after):
        """Report whether a full-state-dict load preserves the checkpoint's precision.

        Gated by ``TWINKLE_LOAD_PRECISION_PROBE`` (a path prefix). A checkpoint saved from a
        float32 master holds values that are almost never representable in bf16, so the fraction
        of loaded elements that ARE bf16-representable is a direct read-out of a downcast: near
        zero means full precision survived, near one means the master was rounded to bf16. The
        destination dtype is reported alongside because a bf16 destination at load time followed
        by a later promotion to float32 leaves exactly that signature -- float32 parameters
        carrying bf16-rounded values -- which no post-hoc inspection of the parameters can
        distinguish from a faithful load of genuinely bf16 weights.
        """
        prefix = os.environ.get('TWINKLE_LOAD_PRECISION_PROBE')
        if not prefix:
            return
        import json

        # torch is not a module-level dependency of this abstract base; os already is, and
        # re-importing it here would make it function-local and shadow the module binding that
        # the guard above just used.
        import torch

        # Sample the parameters the load actually writes, not the first ones the model happens to
        # expose. An adapter checkpoint carries only the LoRA keys, so sampling in
        # ``named_parameters()`` order measures the frozen base -- bfloat16 by construction, hence
        # trivially 100% bfloat16-representable -- and reports a clean load over a truncated one.
        targets = set(state_dict.keys())
        candidates = [(name, param) for name, param in model.named_parameters() if name in targets]
        if not candidates:
            candidates = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        if not candidates:
            candidates = list(model.named_parameters())
        limit = int(os.environ.get('TWINKLE_LOAD_PRECISION_PROBE_PARAMS', 8))
        # Spread the sample evenly over the candidate list, both ends included, rather than taking it
        # from the head. Sampling the first few is what let a load that only wrote part of the targeted
        # parameters read as clean: the head is exactly where a truncated write still succeeds. A fixed
        # stride does not fix that either -- unless it divides the list it stops short of the last
        # parameter, and the limit is reached before the appended tail is ever visited.
        count = len(candidates)
        if count <= max(limit, 1):
            order = list(range(count))
        elif limit <= 1:
            order = [0]
        else:
            order = [round(i * (count - 1) / (limit - 1)) for i in range(limit)]
        sampled = 0
        representable = 0
        total = 0
        sampled_names = []
        for index in order:
            if sampled >= limit:
                break
            name, param = candidates[index]
            local = param.to_local() if hasattr(param, 'to_local') else param
            if local.numel() == 0 or not local.is_floating_point():
                continue
            sampled += 1
            sampled_names.append(name)
            flat = local.detach().float().flatten()
            total += flat.numel()
            representable += int((flat.to(torch.bfloat16).to(torch.float32) == flat).sum().item())
        record = {
            'rank': int(os.environ.get('RANK') or os.environ.get('LOCAL_RANK') or 0),
            'phase': 'after' if after else 'before',
            # The destination dtypes that matter are those of the parameters being written; over the
            # whole model they mix the frozen base in and hide which side the load landed on.
            'dest_dtypes': sorted({str(p.dtype) for _, p in candidates}),
            'src_dtypes': sorted({str(v.dtype) for v in state_dict.values() if torch.is_tensor(v)}),
            'targeted': len(candidates),
            'params_sampled': sampled,
            # A ratio without the names it was taken over cannot be compared between two runs: the
            # same number from two different parameter subsets says nothing. The stride makes the
            # subset reproducible, and naming it makes that checkable rather than assumed.
            'sampled_params': sampled_names,
            'elements': total,
            'bf16_representable_ratio': (representable / total) if total else None,
        }
        with open(f'{prefix}.rank{record["rank"]}.jsonl', 'a') as handle:
            handle.write(json.dumps(record) + '\n')

    @abstractmethod
    def get_adapter_state_dict(self, model, adapter_name: str) -> Dict[str, Any]:
        """Collect LoRA parameters, trained modules_to_save copies and their buffers."""
        ...

    @abstractmethod
    def save_optimizer_checkpoint(self, model, optimizer, output_path: str, *,
                                  param_name_mapping: Optional[Dict[str, str]] = None) -> None:
        """Persist optimizer state under ``output_path``."""
        ...

    @abstractmethod
    def load_optimizer_checkpoint(self, model, optimizer, input_path: str, *,
                                  param_name_mapping: Optional[Dict[str, str]] = None) -> None:
        """Restore optimizer state from ``input_path``."""
        ...

    # ── layer 2: required, with a universal default ────────────────────────

    def init_kwargs(self) -> Dict[str, Any]:
        """Extra kwargs the model construction call needs for this strategy.

        Strategies without engine-specific load kwargs return an empty dict.
        """
        return {}

    def capture_pre_ep_state_if_needed(self, model, *, enable_ep: bool) -> None:
        """Snapshot pre-expert-parallel weights; only needed where EP rewrites modules."""
        return

    def prepare_adapter_config(self, config_or_dir, *, enable_ep: bool):
        """Adjust a PEFT config for expert parallelism; pass-through unless EP-aware."""
        return config_or_dir

    def pretrained_load_context(self):
        """Context manager active while ``from_pretrained`` loads weights."""
        return nullcontext()

    # ── layer 3: optional capabilities ─────────────────────────────────────

    def offload_to_cpu(self, model, optimizer=None, *, offload_model: bool = True) -> None:
        """Weight offload for colocation; strategies without it must refuse loudly.

        ``offload_model=False`` asks to leave parameters resident and move only ``optimizer``
        when one is passed. Strategies that cannot move either piece still refuse.
        """
        raise NotImplementedError(
            f'{type(self).__name__} does not support weight offload, which the colocation '
            'flow requires. Use a strategy with offload support (accelerate / native FSDP / deepspeed / megatron).')

    def reload_to_gpu(self, model, optimizer=None) -> None:
        """Bring back whatever :meth:`offload_to_cpu` moved; a no-op when nothing was moved."""
        from twinkle.model.offload import offload_record
        if not any(offload_record(self)):
            return
        raise NotImplementedError(
            f'{type(self).__name__} does not support weight offload, which the colocation '
            'flow requires. Use a strategy with offload support (accelerate / native FSDP / deepspeed / megatron).')

    def optimizer_checkpoint_exists(self, output_dir: str) -> bool:
        """Whether this strategy's optimizer checkpoint is present in ``output_dir``.

        The artifact's layout belongs to the strategy that wrote it: the default is a single
        ``optimizer.pt``, while DeepSpeed's ``engine.save_checkpoint`` writes its own directory -- a
        ``latest`` tag file plus ``global_step<N>/`` -- and never creates that filename. A caller that
        hardcodes ``optimizer.pt`` therefore reads every DeepSpeed checkpoint as incomplete and refuses
        a resume that would have worked.
        """
        import os

        return os.path.exists(os.path.join(output_dir, 'optimizer.pt'))

    def owns_update_step(self) -> bool:
        """Whether this strategy, rather than the training loop, drives the parameter update.

        Accelerate's DeepSpeed integration hands back a ``DeepSpeedOptimizerWrapper`` whose ``step``
        and ``zero_grad`` are ``pass``: the contract is that the caller drives the engine
        (``engine.backward`` then ``engine.step``), which performs the gradient-accumulation
        accounting, the clipping, the ZeRO optimizer step and the zeroing. A loop that instead calls
        ``loss.backward()`` and ``optimizer.step()`` therefore trains without ever updating a weight,
        and the engine's own step counter never advances -- so the checkpoint it writes is tagged
        ``global_step0`` whatever the run reached. Strategies that own the update say so here, and
        the loop then routes its backward and its gradient clipping through the hooks below.
        """
        return False

    def engine_backward(self, model, loss, *, sync_gradients: bool) -> None:
        """Run the backward for one micro-batch, applying the update at the accumulation boundary."""
        raise NotImplementedError(f'{type(self).__name__} does not own the update step.')

    def engine_grad_norm(self, model):
        """The gradient norm the engine computed while stepping, in place of clipping a second time."""
        raise NotImplementedError(f'{type(self).__name__} does not own the update step.')

    def engine_counters(self, model) -> Dict[str, int]:
        """The owning engine's own step counters, or an empty dict when no engine owns the update.

        An engine that owns the update decides at which micro-batch it applies one, so its counters --
        not the training loop's -- determine whether the first update after a resume aggregates the same
        micro-batches the source run aggregated. A resume that restores the weights and the optimizer
        state but not these counters shifts the accumulation boundary: the reported loss of that step
        still looks right while the update it produced is not.
        """
        return {}

    @staticmethod
    def _chunk_stats(flat, chunks: int = 64, width: int = 4096) -> list:
        """Per-chunk ``[sum, absmax, sum_of_squares]`` over a strided sample of a flat partition.

        The partitions are billions of elements, so digesting them whole each step would cost more than
        the training itself. ``chunks`` samples of ``width`` elements are taken at even strides instead:
        cheap, and -- unlike a prefix window -- able to see a difference anywhere in the partition.

        Every reduction here is exact and therefore independent of the schedule it runs under. That is a
        hard requirement, not a preference: ``torch.sum`` reduces in parallel past a few thousand
        elements, so the addition order follows thread scheduling and identical data yields a different
        last digit between two runs. A digest that drifts on identical input reports differences that are
        not there, which is worse than no digest at all -- measured on a ZeRO-2 resume that restores
        exactly, a per-chunk ``torch.sum`` differed by two float64 ULP on a 5e3-magnitude group whose
        elements cancel, while its absmax and norm matched. ``math.fsum`` is correctly rounded and
        ``max`` is exact, so neither depends on order; converting with ``tolist`` first is what makes
        that available, and float32/bf16 to float64 is lossless so nothing is given up by doing it.
        """
        import math

        numel = flat.numel()
        if numel == 0:
            return []
        stride = max(numel // chunks, width)
        if stride == width:
            # Every element is reachable, so tile exactly. Anchoring a final window at ``numel - width``
            # here would overlap the last strided chunk and count that overlap twice, which corrupts the
            # sum and the norm -- the two quantities a ``covered=True`` digest exists to report exactly,
            # since that mode is what sizes and locates a difference rather than merely detecting one.
            starts = list(range(0, numel, width))
        else:
            starts = list(range(0, numel - width + 1, stride)) or [0]
            # The strided walk stops at the last start that still fits a whole chunk, so an even stride
            # leaves the partition's tail unsampled -- anchored here, because a difference only past the
            # sampled region is exactly what this digest exists to catch. The overlap this introduces is
            # acceptable in a sample, which ``covered=False`` already declares non-authoritative.
            if numel >= width and (numel - width) % stride and starts[-1] != numel - width:
                starts.append(numel - width)
        out = []
        for start in starts:
            values = flat[start:start + width].tolist()
            out.append([math.fsum(values), max(abs(v) for v in values), math.fsum(v * v for v in values)])
        return out

    @staticmethod
    def _aggregate(stats: list) -> list:
        """``[sum, absmax, norm]`` combined from per-chunk stats in a fixed order."""
        import math

        if not stats:
            return [0.0, 0.0, 0.0]
        return [
            math.fsum(s[0] for s in stats),
            max(s[1] for s in stats),
            math.sqrt(math.fsum(s[2] for s in stats)),
        ]

    @staticmethod
    def _bits_fingerprint(flat, block: int = 1 << 24) -> list:
        """Two order-independent exact integers that change if any element's bits change.

        Delegates to :func:`twinkle.utils.torch_utils.bits_fingerprint`, which the weight-sync
        engines use for the same purpose: an exact per-element comparison that does not depend on
        the reduction schedule. See there for why integer accumulation is used instead of the exact
        float reductions, which cost minutes on a billion-element partition.
        """
        from twinkle.utils.torch_utils import bits_fingerprint
        return bits_fingerprint(flat, block=block)

    @staticmethod
    def _full_coverage_chunks(numel: int, width: int = 4096, element_budget: int = 1 << 26) -> int:
        """The chunk count that makes a strided sample cover every element of a ``numel``-long partition.

        ``stride = max(numel // chunks, width)``, so coverage is complete once the stride falls to
        ``width`` -- i.e. once ``chunks >= numel / width``. Sampling is what makes the digest cheap, and
        cheap is what lets it run every step; but a strided sample can only prove a difference exists,
        never that one is absent. Measured on a ZeRO-2 offload resume: a 503,552-element group is
        covered exhaustively by the default count and read identical, while a 4,538,761,984-element
        group is 0.37% sampled -- and the post-update difference appeared only in the sparsely sampled
        one, where "identical" was never actually established. Full coverage costs minutes on a
        partition that size, so it is gated to the single step under investigation rather than run
        every step.

        Bounded by an element budget. ``_chunk_stats`` converts each chunk to a Python list and reduces
        it with ``math.fsum`` so the result cannot depend on thread scheduling, which makes the cost of
        full coverage grow with the partition rather than with the chunk count: a 4,538,761,984-element
        partition asks for ~1.1M chunks, and one such digest measured ~14 minutes on the fsdp2 tier,
        turning a 15-step pair into a seven-hour run. The budget bounds that cost. Callers must then
        report coverage from what the chunk count actually buys (``_coverage_complete``) rather than from
        having asked for full coverage -- a truncated digest that claims ``covered=True`` is the same
        false green as a strided sample claiming to have seen everything.
        """
        needed = max(1, -(-numel // width))
        return min(needed, max(1, element_budget // width))

    @staticmethod
    def _coverage_complete(numel: int, chunks: int, width: int = 4096) -> bool:
        """Whether ``chunks`` strided windows of ``width`` reach every element of a ``numel``-long partition.

        ``_chunk_stats`` uses ``stride = max(numel // chunks, width)``, so the sample is exhaustive
        exactly when the stride has fallen to ``width``, i.e. when ``chunks * width >= numel``. This is
        the honest form of the ``covered`` flag: it reads what was obtained, which the element budget in
        ``_full_coverage_chunks`` can make less than what was requested.
        """
        return chunks * width >= numel

    def engine_master_digest(self, model, *, stepped: bool = False, step: Optional[int] = None) -> Dict[str, Any]:
        """Digest of the float32 master this strategy's optimizer actually updates.

        Offload tiers keep the trained precision in a flat CPU partition that the model's parameters
        are only a bfloat16 view of, so digesting the parameters cannot show whether an update landed:
        they read the same values before and after the step. Sampling is spread across the whole
        partition rather than taken from its head, because a prefix sample is exactly what let a
        partition that differed past its first elements report as identical.

        ``agg`` sizes and locates a difference; ``bits`` decides whether two runs are identical at all.
        A strategy that reports neither leaves its tier without any identity-capable field, and every
        conclusion drawn from the loop's plain parameter digest on that tier is unqualified -- that
        digest reads whatever buffer the parallelism wrapper happens to expose, which depends on how the
        model was constructed rather than on the trained values.

        ``stepped`` tells an engine-owning strategy whether its engine has already run this step's
        update, which is what selects the counter its full-coverage gate compares against. ``step`` is
        the same number supplied by the caller, for a strategy whose update happens in the loop and
        which therefore has no counter of its own to read.
        """
        return {}

    def engine_optim_moment_digest(self, model, optimizer=None, *, step: Optional[int] = None) -> Dict[str, Any]:
        """Digest of the Adam moments the next update will read, or an empty dict.

        The last input to the update that is otherwise unmeasured: with the master, the gradient, lr,
        betas, eps, weight decay and the Adam step counter all reading identically across two runs, the
        post-update master can still differ, and nothing else is left to distinguish "the restore lost
        the moments" from "the update itself is not reproducible".

        Where the moments live depends on who owns the update. A strategy whose engine owns it keeps
        them in an inner optimizer keyed by the engine's own partition tensor, and reads them itself
        around the step. A strategy whose loop owns the update holds them in the loop's optimizer, keyed
        by the model parameters, so the caller passes that optimizer in.

        ``step`` is the step the caller is digesting, for a strategy with no engine counter to read.
        """
        return {}

    def engine_grad_partition_digest(self, model) -> Dict[str, Any]:
        """Digest of the float32 gradient partitions the optimizer consumes, or an empty dict.

        Distinct from the master: on an offload tier the CPU optimizer reads a gradient held alongside
        its own float32 partition, not the ``param.grad`` the model exposes -- which is already consumed
        by the time a post-step probe runs, and reads as absent. The engine's global grad norm is a
        single aggregate over the reduction, so two runs can report the same norm while the partitions
        the update is actually computed from differ.
        """
        return {}

    def engine_grad_partition_layout(self, model) -> Dict[str, Any]:
        """Where each parameter sits inside the engine's flat gradient partition, or an empty dict.

        A digest localizes a difference to a range of a flat partition; the layout turns that range into
        parameter names, which is what points at a code path. Only strategies that own a flat partition
        have one to report.
        """
        return {}

    def take_engine_step_probe(self) -> Dict[str, Any]:
        """The digests bracketing the update this strategy last performed, consumed once.

        A strategy that owns the update step runs it inside its own backward, before the loop reaches
        ``clip_grad_and_step``; a digest taken from the loop would then see an already-updated master on
        both sides and report no change over a step that did change it. Those strategies record around
        the update and hand the record over here.
        """
        return {}
