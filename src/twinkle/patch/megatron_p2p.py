# Copyright (c) ModelScope Contributors. All rights reserved.
"""Force megatron's batched pipeline P2P onto the global (WORLD) process group.

Megatron's ``_batched_p2p_ops`` passes *global* peer ranks (obtained via
``dist.get_global_rank``) together with the pipeline sub-group to
``torch.distributed.P2POp`` / ``batch_isend_irecv``. Whether that peer rank is
interpreted as a global or a group-local rank when a sub-group is supplied has
varied across torch/NCCL versions, so on some stacks the pipeline send/recv
target the wrong rank and pipeline parallelism (notably PP>=4, multi-node)
deadlocks.

Forcing ``group=None`` routes the ops through the already-warmed-up WORLD group,
where a global rank equals the group-local rank, so the interpretation is
unambiguous on every version -- same peers, no semantic change. Mirrors legacy
swift's ``_patch__batched_p2p_ops`` (swift/megatron/init.py, PR #4381).

megatron-core >= 0.20 does not need this: its p2p paths handle the global-rank
semantics correctly on their own, and routing every P2P through WORLD shares
one NCCL communicator with the DP grad all-reduces that overlap the backward
pass -- under VPP the interleaved schedule's batched shape exchange (which
mcore only synchronizes when batch_p2p_comm is set, i.e. NOT when
overlap_p2p_comm is on) then loses the race and reads unfilled shape buffers
(all-zero => "[0, 0, 0]" gradients). So the patch self-disables on >= 0.20 and
kept only for the old stacks it was written for.

Usage (persistent, like other global patches)::

    from twinkle.patch import apply_patch
    from twinkle.patch.megatron_p2p import MegatronBatchedP2PGroupPatch
    apply_patch(None, MegatronBatchedP2PGroupPatch())
"""
from twinkle.patch import Patch
from twinkle.utils import get_logger

logger = get_logger()

_MARKER = '_twinkle_origin_batched_p2p_ops'

# megatron-core resolved the rank-interpretation ambiguity itself at 0.20.
_MIN_MCORE_VERSION_WITH_NATIVE_FIX = (0, 20)


class MegatronBatchedP2PGroupPatch(Patch):
    """Force ``_batched_p2p_ops`` onto the WORLD group to avoid subgroup-P2P PP hangs. Reversible."""

    def __call__(self, module=None, *args, **kwargs):
        from megatron.core.pipeline_parallel import p2p_communication
        if hasattr(p2p_communication, _MARKER):
            return module
        if _mcore_has_native_fix():
            # See the module docstring: on >= 0.20 the workaround does more harm (shared
            # WORLD communicator vs overlapping DP all-reduces) than good (none -- the
            # ambiguity it patched is gone).
            logger.info('Skipping megatron _batched_p2p_ops WORLD-group patch: megatron-core >= 0.20 '
                        'handles the rank semantics natively.')
            return module

        origin = p2p_communication._batched_p2p_ops

        def _batched_p2p_ops(**kw):
            # Peers are global ranks; WORLD makes global == group-local, so the target is unchanged.
            kw['group'] = None
            return origin(**kw)

        setattr(p2p_communication, _MARKER, origin)
        p2p_communication._batched_p2p_ops = _batched_p2p_ops
        logger.info('Patched megatron _batched_p2p_ops to use the WORLD group (pipeline-parallel hang workaround).')
        return module

    def unpatch(self, module=None, *args, **kwargs):
        from megatron.core.pipeline_parallel import p2p_communication
        origin = getattr(p2p_communication, _MARKER, None)
        if origin is not None:
            p2p_communication._batched_p2p_ops = origin
            delattr(p2p_communication, _MARKER)
        return module


def _mcore_has_native_fix() -> bool:
    """Whether the installed megatron-core resolves the P2P rank ambiguity itself."""
    from megatron.core.package_info import __version__

    try:
        parts = tuple(int(p) for p in __version__.split('.')[:2] if p.isdigit())
        return parts >= _MIN_MCORE_VERSION_WITH_NATIVE_FIX
    except (ValueError, TypeError):
        # Unparsable version (dev builds): keep the historical behavior.
        return False
