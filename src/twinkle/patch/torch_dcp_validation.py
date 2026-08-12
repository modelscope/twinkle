# Copyright (c) ModelScope Contributors. All rights reserved.
"""Skip torch DCP shard-overlap validation (too slow, redundant for mcore) during checkpoint save/load.

torch's ``validate_non_overlapping_shards_metadata`` and DCP's ``_validate_global_plan`` do
near-O(n^2) pairwise shard checks that dominate save/load time for large mcore checkpoints (many
shards from high TP/PP/DP/EP). megatron generates shards deterministically, so these checks are
redundant; they are short-circuited here. Mirrors legacy swift's
``_patch_validate_non_overlapping_shards_metadata`` (swift/megatron/init.py), including the fix from
ms-swift PR #9896.

The ``_validate_global_plan`` return contract changed across torch versions: older torch returns a
``bool`` (falsy => invalid), newer torch returns a list of error messages (empty => valid). Returning
the wrong type is not harmless -- a ``bool`` sends the newer caller into ``'; '.join(True)``, which
raises ``TypeError`` and buries the real failure. So the no-op is version-aware.

Usage (persistent, like other global patches)::

    from twinkle.patch import apply_patch
    from twinkle.patch.torch_dcp_validation import MegatronDCPValidationPatch
    apply_patch(None, MegatronDCPValidationPatch())
"""
import inspect
from twinkle.patch import Patch
from twinkle.utils import get_logger

logger = get_logger()

_MARKER = '_twinkle_dcp_validation_patched'
_ORIGIN_GLOBAL = '_twinkle_origin_validate_global_plan'
_ORIGIN_SHARDS = '_twinkle_origin_validate_shards'


def _dcp_validation_returns_errors(default_planner) -> bool:
    """Whether ``_validate_global_plan`` is expected to return a list of error messages (vs a bool)."""
    import torch
    try:
        # The caller is what defines the contract, so it is the most reliable thing to inspect.
        source = inspect.getsource(default_planner.DefaultSavePlanner._create_global_plan)
        return 'validation_errors' in source
    except (OSError, TypeError):
        pass
    annotation = inspect.signature(default_planner._validate_global_plan).return_annotation
    if annotation is inspect.Signature.empty:
        logger.warning(f'Could not determine the `_validate_global_plan` contract of torch=={torch.__version__}; '
                       'assuming the legacy boolean form.')
        return False
    return annotation not in (bool, 'bool')


class MegatronDCPValidationPatch(Patch):
    """No-op torch DCP shard-overlap / global-plan validation (too slow, redundant for mcore). Reversible."""

    def __call__(self, module=None, *args, **kwargs):
        from torch.distributed._shard.sharded_tensor import api
        from torch.distributed._shard.sharding_spec import api as api2
        from torch.distributed.checkpoint import default_planner
        if getattr(default_planner, _MARKER, False):
            return module

        setattr(api, _ORIGIN_SHARDS, api.validate_non_overlapping_shards_metadata)
        setattr(api2, _ORIGIN_SHARDS, api2.validate_non_overlapping_shards_metadata)
        setattr(default_planner, _ORIGIN_GLOBAL, default_planner._validate_global_plan)

        def validate_non_overlapping_shards_metadata(*a, **k):
            pass

        api.validate_non_overlapping_shards_metadata = validate_non_overlapping_shards_metadata
        api2.validate_non_overlapping_shards_metadata = validate_non_overlapping_shards_metadata

        # Match the installed torch's return contract (see module docstring / ms-swift PR #9896).
        if _dcp_validation_returns_errors(default_planner):

            def _validate_global_plan(*a, **k):
                return []
        else:

            def _validate_global_plan(*a, **k):
                return True

        default_planner._validate_global_plan = _validate_global_plan
        setattr(default_planner, _MARKER, True)
        logger.info('Patched torch DCP shard validation to no-op (faster mcore checkpoint save/load).')
        return module

    def unpatch(self, module=None, *args, **kwargs):
        from torch.distributed._shard.sharded_tensor import api
        from torch.distributed._shard.sharding_spec import api as api2
        from torch.distributed.checkpoint import default_planner
        if not getattr(default_planner, _MARKER, False):
            return module
        for mod in (api, api2):
            origin = getattr(mod, _ORIGIN_SHARDS, None)
            if origin is not None:
                mod.validate_non_overlapping_shards_metadata = origin
                delattr(mod, _ORIGIN_SHARDS)
        origin_global = getattr(default_planner, _ORIGIN_GLOBAL, None)
        if origin_global is not None:
            default_planner._validate_global_plan = origin_global
            delattr(default_planner, _ORIGIN_GLOBAL)
        delattr(default_planner, _MARKER)
        return module
