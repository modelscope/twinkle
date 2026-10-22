# Copyright (c) ModelScope Contributors. All rights reserved.
"""NCCL critical-section failure logging.

Single responsibility: inside a Megatron NCCL-critical method, log a
rank-attributed failure and then re-raise it unchanged.

This module does NOT prevent asymmetric-failure blocking -- nothing at this layer
can. A rank that swallows its exception still does not enter the collective, so the
other ranks stay blocked regardless. The time bound for an asymmetric failure comes
from Ray_Get_Timeout (the effective execution timeout applied per future), not from
this decorator. Diagnosability is the only reason this wrapper exists.

Coverage removed together with the former Layer 1 (the loss-instance wrapper) and
Layer 2 (the forward/backward decorator) silent degradation: under FSDP, the window
between ``calculate_loss``'s loss call and its surrounding bookkeeping (metric
accumulation, ``status.num_tokens``), between the three calls inside a
``forward_backward`` body, and numerical problems inside a loss (NaN, shape mismatch)
may each constitute a "forward ran, backward did not" asymmetric-failure window. That
window is no longer covered by any silent degradation; its time bound is the two
bounds documented for the task queue (record-terminal = ``queue_timeout + T``;
resource-release = ``Collect_Width * T``).
"""
import functools

from twinkle.utils.logger import get_logger

logger = get_logger()

# Errors are logged with at most this many trailing characters of traceback.
_TRACEBACK_LIMIT = 8192


def _global_rank() -> int:
    """Best-effort global rank for failure attribution; -1 if unavailable."""
    try:
        from twinkle.utils import Platform
        return Platform.get_rank()
    except Exception:
        return -1


def nccl_safe_megatron(func):
    """Log a rank-attributed failure inside the NCCL critical section, then re-raise.

    This decorator does *not* prevent asymmetric-failure blocking -- nothing at this
    layer can. A rank that swallows its exception still does not enter the collective.
    The time bound for that case comes from Ray_Get_Timeout. Diagnosability is the only
    reason this wrapper still exists. Its behavior is unconditional: no environment
    variable or config switch affects it, and it returns no degraded value.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as exc:
            import traceback
            rank = _global_rank()
            context = f'twinkle backend method={func.__name__}, global_rank={rank}'
            if hasattr(exc, 'add_note'):
                exc.add_note(context)
            elif exc.args:
                exc.args = (f'{exc.args[0]} [{context}]', *exc.args[1:])
            else:
                exc.args = (context, )
            tb = traceback.format_exc()
            if len(tb) > _TRACEBACK_LIMIT:
                tb = tb[-_TRACEBACK_LIMIT:]
            logger.error('[nccl_safe_megatron] %s in %s on global rank %s:\n%s',
                         type(exc).__name__, func.__name__, rank, tb)
            raise

    return wrapper
