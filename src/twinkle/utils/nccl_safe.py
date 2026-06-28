# Copyright (c) ModelScope Contributors. All rights reserved.
"""NCCL-safe utilities for production distributed training.

Provides three layers of protection to prevent NCCL hangs:

Layer 1 - safe_loss():
    Wraps loss instances to catch computation errors and return
    graph-connected zero loss (ensures FSDP ReduceScatter can proceed).

Layer 2 - @nccl_safe decorator:
    Wraps forward_backward methods to ensure backward() always executes
    after forward() has started, even if intermediate code raises.

Layer 3 - @nccl_safe_megatron decorator:
    Wraps Megatron backend methods (forward_only, forward_backward) where
    the entire function body involves NCCL communication (sync=True).
    Catches pre-communication errors (e.g. data preprocessing failures)
    that would otherwise leave other DP ranks waiting at a collective.

Controlled by environment variable:
    TWINKLE_FAIL_FAST=1 (default, development): all protection is transparent,
        exceptions propagate normally.
    TWINKLE_FAIL_FAST=0 (production): protection activated, exceptions in
        NCCL-critical sections are caught and handled gracefully.
"""
import functools
import os

from twinkle.data_format import LossOutput
from twinkle.loss import Loss
from twinkle.utils.logger import get_logger

logger = get_logger()


def _is_fail_fast() -> bool:
    """Check if fail-fast mode is enabled (default: enabled).

    Returns True (fail-fast/development mode) unless TWINKLE_FAIL_FAST
    is explicitly set to a falsy value.
    """
    val = os.getenv('TWINKLE_FAIL_FAST', '1').upper()
    return val not in ('0', 'NO', 'FALSE', 'OFF')


# ─── Layer 1: safe_loss ────────────────────────────────────────────────────


def safe_loss(loss_instance):
    """Wrap loss instance for production graceful degradation.

    Always wraps the loss instance (idempotent). The fail-fast check is deferred
    to call time so that TWINKLE_FAIL_FAST can be set after wrapping (e.g. in
    Ray actor processes where env vars may not be inherited from the launcher).

    When TWINKLE_FAIL_FAST=1 (default, development): wrapper is transparent,
        exceptions propagate normally.
    When TWINKLE_FAIL_FAST=0 (production): wrapper catches exceptions and
        returns a graph-connected zero loss (ensures FSDP ReduceScatter proceeds).

    Idempotent: already-wrapped instances are returned as-is.
    """
    if getattr(loss_instance, '_nccl_safe_wrapped', False):
        return loss_instance
    return SafeLossWrapper(loss_instance)


class SafeLossWrapper(Loss):
    """Loss subclass that catches computation errors and returns graph-connected zero loss.

    Inherits from :class:`twinkle.loss.Loss` so ``isinstance(wrapper, Loss)``
    assertions in the training pipeline continue to pass.
    """

    def __init__(self, loss_instance):
        super().__init__()
        self._loss_instance = loss_instance
        self.require_logps = getattr(loss_instance, 'require_logps', True)
        self.require_entropy = getattr(loss_instance, 'require_entropy', False)
        self.require_logits = getattr(loss_instance, 'require_logits', False)
        self.reduction = getattr(loss_instance, 'reduction', 'mean')
        self._nccl_safe_wrapped = True

    def __call__(self, inputs, outputs, **kwargs):
        if _is_fail_fast():
            return self._loss_instance(inputs, outputs, **kwargs)
        try:
            return self._loss_instance(inputs, outputs, **kwargs)
        except Exception as e:
            import traceback
            logger.warning('[nccl_safe] Loss computation skipped due to error: '
                           '%s: %s\n%s',
                           type(e).__name__, e, traceback.format_exc())
            return _zero_loss(outputs)


def _zero_loss(outputs) -> 'LossOutput':
    """Create a graph-connected zero loss for FSDP compatibility.

    Finds a gradient-bearing tensor from outputs to maintain graph connectivity,
    ensuring backward hooks (ReduceScatter) fire.
    """
    import torch
    if isinstance(outputs, dict):
        for key in ('logps', 'logits', 'loss'):
            t = outputs.get(key)
            if t is not None and isinstance(t, torch.Tensor) and t.requires_grad:
                return LossOutput(loss=(t.flatten()[:1] * 0).sum(), num_tokens=0)
    # Fallback: standalone zero tensor (may not trigger FSDP hooks)
    device = 'cpu'
    if isinstance(outputs, dict):
        for v in outputs.values():
            if hasattr(v, 'device'):
                device = v.device
                break
    return LossOutput(loss=torch.zeros((), device=device, requires_grad=True), num_tokens=0)


# ─── Layer 2: @nccl_safe decorator ──────────────────────────────────────────


def nccl_safe(func=None, *, tinker=False):
    """Decorator ensuring backward() executes if forward() has already run.

    Detects forward completion by comparing train_status.outputs before/after
    the wrapped function call. If an exception occurs after forward has run
    but before backward completes, forces a zero-gradient backward pass to
    prevent NCCL hang (other ranks waiting for ReduceScatter).

    Args:
        func: The function to decorate (when used without arguments).
        tinker: If True, fallback returns ``[[], 0.0]`` (tinker format).
                If False, fallback returns outputs dict with ``loss=0.0``.

    Usage::

        @remote_function(dispatch='slice_dp', collect=...)
        @nccl_safe(tinker=True)
        def tinker_forward_backward(self, *, inputs, adapter_name, ...):
            # method body completely unchanged
            ...

        @remote_function(dispatch='slice_dp', collect=...)
        @nccl_safe
        def forward_backward(self, *, inputs, **kwargs):
            # method body completely unchanged
            ...
    """

    def decorator(fn):

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if _is_fail_fast():
                return fn(self, *args, **kwargs)

            # Extract adapter_name for state tracking
            adapter_name = kwargs.get('adapter_name')
            if adapter_name is None and hasattr(self, '_get_default_group'):
                adapter_name = self._get_default_group()

            og = self.optimizer_group.get(adapter_name) if adapter_name else None
            if og is None:
                # Cannot track state without optimizer group, passthrough
                return fn(self, *args, **kwargs)

            # Snapshot state before call to detect forward completion
            outputs_before = og.train_status.outputs

            try:
                return fn(self, *args, **kwargs)
            except Exception as e:
                outputs_after = og.train_status.outputs
                forward_ran = (outputs_after is not None and outputs_after is not outputs_before)

                if not forward_ran:
                    # Pre-forward failure: no NCCL ops started, safe to propagate
                    raise

                # Forward completed. Check if backward already ran.
                # TransformersModel.backward() clears loss_value to None.
                backward_done = (og.train_status.loss_value is None)

                if backward_done:
                    # Post-backward failure (e.g. output formatting)
                    # No NCCL hang risk, just return gracefully
                    logger.warning(f'[nccl_safe] Post-backward error (no NCCL risk): '
                                   f'{type(e).__name__}: {e}')
                else:
                    # CRITICAL: forward ran but backward didn't → NCCL hang risk!
                    logger.warning(f'[nccl_safe] Forcing zero backward to prevent NCCL hang: '
                                   f'{type(e).__name__}: {e}')
                    _force_zero_backward(self, og, adapter_name, kwargs)

                # Return fallback result
                if tinker:
                    return [[], 0.0]
                outputs_after['loss'] = 0.0
                return outputs_after

        return wrapper

    if func is not None:
        # @nccl_safe without arguments
        return decorator(func)
    # @nccl_safe(tinker=True) with arguments
    return decorator


def _iter_model_params(model):
    """Iterate parameters from ``model.model``, supporting single model or list of models."""
    raw_model = getattr(model, 'model', None)
    if raw_model is None:
        return iter([])
    if isinstance(raw_model, (list, tuple)):
        for m in raw_model:
            yield from m.parameters()
    else:
        yield from raw_model.parameters()


def _force_zero_backward(model, og, adapter_name, kwargs):
    """Force a zero-gradient backward pass to prevent NCCL hang.

    Creates a graph-connected zero loss tensor and calls backward(),
    ensuring FSDP ReduceScatter hooks fire on all ranks.
    """
    import torch

    outputs = og.train_status.outputs

    # Find a graph-connected tensor for zero loss
    zero_loss = None
    if outputs is not None and isinstance(outputs, dict):
        for key in ('logps', 'logits', 'loss'):
            t = outputs.get(key)
            if t is not None and isinstance(t, torch.Tensor) and t.requires_grad:
                zero_loss = (t.flatten()[:1] * 0).sum()
                break

    if zero_loss is None:
        # Fallback: use first model parameter to maintain graph connectivity.
        # Do NOT detach() the parameter -- the zero loss must remain connected
        # to the model's autograd graph so FSDP ReduceScatter hooks fire.
        # Use lazy iteration to avoid materializing the full parameter list.
        try:
            param = next((p for p in _iter_model_params(model) if p.requires_grad), None)
            if param is not None:
                zero_loss = (param.flatten()[0] * 0).sum()
            else:
                zero_loss = torch.zeros((), device='cuda', requires_grad=True)
        except Exception:
            zero_loss = torch.zeros((), device='cuda', requires_grad=True)

    og.train_status.loss_value = zero_loss

    # Call backward with minimal kwargs
    bwd_kwargs = {'adapter_name': adapter_name}
    gas = kwargs.get('gradient_accumulation_steps')
    if gas is not None:
        bwd_kwargs['gradient_accumulation_steps'] = gas
    model.backward(**bwd_kwargs)


# ─── Layer 3: @nccl_safe_megatron decorator ──────────────────────────────────


def nccl_safe_megatron(func=None, *, tinker=False, forward_only=False):
    """Decorator for Megatron backend methods where the entire body is NCCL-critical.

    Unlike @nccl_safe (which detects forward/backward boundaries), this decorator
    treats the **entire function** as a NCCL-critical section. In Megatron,
    forward_only and forward_backward both call get_forward_backward_func() which
    requires all DP ranks to enter synchronously. If one rank fails during data
    preprocessing (before entering Megatron's scheduler), other ranks will hang
    waiting for the collective.

    This decorator catches ALL exceptions (when TWINKLE_FAIL_FAST=0) and returns
    a safe fallback value, preventing NCCL hang from asymmetric failures.

    Args:
        func: The function to decorate (when used without arguments).
        tinker: If True, fallback returns ``[[], 0.0]`` (tinker format).
        forward_only: If True, fallback returns empty dict ``{}`` (forward_only format).

    Usage::

        @remote_function(dispatch='slice_dp', collect=..., sync=True)
        @nccl_safe_megatron
        def forward_backward(self, *, inputs, **kwargs):
            ...

        @remote_function(dispatch='slice_dp', collect=...)
        @nccl_safe_megatron(forward_only=True)
        def forward_only(self, *, inputs, **kwargs):
            ...

        @remote_function(dispatch='slice_dp', collect=..., sync=True)
        @nccl_safe_megatron(tinker=True)
        def tinker_forward_backward(self, *, inputs, **kwargs):
            ...
    """

    def decorator(fn):

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if _is_fail_fast():
                return fn(self, *args, **kwargs)

            try:
                return fn(self, *args, **kwargs)
            except Exception as e:
                import traceback
                logger.warning(f'[nccl_safe_megatron] Exception in Megatron method '
                               f'{fn.__name__}: {type(e).__name__}: {e}\n'
                               f'{traceback.format_exc()}')

                # Return safe fallback to prevent NCCL hang on other ranks
                if tinker:
                    return [[], 0.0]
                if forward_only:
                    return {}
                # forward_backward fallback: return dict with loss=0.0
                return {'loss': 0.0}

        return wrapper

    if func is not None:
        # @nccl_safe_megatron without arguments
        return decorator(func)
    # @nccl_safe_megatron(tinker=True) with arguments
    return decorator
