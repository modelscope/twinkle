# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import torch
from dataclasses import dataclass, field
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Any, Dict, List, Optional

from twinkle import DeviceMesh
from twinkle.data_format import InputFeature, ModelOutput
from twinkle.loss import Loss
from twinkle.metric import Metric
from twinkle.processor import InputProcessor
from twinkle.template import Template


@dataclass
class TrainStatus:
    """Status for training or evaluation.

    Encapsulates inputs, outputs, loss, tokens count and metrics for a training/eval step.
    """
    inputs: List[InputFeature] = None
    outputs: ModelOutput = None
    loss_value: Any = None
    num_tokens: int = 0
    metrics: List[Metric] = field(default_factory=list)
    forward_kwargs: Dict[str, Any] = field(default_factory=dict)
    loss_reduction: Optional[str] = None


@dataclass
class BaseOptimizerGroup:
    """Base optimizer group with common fields for training.

    Subclasses: OptimizerGroup (Transformers), MegatronOptimizerGroup (Megatron)
    """
    adapter_name: str = None
    adapter_config: Any = None
    optimizer: Optimizer = None
    lr_scheduler: LRScheduler = None
    loss_instance: Loss = None
    loss_registry: Dict[str, Loss] = field(default_factory=dict)
    train_status: TrainStatus = None
    eval_status: TrainStatus = None
    template: Template = None
    processor: InputProcessor = None
    gradient_accumulation_steps: int = 1
    cur_step: int = 0
    _dp_group: Any = None
    _device_mesh: DeviceMesh = None
    _last_grad_norm: float = 0.0
    pending_backward: bool = False
    pending_failed: bool = False
    pending_loss: float = 0.0

    def set_loss(self, loss: Loss, loss_key: Optional[str] = None) -> None:
        if loss_key is None:
            self.loss_registry.clear()
            self.loss_instance = loss
            return
        if not isinstance(loss_key, str) or not loss_key:
            raise ValueError('loss_key must be a non-empty string')
        self.loss_registry[loss_key] = loss

    def get_loss(self, loss_key: Optional[str] = None) -> Loss:
        if loss_key is not None:
            try:
                return self.loss_registry[loss_key]
            except KeyError as exc:
                raise ValueError(
                    f'Unknown loss_key={loss_key!r}; registered keys are {list(self.loss_registry)}') from exc
        if len(self.loss_registry) == 1:
            return next(iter(self.loss_registry.values()))
        if len(self.loss_registry) > 1:
            # Multiple keyed losses registered AND a keyless primary exists: the keyless getter means the
            # primary training loss (set_loss(None)'s loss_instance) -- the probe/eval callers that never
            # carry a key must keep working when CHORD registers its second, keyed loss.
            if self.loss_instance is not None:
                return self.loss_instance
            raise ValueError(f'loss_key is required when multiple losses are registered: {list(self.loss_registry)}')
        return self.loss_instance

    def prepare_backward_transaction(self, finalize_gradients: bool, loss_weight: float) -> bool:
        """Validate and open/complete a two-RPC backward transaction.

        Returns whether this call completes an already pending backward.
        """
        if not math.isfinite(loss_weight) or loss_weight < 0:
            raise ValueError(f'loss_weight must be finite and non-negative, got {loss_weight}')
        if self.pending_failed:
            raise RuntimeError(
                'The deferred backward transaction was interrupted; its gradients cannot be reused safely. '
                'Reinitialize the model/optimizer state before continuing.')
        completing_pending = self.pending_backward
        if not finalize_gradients:
            if completing_pending:
                raise RuntimeError('A deferred backward is already pending; finalize it before starting another one.')
            self.pending_backward = True
            self.pending_failed = True
        elif completing_pending:
            self.pending_failed = True
        return completing_pending

    def record_transaction_loss(self, local_loss: float, finalize_gradients: bool) -> float:
        if not finalize_gradients:
            self.pending_loss = float(local_loss)
            self.pending_failed = False
            return float(local_loss)
        total_loss = self.pending_loss + float(local_loss) if self.pending_backward else float(local_loss)
        self.pending_backward = False
        self.pending_failed = False
        self.pending_loss = 0.0
        return total_loss

    def require_finalized(self, operation: str) -> None:
        if self.pending_failed:
            raise RuntimeError(
                f'Cannot {operation}: the deferred backward transaction was interrupted and its gradients '
                'cannot be reused safely. Reinitialize the model/optimizer state before continuing.')
        if self.pending_backward:
            raise RuntimeError(
                f'Cannot {operation} while a deferred backward is pending; call forward_backward with '
                'finalize_gradients=True first.')

    def do_grad_sync(self, gradient_accumulation_steps: Optional[int] = None) -> bool:
        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = self.gradient_accumulation_steps
        else:
            self.gradient_accumulation_steps = gradient_accumulation_steps
        return gradient_accumulation_steps == 1 or ((self.cur_step - 1) % gradient_accumulation_steps == 0
                                                    and self.cur_step > 1)

    def _get_lr(self):
        """Get learning rates from optimizer. Override in subclass."""
        return []

    def accumulate_metrics(self, is_training, include_default: bool = True, metric_step: Optional[int] = None):
        """Accumulate metrics for train/eval status. Override in subclass if needed."""
        status = self.train_status if is_training else self.eval_status
        if metric_step is None:
            metric_step = self.cur_step - 1
        if len(status.metrics) > 0 and status.inputs is not None and status.outputs is not None:
            # forward_kwargs is the raw kwargs the forward was called with. A caller that drives
            # gradient accumulation threads gradient_accumulation_steps through it (the Megatron
            # forward_backward pops it for do_grad_sync, but the dict stored on train_status still
            # carries it). It must not be forwarded to the metric verbatim: this method passes the
            # authoritative self.gradient_accumulation_steps explicitly, so a second copy inside
            # **forward_kwargs raises "got multiple values for keyword argument". Pop it here -- the
            # TransformersOptimizerGroup override does the same -- so both backends accumulate metrics
            # identically (basic principle: backend equivalence, no per-backend special casing).
            forward_kwargs = dict(status.forward_kwargs)
            forward_kwargs.pop('gradient_accumulation_steps', None)
            loss_key = forward_kwargs.pop('_metric_loss_key', None)
            for metric in status.metrics:
                metric_loss_key = getattr(metric, '_twinkle_loss_key', None)
                if metric_loss_key != loss_key or (metric_loss_key is None and not include_default):
                    continue
                metric.accumulate(
                    status.inputs,
                    status.outputs,
                    lr=self._get_lr(),
                    step=metric_step,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    grad_norm=self._last_grad_norm,
                    loss_reduction=status.loss_reduction or getattr(self.get_loss(loss_key), 'reduction', 'mean'),
                    **forward_kwargs)

    def calculate_metrics(self, is_training):
        """Calculate and return metrics."""
        self.accumulate_metrics(is_training)
        status = self.train_status if is_training else self.eval_status
        # The metrics below gather over the DP group, and that gather is the first
        # use of its NCCL communicator: NCCL connects it with its own cudaMalloc,
        # which draws on device memory torch's caching allocator has *not* taken, so
        # it fails on whichever rank has the least left -- and it fails where nothing
        # reports it. Measured on 8xH20 with one padded trajectory per micro batch:
        # 16 mini batches of 7k-19k tokens left the allocator holding 87.8 GiB
        # reserved against 29.0 GiB live, one rank down to 164 MiB free, and that
        # rank raised inside all_gather_object while the other seven waited in it
        # forever -- 54 minutes, no log line, GPUs at 0% with their memory held,
        # because calculate_metric is collected 'last_pp_first' so the driver never
        # fetches the failing rank's exception. Releasing the cache first puts every
        # rank above 54 GiB free and the same step completes in 5 ms. The cost is one
        # re-allocation per optimizer step, which is once per iteration here.
        if status.metrics and torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        results = {}
        for metric in status.metrics:
            results.update(metric.calculate())
        status.inputs = None
        status.outputs = None
        return results
