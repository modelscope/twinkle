# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import Optional


class MtpTrainingState:
    """Per-model MTP training state: pass gating, one-shot trainability check, loss draining.

    The trainability warning must fire once per model and the loss normalizer must be the
    microbatch count ``forward_backward`` itself chose -- a caller counting batches would
    divide by the wrong number.
    """

    def __init__(self):
        # One-shot guard: the trainability warning fires once per model, not per step.
        self._trainable_checked = False
        # Microbatch count of the last forward_backward; starts at 1 so an early pop
        # does not divide by 0.
        self._last_num_microbatches = 1

    @staticmethod
    def feed_enabled(enable_intent: bool, *, forward_only: bool, task: str, disable_lora: bool) -> bool:
        """Whether this pass should feed the MTP heads.

        Config declares the intent; only the call site knows whether a pass is backward'ed --
        ``forward_only`` skips backward but keeps the module in training mode with autograd on.
        """
        if not enable_intent:
            return False
        if forward_only:
            return False  # log-prob pass: no backward, the MTP forward would be pure waste
        if task != 'causal_lm':
            return False  # a pooling head has no next-token target
        if disable_lora:
            return False  # base-policy pass of a LoRA run; its gradients are discarded
        return True

    @staticmethod
    def iter_modules(unwrapped_models):
        """Yield the MTP blocks across model chunks.

        ``mtp`` sits on the language model (one level down on a multimodal model); PP ranks
        other than the last hold none, and a PeftModel forwards the lookup to its base.
        """
        for _model in unwrapped_models:
            mtp = getattr(_model, 'mtp', None)
            if mtp is None:
                language_model = getattr(_model, 'language_model', None)
                mtp = getattr(language_model, 'mtp', None) if language_model is not None else None
            if mtp is not None:
                yield mtp

    @staticmethod
    def warn_if_not_trainable(mtp_modules) -> None:
        """Warn when MTP training is asked for but no MTP parameter is trainable.

        LoRA freezes every base parameter, so MTP layers outside ``target_modules`` yield an
        mtp_loss that never moves -- harder to notice than an error.
        """
        from twinkle.utils import get_logger

        if not mtp_modules:
            return  # expected on every PP rank but the last
        if any(p.requires_grad for mtp in mtp_modules for p in mtp.parameters()):
            return
        get_logger().warning('enable_mtp_training is set, but no MTP parameter has requires_grad=True, so the '
                             'MTP loss will be computed and then discarded. With LoRA this means the MTP '
                             'layers are not covered by target_modules.')

    def training_enabled(self, model, *, forward_only: bool, task: str, disable_lora: bool) -> bool:
        """Gate this pass and, on the first gated-in pass, check MTP trainability.

        *model* is the owning ``MegatronModel`` (config intent, unwrapped chunks, strategy).
        """
        enable_intent = getattr(model.strategy.config, 'enable_mtp_training', False)
        if not self.feed_enabled(enable_intent, forward_only=forward_only, task=task, disable_lora=disable_lora):
            return False
        if not self._trainable_checked:
            self._trainable_checked = True
            self.warn_if_not_trainable(list(self.iter_modules(model.strategy.unwrap_model(model.model))))
        return True

    def note_num_microbatches(self, num_microbatches: int) -> None:
        """Record the microbatch count of the forward_backward currently running."""
        self._last_num_microbatches = num_microbatches

    def pop(self, loss_scale: Optional[float] = None) -> dict[str, float]:
        """Drain megatron's MTP loss tracker into per-depth losses plus a total.

        The tracker accumulates across microbatches and megatron never clears it, so it must be
        drained, not read. *loss_scale* defaults to ``1 / last_num_microbatches``; returns {}
        when MTP never ran, plus ``mtp_{i}_accept`` when the megatron version reports it.
        """
        from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

        tracker = MTPLossLoggingHelper.tracker
        # mcore >= 0.19 renamed 'values' to 'loss_values' and added acceptance counts.
        loss_key = 'loss_values' if 'loss_values' in tracker else 'values'
        if loss_key not in tracker:
            return {}
        if loss_scale is None:
            loss_scale = 1.0 / max(self._last_num_microbatches, 1)

        if hasattr(MTPLossLoggingHelper, 'reduce_metrics_in_tracker'):
            MTPLossLoggingHelper.reduce_metrics_in_tracker()
        else:
            self._reduce_tracker_legacy(tracker, loss_key)

        losses = (tracker[loss_key] * loss_scale).tolist()
        metrics = {f'mtp_{i + 1}_loss': float(v) for i, v in enumerate(losses)}
        metrics['mtp_loss'] = float(sum(losses))

        correct, total = tracker.get('correct_values'), tracker.get('total_values')
        if correct is not None and total is not None:
            for i, (c, t) in enumerate(zip(correct.tolist(), total.tolist(), strict=True)):
                # Acceptance decides whether speculative decoding pays off at rollout time.
                metrics[f'mtp_{i + 1}_accept'] = float(c / t) if t else 0.0

        if hasattr(MTPLossLoggingHelper, 'clean_metrics_in_tracker'):
            MTPLossLoggingHelper.clean_metrics_in_tracker()
        else:
            MTPLossLoggingHelper.clean_loss_in_tracker()
        return metrics

    @staticmethod
    def _reduce_tracker_legacy(tracker, loss_key: str) -> None:
        """Pre-0.19 reduction: sum over the reduce group first, then average over the
        replication group -- swapping the two averages partial sums."""
        import torch.distributed as dist

        values = tracker[loss_key]
        if tracker.get('reduce_group') is not None:
            dist.all_reduce(values, group=tracker['reduce_group'])
        if tracker.get('avg_group') is not None:
            dist.all_reduce(values, group=tracker['avg_group'], op=dist.ReduceOp.AVG)

