# Copyright (c) ModelScope Contributors. All rights reserved.
"""PPO value-function (critic) loss: the clipped value-regression objective.

The PPO critic rides a ``num_labels=1`` sequence-classification head forwarded with ``task='value'``,
which keeps the head's PER-TOKEN output (skipping the last-token pooling ``seq_cls`` does), so
``outputs['logits']`` is a value estimate ``V(s_t)`` at every token (shape ``[B, T]``). It is regressed
toward the per-token ``returns`` over the response tokens only (``labels != ignore_index``) with PPO's
pessimistic value clipping -- the value may not move more than ``cliprange_value`` from the estimate
taken at rollout time, and the larger of the clipped/unclipped squared errors is used, mirroring TRL's
``PPOTrainer``:

    L_V = 0.5 * vf_coef * mean_over_valid( max( (V - R)^2 , (clip(V, V_old ± cliprange_value) - R)^2 ) )

This is the critic half of PPO; the policy half reuses the shared clipped-surrogate GRPOLoss. The
per-token value output is symmetric across backends: the transformers ``TransformersValuePatch`` and
the Megatron ``forward_step`` ``task='value'`` branch both surface it in ``outputs['logits']``.
"""
from typing import TYPE_CHECKING, Optional, Union

from twinkle.data_format import LossOutput
from twinkle.loss.base import Loss

if TYPE_CHECKING:
    import torch


class PPOValueLoss(Loss):
    """Clipped value-regression loss for the PPO critic.

    Reads the per-token value ``outputs['logits']`` of shape ``[B, T]`` (the ``task='value'`` head
    output, before any pooling) and regresses it toward the per-token ``returns`` over the response
    tokens only (``labels != ignore_index``), with PPO's pessimistic value clipping:

        L_V = 0.5 * vf_coef * mean_over_valid( max( (V - R)^2 , (clip(V, V_old ± cliprange_value) - R)^2 ) )

    ``returns`` / ``old_values`` arrive response-only (one value per response token) and are scattered
    onto the response positions, exactly like the policy's advantages / old_logps.

    Args:
        cliprange_value: The value may not move more than this from ``old_values`` (the value at
            rollout time); the loss takes the larger of the clipped and unclipped squared errors.
        vf_coef: Value-loss coefficient (scales the critic loss relative to the policy loss).
        ignore_index: Label value marking non-response tokens, masked out of the per-token loss.
    """

    # Reads the per-token value head output outputs['logits'] ([B, T]); never per-token logps.
    require_logits = True
    require_logps = False

    def __init__(self, cliprange_value: float = 0.2, vf_coef: float = 0.1, ignore_index: int = -100, **kwargs):
        super().__init__()
        self.cliprange_value = cliprange_value
        self.vf_coef = vf_coef
        self.ignore_index = ignore_index

    def __call__(
        self,
        inputs,
        outputs,
        *,
        returns: Union['torch.Tensor', list] | None = None,
        old_values: Union['torch.Tensor', list] | None = None,
        **kwargs,
    ) -> LossOutput:
        """Compute the clipped value loss.

        Args:
            returns: per-token value targets, response-only (one per response token) or ``[B, T]``.
            old_values: the value estimate at rollout time (same shape as ``returns``), the clip
                anchor. Defaults to the current estimate (no clipping) when absent.
        """
        import torch

        assert returns is not None, "PPOValueLoss requires 'returns' (the value targets)."
        values = outputs['logits'] if isinstance(outputs, dict) else outputs
        assert values is not None, ("PPOValueLoss needs the per-token value in outputs['logits']; forward the "
                                    "critic with task='value'.")
        # num_labels=1 head -> squeeze the trailing width to [B, T]; promote a 1-D sample to [1, T].
        if values.dim() == 3 and values.shape[-1] == 1:
            values = values[..., 0]
        if values.dim() == 1:
            values = values.unsqueeze(0)
        return self._per_token_loss(inputs, values, returns, old_values)

    def _clipped(self, values: 'torch.Tensor', returns: 'torch.Tensor',
                 old_values: Union['torch.Tensor', list] | None) -> 'torch.Tensor':
        """Element-wise max(unclipped, clipped) squared value error (no reduction)."""
        import torch

        if old_values is None:
            vpred_clipped = values
        else:
            old = torch.as_tensor(old_values, device=values.device, dtype=values.dtype).reshape(values.shape)
            vpred_clipped = old + torch.clamp(values - old, -self.cliprange_value, self.cliprange_value)
        return torch.max((values - returns)**2, (vpred_clipped - returns)**2)

    def _per_token_loss(self, inputs, values: 'torch.Tensor', returns, old_values) -> LossOutput:
        import torch

        labels = inputs.get('labels') if isinstance(inputs, dict) else None
        assert labels is not None, "per-token PPOValueLoss needs inputs['labels'] to locate the response tokens."
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels, device=values.device)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        mask = (labels.to(values.device) != self.ignore_index)
        # returns / old_values arrive response-only (one value per response token, exactly like the
        # policy's advantages/old_logps); scatter them onto the response positions so they line up with
        # the per-token values the head emits over the WHOLE sequence.
        returns_t = self._align_to_mask(returns, mask, values.dtype)
        old_t = None if old_values is None else self._align_to_mask(old_values, mask, values.dtype)
        maskf = mask.to(values.dtype)
        sq = self._clipped(values, returns_t, old_t)
        loss = 0.5 * self.vf_coef * (sq * maskf).sum() / maskf.sum().clamp(min=1.0)
        return LossOutput(loss=loss, num_tokens=0)

    def _align_to_mask(self, data, mask: 'torch.Tensor', dtype) -> 'torch.Tensor':
        """Scatter per-sample response-only (or full-length) targets onto the response mask -> ``[B, T]``.

        Mirrors ``GRPOLoss._pad_and_align_to_batch``: a per-sample sequence whose length equals the
        number of masked positions fills them directly; a full-length (``>= T``) sequence is sliced
        then masked; a ``[B, T]`` tensor passes through unchanged.
        """
        import torch

        batch_size, seq_len = mask.shape
        if torch.is_tensor(data) and tuple(data.shape) == (batch_size, seq_len):
            return data.to(device=mask.device, dtype=dtype)
        if torch.is_tensor(data):
            data = [data[i] for i in range(batch_size)] if data.dim() == 2 else [data]
        result = torch.zeros((batch_size, seq_len), dtype=dtype, device=mask.device)
        for i in range(batch_size):
            pos = mask[i].nonzero(as_tuple=True)[0]
            seq = torch.as_tensor(data[i], dtype=dtype, device=mask.device).flatten()
            n = seq.numel()
            if n == len(pos):
                result[i, pos] = seq
            elif n >= seq_len:
                result[i, pos] = seq[:seq_len][mask[i]]
            else:
                raise AssertionError(f'value-target length {n} at sample {i} matches neither the response '
                                     f'length {len(pos)} nor the full sequence length {seq_len}.')
        return result
