# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Union

from twinkle.data_format import LossOutput
from twinkle.loss.base import Loss
from twinkle.utils.torch_utils import selective_log_softmax

if TYPE_CHECKING:
    import torch


class GRPOLoss(Loss):
    """
    GRPO (Group Relative Policy Optimization) Loss.

    Args:
        epsilon: Clipping epsilon for PPO objective (lower bound)
        epsilon_high: Clipping epsilon for high importance sampling ratio (upper bound)
        beta: KL penalty coefficient (0.0 = no KL penalty)
        entropy_coef: Entropy bonus coefficient (0.0 = disabled). When > 0, the loss
            subtracts ``entropy_coef * H(pi)`` per token to encourage exploration and
            prevent mode-collapse / repetition. Requires the model forward to supply
            ``outputs['entropies']`` — enabled automatically via ``require_entropy``.
        ignore_index: Index to ignore in labels (default: -100)
    """

    def __init__(
        self,
        epsilon: float = 0.2,
        epsilon_high: Optional[float] = None,
        beta: float = 0.0,
        entropy_coef: float = 0.0,
        ignore_index: int = -100,
        enable_sampling_replay: bool = False,
        *,
        # -- advanced on-policy controls --------------------------------------------------------
        # Every one defaults to "off", so a GRPOLoss built with none of them is byte-for-byte the
        # plain surrogate above. They are the constructor-parameterised form of the controls the
        # legacy ``GRPOTrainer`` applied inline (and dev carried in ``_ConfiguredGRPOLoss`` /
        # ``_AdvancedGRPOLoss``): sequence-level IS, dual-clip, entropy filter/logging, truncated-IS
        # rollout correction, off-policy sequence masking, FIPO future-KL weighting, SDAR distillation
        # and CHORD's auxiliary SFT objective.
        importance_sampling_level: str = 'token',
        delta: Optional[float] = None,
        top_entropy_quantile: float = 1.0,
        log_entropy: bool = False,
        rollout_importance_sampling_mode: Optional[str] = None,
        rollout_importance_sampling_threshold: float = 2.0,
        log_rollout_offpolicy_metrics: bool = False,
        off_policy_sequence_mask_delta: Optional[float] = None,
        overlong_filter: bool = False,
        loss_type: str = 'grpo',
        fipo_decay_rate: float = 32.0,
        fipo_clip_range: Optional[float] = 0.2,
        fipo_clip_high_only: bool = True,
        fipo_safety_threshold: Optional[float] = 4.0,
        sdar_loss_coef: float = 0.0,
        sdar_gate_beta: float = 5.0,
        enable_chord: bool = False,
        **kwargs,
    ):
        self.epsilon = epsilon
        self.epsilon_high = epsilon_high if epsilon_high is not None else epsilon
        self.beta = beta
        self.entropy_coef = entropy_coef
        self.enable_sampling_replay = enable_sampling_replay
        if enable_sampling_replay and beta != 0.0:
            raise ValueError('sampling replay does not support a GRPO KL penalty (beta must be 0)')
        if enable_sampling_replay and entropy_coef != 0.0:
            raise ValueError('sampling replay does not support a GRPO entropy bonus')
        self.ignore_index = ignore_index

        self.importance_sampling_level = importance_sampling_level
        self.delta = delta
        self.top_entropy_quantile = top_entropy_quantile
        self.log_entropy = log_entropy
        self.rollout_importance_sampling_mode = rollout_importance_sampling_mode
        self.rollout_importance_sampling_threshold = rollout_importance_sampling_threshold
        self.log_rollout_offpolicy_metrics = log_rollout_offpolicy_metrics
        self.off_policy_sequence_mask_delta = off_policy_sequence_mask_delta
        self.overlong_filter = overlong_filter
        self.loss_type = loss_type
        self.fipo_gamma = 2.0**(-1.0 / fipo_decay_rate)
        self.fipo_clip_range = fipo_clip_range
        self.fipo_clip_high_only = fipo_clip_high_only
        self.fipo_safety_threshold = fipo_safety_threshold
        self.sdar_loss_coef = sdar_loss_coef
        self.sdar_gate_beta = sdar_gate_beta
        self.enable_chord = enable_chord
        # ``_advanced`` selects the configured surrogate pipeline (see ``_configured_grpo``). It covers
        # exactly the controls the base ``_reduce_loss`` path cannot express: dual-clip, entropy
        # filter/logging, overlong filtering, truncated-IS rollout correction / off-policy diagnostics,
        # off-policy sequence masking and FIPO. ``importance_sampling_level`` is deliberately NOT here --
        # the base ``_compute_log_importance_weights`` already branches on it (token/sequence/sequence_token),
        # so a plain GSPO run stays on the ±5-clamped base path exactly as before; only when one of the
        # controls below is active does the ±20-clamped configured pipeline take over (which then also
        # reads ``importance_sampling_level``).
        self._advanced = (loss_type == 'fipo' or delta is not None or top_entropy_quantile < 1.0 or log_entropy
                          or overlong_filter or rollout_importance_sampling_mode is not None
                          or log_rollout_offpolicy_metrics or off_policy_sequence_mask_delta is not None)
        # ``_distill_wrap`` marks the SDAR/CHORD auxiliary objectives, which slice the batch and read
        # ``outputs['logits']``/``outputs['logps']`` around the policy loss.
        self._distill_wrap = enable_chord or sdar_loss_coef > 0.0
        # The configured pipeline recomputes the surrogate and aggregation itself and never calls
        # ``_reduce_loss``, so pairing it with a variant whose objective LIVES in an overridden
        # ``_reduce_loss`` (REAL / DRO / DPPO / GeoMean / ClipCov / KLCov) would silently downgrade that
        # variant to plain GRPO. Refuse the combo here -- one drift-free place, and a structural check so
        # a future ``_reduce_loss`` variant is covered without editing a name list. Variants that override
        # only ``_compute_per_token_loss`` (e.g. GPG) still compose, since the pipeline calls that hook.
        if self._advanced and type(self)._reduce_loss is not GRPOLoss._reduce_loss:
            raise ValueError(
                f'{type(self).__name__} is a group-/sequence-level objective (it overrides `_reduce_loss`) '
                'that the advanced GRPO ratio/entropy/rollout controls would bypass. Drop the conflicting '
                'control (importance_sampling_level / delta / top_entropy_quantile / log_entropy / '
                'overlong_filter / rollout_importance_sampling_mode / log_rollout_offpolicy_metrics / '
                'off_policy_sequence_mask_delta / fipo) to use this loss_type.')
        # Gate the expensive forward paths: entropy for the bonus OR the advanced entropy
        # logging/filtering; logits for the SDAR/CHORD auxiliary terms.
        self.require_entropy = entropy_coef > 0.0 or log_entropy or top_entropy_quantile < 1.0
        self.require_logits = self._distill_wrap

    def micro_batch_scale(self, inputs, indices):
        return len(indices) / len(inputs)

    def _compute_log_importance_weights(
        self,
        per_token_logps: 'torch.Tensor',
        per_token_old_logps: 'torch.Tensor',
        loss_mask: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """
        Compute log importance sampling weights at ``self.importance_sampling_level``.

        Three levels share this one base-path implementation (the advanced ``_configured_grpo``
        pipeline has its own ±20-clamped ``_advanced_importance_weights`` twin):
          * ``'token'`` (default): per-token log ratio.
          * ``'sequence'``: the mask-averaged log ratio broadcast to every token (GSPO).
          * ``'sequence_token'``: GSPO-token -- grad flows through the per-token logp while the
            sequence weight rides along detached (``logp - sg[logp] + sg[seq]``).

        Args:
            per_token_logps: [batch, seq_len] current policy log probabilities
            per_token_old_logps: [batch, seq_len] old policy log probabilities
            loss_mask: [batch, seq_len] mask for valid tokens

        Returns:
            log_weights: [batch, seq_len] log importance weights
        """
        import torch
        log_ratio = per_token_logps - per_token_old_logps
        # Clamp for numerical stability (±5 bounds ratio to [exp(-5), exp(5)] ≈ [0.007, 148])
        log_ratio = torch.clamp(log_ratio, min=-5.0, max=5.0)
        if self.importance_sampling_level == 'token':
            return log_ratio
        sequence = ((log_ratio * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1.0)).unsqueeze(-1)
        if self.importance_sampling_level == 'sequence':
            return sequence
        if self.importance_sampling_level == 'sequence_token':
            return per_token_logps - per_token_logps.detach() + sequence.detach()
        raise ValueError(f'Unknown importance_sampling_level={self.importance_sampling_level!r}. '
                         "Expected 'token', 'sequence' or 'sequence_token'.")

    def _compute_per_token_loss(
        self,
        ratio: 'torch.Tensor',
        advantages: 'torch.Tensor',
        per_token_logps: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """
        Compute per-token loss with PPO clipping.

        Override this method in subclasses for different loss formulations.

        Args:
            ratio: [batch, seq_len] importance sampling ratio
            advantages: [batch, 1] or [batch, seq_len] advantage values (already expanded)
            per_token_logps: [batch, seq_len] current policy log probabilities

        Returns:
            per_token_loss: [batch, seq_len] loss for each token
        """
        import torch
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon_high)
        loss1 = ratio * advantages
        loss2 = clipped_ratio * advantages
        return -torch.min(loss1, loss2)

    def _aggregate_loss(
        self,
        per_token_loss: 'torch.Tensor',
        loss_mask: 'torch.Tensor',
        **kwargs,
    ) -> 'torch.Tensor':
        """
        Aggregate per-token loss to scalar.

        Override this method in subclasses for different normalization.
        Default: mean over sequences, then mean over batch.

        Args:
            per_token_loss: [batch, seq_len] per-token loss values
            loss_mask: [batch, seq_len] mask for valid tokens
            **kwargs: Additional arguments for subclass implementations

        Returns:
            loss: scalar loss value
        """
        # Per-sequence mean, then batch mean (aligned with Swift/TRL GRPO).
        # Each sequence contributes equally regardless of length.
        return ((per_token_loss * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1.0)).mean()

    def _loss_num_tokens(self, loss_mask: 'torch.Tensor'):
        """Token denominator reported in ``LossOutput.num_tokens``.

        0 (default) => framework uses the PER-TOKEN-MEAN accumulation path, where each
        micro/dp group is equal-weighted. Subclasses that want a strict GLOBAL token-mean
        (the SUM-loss path in transformers.py / megatron.py) return ``Σmask`` instead, so
        the accumulated gradient is divided by the global token count and the result is
        invariant to how the batch is split into micro/dp groups.
        """
        return 0

    def _pad_and_align_to_batch(
        self,
        data: 'Union[torch.Tensor, List, np.ndarray]',
        mask: 'torch.Tensor',
        device: 'torch.device',
        dtype: 'torch.dtype',
        fill_value: float = 0.0,
    ) -> 'torch.Tensor':
        """Align data to mask: scalars broadcast, sequences scatter.

        Two valid per-sample sequence forms are supported and disambiguated
        by length:
          * Response-only form (e.g. ``old_logps`` from vLLM): length equals
            the number of trainable positions in ``mask[i]`` and is scattered
            directly onto those positions.
          * Full-sequence form (e.g. ``ref_logps`` from a ref-model forward,
            right-padded to ``mask.shape[1]``): length ``>= mask.shape[1]``;
            we slice to ``seq_len`` and index by ``mask[i]`` to extract the
            trainable positions, then scatter.

        Any other length is a real bug and triggers a hard assert — never
        silently truncate, since that misaligns IS ratios to the wrong tokens.
        """
        import torch

        batch_size, seq_len = mask.shape

        # Convert to tensor if possible
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if isinstance(data, torch.Tensor):
            data = data.to(device=device, dtype=dtype)
            if data.shape == (batch_size, seq_len):
                return data  # Already aligned
            if data.dim() == 1:
                data = data.unsqueeze(1)
            if data.shape[1] == 1:
                assert data.shape[0] == batch_size, (
                    f'scalar broadcast expects data.shape[0]==batch_size, '
                    f'got data.shape={tuple(data.shape)} mask.shape={(batch_size, seq_len)}')
                fill = torch.full((batch_size, seq_len), fill_value, dtype=dtype, device=device)
                expanded = data.expand(batch_size, seq_len)
                return torch.where(mask, expanded, fill)
            data = [data[i] for i in range(batch_size)]  # To list

        # Handle list (scalars or sequences)
        if isinstance(data, (list, tuple)):
            if all(isinstance(x, (int, float)) for x in data):  # Flat scalars
                return self._pad_and_align_to_batch(
                    torch.tensor(data, dtype=dtype, device=device), mask, device, dtype, fill_value)
            data = [torch.as_tensor(s, dtype=dtype, device=device) for s in data]

        # Scatter sequences
        result = torch.full((batch_size, seq_len), fill_value, dtype=dtype, device=device)
        for i, sample in enumerate(data):
            sample = sample.flatten()
            pos = mask[i].nonzero(as_tuple=True)[0]
            n_pos = len(pos)
            n_sample = sample.numel()

            if n_sample == 1:
                result[i, pos] = sample.item()
            elif n_sample == n_pos:
                # Response-only form (e.g. old_logps from vLLM).
                result[i, pos] = sample
            elif n_sample >= seq_len:
                # Full-sequence form (e.g. ref_logps right-padded with ignore-value).
                result[i, pos] = sample[:seq_len][mask[i]]
            else:
                raise AssertionError(f'data/mask length mismatch at sample {i}: '
                                     f'n_pos={n_pos}, n_sample={n_sample}, seq_len={seq_len} '
                                     '(expected n_sample == n_pos for response-only form, '
                                     'or n_sample >= seq_len for full-sequence form)')

        return result

    def _resolve_loss_mask(self, inputs: Dict, labels: 'torch.Tensor') -> 'torch.Tensor':
        """Positions this loss may score: trainable *and* log-prob-bearing.

        ``labels`` alone answers "should this token be scored", which is all SFT
        needs. A policy-gradient loss also needs a sampling log-prob per token to
        form an importance ratio, and a turn produced outside the sampled policy
        (an API, a human, a replayed demonstration) has none. Such turns carry
        ``completion_mask == 0``: excluded here, yet still trainable for SFT.

        A feature without ``completion_mask`` predates the field, and there every
        trainable token was the policy's own, so the mask degenerates to
        ``labels != ignore_index`` and old trajectories train exactly as before.
        """
        import torch
        trainable = (labels != self.ignore_index).bool()
        completion_mask = inputs.get('completion_mask')
        if completion_mask is None:
            return trainable
        if not torch.is_tensor(completion_mask):
            completion_mask = torch.as_tensor(completion_mask)
        completion_mask = completion_mask.to(trainable.device)
        if completion_mask.dim() == 1:
            completion_mask = completion_mask.unsqueeze(0)
        if completion_mask.shape != trainable.shape:
            raise ValueError(f'completion_mask shape {tuple(completion_mask.shape)} does not match labels shape '
                             f'{tuple(trainable.shape)}. A misaligned mask would apply importance ratios to '
                             'the wrong tokens, so it is refused rather than broadcast.')
        loss_mask = trainable & completion_mask.bool()
        if self.enable_sampling_replay and not bool((loss_mask == trainable).all()):
            raise ValueError('sampling replay does not support turns generated outside the sampled policy: '
                             'they are trainable but have no sampling mask to replay against.')
        return loss_mask

    def _base_grpo(
        self,
        inputs: Dict,
        outputs: Dict,
        *,
        old_logps: Optional[Union['torch.Tensor', List[List[float]]]] = None,
        ref_logps: Optional['torch.Tensor'] = None,
        advantages: Optional[Union['torch.Tensor', List[float], np.ndarray]] = None,
        **kwargs,
    ):
        """
        Compute GRPO loss.

        Args:
            inputs: Dict containing 'labels' [batch, seq_len].
                In packing mode, the processor has already unpacked
                logps and labels to [num_sequences, max_seq_len].
            outputs: Dict containing either:
                - 'logps': [batch, seq_len] pre-computed log probs, OR
                - 'logits': [batch, seq_len, vocab] from which logps will be computed
            old_logps: [batch, seq_len] or List[List[float]] log probs from old/sampling policy.
                      Can have ragged per-sample lengths — will be padded and aligned
                      automatically.  If None, uses current logps (on-policy, ratio=1).
            ref_logps: Optional [batch, seq_len] reference model log probs for KL penalty.
                      Same padding/alignment rules as old_logps.
            advantages: advantage values
            **kwargs: Additional arguments
        """
        import torch
        if self.enable_sampling_replay:
            if old_logps is None:
                raise ValueError('old_logps are required when sampling replay is enabled')
        labels = inputs.get('labels')
        assert labels is not None, "inputs must contain 'labels'"
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        logps = outputs.get('logps')
        if self.enable_sampling_replay and logps is None:
            raise RuntimeError('sampling replay logps must be computed by the model forward')
        loss_mask = self._resolve_loss_mask(inputs, labels)
        if logps is None:
            logits = outputs.get('logits')
            if logits.shape[1] != labels.shape[1]:
                # some mllm return logits with image tokens, exclude here
                logits = logits[:, -labels.shape[1]:]
            masked_labels = labels.clone()
            masked_labels[~loss_mask] = 0
            logps = selective_log_softmax(logits, masked_labels)

        device = logps.device

        # ── Prepare old_logps ────────────────────────────────────────────
        # old_logps may be ragged (List[List[float]]) containing only
        # response-token log-probs, whereas logps covers the full padded
        # sequence.  _pad_and_align_logps scatters them into the correct
        # positions using loss_mask.
        if old_logps is None:
            old_logps = logps.detach()
        else:
            old_logps = self._pad_and_align_to_batch(
                old_logps,
                loss_mask,
                device,
                logps.dtype,
            )

        # ── Prepare ref_logps (same treatment) ──────────────────────────
        if ref_logps is not None:
            ref_logps = self._pad_and_align_to_batch(
                ref_logps,
                loss_mask,
                device,
                logps.dtype,
            )

        # GRPO loss is ill-defined without advantages (e.g. ref-logps-only forward,
        # or eval/validation forwards). Return a zero loss that still flows through
        # autograd so DDP/FSDP do not see unused params, and callers can harvest
        # outputs['logps'] freely.
        if advantages is None:
            zero = logps.sum() * 0.0
            return LossOutput(loss=zero, num_tokens=0)

        advantages = self._pad_and_align_to_batch(
            advantages,
            loss_mask,
            device,
            logps.dtype,
        )

        # ── Compute loss ────────────────────────────────────────────────
        log_importance_weights = self._compute_log_importance_weights(logps, old_logps, loss_mask)
        return self._reduce_loss(logps, old_logps, ref_logps, advantages, loss_mask, log_importance_weights,
                                 outputs, **kwargs)

    def _reduce_loss(
        self,
        logps: 'torch.Tensor',
        old_logps: 'torch.Tensor',
        ref_logps: Optional['torch.Tensor'],
        advantages: 'torch.Tensor',
        loss_mask: 'torch.Tensor',
        log_importance_weights: 'torch.Tensor',
        outputs: Dict,
        **kwargs,
    ) -> LossOutput:
        """Reduce the per-token importance weights to the scalar ``LossOutput``.

        This is the extension seam for objectives that reshape the surrogate at the sequence / group
        level instead of per token (e.g. ``REALLoss``). The base implementation is the standard clipped
        PPO surrogate plus the optional KL penalty and entropy bonus, aggregated by ``_aggregate_loss``
        -- byte-for-byte identical to inlining it at the tail of ``__call__``.
        """
        import torch
        ratio = torch.exp(log_importance_weights)

        per_token_loss = self._compute_per_token_loss(ratio, advantages, logps)

        if self.beta > 0.0 and ref_logps is not None:
            per_token_kl = (torch.exp(ref_logps - logps) - (ref_logps - logps) - 1)
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.entropy_coef > 0.0:
            entropies = outputs.get('entropies')
            assert entropies is not None, ('entropy_coef > 0 requires outputs[\'entropies\'] — make sure the '
                                           "loss instance's require_entropy flag was set before the forward call.")
            # entropies may come in fp32 from the kernel; cast to match logps dtype
            # so the final per_token_loss stays consistent (bf16 under amp).
            per_token_loss = per_token_loss - self.entropy_coef * entropies.to(per_token_loss.dtype)

        loss = self._aggregate_loss(per_token_loss, loss_mask, **kwargs)

        return LossOutput(loss=loss, num_tokens=self._loss_num_tokens(loss_mask))

    # ── Advanced on-policy controls ────────────────────────────────────────────
    # A faithful port of the legacy ``GRPOTrainer`` surrogate (carried in dev as ``_ConfiguredGRPOLoss``
    # / ``_AdvancedGRPOLoss``). It runs on the legacy numerics -- a ±20 stability clamp on the log
    # importance ratio and a clamped (k3) reference-KL -- which deliberately differ from the base path's
    # ±5 clamp and unclamped KL, so it is kept separate from ``_reduce_loss`` rather than merged into
    # it. Reached only when an advanced control is on (``self._advanced`` / ``self._distill_wrap``).

    def __call__(
        self,
        inputs: Dict,
        outputs: Dict,
        *,
        old_logps: Optional[Union['torch.Tensor', List[List[float]]]] = None,
        ref_logps: Optional['torch.Tensor'] = None,
        advantages: Optional[Union['torch.Tensor', List[float], np.ndarray]] = None,
        rollout_logps: Optional[Union['torch.Tensor', List[List[float]]]] = None,
        truncated: Optional[List[bool]] = None,
        chord_count: int = 0,
        chord_mu: float = 0.0,
        chord_phi: bool = False,
        teacher_logps: Optional[Union['torch.Tensor', List[List[float]]]] = None,
        **kwargs,
    ):
        """Compute the GRPO loss, applying any advanced controls this instance was built with.

        The extra keyword arguments are supplied by the RL training loop for every GRPO step (so the
        signature is stable whether or not a control is active): ``rollout_logps`` (the sampler's own
        log-probs, for truncated-IS correction and off-policy diagnostics), ``truncated`` (per-row
        overlong flags), ``chord_count``/``chord_mu``/``chord_phi`` (CHORD's auxiliary-SFT rows appended
        to the batch and their mixing weight), and ``teacher_logps`` (SDAR's privileged teacher). With
        every control off this is exactly :meth:`_base_grpo`.
        """
        chord_count = int(chord_count)
        if not (self._advanced or self._distill_wrap):
            return self._base_grpo(inputs, outputs, old_logps=old_logps, ref_logps=ref_logps,
                                   advantages=advantages, **kwargs)

        rl_inputs, rl_outputs, rl_kwargs = inputs, outputs, kwargs
        if chord_count:
            # CHORD appends ``chord_count`` auxiliary-SFT rows to the batch; the policy loss sees only
            # the leading rollout rows, the SFT term only the tail.
            labels = inputs['labels']
            rl_count = labels.shape[0] - chord_count
            if rl_count < 1:
                raise ValueError('advanced GRPO batches must contain at least one rollout sample.')
            rl_inputs = dict(inputs)
            rl_inputs['labels'] = labels[:rl_count]
            rl_outputs = {name: self._slice_batch(value, rl_count) for name, value in outputs.items()}
            rl_kwargs = {name: self._slice_batch(value, rl_count) for name, value in kwargs.items()}

        if self._advanced:
            result = self._configured_grpo(
                rl_inputs, rl_outputs, old_logps=old_logps, ref_logps=ref_logps, advantages=advantages,
                rollout_logps=rollout_logps, truncated=truncated, **rl_kwargs)
        else:
            result = self._base_grpo(
                rl_inputs, rl_outputs, old_logps=old_logps, ref_logps=ref_logps, advantages=advantages,
                **rl_kwargs)

        loss = result['loss']
        channel_loss = result.get('channel_loss')
        if self._distill_wrap or chord_count:
            loss = loss * (1.0 - float(chord_mu))
            loss = self._add_sdar(loss, rl_inputs, rl_outputs, teacher_logps)
            loss = self._add_chord(loss, outputs, inputs['labels'], chord_count, chord_mu, chord_phi)
            result = LossOutput(loss=loss, num_tokens=0)
        else:
            result = LossOutput(loss=loss, num_tokens=result['num_tokens'])
        if channel_loss is not None:
            result['channel_loss'] = channel_loss
        return result

    @staticmethod
    def _slice_batch(values, end: int):
        """The leading ``end`` rows of a batch-parallel value, tolerating non-sliceable entries.

        Per-row lists/tensors slice normally; a scalar broadcast across the batch (or a value with no
        row axis) has no leading dimension to trim, so it is returned unchanged.
        """
        if values is None:
            return None
        try:
            return values[:end]
        except (TypeError, KeyError):
            return values

    def _configured_grpo(
        self,
        inputs: Dict,
        outputs: Dict,
        *,
        old_logps=None,
        ref_logps=None,
        advantages=None,
        rollout_logps=None,
        truncated=None,
        **kwargs,
    ) -> LossOutput:
        """The configured surrogate: dual-clip / entropy / FIPO / rollout-IS / off-policy controls.

        Requires ``outputs['logps']`` (the forward supplies them under ``require_logps``) and scores the
        trainable positions given by ``labels`` alone -- matching the ported legacy path, which does not
        intersect with ``completion_mask`` the way the base :meth:`_resolve_loss_mask` does.
        """
        import torch
        labels = inputs['labels']
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        logps = outputs.get('logps')
        if logps is None:
            raise RuntimeError('The advanced GRPO controls require outputs["logps"].')
        alignment_mask = labels.ne(self.ignore_index)
        device, dtype = logps.device, logps.dtype
        old = (logps.detach() if old_logps is None else self._pad_and_align_to_batch(
            old_logps, alignment_mask, device, dtype))
        advantages = self._pad_and_align_to_batch(advantages, alignment_mask, device, dtype)
        mask = alignment_mask.clone()

        channel_loss: Dict[str, 'torch.Tensor'] = {}
        entropy_mask = None
        if self.require_entropy:
            entropies = outputs.get('entropies')
            if entropies is None:
                raise RuntimeError('Entropy logging/filtering requires outputs["entropies"].')
            if self.log_entropy:
                count = alignment_mask.sum().detach().float()
                channel_loss['entropy'] = torch.stack(
                    ((entropies * alignment_mask).sum().detach().float(), count))
            if self.top_entropy_quantile < 1.0 and alignment_mask.any():
                threshold = torch.quantile(
                    entropies[alignment_mask].float(), 1.0 - self.top_entropy_quantile)
                entropy_mask = entropies.ge(threshold)

        if truncated is not None:
            truncated_mask = torch.as_tensor(truncated, dtype=torch.bool, device=device).reshape(-1, 1)
            mask = mask & ~truncated_mask

        log_ratio, ratio = self._advanced_importance_weights(logps, old, mask)
        per_token_loss = self._dual_clip_per_token_loss(ratio, advantages, logps)
        if self.loss_type == 'fipo':
            per_token_loss = per_token_loss * self._fipo_weights(log_ratio, ratio, advantages, mask)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.beta > 0.0 and ref_logps is not None:
            ref = self._pad_and_align_to_batch(ref_logps, alignment_mask, device, dtype)
            ref_delta = torch.clamp(ref - logps, min=-20.0, max=20.0)
            per_token_kl = torch.clamp(torch.exp(ref_delta) - ref_delta - 1.0, min=-10.0, max=10.0)
            per_token_loss = per_token_loss + self.beta * per_token_kl

        rollout = None
        if rollout_logps is not None:
            rollout = self._pad_and_align_to_batch(rollout_logps, alignment_mask, device, dtype)
            rollout_log_ratio = old - rollout
            if self.log_rollout_offpolicy_metrics:
                count = mask.sum().detach().float()
                channel_loss['rollout_log_ratio'] = torch.stack(
                    ((rollout_log_ratio.abs() * mask).sum().detach().float(), count))
            if self.rollout_importance_sampling_mode is not None:
                per_token_loss = per_token_loss * self._rollout_is_weights(rollout_log_ratio, mask)
        elif self.rollout_importance_sampling_mode is not None:
            raise ValueError('rollout_importance_sampling_mode requires rollout_logps from the sampler.')

        if self.off_policy_sequence_mask_delta is not None:
            old_policy = rollout if rollout is not None else old
            sequence_delta = ((old_policy - logps) * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            sequence_advantage = (advantages * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            keep = ~((sequence_delta > self.off_policy_sequence_mask_delta) & (sequence_advantage < 0))
            mask = mask & keep.unsqueeze(-1)

        if self.loss_type == 'fipo':
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            loss = self._aggregate_loss(per_token_loss, mask, **kwargs)
        result = LossOutput(loss=loss, num_tokens=0)
        if channel_loss:
            result['channel_loss'] = channel_loss
        return result

    def _advanced_importance_weights(self, logps, old_logps, mask):
        """Token / sequence / sequence-token log importance weights, plus the raw clamped log ratio.

        The raw ``log_ratio`` is returned alongside the weights because FIPO's future-KL weighting
        consumes it directly. The ±20 clamp is the legacy stability bound for this pipeline.
        """
        import torch
        log_ratio = torch.clamp(logps - old_logps, min=-20.0, max=20.0)
        if self.importance_sampling_level == 'token':
            log_weights = log_ratio
        else:
            sequence = ((log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).unsqueeze(-1)
            if self.importance_sampling_level == 'sequence':
                log_weights = sequence
            elif self.importance_sampling_level == 'sequence_token':
                # GSPO-token: grad flows through the per-token logp, the sequence weight is a constant.
                log_weights = logps - logps.detach() + sequence.detach()
            else:
                raise ValueError(f'Unknown importance_sampling_level={self.importance_sampling_level!r}.')
        return log_ratio, torch.exp(log_weights)

    def _dual_clip_per_token_loss(self, ratio, advantages, logps):
        """PPO clip with an optional dual-clip floor ``delta`` on the raw ratio (legacy ordering)."""
        import torch
        if self.delta is None or self.loss_type not in {'grpo', 'dapo', 'fipo', 'bnpo', 'dr_grpo'}:
            return self._compute_per_token_loss(ratio, advantages, logps)
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon_high)
        dual_clipped_ratio = torch.clamp(ratio, max=self.delta)
        return -torch.min(dual_clipped_ratio * advantages, clipped_ratio * advantages)

    def _rollout_is_weights(self, log_ratio, mask):
        """Truncated importance-sampling weights correcting the policy-vs-sampler (rollout) gap."""
        import torch
        ratio = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))
        mode = self.rollout_importance_sampling_mode
        threshold = self.rollout_importance_sampling_threshold
        if mode == 'token_truncate':
            return torch.clamp(ratio, max=threshold)
        if mode == 'token_mask':
            return torch.where(ratio <= threshold, ratio, torch.zeros_like(ratio))
        sequence = torch.exp((torch.log(ratio.clamp(min=1e-10)) * mask).sum(-1)
                             / mask.sum(-1).clamp(min=1.0))
        if mode == 'sequence_truncate':
            return torch.clamp(sequence, max=threshold).unsqueeze(-1).expand_as(ratio)
        if mode == 'sequence_mask':
            return ratio * (sequence <= threshold).unsqueeze(-1)
        return ratio

    def _fipo_weights(self, log_ratio, ratio, advantages, mask):
        """FIPO's detached future-KL influence weight (block-matmul of the decayed future log-ratio)."""
        import torch
        future_delta = log_ratio.masked_fill(~mask, 0.0)
        if self.delta is not None:
            # Dual-clip participation: high-ratio tokens do not contribute to the future KL.
            future_delta = torch.where(ratio > self.delta, torch.zeros_like(future_delta), future_delta)
        seq_len = future_delta.shape[1]
        positions = torch.arange(seq_len, device=log_ratio.device).unsqueeze(1)
        future_kl = torch.zeros_like(future_delta)
        for start in range(0, seq_len, 128):
            end = min(seq_len, start + 128)
            block_positions = torch.arange(start, end, device=log_ratio.device).unsqueeze(0)
            distance = block_positions - positions
            decay = torch.pow(
                torch.as_tensor(self.fipo_gamma, dtype=log_ratio.dtype, device=log_ratio.device),
                distance.clamp(min=0))
            decay = decay * (distance >= 0).to(log_ratio.dtype)
            future_kl += torch.matmul(future_delta[:, start:end], decay.t())
        weights = torch.exp(future_kl.masked_fill(~mask, 0.0))
        if self.fipo_clip_range:
            lower = 1.0 if self.fipo_clip_high_only else 1.0 - self.fipo_clip_range
            weights = torch.clamp(weights, min=lower, max=1.0 + self.fipo_clip_range)
        if self.fipo_safety_threshold is not None:
            unsafe = (advantages < 0) & (ratio > self.fipo_safety_threshold)
            weights = torch.where(unsafe, torch.clamp(weights, min=0.8, max=1.0), weights)
        return weights.detach()

    def _add_sdar(self, loss, rl_inputs, rl_outputs, teacher_logps):
        """Add SDAR's confidence-gated teacher-distillation term on the rollout rows."""
        if teacher_logps is None or self.sdar_loss_coef <= 0:
            return loss
        from twinkle.loss.sdar import compute_sdar_loss
        student_logps = rl_outputs['logps']
        response_mask = rl_inputs['labels'].ne(self.ignore_index)
        teacher = self._pad_and_align_to_batch(
            teacher_logps, response_mask, student_logps.device, student_logps.dtype)
        sdar_loss, _ = compute_sdar_loss(student_logps, teacher, response_mask, self.sdar_gate_beta)
        return loss + self.sdar_loss_coef * sdar_loss

    def _add_chord(self, loss, outputs, labels, chord_count, chord_mu, chord_phi):
        """Add CHORD's auxiliary SFT cross-entropy over the trailing ``chord_count`` rows."""
        if not chord_count:
            return loss
        import torch
        import torch.nn.functional as F
        rl_count = labels.shape[0] - chord_count
        chord_logits = outputs['logits'][rl_count:]
        chord_labels = labels[rl_count:]
        token_loss = F.cross_entropy(
            chord_logits.reshape(-1, chord_logits.shape[-1]),
            chord_labels.reshape(-1),
            ignore_index=self.ignore_index,
            reduction='none')
        valid = chord_labels.reshape(-1).ne(self.ignore_index)
        if chord_phi:
            # CHORD's phi-function reweights each token by p(1-p), peaking at medium confidence.
            probability = torch.exp(-token_loss.detach())
            token_loss = token_loss * probability * (1.0 - probability)
        chord_loss = token_loss[valid].mean() if valid.any() else token_loss.sum() * 0.0
        return loss + float(chord_mu) * chord_loss


class PPOLoss(GRPOLoss):
    """PPO clipped policy loss.

    PPO and GRPO share the same clipped policy objective. The algorithms differ
    in how advantages are produced, so this class provides the PPO-facing name
    while using PPO-specific loss aggregation.

    Args:
        loss_agg_mode: ``'token-mean'`` averages over every valid response
            token. ``'seq-mean-token-mean'`` averages tokens within each
            sequence and then averages the sequences.
    """

    _LOSS_AGG_MODES = {'token-mean', 'seq-mean-token-mean'}

    def __init__(self, loss_agg_mode: Literal['token-mean', 'seq-mean-token-mean'] = 'token-mean', **kwargs):
        super().__init__(**kwargs)
        if loss_agg_mode not in self._LOSS_AGG_MODES:
            raise ValueError(f'Unsupported PPO loss_agg_mode: {loss_agg_mode}. '
                             f'Expected one of {sorted(self._LOSS_AGG_MODES)}.')
        self.loss_agg_mode = loss_agg_mode

    def _aggregate_loss(
        self,
        per_token_loss: 'torch.Tensor',
        loss_mask: 'torch.Tensor',
        **kwargs,
    ) -> 'torch.Tensor':
        if self.loss_agg_mode == 'token-mean':
            return (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
        return super()._aggregate_loss(per_token_loss, loss_mask, **kwargs)


class GSPOLoss(GRPOLoss):
    """GRPO with sequence-level importance sampling (GSPO).

    A thin alias for ``GRPOLoss(importance_sampling_level='sequence')``: the base
    ``_compute_log_importance_weights`` already branches on the level, so GSPO only pins the default.
    Kept as a distinct class because twinkle's loss registry resolves the ``'gspo'`` name to it.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('importance_sampling_level', 'sequence')
        super().__init__(*args, **kwargs)


class SAPOLoss(GRPOLoss):
    """
    SAPO (Soft-gated Advantage Policy Optimization) Loss.

    Uses soft gating instead of hard clipping.
    """

    def __init__(
        self,
        epsilon: float = 0.2,
        beta: float = 0.0,
        tau_pos: float = 1.0,
        tau_neg: float = 1.0,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(epsilon=epsilon, beta=beta, ignore_index=ignore_index, **kwargs)
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg

    def _compute_per_token_loss(
        self,
        ratio: 'torch.Tensor',
        advantages: 'torch.Tensor',
        per_token_logps: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Soft-gated loss."""
        import torch
        gate_pos = torch.sigmoid(self.tau_pos * (ratio - 1)) * (4.0 / self.tau_pos)
        gate_neg = torch.sigmoid(self.tau_neg * (ratio - 1)) * (4.0 / self.tau_neg)
        is_positive = advantages > 0
        soft_gate = torch.where(is_positive, gate_pos, gate_neg)
        return -soft_gate * advantages


class CISPOLoss(GRPOLoss):
    """
    CISPO (Clipped Importance Sampling Policy Optimization) Loss.

    Clamps the IS weight and uses policy gradient.
    """

    def micro_batch_scale(self, inputs, indices):
        token_counts = []
        for model_input in inputs:
            labels = model_input['labels']
            if hasattr(labels, 'ne'):
                token_counts.append(int(labels.ne(self.ignore_index).sum().item()))
            else:
                token_counts.append(sum(int(token != self.ignore_index) for token in labels))
        total_tokens = sum(token_counts)
        if total_tokens == 0:
            return 0.0
        return sum(token_counts[index] for index in indices) / total_tokens

    def _compute_per_token_loss(
        self,
        ratio: 'torch.Tensor',
        advantages: 'torch.Tensor',
        per_token_logps: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Clamped ratio * advantage * log_prob."""
        import torch

        clamped_ratios = torch.clamp(ratio, max=1 + self.epsilon).detach()
        return -clamped_ratios * advantages * per_token_logps

    def _aggregate_loss(
        self,
        per_token_loss: 'torch.Tensor',
        loss_mask: 'torch.Tensor',
        **kwargs,
    ) -> 'torch.Tensor':
        """Sum over all tokens, divide by total token count."""
        # Use provided num_items_in_batch if available, otherwise use mask sum
        num_items = kwargs.get('num_items_in_batch', loss_mask.sum())
        return (per_token_loss * loss_mask).sum() / num_items


class BNPOLoss(GRPOLoss):
    """
    BNPO (Batch-Normalized Policy Optimization) Loss.

    Normalizes by total completion tokens across batch.

    ``token_mean_scope``:
      'micro' (default, matches verl/SEAM): per-(micro÷dp)-group token-mean,
        equal-weighted across groups (``num_tokens=0`` => PER-TOKEN-MEAN accumulation).
        This is what verl actually does -- see verl/workers/actor/dp_actor.py: pg_loss =
        agg_loss(..., 'token-mean') is ``masked_mean`` computed WITHIN each micro-batch,
        then ``loss = policy_loss * (1/gradient_accumulation)`` before ``backward()``.
        So verl's effective gradient is the equal-weighted mean of per-micro token-means,
        NOT a global token-mean.
      'global': return the UN-normalized token sum and report ``num_tokens=Σmask``, so the
        framework's SUM-loss path divides the accumulated gradient by the GLOBAL token
        count => strict token-mean, invariant to micro/dp splitting.

    Why 'global' is NOT the default, despite being the "textbook" token-mean
    (measured on skill2lora E13, 2026-08-01):
      Group-relative advantages cancel exactly per group (mean A = 0), but the
      TOKEN-weighted mean does not: it equals -cov(len, A)/mean(len). With
      corr(len, A) = -0.42 (long skill-gen responses hit the 8192 budget, lose their
      closing tag, and score 0), 'global' yields a per-token pg_loss of +0.031 versus
      verl/SEAM's +3.2e-4 -- a ~100x coherent "emit fewer tokens" gradient. Under
      'global', E13 collapsed its <think> from 3977 to 1942 tokens in 25 updates
      (SEAM: -17% in 76 updates) and overshot the optimum: corr(len, correct) flipped
      from -0.42 to +0.23 while reward fell 0.816 -> 0.734. 'micro' localizes the
      normalization, so the length coupling largely cancels (it degenerates to
      sequence-mean as the micro size approaches 1).
    """

    def __init__(self, *args, token_mean_scope: str = 'micro', **kwargs):
        super().__init__(*args, **kwargs)
        assert token_mean_scope in ('global', 'micro'), \
            f'token_mean_scope must be global|micro, got {token_mean_scope!r}'
        self.token_mean_scope = token_mean_scope
        # 'global' 返回的是 token 和（梯度在下游按 num_tokens=Σmask 归一）。必须同步告诉展示层
        # 这是 sum-reduction，否则 LossMetric（metric/loss.py）不会除以 num_tokens，会把每个 micro 的
        # token 和当均值直接平均，展示出一个被 token 数放大的巨大 loss（梯度不受影响，纯展示失真）。
        self.reduction = 'sum' if token_mean_scope == 'global' else 'mean'

    def micro_batch_scale(self, inputs, indices):
        """The weight one micro-batch carries, which has to follow ``token_mean_scope``.

        'global' already returns the token SUM and reports ``num_tokens=Σmask``, so the
        global division happens downstream; scaling here as well would divide twice. Same
        contract as ``CrossEntropyLoss(reduction='sum')``.

        'micro' *is* the equal-weighted mean of per-micro token-means, so the inherited
        sample fraction is its weight. A token fraction would make the micro losses sum to
        the global token-mean and erase the distinction this scope exists to make -- which
        is the +3.2e-4 vs +0.031 per-token pg_loss measured above.
        """
        if self.token_mean_scope == 'global':
            return 1.0
        return super().micro_batch_scale(inputs, indices)

    def _aggregate_loss(
        self,
        per_token_loss: 'torch.Tensor',
        loss_mask: 'torch.Tensor',
        **kwargs,
    ) -> 'torch.Tensor':
        """global: return the token SUM (the global division is done downstream via
        num_tokens=Σmask). micro (legacy): local token-mean, later equal-weighted across
        micro/dp groups."""
        summed = (per_token_loss * loss_mask).sum()
        if self.token_mean_scope == 'global':
            return summed
        return summed / loss_mask.sum().clamp(min=1.0)

    def _loss_num_tokens(self, loss_mask: 'torch.Tensor'):
        if self.token_mean_scope == 'global':
            return loss_mask.sum().clamp(min=1.0)
        return 0


class DRGRPOLoss(GRPOLoss):
    """
    DR-GRPO (Dynamic Ratio GRPO) Loss.

    Normalizes by batch_size * max_completion_length for consistent gradients.
    """

    def __init__(
        self,
        epsilon: float = 0.2,
        beta: float = 0.0,
        max_completion_length: int = 1024,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(epsilon=epsilon, beta=beta, ignore_index=ignore_index, **kwargs)
        self.max_completion_length = max_completion_length

    def _aggregate_loss(
        self,
        per_token_loss: 'torch.Tensor',
        loss_mask: 'torch.Tensor',
        **kwargs,
    ) -> 'torch.Tensor':
        """Normalize by batch_size * max_completion_length."""
        batch_size = loss_mask.shape[0]
        return (per_token_loss * loss_mask).sum() / (batch_size * self.max_completion_length)


class REALLoss(GRPOLoss):
    """REAL: a group-level soft-constraint objective (https://arxiv.org/abs/2602.05630).

    Instead of the clipped per-token surrogate, REAL reduces each sequence to a scalar score (its mean
    token log importance ratio), splits every generation group into positive- and negative-advantage
    members, and applies a symmetric log-sum-exp soft constraint that pushes positive scores up and
    negative scores down against a zero baseline, temperature-scaled by ``real_tau``. A group that is
    all-positive or all-negative carries no contrastive signal and is skipped; if every group is
    degenerate the loss collapses to a zero that stays connected to autograd.

    Args:
        real_tau: Softmax temperature scaling the per-group scores (default: 0.5).
        num_generations: Group size; must divide the batch so ``view(-1, num_generations)`` is exact.
    """

    def __init__(self, real_tau: float = 0.5, num_generations: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.real_tau = real_tau
        self.num_generations = num_generations

    def _reduce_loss(
        self,
        logps: 'torch.Tensor',
        old_logps: 'torch.Tensor',
        ref_logps: Optional['torch.Tensor'],
        advantages: 'torch.Tensor',
        loss_mask: 'torch.Tensor',
        log_importance_weights: 'torch.Tensor',
        outputs: Dict,
        **kwargs,
    ) -> LossOutput:
        import torch
        batch_size = loss_mask.shape[0]
        assert self.num_generations > 0 and batch_size % self.num_generations == 0, (
            f'REAL needs num_generations ({self.num_generations}) to divide the batch ({batch_size}) '
            'into whole groups.')
        mask_sum = loss_mask.sum(-1).clamp(min=1.0)

        # Per-sequence score = mean token log importance ratio. Legacy REAL uses the UNCLAMPED token log
        # ratio (grpo_trainer.py:1022,1105), so recompute it from logps/old_logps rather than reuse the
        # base class's clamped log_importance_weights; the group scores then match legacy element-wise.
        log_ratio = logps - old_logps
        global_scores = (log_ratio * loss_mask).sum(-1) / mask_sum
        # advantages is per-token [B, T] (constant across tokens without a teacher); reduce to a scalar.
        seq_advantages = (advantages * loss_mask).sum(-1) / mask_sum

        group_scores = global_scores.view(-1, self.num_generations)
        group_rewards = seq_advantages.view(-1, self.num_generations)

        pos_mask = group_rewards > 0
        neg_mask = group_rewards <= 0
        valid_mask = (pos_mask.sum(dim=1) != 0) & (neg_mask.sum(dim=1) != 0)

        if not bool(valid_mask.any()):
            loss = global_scores.mean() * 0.0
        else:
            batch_scores = group_scores[valid_mask]
            batch_pos_mask = pos_mask[valid_mask]
            batch_neg_mask = neg_mask[valid_mask]

            scaled_scores = batch_scores / self.real_tau
            zeros = torch.zeros(batch_scores.size(0), 1, device=batch_scores.device, dtype=batch_scores.dtype)

            # Negative loss: log(1 + sum(exp(S_neg)))
            neg_input = scaled_scores.masked_fill(~batch_neg_mask, float('-inf'))
            neg_loss = torch.logsumexp(torch.cat([neg_input, zeros], dim=1), dim=1)
            # Positive loss: log(1 + sum(exp(-S_pos)))
            pos_input = (-scaled_scores).masked_fill(~batch_pos_mask, float('-inf'))
            pos_loss = torch.logsumexp(torch.cat([pos_input, zeros], dim=1), dim=1)

            loss = (neg_loss + pos_loss).sum() / group_rewards.size(0)

        # REAL's KL is a standalone global token-mean term (legacy grpo_trainer.py:1137-1139), not the
        # per-token penalty the base surrogate folds into per_token_loss.
        if self.beta != 0.0 and ref_logps is not None:
            per_token_kl = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1
            kl_loss = (per_token_kl * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
            loss = loss + self.beta * kl_loss

        return LossOutput(loss=loss, num_tokens=self._loss_num_tokens(loss_mask))
