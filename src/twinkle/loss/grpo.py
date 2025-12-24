from twinkle.loss.base import Loss
from accelerate.utils import gather_object
import torch


class GRPOLoss(Loss):

    def __init__(self, compute_entropy: bool = False,
                 top_entropy_quantile: float = 0.95,
                 overlong_filter: bool = False,
                 importance_sampling_level: str = 'token',
                 ):
        self.compute_entropy = compute_entropy
        self.top_entropy_quantile = top_entropy_quantile
        self.overlong_filter = overlong_filter
        self.importance_sampling_level = importance_sampling_level

    def __call__(self, inputs, outputs, ref_logits, rewards):
        completion_mask = inputs['completion_mask']
        truncated_mask = inputs['truncated_mask']
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, inputs, compute_entropy=self.compute_entropy)

        entropy_mask = None
        entropy_metrics = {}

        if self.compute_entropy:
            entropies = entropies.masked_fill(completion_mask == 0, float('nan'))
            if self.top_entropy_quantile < 1.0:
                entropy_threshold = torch.nanquantile(entropies.flatten().float(), 1 - self.top_entropy_quantile)
                entropy_metrics['entropy_threshold'] = entropy_threshold.item()
                entropy_mask = entropies >= entropy_threshold

        # apply the completion_mask to exclude loss and metrics for overlong completions
        if self.overlong_filter and any(truncated_mask):
            truncated_mask = truncated_mask.unsqueeze(-1).expand_as(completion_mask)
            completion_mask = completion_mask & (~truncated_mask)

        # Compute the KL divergence between the model and the reference model
        # Only compute KL for loss if kl_in_reward=False (GRPO style)
        if self.beta != 0.0 and not self.kl_in_reward:
            ref_per_token_logps = inputs['ref_per_token_logps']
            per_token_kl = (
                    torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1)
        else:
            per_token_kl = None

        advantages = inputs['advantages']
        # When under on-policy training
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            per_token_logps.detach() if inputs['old_per_token_logps'] is None else inputs['old_per_token_logps'])

        # Compute rollout diagnostic metrics and apply IS correction if enabled
        rollout_correction_metrics = {}
        should_compute_rollout_metrics = (
                self.rollout_importance_sampling_mode is not None or self.log_rollout_offpolicy_metrics)

        local_has_rollout_per_token_logps = inputs.get('rollout_per_token_logps') is not None
        all_has_rollout_per_token_logps = gather_object([local_has_rollout_per_token_logps])

        should_compute_rollout_metrics = should_compute_rollout_metrics and all(all_has_rollout_per_token_logps)
        if (not self.disable_rollout_importance_sampling and should_compute_rollout_metrics):
            rollout_per_token_logps = inputs['rollout_per_token_logps']

            # Compute diagnostic metrics (KL, PPL, etc.) for monitoring off-policy gap
            rollout_correction_metrics = self._compute_rollout_offpolicy_metrics(old_per_token_logps,
                                                                                 rollout_per_token_logps,
                                                                                 completion_mask)

            # Apply importance sampling correction if mode is enabled
            if self.rollout_importance_sampling_mode is not None:
                # Compute the log ratio between policy model and rollout model
                # log π_θ(y|x) - log π_rollout(y|x)
                rollout_log_ratio = old_per_token_logps - rollout_per_token_logps

                # Apply importance sampling correction based on mode
                rollout_is_weights = self._apply_rollout_importance_sampling(rollout_log_ratio, completion_mask)

                # Compute additional IS-specific metrics (ESS, clipped_frac, is_weight_mean)
                is_metrics = self._compute_is_correction_metrics(rollout_log_ratio, rollout_is_weights, completion_mask)
                rollout_correction_metrics.update(is_metrics)

                # Store IS weights for loss computation
                inputs['rollout_is_weights'] = rollout_is_weights
            else:
                inputs['rollout_is_weights'] = None
        else:
            inputs['rollout_is_weights'] = None

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == 'token':
            log_importance_weights = log_ratio
        elif self.importance_sampling_level in ['sequence', 'sequence_token']:
            seq_level_log_weights = ((log_ratio * completion_mask).sum(-1)
                                     / completion_mask.sum(-1).clamp(min=1.0)).unsqueeze(-1)
            if self.importance_sampling_level == 'sequence':
                log_importance_weights = seq_level_log_weights
            else:
                # GSPO-token: sg[si(θ)] * πθ(yi,t)/sg[πθ(yi,t)]
                seq_level_log_weight = seq_level_log_weights.detach()
                log_importance_weights = per_token_logps - per_token_logps.detach() + seq_level_log_weight
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'.")

        coef_1 = torch.exp(log_importance_weights)

        if self.loss_type == 'cispo':
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages.unsqueeze(1) * per_token_logps
        elif self.loss_type == 'sapo':
            advantages_expanded = advantages.unsqueeze(1)
            gate_pos = torch.sigmoid(self.tau_pos * (coef_1 - 1))
            gate_neg = torch.sigmoid(self.tau_neg * (coef_1 - 1))
            is_positive = advantages_expanded > 0
            soft_gate = torch.where(is_positive, gate_pos, gate_neg)

            per_token_loss = -soft_gate * advantages_expanded
        elif self.loss_type in ['grpo', 'bnpo', 'dr_grpo', 'dapo']:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if per_token_kl is not None:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # Apply vLLM importance sampling weights if available
        if inputs.get('rollout_is_weights') is not None and self.rollout_importance_sampling_mode is not None:
            rollout_is_weights = inputs['rollout_is_weights']
            per_token_loss = per_token_loss * rollout_is_weights

        # Apply off-policy sequence masking if enabled
        # Mask out sequences where delta > threshold AND advantage < 0
        if self.off_policy_sequence_mask_delta is not None:
            rollout_per_token_logps = inputs.get('rollout_per_token_logps')
            old_policy_per_token_logps = rollout_per_token_logps if rollout_per_token_logps is not None \
                else old_per_token_logps
            off_policy_seq_mask = self._compute_off_policy_sequence_mask(per_token_logps, old_policy_per_token_logps,
                                                                         completion_mask, advantages)
            # Expand sequence mask to token level and apply to completion_mask
            off_policy_seq_mask_expanded = off_policy_seq_mask.unsqueeze(-1).expand_as(completion_mask)
            completion_mask = completion_mask & off_policy_seq_mask_expanded

        if self.loss_type in ['grpo', 'sapo']:
            # completion_mask is now always [batch_size, seq_len] after pad_back
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == 'bnpo':
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == 'dr_grpo':
            batch_size = completion_mask.shape[0]
            loss = (per_token_loss * completion_mask).sum() / (batch_size * self.max_completion_length)
        elif self.loss_type in ['cispo', 'dapo']:
            # CISPO and DAPO: Normalize by total completion tokens across all processes
            normalizer = inputs['num_items_in_batch'] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            # compute for token-level average
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        # Prepare metrics data
        metrics_data = {
            'mode': mode,
            'entropy': entropy_metrics,
            'completion_mask': completion_mask,
            'completion_token_count': completion_token_count,
        }

        if per_token_kl is not None:
            mean_kl = masked_batch_mean(per_token_kl)
            metrics_data['kl'] = self.accelerator.gather_for_metrics(mean_kl).nanmean().item()

        # Add rollout correction metrics
        if rollout_correction_metrics:
            metrics_data['rollout_correction'] = rollout_correction_metrics

        # Compute the clipped probability ratios
        if self.loss_type == 'cispo':
            # CISPO: Only track upper bound clipping
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages.unsqueeze(1) > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather_for_metrics(cispo_clip_ratio)
            metrics_data['clipping'] = {'cispo_clip_ratio': gathered_cispo_clip_ratio.nanmean().item()}
        elif self.loss_type == 'sapo':
            pass
        else:
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
            gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
            gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)

            metrics_data['clipping'] = {
                'low_clip_mean': gathered_low_clip.nanmean().item(),
                'low_clip_min': nanmin(gathered_low_clip).item(),
                'high_clip_mean': gathered_high_clip.nanmean().item(),
                'high_clip_max': nanmax(gathered_high_clip).item(),
                'region_clip_mean': gathered_clip_ratio.nanmean().item()
            }
        if mode == 'train' and self.chord_sft_iterator is not None:
            loss = compute_chord_loss(self, grpo_loss=loss)

        return loss, metrics_data