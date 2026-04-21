# Copyright (c) ModelScope Contributors. All rights reserved.
"""DPO-specific metrics for preference optimization training."""
from typing import List, Union

from twinkle.data_format import InputFeature, ModelOutput
from twinkle.utils import pad_and_stack_tensors
from .base import Metric


class DPOMetric(Metric):
    """Metrics for DPO (Direct Preference Optimization) training.

    Computes TRL-style metrics:
        - logps/chosen: Average sequence-level log prob of chosen responses
        - logps/rejected: Average sequence-level log prob of rejected responses
        - rewards/chosen: β * (policy_chosen - ref_chosen)
        - rewards/rejected: β * (policy_rejected - ref_rejected)
        - rewards/margins: chosen_reward - rejected_reward
        - rewards/accuracies: Percentage where chosen_reward > rejected_reward

    Args:
        device_mesh: The device mesh
        process_group: The process group to collect data from
        ignore_index: Label index to ignore (default: -100)
        beta: DPO beta parameter for reward scaling (default: 0.1)
    """

    def __init__(self, device_mesh, process_group, ignore_index: int = -100, beta: float = 0.1, **kwargs):
        super().__init__(device_mesh, process_group, **kwargs)
        self.ignore_index = ignore_index
        self.beta = beta
        self.reset()

    def _compute_sequence_logps(self, per_token_logps, labels):
        """Compute sequence-level log probs by summing valid token logps."""
        import torch
        loss_mask = (labels != self.ignore_index).float()
        return (per_token_logps * loss_mask).sum(dim=-1)

    def _align_logps(self, logps, target_shape, device, dtype):
        """Align per-token logps to target shape by padding or truncating.

        Args:
            logps: [batch, seq_len] tensor to align
            target_shape: Target shape (batch, target_seq_len)
            device: Target device
            dtype: Target dtype

        Returns:
            Aligned tensor with shape matching target_shape
        """
        import torch

        if not torch.is_tensor(logps):
            logps = torch.as_tensor(logps)
        logps = logps.to(device=device, dtype=dtype)
        batch_size, src_len = logps.shape
        _, target_len = target_shape

        if src_len == target_len:
            return logps
        elif src_len < target_len:
            raise ValueError(f'ref_logps seq_len ({src_len}) < target seq_len ({target_len}). '
                             f'This should not happen when both models process the same batch.')
        else:
            return logps[:, :target_len]

    def _split_chosen_rejected(self, tensor):
        """Split interleaved tensor into chosen and rejected.

        Input format: [pos_1, neg_1, pos_2, neg_2, ...] (interleaved for DP-safe slicing)
        Output: (chosen [pos_1, pos_2, ...], rejected [neg_1, neg_2, ...])
        """
        return tensor[0::2], tensor[1::2]

    def accumulate(self, inputs: Union[InputFeature, List[InputFeature]], outputs: ModelOutput, **kwargs):
        """Accumulate DPO metrics from model outputs.

        Expects:
            - outputs['logps']: [batch, seq_len] per-token log probabilities
            - inputs['labels']: [batch, seq_len] labels with ignore_index for non-target tokens
            - kwargs['ref_outputs']: Optional reference model outputs with 'logps'
        """
        import torch
        logps = outputs.get('logps')
        if logps is None or len(logps) == 0:
            return

        if isinstance(logps, list) and logps:
            logps = pad_and_stack_tensors(logps)

        # Get labels from inputs
        if isinstance(inputs, list):
            labels = [input['labels'] for input in inputs]
            if len(labels) == 1:
                labels = labels[0]
            else:
                labels = pad_and_stack_tensors(labels)
            inputs = {'labels': labels}

        labels = torch.as_tensor(inputs['labels'])
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        # Ensure logps and labels have same device
        if logps.device != labels.device:
            labels = labels.to(logps.device)

        # Align sequence lengths if needed (truncate right)
        if logps.shape[1] != labels.shape[1]:
            min_len = min(logps.shape[1], labels.shape[1])
            logps = logps[:, :min_len]
            labels = labels[:, :min_len]

        # Compute sequence-level logps
        seq_logps = self._compute_sequence_logps(logps, labels)

        # DPO requires interleaved [chosen, rejected, ...] pairs → batch must be even
        assert seq_logps.shape[0] % 2 == 0, (
            f'DPO metric requires an even batch size (interleaved chosen/rejected pairs), '
            f'but got batch_size={seq_logps.shape[0]}.')

        # Split into chosen and rejected (interleaved format)
        chosen_logps, rejected_logps = self._split_chosen_rejected(seq_logps)
        chosen_labels, rejected_labels = self._split_chosen_rejected(labels)

        # Accumulate policy logps
        self.total_chosen_logps += chosen_logps.sum().item()
        self.total_rejected_logps += rejected_logps.sum().item()

        # Compute rewards if ref_outputs available
        ref_outputs = kwargs.get('ref_outputs')
        if ref_outputs is not None:
            ref_logps = ref_outputs.get('logps')
            if ref_logps is not None:
                # Align ref_logps to match labels shape (handles different seq lengths)
                ref_logps = self._align_logps(ref_logps, labels.shape, labels.device, logps.dtype)

                ref_seq_logps = self._compute_sequence_logps(ref_logps, labels)
                ref_chosen_logps, ref_rejected_logps = self._split_chosen_rejected(ref_seq_logps)

                # Accumulate ref logps
                self.total_ref_chosen_logps += ref_chosen_logps.sum().item()
                self.total_ref_rejected_logps += ref_rejected_logps.sum().item()

                # Compute rewards: β * (policy - ref)
                chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps)
                rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps)

                self.total_chosen_rewards += chosen_rewards.sum().item()
                self.total_rejected_rewards += rejected_rewards.sum().item()
                margins = chosen_rewards - rejected_rewards
                self.total_reward_margin += margins.sum().item()
                self.total_reward_correct += (margins > 0).sum().item()
                self.has_rewards = True

        self.total_count += chosen_logps.shape[0]

    def reset(self):
        """Reset all accumulated values."""
        self.total_chosen_logps = 0.0
        self.total_rejected_logps = 0.0
        self.total_ref_chosen_logps = 0.0
        self.total_ref_rejected_logps = 0.0
        self.total_chosen_rewards = 0.0
        self.total_rejected_rewards = 0.0
        self.total_reward_margin = 0.0
        self.total_reward_correct = 0
        self.total_count = 0
        self.has_rewards = False

    def calculate(self):
        """Calculate and return aggregated metrics."""
        local_results = [{
            'chosen_logps': self.total_chosen_logps,
            'rejected_logps': self.total_rejected_logps,
            'ref_chosen_logps': self.total_ref_chosen_logps,
            'ref_rejected_logps': self.total_ref_rejected_logps,
            'chosen_rewards': self.total_chosen_rewards,
            'rejected_rewards': self.total_rejected_rewards,
            'reward_margin': self.total_reward_margin,
            'reward_correct': self.total_reward_correct,
            'count': self.total_count,
            'has_rewards': self.has_rewards,
        }]
        all_results = self.gather_results(local_results)

        total_chosen_logps = sum(r['chosen_logps'] for r in all_results)
        total_rejected_logps = sum(r['rejected_logps'] for r in all_results)
        total_ref_chosen_logps = sum(r['ref_chosen_logps'] for r in all_results)
        total_ref_rejected_logps = sum(r['ref_rejected_logps'] for r in all_results)
        total_chosen_rewards = sum(r['chosen_rewards'] for r in all_results)
        total_rejected_rewards = sum(r['rejected_rewards'] for r in all_results)
        total_reward_margin = sum(r['reward_margin'] for r in all_results)
        total_reward_correct = sum(r['reward_correct'] for r in all_results)
        total_count = sum(r['count'] for r in all_results)
        has_rewards = any(r['has_rewards'] for r in all_results)

        if total_count == 0:
            return {}

        results = {
            'logps/chosen': f'{total_chosen_logps / total_count:.2f}',
            'logps/rejected': f'{total_rejected_logps / total_count:.2f}',
        }

        if has_rewards:
            results['logps/ref_chosen'] = f'{total_ref_chosen_logps / total_count:.2f}'
            results['logps/ref_rejected'] = f'{total_ref_rejected_logps / total_count:.2f}'
            results['rewards/chosen'] = f'{total_chosen_rewards / total_count:.4f}'
            results['rewards/rejected'] = f'{total_rejected_rewards / total_count:.4f}'
            results['rewards/margins'] = f'{total_reward_margin / total_count:.4f}'
            results['rewards/accuracies'] = f'{total_reward_correct / total_count * 100:.1f}%'
        self.reset()
        return results
