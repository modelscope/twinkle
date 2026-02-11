import torch
import numpy as np
from typing import List
from tinker import types
from twinkle.template import Template
from twinkle.utils.torch_utils import selective_log_softmax
from twinkle.utils.platform import DeviceMesh


def collect_forward_backward_results(results, device_mesh: DeviceMesh):
    """Custom collect function for forward_backward that handles list [outputs, loss].

    Args:
        results: List of lists from each worker, where each list is [outputs_list, loss_float]

    Returns:
        List of [flattened_outputs, averaged_loss]
    """
    if not results:
        return results

    # Filter for last pipeline stage if PP is enabled
    pp_last_ranks = None
    if device_mesh.pp_world_size > 1:
        pp_last_ranks = set(device_mesh.get_pp_last_ranks())

    # Filter for last tp rank if TP is enabled
    tp_last_ranks = None
    if device_mesh.tp_world_size > 1:
        tp_last_ranks = set(device_mesh.get_tp_last_ranks())

    mesh_flat = device_mesh.mesh.flatten()

    # results is a list of lists: [[outputs1, loss1], [outputs2, loss2], ...]
    # Flatten outputs (first element of each list)
    all_outputs = []
    all_losses = []
    for i, result in enumerate(results):
        rank = mesh_flat[i] if i < len(mesh_flat) else -1

        # Only collect from the last PP rank to avoid duplicates
        if pp_last_ranks is not None:
            if rank not in pp_last_ranks:
                continue

        # Only collect from the last TP rank to avoid duplicates
        if tp_last_ranks is not None:
            if rank not in tp_last_ranks:
                continue

        if result is None:
            continue

        outputs, loss = result
        if outputs is None or loss is None:
            continue
        all_outputs.extend(outputs)
        all_losses.append(loss)

    # Average the losses
    if all_losses:
        avg_loss = float(np.mean(all_losses))
    else:
        avg_loss = 0.0

    return [all_outputs, avg_loss]


def clean_metrics(metrics: dict) -> dict:
    cleaned = {}
    for key, value in metrics.items():
        if isinstance(value, str):
            import re
            match = re.match(r'^([+-]?\d*\.?\d+)', value.strip())
            if match:
                cleaned[key] = float(match.group(1))
        else:
            cleaned[key] = value
    return cleaned


class TwinkleCompatModelBase:
    """Base class containing common logic for Twinkle compatibility wrappers."""

    def get_template(self, adapter_name: str) -> Template:
        return self.optimizer_group[adapter_name].template

    @staticmethod
    def _get_forward_output(inputs: List[types.Datum], logits: torch.Tensor) -> List[dict]:
        """Convert raw logits to the expected output format with logprobs and elementwise_loss."""
        results = []
        for feature, logit in zip(inputs, logits):
            # Ensure 1D shape and correct device to avoid dimension mismatch and device errors
            labels = feature.loss_fn_inputs['target_tokens'].to_torch(
            ).long().view(-1).to(logit.device)  # shape (seq_len,)
            weights = feature.loss_fn_inputs['weights'].to_torch(
            ).view(-1).to(logit.device)  # shape (seq_len,)

            # Slice logits to match the sequence length of labels
            # Labels are assumed to be already shifted/aligned with logits
            seq_len = labels.numel()

            # Check if index is within logits bounds
            feature_logits = logit[:seq_len, :]

            # Calculate log probs for all labels
            token_log_probs = selective_log_softmax(feature_logits, labels)

            # elementwise_loss: positive NLL loss (0.0 where masked)
            elementwise_loss = -token_log_probs * weights

            results.append({
                'logprobs': types.TensorData.from_torch(token_log_probs.cpu()),
                'elementwise_loss': types.TensorData.from_torch(elementwise_loss.cpu())
            })
        return results
