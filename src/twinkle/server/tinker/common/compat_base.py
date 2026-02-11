import torch
import numpy as np
from typing import List
from tinker import types
from twinkle.template import Template


def collect_forward_backward_results(results):
    """Custom collect function for forward_backward that handles list [outputs, loss].

    Args:
        results: List of lists from each worker, where each list is [outputs_list, loss_float]

    Returns:
        List of [flattened_outputs, averaged_loss]
    """
    if not results:
        return results

    # results is a list of lists: [[outputs1, loss1], [outputs2, loss2], ...]
    # Flatten outputs (first element of each list)
    all_outputs = []
    all_losses = []
    for result in results:
        outputs, loss = result
        all_outputs.extend(outputs)
        all_losses.append(loss)

    # Average the losses
    avg_loss = float(np.mean(all_losses))

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
        for i, feature in enumerate(inputs):
            # Ensure 1D shape and correct device to avoid dimension mismatch and device errors
            labels = feature.loss_fn_inputs['target_tokens'].to_torch(
            ).long().view(-1).to(logits.device)  # shape (seq_len,)
            weights = feature.loss_fn_inputs['weights'].to_torch(
            ).view(-1).to(logits.device)  # shape (seq_len,)

            # Slice logits to match the sequence length of labels
            # Labels are assumed to be already shifted/aligned with logits
            seq_len = labels.numel()

            # Check if index is within logits bounds
            if i < logits.shape[0]:
                feature_logits = logits[i, :seq_len, :]

                # Calculate log probs for all labels
                # Apply log_softmax to convert raw logits to log-probabilities
                feature_log_probs = torch.log_softmax(feature_logits, dim=-1)
                token_log_probs = feature_log_probs.gather(
                    dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

                # elementwise_loss: positive NLL loss (0.0 where masked)
                elementwise_loss = -token_log_probs * weights

                results.append({
                    'logprobs': types.TensorData.from_torch(token_log_probs),
                    'elementwise_loss': types.TensorData.from_torch(elementwise_loss)
                })
            else:
                # Handle case where batch index exceeds logits batch size
                results.append({
                    'logprobs': None,
                    'elementwise_loss': None
                })
        return results
