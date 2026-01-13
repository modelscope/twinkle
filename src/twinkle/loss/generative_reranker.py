# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import Loss
import torch


class GenerativeRerankerLoss(Loss):

    def __init__(self, tokenizer, positive_token='yes', negative_token='no'):
        self.tokenizer = tokenizer
        self.positive_token = positive_token
        self.negative_token = negative_token

    def __call__(self, inputs, outputs, last_valid_indices, **kwargs):
        """
            Generative reranker loss function.

            This loss function is designed for generative rerankers that use token probabilities
            (e.g., "yes"/"no") to determine relevance scores. It only computes loss on the
            last token position for specific tokens.

            Args:
                outputs: Model outputs containing logits
                labels: Binary labels (0/1) for irrelevant/relevant pairs
                last_valid_indices: The last valid indices to compute loss

            Returns:
                torch.Tensor: Cross entropy loss for yes/no classification
            """
        logits = outputs['logits']
        labels = inputs['labels']
        # Get token IDs for positive and negative tokens
        # Default to "yes"/"no", but can be configured via environment variables
        try:
            positive_token_id = self.tokenizer.convert_tokens_to_ids(self.positive_token)
            negative_token_id = self.tokenizer.convert_tokens_to_ids(self.negative_token)
        except Exception as e:
            raise ValueError(f"Failed to convert tokens '{self.positive_token}'/'{self.negative_token}' to IDs. "
                             f'Please check if these tokens exist in the tokenizer vocabulary. Error: {e}')

        # Extract logits at the last valid (non-padding) token position for each sample
        batch_size = logits.shape[0]
        batch_indices = torch.arange(batch_size, device=logits.device)
        last_valid_logits = logits[batch_indices, last_valid_indices, :]

        positive_logits = last_valid_logits[:, positive_token_id]  # [batch_size]
        negative_logits = last_valid_logits[:, negative_token_id]  # [batch_size]

        # Stack to create binary classification logits
        # Shape: [batch_size, 2] where dim=1 represents [negative, positive]
        binary_logits = torch.stack([negative_logits, positive_logits], dim=1)

        # Convert labels to the correct device and type
        binary_labels = labels.to(binary_logits.device).long()

        # Compute cross entropy loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(binary_logits, binary_labels)

        return loss