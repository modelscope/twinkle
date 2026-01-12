# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import Loss
import torch


class ListwiseGenerativeRerankerLoss(Loss):

    def __init__(self, tokenizer, positive_token='yes', negative_token='no', temperature=1.0, min_group_size=2
                 ):
        self.tokenizer = tokenizer
        self.positive_token = positive_token
        self.negative_token = negative_token
        self.temperature = temperature
        self.min_group_size = min_group_size

    def __call__(self, inputs, outputs, last_valid_indices, **kwargs):
        """
        List-wise generative reranker loss function.

        This loss function combines the generative reranker approach (using token probabilities)
        with list-wise ranking. It groups samples by query based on the pattern where each group
        consists of 1 positive document followed by n negative documents, then uses the
        probabilities of specific tokens (e.g., "yes"/"no") to perform ranking within each group.

        Data format expected:
        - labels: [1, 0, 0, 0, 1, 0, 0, ...] where 1 indicates positive, 0 indicates negative
        - Each 1 is followed by its corresponding negative documents until the next 1

        Environment variables for configuration:
        - GENERATIVE_RERANKER_POSITIVE_TOKEN: Token for positive relevance (default: "yes")
        - GENERATIVE_RERANKER_NEGATIVE_TOKEN: Token for negative relevance (default: "no")
        - LISTWISE_RERANKER_TEMPERATURE: Temperature for softmax (default: 1.0)
        - LISTWISE_RERANKER_MIN_GROUP_SIZE: Minimum group size to include (default: 2)

        Args:
            outputs: Model outputs containing logits [batch_size, seq_len, vocab_size]
            labels: Binary labels (1 for positive, 0 for negative) [batch_size]
            loss_scale: Not used for listwise generative reranker
            num_items_in_batch: Not used for listwise generative reranker
            trainer: Trainer instance to access tokenizer

        Returns:
            torch.Tensor: Cross entropy loss for ranking classification based on token probabilities
        """
        logits = outputs['logits']
        labels = inputs['labels']
        # Get token IDs for positive and negative tokens
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

        logits = torch.nn.functional.logsigmoid(positive_logits - negative_logits)

        # Find positive sample indices to determine group boundaries
        positive_indices = torch.nonzero(labels == 1, as_tuple=False).squeeze(-1)

        if len(positive_indices) == 0:
            # No positive samples in this batch, return zero loss
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Ensure positive_indices is 1D
        if positive_indices.dim() == 0:
            positive_indices = positive_indices.unsqueeze(0)

        total_loss = 0.0
        num_groups = 0

        for i, pos_idx in enumerate(positive_indices):
            # Determine group boundaries
            group_start = pos_idx.item()

            # Find the end of current group (start of next group or end of batch)
            if i + 1 < len(positive_indices):
                group_end = positive_indices[i + 1].item()
            else:
                group_end = len(labels)

            # Extract group relevance scores and labels
            group_scores = logits[group_start:group_end]  # [group_size]
            group_labels = labels[group_start:group_end]  # [group_size]

            # Skip groups that are too small
            if len(group_scores) < self.min_group_size:
                continue

            # Verify that the first sample in the group is positive
            if group_labels[0] != 1:
                continue  # Skip malformed groups

            group_logits = group_scores / self.temperature

            # The positive document is always at index 0 within the group
            target = torch.tensor(0, dtype=torch.long, device=logits.device)

            # Apply cross-entropy loss: positive document should have highest relevance score
            loss_fct = torch.nn.CrossEntropyLoss()
            group_loss = loss_fct(group_logits.unsqueeze(0), target.unsqueeze(0))

            total_loss += group_loss
            num_groups += 1

        if num_groups == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Return average loss across all groups
        return total_loss / num_groups