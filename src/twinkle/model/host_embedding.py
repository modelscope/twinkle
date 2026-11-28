# Copyright (c) ModelScope Contributors. All rights reserved.
"""Frozen host-resident embeddings excluded from accelerator placement and sharding."""
from torch import nn


class HostEmbedding(nn.Module):
    """Keep a frozen table on CPU while returning lookups to the input device.

    The table is deliberately not a child parameter or buffer: FSDP must not
    broadcast or all-gather it onto the accelerator. Full checkpoint collection
    includes it explicitly; adapter checkpoints retain the base checkpoint reference.
    """

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        if embedding.weight.requires_grad or embedding.weight.is_meta:
            raise ValueError("HostEmbedding requires a materialized, frozen embedding table.")
        embedding.to(device="cpu")
        object.__setattr__(self, "_embedding", embedding)

    @property
    def weight(self):
        return self._embedding.weight

    def forward(self, input_ids):
        return self._embedding(input_ids.to("cpu")).to(input_ids.device)

    @staticmethod
    def load_host_state(model, state_dict):
        """Restore host tables locally and exclude them from device state-dict broadcasts."""
        remaining = dict(state_dict)
        for name, module in model.named_modules():
            if isinstance(module, HostEmbedding):
                key = f'{name}.weight' if name else 'weight'
                if key not in remaining:
                    raise ValueError(f'Full checkpoint is missing host embedding {key!r}.')
                module.load_state_dict({'weight': remaining.pop(key)})
        return remaining

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + "weight"
        if key in state_dict:
            weight = state_dict.pop(key)
            if weight.shape != self.weight.shape or weight.dtype != self.weight.dtype:
                error_msgs.append(f'{key}: host embedding checkpoint shape or dtype does not match.')
            else:
                # Rebind immutable CPU storage rather than copying a huge table into each rank's old mapping.
                self._embedding.weight = nn.Parameter(weight.detach().to('cpu'), requires_grad=False)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
