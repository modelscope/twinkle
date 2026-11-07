"""Keep only supervised positions in lm_head while preserving full-vocabulary CE."""
import torch
import torch.nn.functional as F
from twinkle.processor import InputProcessor


class DecisionProcessor(InputProcessor):
    def __init__(self, *, pad_token_id, pad_multiple=128, **kwargs):
        super().__init__(padding_free=False, **kwargs)
        self.pad_token_id = pad_token_id
        self.pad_multiple = pad_multiple
        # Append after the base whitelist, which would discard logits_to_keep.
        self.process_pipeline.append(self.select_decision_logits)

    def select_decision_logits(self, inputs, **kwargs):
        if self.framework != 'transformers':
            raise ValueError('This adapter is for TransformersModel without sequence parallelism')
        if kwargs.get('enable_sp'):
            raise ValueError('Sequence parallelism has not been validated for sparse decision logits')
        labels = inputs['labels']
        if labels.ndim != 2 or not torch.all((labels != -100).sum(-1) > 0):
            raise ValueError('Every decision example must contain supervision')
        length = inputs['input_ids'].shape[-1]
        multiple = self.pad_multiple
        padding = (-length) % multiple if multiple else 0
        for key, value in [('input_ids', self.pad_token_id), ('attention_mask', 0)]:
            if padding:
                inputs[key] = F.pad(inputs[key], (0, padding), value=value)
        # Positions derive from labels already shifted by encode_decision.
        selected = (labels != -100).any(dim=0).nonzero(as_tuple=True)[0]
        inputs['labels'] = labels.index_select(-1, selected)
        inputs['logits_to_keep'] = selected
        # Recomputing a checkpointed layer must not append to a mutable KV cache.
        inputs['use_cache'] = False
        return inputs
