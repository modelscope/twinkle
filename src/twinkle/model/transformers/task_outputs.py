"""Model-side task reductions, followed by sequence-parallel loss collection."""
from copy import copy

import torch.nn.functional as F

from .strategy.sequence_parallel.pooling import pool_last_valid_token


def postprocess_task_outputs(inputs, outputs, *, task='causal_lm', sp_strategy=None, sp_gather_keys=()):
    """Reduce task features before loss; processors only restore batch layouts."""
    if task in ('embedding', 'generative_reranker', 'seq_cls', 'reranker'):
        features = outputs.get('features') if task == 'embedding' else outputs.get('logits')
        if features is None:
            raise ValueError(f'Missing per-token features for task={task!r}')
        pool = sp_strategy.pool_last_valid_token if sp_strategy is not None else pool_last_valid_token
        pooled, _ = pool(features, inputs)
        outputs = copy(outputs)
        if task == 'embedding':
            outputs.pop('features', None)
            outputs['embeddings'] = F.normalize(pooled, p=2, dim=-1).contiguous()
        else:
            outputs['logits'] = pooled.contiguous()
        return inputs, outputs
    if sp_strategy is not None:
        return sp_strategy.gather_loss_tensors(inputs, outputs, sp_gather_keys=sp_gather_keys)
    return inputs, outputs
