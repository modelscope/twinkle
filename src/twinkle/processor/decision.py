# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from copy import deepcopy

from .base import InputProcessor


class DecisionInputProcessor(InputProcessor):
    """Collate token rows while keeping question metadata outside the token axis."""

    def __init__(self, pad_token_id, padding_side='right', **kwargs):
        if kwargs.get('padding_free'):
            raise ValueError('Decision packing is not supported')
        super().__init__(**kwargs)
        if padding_side not in ('left', 'right'):
            raise ValueError('padding_side must be left or right')
        self.padding_side = padding_side
        self.pad_token_id = pad_token_id

    def __call__(self, inputs, **kwargs):
        strategy = kwargs.get('sp_strategy')
        if kwargs.get('enable_sp') or (strategy is not None and getattr(strategy, 'world_size', 1) > 1):
            raise ValueError('Decision sequence parallelism is not supported')
        if self.device_mesh is not None and self.device_mesh.cp_world_size > 1:
            raise ValueError('Decision context parallelism is not supported')
        rows = deepcopy(inputs if isinstance(inputs, list) else [inputs])
        if not rows:
            raise ValueError('Empty decision batch')
        lengths = [len(row['input_ids']) for row in rows]
        if not min(lengths):
            raise ValueError('Empty decision input')
        width = max(lengths)
        ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        attention = torch.zeros_like(ids)
        meta = {
            key: []
            for key in ('counts', 'record_index', 'n_options', 'kinds', 'question_spans', 'option_spans',
                        'option_markers', 'mm_counts')
        }
        targets = []
        labelled = [row.get('target_probs') is not None for row in rows]
        if any(labelled) and not all(labelled):
            raise ValueError('Do not mix labelled and unlabelled decision records')
        batch = {'input_ids': ids, 'attention_mask': attention}
        for index, row in enumerate(rows):
            length = lengths[index]
            offset = width - length if self.padding_side == 'left' else 0
            ids[index, offset:offset + length] = torch.as_tensor(row['input_ids'])
            attention[index, offset:offset + length] = 1
            info = row['decision_meta']
            sizes = info['n_options']
            if not sizes or len(sizes) != len(info['kinds']) or min(sizes) < 1:
                raise ValueError('Invalid decision question metadata')
            meta['counts'].append(len(sizes))
            meta['record_index'].extend([index] * len(sizes))
            for key in ('n_options', 'kinds', 'question_spans', 'option_spans', 'option_markers'):
                meta[key].extend(info.get(key, []))
            meta['mm_counts'].append(info.get('mm_counts', {'image': 0, 'video': 0}))
            if all(labelled):
                if len(row['target_probs']) != len(sizes):
                    raise ValueError('Missing question targets')
                for target, size in zip(row['target_probs'], sizes):
                    if len(target) != size:
                        raise ValueError('Target length differs from legal option count')
                    targets.append(target)
            for key in self.VLM_CONCAT_FIELDS:
                if key in row:
                    batch.setdefault(key, []).append(torch.as_tensor(row[key]))
            if 'mm_token_type_ids' in row:
                batch.setdefault('mm_token_type_ids', torch.zeros_like(ids))[index, offset:offset + length] = (
                    torch.as_tensor(row['mm_token_type_ids']).flatten())
        meta.update(max_opt=max(meta['n_options']), seq_lens=lengths, left_padding=self.padding_side == 'left')
        if targets:
            batch['labels'] = torch.tensor([t + [0.] * (meta['max_opt'] - len(t)) for t in targets])
        for key in self.VLM_CONCAT_FIELDS:
            if key in batch:
                batch[key] = torch.cat(batch[key], dim=0)
        model = kwargs.get('model')
        device = next(model.parameters()).device if model is not None else torch.device('cpu')
        batch = {key: value.to(device) for key, value in batch.items()}
        meta['input_ids'] = batch['input_ids']
        batch['decision_meta'] = meta
        return batch

    def postprocess_tensor_sp(self, inputs, outputs, **kwargs):
        return inputs, outputs

    def unpack_packed_sequences(self, inputs, outputs=None, **kwargs):
        return inputs, outputs
