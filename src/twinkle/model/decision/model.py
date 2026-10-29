# Copyright (c) ModelScope Contributors. All rights reserved.
# Branch execution adapted from ms-swift fbc61cf2 (Apache-2.0).
import torch
from contextlib import contextmanager
from types import MethodType

from .heads import ClefJointSchemaHead, JevVerbalizerHead, OmniJevHead, StartLuxHead
from .omnijev_branch import (OPT_CLOSE, OPT_OPEN, add_option_tokens, branch_questions, find_rope_owner, is_hybrid,
                             prefix_inputs, rope_positions, rows_from_ids)


def attach_decision_head(model, processor, family='generic', model_dir=None):
    """Attach a scoring head before adding PEFT adapters or sharding the backbone."""
    if hasattr(model, 'scoring_head'):
        raise ValueError('A scoring head is already attached')
    tokenizer = getattr(processor, 'tokenizer', processor)
    if family == 'generic':
        head = JevVerbalizerHead.from_tokenizer(tokenizer)
    else:
        classes = {
            'jev': JevVerbalizerHead,
            'clef': ClefJointSchemaHead,
            'omnijev': OmniJevHead,
            'startlux': StartLuxHead
        }
        if family not in classes or model_dir is None:
            raise ValueError('A supported family and checkpoint directory are required')
        if family == 'startlux':
            import json
            from pathlib import Path
            from startlux_decision import jevfmt
            data = json.loads((Path(model_dir) / 'decision_config.json').read_text())
            if jevfmt.check_tokenizer(tokenizer) != data['letter_token_ids']:
                raise ValueError('StartLux verbalizer does not match tokenizer')
        if family == 'omnijev':
            if not is_hybrid(model):
                raise ValueError('OmniJev requires a hybrid Qwen3.5 backbone')
            emb = OmniJevHead._locate(model_dir, 'new_tok_emb.pt', 'OMNIJEV_NEW_TOK_EMB')
            if emb is None:
                raise FileNotFoundError('Missing OmniJev option-token embeddings')
            add_option_tokens(model, processor, emb)
        head = classes[family].from_pretrained(model_dir, model=model, config=model.config)
    device = next(model.parameters()).device
    model.scoring_head = head.to(device=device, dtype=torch.float32)
    if head.needs_output_embeddings:
        head.set_output_embeddings(model.get_output_embeddings())
    origin_forward = model.forward
    open_ids, close_ids = None, None
    if family == 'omnijev':
        open_ids = tokenizer.encode(OPT_OPEN, add_special_tokens=False)
        close_ids = tokenizer.encode(OPT_CLOSE, add_special_tokens=False)
        head.set_marker_lengths(len(open_ids), len(close_ids))

    def forward(self, *args, **kwargs):
        active_head = self.scoring_head
        if family == 'omnijev':
            # A PEFT ModulesToSaveWrapper delegates __call__, not custom methods.
            actual = active_head
            if hasattr(active_head, 'modules_to_save'):
                actual = active_head.modules_to_save[active_head.active_adapter]
            return _omnijev_branch_forward(
                self,
                *args,
                head=actual,
                origin_forward=origin_forward,
                backbone_model=self,
                tokenizer=tokenizer,
                open_ids=open_ids,
                close_ids=close_ids,
                **kwargs)
        meta = kwargs.pop('decision_meta', None)
        if meta is None:
            raise ValueError('Decision inference requires decision_meta')
        kwargs.pop('labels', None)
        kwargs.update(output_hidden_states=True, return_dict=True, use_cache=False)
        output = origin_forward(*args, **kwargs)
        logits = output.logits if family in ('generic', 'jev') else None
        with torch.autocast(device_type=output.hidden_states[-1].device.type, enabled=False):
            return active_head(output.hidden_states[-1], logits, meta)

    model.forward = MethodType(forward, model)
    return head


def decision_lora_config(**kwargs):
    """Include fp32 head weights in the adapter's normal save/load lifecycle."""
    from peft import LoraConfig
    supplied = list(kwargs.pop('modules_to_save', None) or [])
    return LoraConfig(modules_to_save=list(dict.fromkeys(supplied + ['scoring_head'])), **kwargs)


def activate_decision_head(model):
    """Call after PEFT wrapping and before constructing the optimizer."""
    found = False
    for name, module in model.named_modules():
        if name.endswith('scoring_head'):
            if hasattr(module, 'modules_to_save'):
                module = module.modules_to_save[module.active_adapter]
            module.float().requires_grad_(True)
            if module.needs_output_embeddings:
                module.set_output_embeddings(model.get_output_embeddings())
            found = True
    if not found:
        raise ValueError('No decision head found')


@contextmanager
def _grad_ckpt_disabled(backbone_model):
    touched = [m for m in backbone_model.modules() if getattr(m, 'gradient_checkpointing', False)]
    for m in touched:
        m.gradient_checkpointing = False
    try:
        yield
    finally:
        for m in touched:
            m.gradient_checkpointing = True


def _omnijev_branch_forward(self, *args, head, origin_forward, backbone_model, tokenizer, open_ids, close_ids,
                            **kwargs):
    meta = kwargs.pop('decision_meta', None)
    kwargs.pop('labels', None)
    if meta is None:
        raise ValueError('OmniJev branch forward requires decision_meta from OmniJevTemplate._encode.')
    input_ids = kwargs.get('input_ids')
    if input_ids is None:
        raise ValueError('OmniJev branch forward requires input_ids; OmniJevTemplate._post_encode must NOT convert the '
                         'batch to inputs_embeds (the branch re-runs the backbone on the real token ids + pixels).')
    attention_mask = kwargs.get('attention_mask')
    pixel_values = kwargs.get('pixel_values')
    image_grid_thw = kwargs.get('image_grid_thw')
    mm_token_type_ids = kwargs.get('mm_token_type_ids')
    dev = input_ids.device
    B, T = input_ids.shape[0], input_ids.shape[1]
    left_padding = bool(meta.get('left_padding', False))
    seq_lens = meta.get('seq_lens')
    counts = [int(c) for c in meta['counts']]
    option_markers = meta['option_markers']
    mm_counts = meta.get('mm_counts') or [{'image': 0, 'video': 0}] * B
    lm_head = backbone_model.get_output_embeddings()
    rope_owner = find_rope_owner(backbone_model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    img_off = 0
    patch_off = 0
    q_cursor = 0
    per_question_features = []
    for r in range(B):
        if attention_mask is not None:
            sl = int(attention_mask[r].sum())
        elif seq_lens is not None:
            sl = int(seq_lens[r])
        else:
            sl = T
        if left_padding:
            real_ids = input_ids[r, T - sl:]
            real_mm = mm_token_type_ids[r, T - sl:] if mm_token_type_ids is not None else None
        else:
            real_ids = input_ids[r, :sl]
            real_mm = mm_token_type_ids[r, :sl] if mm_token_type_ids is not None else None
        enc = {'input_ids': real_ids[None]}
        n_img = int(mm_counts[r].get('image', 0))
        if image_grid_thw is not None and n_img > 0:
            grid_r = image_grid_thw[img_off:img_off + n_img]
            n_patch = int(grid_r.prod(-1).sum())
            if pixel_values is not None:
                enc['pixel_values'] = pixel_values[patch_off:patch_off + n_patch]
            enc['image_grid_thw'] = grid_r
            img_off += n_img
            patch_off += n_patch
        if real_mm is not None:
            enc['mm_token_type_ids'] = real_mm[None]

        nq = counts[r]
        real_list = real_ids.tolist()
        per_q_spans = []
        for qi in range(nq):
            markers = option_markers[q_cursor + qi]
            opens = [int(o) for o, _ in markers]
            closes = [int(c) for _, c in markers]
            per_q_spans.append((opens, closes))
        L_use = max(1, min(opens[0] - 1 for opens, _ in per_q_spans))
        penc = prefix_inputs(enc, L_use)
        ppos = rope_positions(rope_owner, penc)
        qrows = [rows_from_ids(real_list, L_use, opens, closes, k=len(opens)) for opens, closes in per_q_spans]
        with _grad_ckpt_disabled(backbone_model):
            outs = branch_questions(
                origin_forward, penc, qrows, dev, open_ids, close_ids, lm_head=lm_head, prefix_pos=ppos, pad_id=pad_id)
        per_question_features.extend(outs)
        q_cursor += nq

    with torch.autocast(device_type=dev.type, enabled=False):
        return head.forward_features(per_question_features, meta)


def initialize_missing_adapter_head(model, state, adapter_name):
    """Seed a released adapter without a saved head from explicitly loaded factory artifacts.

    Only call when the adapter configuration did not declare scoring_head in modules_to_save.
    Trained checkpoints declaring a saved head must never silently fall back to factory values.
    """
    if any('.scoring_head.' in key for key in state):
        raise ValueError('Adapter contains head state but does not declare scoring_head in modules_to_save')
    result = dict(state)
    for name, module in model.named_modules():
        if name.endswith('scoring_head') and hasattr(module, 'modules_to_save'):
            head = module.modules_to_save[adapter_name]
            for key, value in head.state_dict().items():
                result[f'{name}.{key}'] = value.detach().cpu()
            return result
    raise ValueError('Active decision head is unavailable')
