# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pure key-shaping helpers for the checkpoint-engine weight stream.

These transforms sit between a model's weight source (``_get_weight_generator`` --
backend-specific, lives in the model classes) and the transport engines: they
answer "what name does the receiving engine expect for this tensor", which is
receiver vocabulary (PEFT wrapper segments, vLLM's LoRA module naming), not
model knowledge. Both backends previously carried identical private copies
inside their generators; they live here once now.
"""
from __future__ import annotations

from typing import List, Optional

from twinkle.utils import get_logger

logger = get_logger()


def is_lora_key(name: str) -> bool:
    """True for PEFT LoRA delta parameters."""
    return 'lora_A' in name or 'lora_B' in name or 'lora_embedding' in name


def normalize_param_name(name: str, keep_base_layer: bool) -> str:
    """Strip the PEFT wrapper prefixes a merged-weight receiver does not use.

    ``base_model.model.`` always goes; ``.base_layer`` (the frozen original under
    a LoRA layer) goes unless the receiver addresses it explicitly.
    """
    name = name.replace('base_model.model.', '')
    if not keep_base_layer:
        name = name.replace('.base_layer', '')
    return name


def add_base_layer_suffix(name: str, model_keys: Optional[List[str]]) -> str:
    """Rename to the ``.base_layer.*`` variant whenever the sampler exposes it.

    vLLM (enable_lora) wraps some modules as ``*WithLoRA`` and exposes
    only their ``.base_layer.*`` param, even when PEFT does not target
    them on the training side (e.g. linear-attn ``conv1d`` /
    ``in_proj_qkvz``).  ``model_keys`` is the receiver's key set; when it
    is None the rename is applied unconditionally.
    """
    base_layer_name = None
    if name.endswith('.weight'):
        base_layer_name = f'{name[:-7]}.base_layer.weight'
        if not model_keys or base_layer_name in model_keys:
            name = base_layer_name
    elif name.endswith('.bias'):
        base_layer_name = f'{name[:-5]}.base_layer.bias'
        if not model_keys or base_layer_name in model_keys:
            name = base_layer_name
    if 'experts' in name and base_layer_name is not None:
        return base_layer_name
    return name


def print_weight_example(names) -> None:
    """Log the first few synced names as a wiring sanity check."""
    for name in names[:3]:
        logger.info(f'Sync weight: {name}')


def record_weight_fingerprint(via: str, side: str, seq: int, name: str, tensor, *, tag: str = '') -> None:
    """Append one weight's exact fingerprint to this side's reconciliation file, if the probe is on.

    Set ``TWINKLE_WEIGHT_SYNC_FINGERPRINT`` to a directory to enable it. Publisher and receiver are
    different processes, so each appends its own file and the two are compared offline: the
    fingerprint covers every element exactly, so equal values for a name mean the bytes that arrived
    are the bytes that were published. A checkpoint cannot answer that question -- it shows the
    weights changed, not that the transport preserved them.

    ``via`` names the call site rather than the transport, so a run whose sampler relays through more
    than one layer records each layer separately instead of overwriting one another's rows; ``tag``
    carries whatever identity the site has (channel, rank) so rows can be paired across processes.
    """
    import json
    import os

    root = os.environ.get('TWINKLE_WEIGHT_SYNC_FINGERPRINT')
    if not root:
        return
    from twinkle.utils.torch_utils import bits_fingerprint
    os.makedirs(root, exist_ok=True)
    safe_tag = ''.join(ch if ch.isalnum() or ch in '-_' else '.' for ch in str(tag))
    path = os.path.join(root, f'weightsync.{via}.{side}{("." + safe_tag) if safe_tag else ""}.jsonl')
    with open(path, 'a') as handle:
        handle.write(json.dumps({
            'via': via,
            'side': side,
            'tag': str(tag),
            'seq': seq,
            'name': name,
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'bits': bits_fingerprint(tensor),
        }) + '\n')
