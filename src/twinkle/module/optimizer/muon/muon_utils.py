# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from twinkle.utils import get_logger

logger = get_logger()

# Parameters that stay on AdamW even though they are 2-D. Orthogonalisation itself would be well defined
# on them, but Muon's `sqrt(max(rows, cols))` step scaling is calibrated for a hidden-by-hidden matrix, and
# with a vocabulary as one dimension it inflates the step several-fold. Their gradients are also sparse
# per step -- only the tokens in the batch -- while the iteration normalises by the whole matrix's norm, so
# the step a token receives would depend on how many other tokens the batch happened to touch.
DEFAULT_EXCLUDE_KEYS = ('embed_tokens', 'lm_head')
# Q and K projections, the only ones QK-Clip scales. Covers the `*_proj` naming used by the Qwen/Llama
# families and the `wq`/`wk` naming used by others.
DEFAULT_QK_KEYS = ('q_proj', 'k_proj', '.wq', '.wk')


@dataclass
class MuonConfig:
    """The configuration of the Muon/QK-Clip param grouping.

    See https://arxiv.org/abs/2502.16982 (Moonlight) for Muon at scale and the Kimi K2 report for QK-Clip.

    Args:
        momentum: Momentum coefficient.
        nesterov: Blend the raw gradient into the buffer before orthogonalising it. Muon groups only.
        newton_schulz_steps: Iterations of the orthogonalisation.
        qk_clip_tau: The attention logit ceiling QK-Clip aims to keep the model under.
        qk_clip_enabled: Whether Q/K groups should clip. Off also means the attention wrappers that
            observe the logits are never installed.
        rms_scale_factor: Scales the orthogonalised update, relative to `sqrt(max(rows, cols))`.
        adamw_betas: The AdamW coefficients for the groups Muon does not apply to.
        adamw_eps: The AdamW denominator epsilon for those same groups.
        exclude_keys: Name keys whose parameters take the AdamW step instead of the Muon update.
        qk_keys: Name keys identifying the Q/K projections QK-Clip applies to.
    """
    momentum: float = 0.95
    nesterov: bool = False
    newton_schulz_steps: int = 5
    qk_clip_tau: float = 100.0
    qk_clip_enabled: bool = True
    rms_scale_factor: float = 0.2
    adamw_betas: Tuple[float, float] = (0.9, 0.999)
    adamw_eps: float = 1e-8
    exclude_keys: Sequence[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_KEYS))
    qk_keys: Sequence[str] = field(default_factory=lambda: list(DEFAULT_QK_KEYS))

    def __post_init__(self):
        if isinstance(self.exclude_keys, str):
            self.exclude_keys = [self.exclude_keys]
        if isinstance(self.qk_keys, str):
            self.qk_keys = [self.qk_keys]
        self.adamw_betas = tuple(self.adamw_betas)

    @property
    def group_defaults(self) -> Dict[str, Any]:
        """The keys :class:`MuonClip` reads off each param group."""
        return {
            'momentum': self.momentum,
            'nesterov': self.nesterov,
            'newton_schulz_steps': self.newton_schulz_steps,
            'qk_clip_tau': self.qk_clip_tau,
            'qk_clip_enabled': self.qk_clip_enabled,
            'rms_scale_factor': self.rms_scale_factor,
            'adamw_betas': self.adamw_betas,
            'adamw_eps': self.adamw_eps,
        }


def create_muon_param_groups(param_groups: List[Dict[str, Any]], config: MuonConfig) -> List[Dict[str, Any]]:
    """Split the given param groups by how :class:`MuonClip` should treat each parameter.

    Each incoming group can yield up to three: the Q/K weights (`apply_muon=True, is_qk=True`), the other
    matrices (`apply_muon=True`), and everything Muon does not apply to (`apply_muon=False`) -- vectors,
    plus whatever `config.exclude_keys` matches. Splitting rather than rebuilding keeps each group's own
    `lr` and `weight_decay`, which a caller may have set per group and which a flat regrouping would drop.

    Args:
        param_groups: The param groups built by the caller, each with `params` and `param_names`.
        config: The Muon config.
    Returns:
        The new param groups, ready to be passed to :class:`MuonClip`.
    """
    defaults = config.group_defaults
    new_param_groups = []
    counts = {'qk': 0, 'muon': 0, 'rest': 0}
    qk_examples = []

    for group in param_groups:
        names = group.get('param_names')
        assert names is not None, 'create_muon_param_groups requires `param_names` in every param group.'
        assert len(names) == len(group['params']), '`param_names` and `params` must have the same length.'

        buckets = {'qk': [], 'muon': [], 'rest': []}
        for index, name in enumerate(names):
            param = group['params'][index]
            if param.ndim < 2 or any(key in name for key in config.exclude_keys):
                kind = 'rest'
            elif any(key in name.lower() for key in config.qk_keys):
                kind = 'qk'
            else:
                kind = 'muon'
            buckets[kind].append(index)

        for kind in ('qk', 'muon', 'rest'):
            indices = buckets[kind]
            if not indices:
                continue
            new_group = {key: value for key, value in group.items() if key not in ('params', 'param_names')}
            new_group.update(defaults)
            new_group['apply_muon'] = kind != 'rest'
            new_group['is_qk'] = kind == 'qk'
            new_group['params'] = [group['params'][i] for i in indices]
            new_group['param_names'] = [names[i] for i in indices]
            new_param_groups.append(new_group)
            counts[kind] += len(indices)
            if kind == 'qk' and len(qk_examples) < 3:
                qk_examples.extend(new_group['param_names'][:3 - len(qk_examples)])

    if counts['muon'] + counts['qk'] == 0:
        logger.warning('Muon is enabled but no parameter takes the Muon update, so the optimizer will '
                       'behave like plain AdamW. Check `exclude_keys` against the parameter names.')
    else:
        logger.info(f"Enable Muon for {counts['muon'] + counts['qk']} weights "
                    f"({counts['qk']} of them Q/K), AdamW for {counts['rest']}.")
    if config.qk_clip_enabled and counts['qk'] == 0:
        logger.warning(f'QK-Clip is enabled but no parameter matched qk_keys={list(config.qk_keys)}, '
                       'so nothing will ever be clipped.')
    elif config.qk_clip_enabled:
        logger.info(f'QK-Clip is enabled for {counts["qk"]} weights, e.g. {qk_examples}, '
                    f'tau={config.qk_clip_tau}')
    return new_param_groups
