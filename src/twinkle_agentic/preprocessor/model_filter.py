import re
from typing import Any, Dict, Optional, Sequence

from twinkle.preprocessor import Filter

# Each entry is the discriminating prefix only; a shared variant tail is appended uniformly
# so suffixes like -Instruct, -Thinking-2507, -Distill-Qwen-7B, -Air are accepted everywhere.
#
# DESIGN INTENT — only large text models are admitted (this is deliberate, not a bug):
#   * Sub-100B models are excluded on purpose. The size gate is baked into the
#     patterns: e.g. ``-[123]\d{2}b`` requires a 3-digit "1xx/2xx/3xxB" family, so
#     27B / 35B / 8B variants intentionally FAIL to match and get dropped.
#   * Vision-Language (VL / multimodal) models are excluded on purpose. There is no
#     VL entry in the allow-list, so e.g. ``Qwen3-VL-*`` is dropped even at 235B.
# Consequence: on mixed dumps a large share of rows (small + VL models) land in
# ``dropped.jsonl`` with reason ``model_not_allowed`` — expected by design.
# Revisit here (add a pattern / relax the size digits) only when we decide to
# start distilling from small or multimodal teachers.
_DEFAULT_PATTERNS = [
    r'minimax/minimax-m[23][\d.]*',
    r'opengvlab/internvl[\d._]+-2\d{2}b',
    r'qwen/qwen3[\d.]*-[123]\d{2}b(?:-a\d+b)?',
    r'qwen/qwen3-coder',
    r'xiaomimimo/mimo-v[\d.]+',
    r'(?:zhipuai|z-ai)/glm-[56][\d.]*',
    r'deepseek-ai/deepseek-(?:r1|v[34])',
    r'moonshotai/kimi',
    r'stepfun-ai/step',
]

# Hyphen MUST be inside the class (e.g. Kimi-K2-Instruct), '*' allows bare-prefix matches.
_VARIANT_TAIL = r'[-\w.]*'


class ModelFilter(Filter):
    """Keep only rows whose model_id matches an allowed family (case-insensitive)."""

    def __init__(self, patterns: Optional[Sequence[str]] = None, field: str = 'model_id'):
        self._field = field
        pats = patterns if patterns is not None else _DEFAULT_PATTERNS
        self._re = re.compile('|'.join(f'(?:{p}{_VARIANT_TAIL})' for p in pats), re.IGNORECASE)

    def keep(self, row: Dict[str, Any]) -> bool:
        return bool(self._re.fullmatch(row.get(self._field) or ''))

    def drop_reason(self, row: Dict[str, Any]) -> str:
        return 'model_not_allowed'
