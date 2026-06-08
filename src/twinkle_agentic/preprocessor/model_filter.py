import re
from typing import Any, Dict, List, Optional, Sequence

from twinkle.preprocessor import Preprocessor

# Case-insensitive regexes covering all known allowed model-id variants.
_DEFAULT_PATTERNS = [
    r'minimax/minimax-m[23][\d.]*',
    r'opengvlab/internvl[\d._]+-2\d{2}b.*',
    r'qwen/qwen3[\d.]*-[123]\d{2}b(-a\d+b)?',
    r'qwen/qwen3-coder[-\w.]*',
    r'xiaomimimo/mimo-v[\d.]+(-.+)?',
    r'(zhipuai|z-ai)/glm-[56][\d.]*',
    r'deepseek-ai/deepseek-(r1|v[34])[-\w.]*(\[\w+\])?',
    r'moonshotai/kimi-[\w.]+',
    r'stepfun-ai/step-[\w.-]+',
]


class ModelFilter(Preprocessor):
    """Keep only rows whose model_id matches at least one allowed pattern."""

    def __init__(self, patterns: Optional[Sequence[str]] = None, field: str = 'model_id'):
        self._field = field
        pats = patterns if patterns is not None else _DEFAULT_PATTERNS
        self._re = re.compile('|'.join(f'(?:{p})' for p in pats), re.IGNORECASE)

    def __call__(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = self.map_col_to_row(rows)
        return [r for r in rows if self._re.fullmatch(r.get(self._field) or '')]
