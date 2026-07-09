"""Backward-compat re-export shim (AUDIT A2).

``utils.py`` was split into two focused modules:
- :mod:`logprob_utils` — log-prob data-selection math (IFD / S-IFD / chr_min),
  used only by the experimental log-prob scorers.
- :mod:`message_utils` — message-format helpers used by every active step.

This shim keeps historical ``from .utils import ...`` imports working. Prefer
importing from the focused modules directly in new code.
"""
from .logprob_utils import (_chr_min_distinct, _chr_min_weighted,  # noqa: F401
                            _extract_logprob, _ifd_family_metrics,
                            _lp_to_jsonable, _mean_logprob_delta, _pad_batch,
                            _to_int_list)
from .message_utils import (CJK_CHARS_RE, build_sensitive_regex,  # noqa: F401
                            cjk_ratio, is_agent_row, load_sensitive_words,
                            msg_content_text, msg_has_media, msg_has_payload,
                            normalize_tool_calls)

__all__ = [
    # message utils
    'msg_content_text', 'msg_has_media', 'msg_has_payload', 'normalize_tool_calls',
    'cjk_ratio', 'CJK_CHARS_RE', 'load_sensitive_words', 'build_sensitive_regex',
    'is_agent_row',
    # logprob utils
    '_extract_logprob', '_to_int_list', '_chr_min_distinct', '_chr_min_weighted',
    '_ifd_family_metrics', '_mean_logprob_delta', '_lp_to_jsonable', '_pad_batch',
]
