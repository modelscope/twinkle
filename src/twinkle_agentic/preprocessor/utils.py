"""Pure helpers shared across preprocessor modules."""
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple


def _extract_logprob(lp, token_id: Optional[int] = None) -> Optional[float]:
    if lp is None:
        return None
    if isinstance(lp, (int, float)):
        return float(lp)
    if not isinstance(lp, dict):
        return None
    # vLLM with prompt_logprobs=1 returns top-1 PLUS actual token if they differ;
    # actual is appended LAST, so iter-first picks the wrong (top-1) one.
    entry = None
    if token_id is not None:
        entry = lp.get(token_id)
        if entry is None:
            entry = lp.get(str(token_id))
    if entry is None:
        entry = next(iter(lp.values()), None)
    if entry is None:
        return None
    if hasattr(entry, 'logprob'):
        return float(entry.logprob)
    if isinstance(entry, dict):
        v = entry.get('logprob')
        return float(v) if v is not None else None
    if isinstance(entry, (int, float)):
        return float(entry)
    return None


def _to_int_list(x) -> List[int]:
    if hasattr(x, 'tolist'):
        return x.tolist()
    return list(x)


def _chr_min_distinct(
    cond_lp: List,
    asst_lp: List,
    cond_ids: List[int],
    asst_ids: List[int],
    n_prompt: int,
    exclude_ids: Optional[Set[int]] = None,
) -> Optional[float]:
    """chr_dist_min_pos: fraction of distinct asst-token ids whose
    per-occurrence min(cond_lp - asst_lp) is strictly positive."""
    if not asst_lp or not cond_lp or not asst_ids:
        return None
    n_a = min(len(asst_lp), len(asst_ids))
    n_c = len(cond_lp)
    by_tok: Dict[int, List[float]] = {}
    for i in range(n_a):
        ci = n_prompt + i
        if ci >= n_c:
            break
        tid = asst_ids[i]
        if tid is None:
            continue
        if exclude_ids is not None and int(tid) in exclude_ids:
            continue
        a = _extract_logprob(asst_lp[i], tid)
        c_tok = cond_ids[ci] if ci < len(cond_ids) else None
        c = _extract_logprob(cond_lp[ci], c_tok)
        if a is None or c is None:
            continue
        by_tok.setdefault(int(tid), []).append(c - a)
    if not by_tok:
        return None
    pos = sum(1 for diffs in by_tok.values() if min(diffs) > 0)
    return pos / len(by_tok)


def _chr_min_weighted(
    cond_lp: List,
    asst_lp: List,
    cond_ids: List[int],
    asst_ids: List[int],
    n_prompt: int,
) -> Optional[float]:
    """Magnitude-weighted chr_min: each distinct token contributes |min_delta|
    as weight; returns sum(pos_weights) / sum(all_weights)."""
    if not asst_lp or not cond_lp or not asst_ids:
        return None
    n_a = min(len(asst_lp), len(asst_ids))
    n_c = len(cond_lp)
    by_tok: Dict[int, List[float]] = {}
    for i in range(n_a):
        ci = n_prompt + i
        if ci >= n_c:
            break
        tid = asst_ids[i]
        if tid is None:
            continue
        a = _extract_logprob(asst_lp[i], tid)
        c_tok = cond_ids[ci] if ci < len(cond_ids) else None
        c = _extract_logprob(cond_lp[ci], c_tok)
        if a is None or c is None:
            continue
        by_tok.setdefault(int(tid), []).append(c - a)
    if not by_tok:
        return None
    total_w = 0.0
    pos_w = 0.0
    for diffs in by_tok.values():
        md = min(diffs)
        w = abs(md)
        total_w += w
        if md > 0:
            pos_w += w
    if total_w == 0:
        return None
    return pos_w / total_w


def _ifd_family_metrics(
    cond_lp: List,
    asst_lp: List,
    cond_ids: List[int],
    asst_ids: List[int],
    n_prompt: int,
) -> Dict[str, Any]:
    """IFD (Cherry-LLM) and S-IFD-{50,75} (T-SHIRT) for one round."""
    if not asst_lp or not cond_lp or not asst_ids:
        return {}
    n_a = min(len(asst_lp), len(asst_ids))
    n_c = len(cond_lp)
    deltas: List[float] = []
    for i in range(n_a):
        ci = n_prompt + i
        if ci >= n_c:
            break
        tid = asst_ids[i]
        if tid is None:
            continue
        a = _extract_logprob(asst_lp[i], tid)
        c_tok = cond_ids[ci] if ci < len(cond_ids) else None
        c = _extract_logprob(cond_lp[ci], c_tok)
        if a is None or c is None:
            continue
        deltas.append(c - a)
    if not deltas:
        return {}
    n = len(deltas)
    mean_delta = sum(deltas) / n
    out: Dict[str, Any] = {
        'n_tokens': n,
        'mean_delta': mean_delta,
        'ifd': math.exp(-mean_delta),
    }
    abs_sorted = sorted(range(n), key=lambda i: abs(deltas[i]), reverse=True)
    for k_pct in (50, 75):
        keep = max(1, int(round(n * k_pct / 100)))
        sub = [deltas[i] for i in abs_sorted[:keep]]
        out[f's_ifd_{k_pct}'] = math.exp(-sum(sub) / len(sub))
    return out


def _mean_logprob_delta(
    cond_lp: List,
    asst_lp: List,
    cond_ids: List[int],
    asst_ids: List[int],
    n_prompt: int,
) -> Optional[float]:
    """Mean per-token (cond_lp - asst_lp) over the response span."""
    if not asst_lp or not cond_lp or not asst_ids:
        return None
    n_a = min(len(asst_lp), len(asst_ids))
    n_c = len(cond_lp)
    deltas: List[float] = []
    for i in range(n_a):
        ci = n_prompt + i
        if ci >= n_c:
            break
        tid = asst_ids[i]
        if tid is None:
            continue
        a = _extract_logprob(asst_lp[i], tid)
        c_tok = cond_ids[ci] if ci < len(cond_ids) else None
        c = _extract_logprob(cond_lp[ci], c_tok)
        if a is None or c is None:
            continue
        deltas.append(c - a)
    if not deltas:
        return None
    return sum(deltas) / len(deltas)


def _lp_to_jsonable(lp_list):
    """Convert per-position prompt_logprobs into JSON-safe form."""
    out = []
    for lp in (lp_list or []):
        if lp is None:
            out.append(None)
            continue
        if isinstance(lp, (int, float)):
            out.append(float(lp))
            continue
        if not isinstance(lp, dict):
            out.append(repr(lp))
            continue
        d = {}
        for k, v in lp.items():
            if hasattr(v, 'logprob'):
                d[str(k)] = {
                    'logprob': float(v.logprob),
                    'rank': getattr(v, 'rank', None),
                    'decoded': getattr(v, 'decoded_token', None)
                }
            elif isinstance(v, dict):
                d[str(k)] = v
            else:
                d[str(k)] = repr(v)
        out.append(d)
    return out


def _pad_batch(batch: List[List[int]], floor: int) -> Tuple[List[List[int]], int]:
    n = len(batch)
    if n >= floor or not batch:
        return batch, n
    return list(batch) + [batch[-1]] * (floor - n), n


# ══════════════════════════════════════════════════════════════════════════════
# Message-format utilities
# ══════════════════════════════════════════════════════════════════════════════


def msg_content_text(msg: Dict[str, Any]) -> str:
    """Extract plain text from a message's content (str | list | dict)."""
    c = msg.get('content')
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return ' '.join(p.get('text', '') for p in c if isinstance(p, dict) and p.get('type') == 'text')
    if isinstance(c, dict) and c.get('type') == 'text':
        return c.get('text', '')
    return ''


def msg_has_media(msg: Dict[str, Any]) -> bool:
    """True if message content contains non-text parts (image/audio/video)."""
    c = msg.get('content')
    return isinstance(c, list) and any(isinstance(p, dict) and p.get('type') not in ('text', None) for p in c)


def msg_has_payload(msg: Dict[str, Any]) -> bool:
    """True if a message carries any substantive payload (text, tool_calls, reasoning, or media)."""
    return bool(
        msg_content_text(msg).strip() or msg.get('tool_calls') or msg.get('reasoning_content') or msg.get('thinking')
        or msg_has_media(msg))


_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7a3]')


def normalize_tool_calls(msg: Dict[str, Any]) -> Optional[List[Any]]:
    """Return ``tool_calls`` as a list of dicts, handling PyArrow/HF serialization artifacts."""
    tcs = msg.get('tool_calls')
    if isinstance(tcs, str):
        s = tcs.strip()
        if not s:
            return None
        try:
            decoded = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(decoded, list) or not decoded:
            return None
        tcs = decoded
    if not isinstance(tcs, list) or not tcs:
        return None
    result = []
    for tc in tcs:
        if isinstance(tc, str):
            try:
                tc = json.loads(tc)
            except (json.JSONDecodeError, ValueError):
                return None
        if not isinstance(tc, dict):
            return None
        func = tc.get('function')
        if isinstance(func, str):
            try:
                func = json.loads(func)
            except (json.JSONDecodeError, ValueError):
                return None
            tc = dict(tc, function=func)
        result.append(tc)
    return result


CJK_CHARS_RE = _CJK_RE


def cjk_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that are CJK."""
    chars = text.replace(' ', '').replace('\n', '').replace('\t', '')
    if not chars:
        return 0.0
    return len(CJK_CHARS_RE.findall(chars)) / len(chars)


def load_sensitive_words(path: Optional[str]) -> Set[str]:
    """Load from external file (one word per line). Blank lines and #-comments ignored."""
    if not path or not os.path.isfile(path):
        return set()
    words: Set[str] = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                words.add(line)
    return words


def build_sensitive_regex(words: Set[str]) -> Optional['re.Pattern']:
    """Build a compiled regex from a set of words. Returns None if empty."""
    if not words:
        return None
    cjk_words = []
    latin_words = []
    cjk_re = re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7a3]')
    for w in sorted(words):
        if cjk_re.search(w):
            cjk_words.append(re.escape(w))
        else:
            latin_words.append(re.escape(w))
    parts = []
    if latin_words:
        parts.append(r'\b(' + '|'.join(latin_words) + r')\b')
    if cjk_words:
        parts.append('(' + '|'.join(cjk_words) + ')')
    return re.compile('|'.join(parts), re.IGNORECASE)


def is_agent_row(messages) -> bool:
    """Return True if the conversation contains tool interactions (agent trace).

    After MessageNormalizer runs, all non-standard formats are already converted
    to standard tool_calls / role=tool — so checking those two signals suffices.
    """
    if not isinstance(messages, list):
        return False
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get('role') == 'tool':
            return True
        if normalize_tool_calls(m):
            return True
    return False
