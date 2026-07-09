"""Log-probability data-selection math (IFD / S-IFD / chr_min).

Split out of ``utils.py`` (AUDIT A2): these helpers are consumed only by the
log-prob based scorers (the experimental ``ScoreFilter`` family). Keeping them
separate from the message-format utilities means editing scoring math never
risks touching the message helpers used across every active cleaning step.
"""
import math
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
