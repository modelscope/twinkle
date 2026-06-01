# Copyright (c) ModelScope Contributors. All rights reserved.
import math
from typing import Any, Dict, List, Optional, Tuple

from twinkle.preprocessor import Preprocessor
from twinkle.template import Template
from twinkle.utils import get_logger

from .llm_backend import LLMBackend, OpenAIBackend

logger = get_logger(only_local_master=False)

_MIN_RESPONSE_TOKENS = 5
_DEFAULT_IFD_THRESHOLD = 0.8


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
    """Coerce ndarray / tensor / list to a flat Python int list."""
    if hasattr(x, 'tolist'):
        return x.tolist()
    return list(x)


def _avg_nll(prompt_logprobs: List, token_ids: List[int], start: int) -> Optional[float]:
    """Avg NLL over token_ids[start:], looking up each position's actual-token logprob."""
    n = min(len(prompt_logprobs), len(token_ids))
    lps: List[float] = []
    for i in range(start, n):
        lp = _extract_logprob(prompt_logprobs[i], token_ids[i])
        if lp is not None:
            lps.append(lp)
    if len(lps) < _MIN_RESPONSE_TOKENS:
        return None
    return -sum(lps) / len(lps)


class IFDFilter(Preprocessor):
    """Filter key rounds by Instruction-Following Difficulty (IFD).

    Requires rows pre-annotated by IntentClassifier (user_data.key_rounds).
    For each key round, computes IFD = L(A|Q) / L(A):
      - IFD > threshold → hard example → keep
      - IFD <= threshold → easy example → remove from key_rounds

    Rows with all key_rounds removed are discarded entirely.
    Rows without key_rounds are passed through unchanged.

    Tokenization MUST go through ``template.encode`` so the prompt/response
    boundary matches the exact byte stream the sampler would emit.
    Backend calls are batched in one shot so distributed samplers can keep
    every DP worker busy (slice_dp dispatch).
    """

    def __init__(
        self,
        backend: LLMBackend = None,
        template: Optional[Template] = None,
        ifd_threshold: float = _DEFAULT_IFD_THRESHOLD,
        keep_if_no_key_rounds: bool = False,
        # Legacy params (used to create OpenAIBackend if backend is None)
        api_endpoint: str = '',
        model: str = 'default',
    ):
        super().__init__()
        if backend is not None:
            self._backend = backend
        else:
            self._backend = OpenAIBackend(endpoint=api_endpoint, model=model)
        if not isinstance(template, Template):
            raise TypeError(
                f'IFDFilter requires a `Template` instance, got {type(template).__name__}.')
        self._template = template
        self._ifd_threshold = ifd_threshold
        self._keep_if_no_key_rounds = keep_if_no_key_rounds

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = self.ifd_filter(rows)
        return self.map_row_to_col(rows)

    def _prepare_round(
        self,
        messages: List[Dict[str, Any]],
        assistant_idx: int,
    ) -> Optional[Tuple[List[int], int, List[int]]]:
        """Tokenize one round; return (cond_ids, n_prompt, asst_ids) or None if invalid."""
        if assistant_idx >= len(messages):
            return None
        asst_msg = messages[assistant_idx]
        if not isinstance(asst_msg, dict) or asst_msg.get('role') != 'assistant':
            return None
        assistant_text = asst_msg.get('content') or ''
        if isinstance(assistant_text, list):
            assistant_text = ' '.join(
                p.get('text', '') for p in assistant_text
                if isinstance(p, dict) and p.get('type') == 'text'
            )
        if not assistant_text.strip():
            return None
        context_messages = messages[:assistant_idx]
        if not context_messages:
            return None

        prompt_traj = {'messages': list(context_messages)}
        prompt_feat = self._template.encode(prompt_traj, add_generation_prompt=True)
        prompt_ids = _to_int_list(prompt_feat['input_ids'])
        # Use raw asst_ids (no chat-template wrapping) so numerator/denominator
        # average over byte-equal A token sequences; otherwise IFD ratio collapses to ~1.
        asst_ids = _to_int_list(self._template.tokenizer(assistant_text, add_special_tokens=False)['input_ids'])
        if len(asst_ids) < _MIN_RESPONSE_TOKENS + 1:
            return None
        cond_ids = prompt_ids + asst_ids
        n_prompt = len(prompt_ids)
        return cond_ids, n_prompt, asst_ids

    def _batch_floor(self) -> int:
        """Minimum batch size to keep all DP workers busy (1 for HTTP backends)."""
        sampler = getattr(self._backend, '_sampler', None)
        device_mesh = getattr(sampler, 'device_mesh', None)
        return getattr(device_mesh, 'dp_world_size', 1) or 1

    @staticmethod
    def _pad_batch(batch: List[List[int]], floor: int) -> Tuple[List[List[int]], int]:
        """Repeat last item until len(batch) ≥ floor; returns padded list and original length."""
        n = len(batch)
        if n >= floor or not batch:
            return batch, n
        return list(batch) + [batch[-1]] * (floor - n), n

    def ifd_filter(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score key rounds by IFD, remove easy rounds, discard rows with none left."""
        if not rows:
            return rows

        # Phase 1: tokenize all rounds upfront.
        prepared: Dict[Tuple[int, int], Tuple[List[int], int, List[int]]] = {}
        for ri, row in enumerate(rows):
            user_data = row.get('user_data')
            if not isinstance(user_data, dict):
                continue
            key_rounds = user_data.get('key_rounds')
            if not isinstance(key_rounds, list) or not key_rounds:
                continue
            messages = row.get('messages') or []
            for rnd_idx, asst_idx in enumerate(key_rounds):
                if not isinstance(asst_idx, int):
                    continue
                result = self._prepare_round(messages, asst_idx)
                if result is not None:
                    prepared[(ri, rnd_idx)] = result

        # Phase 2: one batched dispatch for cond, one for asst.
        scores: Dict[Tuple[int, int], float] = {}
        if prepared:
            keys = list(prepared.keys())
            cond_batch = [prepared[k][0] for k in keys]
            asst_batch = [prepared[k][2] for k in keys]
            floor = self._batch_floor()
            cond_padded, cond_n = self._pad_batch(cond_batch, floor)
            asst_padded, asst_n = self._pad_batch(asst_batch, floor)
            cond_logprobs = self._backend.prompt_logprobs_ids(cond_padded)[:cond_n]
            asst_logprobs = self._backend.prompt_logprobs_ids(asst_padded)[:asst_n]
            for key, cond_lp, asst_lp in zip(keys, cond_logprobs, asst_logprobs):
                cond_ids, n_prompt, asst_ids = prepared[key]
                # Skip A[0] in BOTH paths: asst_lp[0] is None (no prior context),
                # so cond must skip its A[0] too to average over the same token set.
                l_a_given_q = _avg_nll(cond_lp, cond_ids, n_prompt + 1)
                l_a = _avg_nll(asst_lp, asst_ids, 1)
                if l_a_given_q is None or l_a is None or l_a < 1e-8:
                    continue
                ifd = l_a_given_q / l_a
                if math.isfinite(ifd):
                    scores[key] = ifd

        # Phase 3: apply scores.
        out = []
        n_removed_rounds = 0
        n_removed_rows = 0
        for ri, row in enumerate(rows):
            user_data = row.get('user_data')
            if not isinstance(user_data, dict):
                n_removed_rows += 1
                continue
            key_rounds = user_data.get('key_rounds')
            if not isinstance(key_rounds, list) or not key_rounds:
                if self._keep_if_no_key_rounds:
                    out.append(row)
                else:
                    n_removed_rows += 1
                continue
            kept_rounds = []
            for rnd_idx, asst_idx in enumerate(key_rounds):
                ifd = scores.get((ri, rnd_idx))
                # Unscored rounds (failed prepare) are kept conservatively.
                if ifd is None or ifd > self._ifd_threshold:
                    kept_rounds.append(asst_idx)
                else:
                    n_removed_rounds += 1
            if not kept_rounds:
                n_removed_rows += 1
                continue
            row = dict(row)
            row['user_data'] = dict(user_data, key_rounds=kept_rounds)
            out.append(row)

        logger.info(
            f'[IFDFilter] removed {n_removed_rounds} easy rounds, '
            f'dropped {n_removed_rows} rows, kept {len(out)}/{len(rows)}')
        return out
