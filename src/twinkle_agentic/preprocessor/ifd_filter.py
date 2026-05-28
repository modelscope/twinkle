# Copyright (c) ModelScope Contributors. All rights reserved.
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from twinkle.preprocessor import Preprocessor
from twinkle.utils import get_logger

from .llm_backend import LLMBackend, OpenAIBackend

logger = get_logger(only_local_master=False)

_MIN_RESPONSE_TOKENS = 5
_DEFAULT_IFD_THRESHOLD = 0.8


def _extract_logprob(lp) -> Optional[float]:
    if lp is None:
        return None
    if isinstance(lp, (int, float)):
        return float(lp)
    if isinstance(lp, dict):
        v = next(iter(lp.values()), None)
        if isinstance(v, dict):
            return float(v['logprob'])
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _avg_nll(prompt_logprobs: List, start: int) -> Optional[float]:
    """Compute average negative log-likelihood from position `start` onward."""
    lps = [_extract_logprob(lp) for lp in prompt_logprobs[start:]]
    lps = [lp for lp in lps if lp is not None]
    if len(lps) < _MIN_RESPONSE_TOKENS:
        return None
    return -sum(lps) / len(lps)


def _get_prompt_logprobs(
    backend: LLMBackend,
    messages: List[Dict[str, Any]],
) -> Optional[List]:
    return backend.prompt_logprobs(messages)


def _compute_ifd(
    backend: LLMBackend,
    tokenizer,
    context_messages: List[Dict[str, Any]],
    assistant_text: str,
) -> Optional[float]:
    """Compute IFD = L(A|Q) / L(A) for a single (context, response) pair."""
    # L(A|Q): conditional loss — full context + assistant response
    cond_messages = context_messages + [{'role': 'assistant', 'content': assistant_text}]
    try:
        prompt_part = tokenizer.apply_chat_template(
            context_messages, tokenize=False, add_generation_prompt=True)
        full_part = tokenizer.apply_chat_template(
            cond_messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        return None

    n_prompt = len(tokenizer(prompt_part, add_special_tokens=False)['input_ids'])
    n_full = len(tokenizer(full_part, add_special_tokens=False)['input_ids'])
    if n_full - n_prompt < _MIN_RESPONSE_TOKENS:
        return None

    cond_logprobs = _get_prompt_logprobs(backend, cond_messages)
    if cond_logprobs is None:
        return None
    l_a_given_q = _avg_nll(cond_logprobs, n_prompt)
    if l_a_given_q is None:
        return None

    # L(A): unconditional loss — just the assistant text as a standalone message
    uncond_messages = [{'role': 'user', 'content': ''}, {'role': 'assistant', 'content': assistant_text}]
    try:
        uncond_prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': ''}], tokenize=False, add_generation_prompt=True)
    except Exception:
        return None

    n_uncond_prompt = len(tokenizer(uncond_prompt, add_special_tokens=False)['input_ids'])
    uncond_logprobs = _get_prompt_logprobs(backend, uncond_messages)
    if uncond_logprobs is None:
        return None
    l_a = _avg_nll(uncond_logprobs, n_uncond_prompt)
    if l_a is None or l_a < 1e-8:
        return None

    return l_a_given_q / l_a


class IFDFilter(Preprocessor):
    """Filter key rounds by Instruction-Following Difficulty (IFD).

    Requires rows pre-annotated by IntentClassifier (user_data.key_rounds).
    For each key round, computes IFD = L(A|Q) / L(A):
      - IFD > threshold → hard example → keep
      - IFD <= threshold → easy example → remove from key_rounds

    Rows with all key_rounds removed are discarded entirely.
    Rows without key_rounds are passed through unchanged.
    """

    def __init__(
        self,
        backend: LLMBackend = None,
        tokenizer_name_or_path: str = '',
        ifd_threshold: float = _DEFAULT_IFD_THRESHOLD,
        max_workers: int = 8,
        keep_if_no_key_rounds: bool = False,
        # Legacy params (used to create OpenAIBackend if backend is None)
        api_endpoint: str = '',
        model: str = 'default',
    ):
        from transformers import AutoTokenizer

        super().__init__()
        if backend is not None:
            self._backend = backend
        else:
            self._backend = OpenAIBackend(endpoint=api_endpoint, model=model)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self._ifd_threshold = ifd_threshold
        self._max_workers = max_workers
        self._keep_if_no_key_rounds = keep_if_no_key_rounds

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = self.ifd_filter(rows)
        return self.map_row_to_col(rows)

    def _score_round(
        self,
        messages: List[Dict[str, Any]],
        assistant_idx: int,
    ) -> Optional[float]:
        """Compute IFD for a single key round."""
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

        # Context = everything before this assistant message
        context_messages = messages[:assistant_idx]
        if not context_messages:
            return None

        return _compute_ifd(
            self._backend, self._tokenizer, context_messages, assistant_text,
        )

    def ifd_filter(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score key rounds by IFD, remove easy rounds, discard rows with none left."""
        if not rows:
            return rows

        # Collect all (row_idx, round_idx, assistant_idx) tasks
        tasks: List[Tuple[int, int, int, List[Dict[str, Any]]]] = []
        for ri, row in enumerate(rows):
            user_data = row.get('user_data')
            if not isinstance(user_data, dict):
                continue
            key_rounds = user_data.get('key_rounds')
            if not isinstance(key_rounds, list) or not key_rounds:
                continue
            messages = row.get('messages') or []
            for rnd_idx, asst_idx in enumerate(key_rounds):
                if isinstance(asst_idx, int):
                    tasks.append((ri, rnd_idx, asst_idx, messages))

        # Parallel IFD scoring
        scores: Dict[Tuple[int, int], Optional[float]] = {}
        if tasks:
            n_workers = min(self._max_workers, len(tasks))
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                future_to_key = {
                    pool.submit(self._score_round, msgs, asst_idx): (ri, rnd_idx)
                    for ri, rnd_idx, asst_idx, msgs in tasks
                }
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        scores[key] = future.result()
                    except Exception:
                        scores[key] = None

        # Filter key_rounds and rows
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

            # Keep only hard rounds (IFD > threshold or score unavailable)
            kept_rounds = []
            for rnd_idx, asst_idx in enumerate(key_rounds):
                ifd = scores.get((ri, rnd_idx))
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
