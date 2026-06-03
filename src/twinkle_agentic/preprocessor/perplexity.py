# Copyright (c) ModelScope Contributors. All rights reserved.
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from twinkle.preprocessor import Preprocessor

from .llm_backend import LLMBackend, OpenAIBackend

_MIN_RESPONSE_TOKENS = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def _encode_pair(
    tokenizer,
    messages: List[Dict[str, Any]],
) -> Optional[Tuple[List[Dict[str, Any]], int]]:
    """Return (messages, n_prompt_tokens) or None."""
    last_asst = next(
        (i for i in range(len(messages) - 1, -1, -1)
         if isinstance(messages[i], dict) and messages[i].get('role') == 'assistant'),
        None,
    )
    if last_asst is None:
        return None

    try:
        prompt_text = tokenizer.apply_chat_template(
            messages[:last_asst], tokenize=False, add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
    except Exception:
        return None

    # Template already embeds special tokens as text; avoid double-adding them
    n_prompt = len(tokenizer(prompt_text, add_special_tokens=False)['input_ids'])
    n_full   = len(tokenizer(full_text,   add_special_tokens=False)['input_ids'])
    if n_full - n_prompt < _MIN_RESPONSE_TOKENS:
        return None
    return messages, n_prompt


def _extract_logprob(lp) -> Optional[float]:
    """Extract scalar log-prob from a vLLM prompt_logprobs element after JSON round-trip."""
    if lp is None:
        return None
    if isinstance(lp, (int, float)):
        return float(lp)
    # vLLM JSON format: {str(token_id): {"logprob": float, "rank": int, "decoded_token": str}}
    if isinstance(lp, dict):
        v = next(iter(lp.values()), None)
        if isinstance(v, dict):
            return float(v['logprob'])
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _ppl_from_logprobs(
    prompt_logprobs: List,
    n_prompt: int,
) -> Optional[float]:
    response_lps = [_extract_logprob(lp) for lp in prompt_logprobs[n_prompt:]]
    response_lps = [lp for lp in response_lps if lp is not None]
    if len(response_lps) < _MIN_RESPONSE_TOKENS:
        return None
    return math.exp(-sum(response_lps) / len(response_lps))


def _score_one(
    backend: LLMBackend,
    messages: List[Dict[str, Any]],
) -> List[Optional[float]]:
    return backend.prompt_logprobs(messages)


# ── Preprocessor ─────────────────────────────────────────────────────────────

class PerplexityFilter(Preprocessor):
    """Filter dataset rows by model perplexity on the assistant response.

    Uses the OpenAI-compatible /v1/chat/completions endpoint with prompt_logprobs
    so it is safe to use in multiprocessing contexts — no shared GPU state.

    ppl_min / ppl_max define the keep window:
      - Too low  → trivially memorized / degenerate output.
      - Too high → out-of-distribution, garbled, or badly formatted.

    Requirement: tokenizer_name_or_path must match the model served at api_endpoint.
    """

    def __init__(
        self,
        backend: LLMBackend = None,
        tokenizer_name_or_path: str = '',
        ppl_min: float = 2.0,
        ppl_max: float = 100.0,
        max_workers: int = 8,
        # Legacy params
        api_endpoint: str = '',
        model: str = 'default',
    ):
        from transformers import AutoTokenizer

        if backend is not None:
            self._backend = backend
        else:
            self._backend = OpenAIBackend(endpoint=api_endpoint, model=model)
        self._tokenizer   = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.ppl_min      = ppl_min
        self.ppl_max      = ppl_max
        self._max_workers = max_workers

    def __call__(self, rows) -> List[Dict[str, Any]]:
        """Parallel-score rows via chat completions; keep rows with PPL in [ppl_min, ppl_max]."""
        scoreable: List[Tuple[int, List[Dict[str, Any]], int]] = []  # (row_idx, messages, n_prompt)
        for i, row in enumerate(rows):
            messages = row.get('messages') or []
            result = _encode_pair(self._tokenizer, messages)
            if result is not None:
                scoreable.append((i, result[0], result[1]))

        if not scoreable:
            return rows

        drop: set = set()
        n_workers = min(self._max_workers, len(scoreable))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_to_meta = {
                pool.submit(_score_one, self._backend, messages): (row_idx, n_prompt)
                for row_idx, messages, n_prompt in scoreable
            }
            for future in as_completed(future_to_meta):
                row_idx, n_prompt = future_to_meta[future]
                try:
                    prompt_logprobs = future.result()
                except Exception:
                    continue
                ppl = _ppl_from_logprobs(prompt_logprobs, n_prompt)
                if ppl is not None and not (self.ppl_min <= ppl <= self.ppl_max):
                    drop.add(row_idx)

        return [row for i, row in enumerate(rows) if i not in drop]
