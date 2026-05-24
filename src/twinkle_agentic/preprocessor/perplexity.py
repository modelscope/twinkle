# Copyright (c) ModelScope Contributors. All rights reserved.
import math
from typing import Any, Dict, List, Optional, Tuple

from twinkle.data_format import InputFeature, SamplingParams
from twinkle.preprocessor import Preprocessor
from twinkle.sampler.base import Sampler

# ── Defaults ──────────────────────────────────────────────────────────────────

# PPL range that indicates the data is a good fit for the current model.
# Too low  → trivially memorized / degenerate output.
# Too high → out-of-distribution, garbled, or badly formatted.
_DEFAULT_PPL_MIN = 2.0
_DEFAULT_PPL_MAX = 100.0

# Ignore response tokens shorter than this (stats unreliable)
_MIN_RESPONSE_TOKENS = 5

# Reusable sampling params: generate no tokens, only score prompt logprobs.
# max_tokens=0 triggers vLLMSampler's logprobs_only path.
_SCORE_SP = SamplingParams(max_tokens=0, prompt_logprobs=1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _encode_pair(
    sampler: Sampler,
    messages: List[Dict[str, Any]],
) -> Optional[Tuple[InputFeature, int]]:
    """Encode (prompt, full_sequence) and return (full_feat, prompt_length).

    Returns None if the trajectory has no assistant turn or encoding fails.
    """
    # Find last assistant message index
    last_asst = next(
        (i for i in range(len(messages) - 1, -1, -1)
         if isinstance(messages[i], dict) and messages[i].get('role') == 'assistant'),
        None,
    )
    if last_asst is None:
        return None

    prompt_traj = {'messages': messages[:last_asst]}
    full_traj   = {'messages': messages}

    try:
        prompt_feat = sampler.encode_trajectory(prompt_traj, add_generation_prompt=True)
        full_feat   = sampler.encode_trajectory(full_traj,   add_generation_prompt=False)
    except Exception:
        return None

    n_prompt   = len(prompt_feat['input_ids'])
    n_response = len(full_feat['input_ids']) - n_prompt
    if n_response < _MIN_RESPONSE_TOKENS:
        return None
    return full_feat, n_prompt


def _ppl_from_logprobs(
    prompt_logprobs: List[Optional[float]],
    n_prompt: int,
) -> Optional[float]:
    """Compute PPL from a response-token slice of prompt_logprobs."""
    response_lps = [lp for lp in prompt_logprobs[n_prompt:] if lp is not None]
    if len(response_lps) < _MIN_RESPONSE_TOKENS:
        return None
    avg_nll = -sum(response_lps) / len(response_lps)
    return math.exp(avg_nll)


# ── Preprocessor ─────────────────────────────────────────────────────────────

class PerplexityFilter(Preprocessor):
    """Filter dataset rows by model perplexity on the assistant response.

    The sampler scores the assistant's tokens conditioned on the prompt
    (prompt_logprobs mode, no tokens generated). PPL outside [ppl_min, ppl_max]
    is treated as low quality:
      - PPL too low  → trivial / highly memorized content
      - PPL too high → out-of-distribution, garbled, or badly formatted

    Requirements:
      - ``sampler.set_template(...)`` must be called before using this filter.
      - Works with any Sampler subclass that supports ``sample()`` with
        ``SamplingParams(max_tokens=0, prompt_logprobs=1)``.
    """

    def __init__(
        self,
        sampler: Sampler,
        ppl_min: float = _DEFAULT_PPL_MIN,
        ppl_max: float = _DEFAULT_PPL_MAX,
    ):
        self.sampler = sampler
        self.ppl_min = ppl_min
        self.ppl_max = ppl_max

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = self.ppl_filter(rows)
        rows = self.map_row_to_col(rows)
        return rows

    def ppl_filter(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score a batch via one sampler call; keep rows with PPL in [ppl_min, ppl_max]."""
        # Encode each row; track which rows are scoreable
        scoreable: List[Tuple[int, InputFeature, int]] = []  # (row_idx, full_feat, n_prompt)
        for i, row in enumerate(rows):
            messages = row.get('messages') or []
            result = _encode_pair(self.sampler, messages)
            if result is not None:
                scoreable.append((i, result[0], result[1]))

        if not scoreable:
            return rows

        # One batched sampler call for all scoreable rows
        try:
            responses = self.sampler.sample(
                [s[1] for s in scoreable],
                sampling_params=_SCORE_SP,
            )
        except Exception:
            return rows  # pass through on sampler error

        # Determine which rows to drop
        drop = set()
        for (row_idx, _, n_prompt), resp in zip(scoreable, responses):
            lps = resp.prompt_logprobs
            if not lps:
                continue
            ppl = _ppl_from_logprobs(lps, n_prompt)
            if ppl is None:
                continue
            if not (self.ppl_min <= ppl <= self.ppl_max):
                drop.add(row_idx)

        return [row for i, row in enumerate(rows) if i not in drop]
