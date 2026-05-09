# Copyright (c) ModelScope Contributors. All rights reserved.
"""LLM-backed passage condenser.

Delegates compression to a :class:`twinkle.sampler.base.Sampler`. For
each eligible chunk, builds a compression prompt, samples from the
LLM, parses the markdown response into ``## Summary / ## Key Facts /
## More`` sections, and strictly clamps the final output to
``ceil(len(input) / compression_ratio)`` characters via progressive
section-drop + word-boundary truncation.
"""
from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from twinkle_agentic.condenser.base import Condenser
from twinkle_agentic.data_format import Chunk, Chunks

if TYPE_CHECKING:  # only used for type hints, keep runtime deps minimal
    from twinkle.data_format import SamplingParams, Trajectory
    from twinkle.sampler.base import Sampler


def _sampling_params_cls():
    """Lazy import to avoid coupling module import to twinkle.sampler."""
    from twinkle.data_format.sampling import SamplingParams
    return SamplingParams

# Markdown headers emitted by the condenser.
_SUMMARY_HEADER = '## Summary'
_FACTS_HEADER = '## Key Facts'
_MORE_HEADER = '## More'

_DEFAULT_SYSTEM_PROMPT = (
    'You are a precise text compression assistant. Summarize the user'
    ' passage into the required markdown structure without inventing'
    ' any information. Preserve named entities, dates, numbers, and'
    ' factual relations.'
)

_DEFAULT_USER_PROMPT_TEMPLATE = (
    'Compress the passage below into markdown with EXACTLY three'
    ' sections in this order:\n\n'
    '## Summary\n<one or two sentences describing the passage>\n\n'
    '## Key Facts\n<3-5 bullet lines, each starting with "- ">\n\n'
    '## More\n<comma-separated keywords useful for expansion>\n\n'
    'Hard rule: the total output MUST NOT exceed {budget} characters.'
    ' Do not add extra sections, preambles, or closing remarks.\n\n'
    'Passage:\n{text}')


# ---------------------------------------------------------------------------
# ModelCondenser
# ---------------------------------------------------------------------------
class ModelCondenser(Condenser):
    """Condenser that delegates compression to an LLM via a :class:`Sampler`.

    Args:
        sampler: A configured :class:`Sampler`. The sampler must already
            have a ``template`` set so it can encode ``Trajectory``
            inputs. The sampler is reused across chunks (batched).
        compression_ratio: Target factor, must be ``> 1``. For chunks
            that pass ``min_chars``,
            ``len(output) <= ceil(len(input) / compression_ratio)`` is
            strictly enforced via post-sampling truncation (the model
            cannot be trusted to obey a soft word count).
        sampling_params: Override for per-call sampling. Defaults to
            greedy (temperature 0) with ``max_tokens`` derived from the
            budget.
        system_prompt: Override the default system prompt.
        user_prompt_template: Override the default user prompt.
            Supported placeholders: ``{budget}`` and ``{text}``.
        min_chars: Pre-filter. Chunks shorter than this are passed
            through unchanged (the ratio contract does not apply to
            them).
        skip_roles: Roles whose chunks are never compressed.
        rounds: Optional set/list of conversation-turn numbers to
            compress. ``None`` (default) = no round-based filtering;
            when provided, chunks whose ``round`` is not in this set
            are passed through unchanged. Chunks that lack a ``round``
            field are also skipped when this filter is active.
        batch_size: Max chunks per sampler call. Larger values amortize
            LLM prefill / worker-dispatch overhead.
        use_base_model: When ``True``, compression is done WITHOUT the
            currently-synced LoRA adapter (i.e. the frozen base model).
            This breaks the closed-loop "policy compresses its own
            context" drift during RL training — strongly recommended
            when ``sampler`` is also the training policy. The flag is
            forwarded to :meth:`Sampler.sample` as ``use_base_model``;
            samplers that do not support it will raise a
            ``TypeError``.

    The condenser marks every produced chunk with ``raw.condensed=True``
    so :meth:`Chunks.to_trajectory` wraps it in ``<block_N>...</block_N>``.

    Example:
        >>> from twinkle.sampler import vLLMSampler
        >>> sampler = vLLMSampler(model_id='Qwen/Qwen2.5-3B-Instruct',
        ...                       engine_args={'dtype': 'bfloat16'})
        >>> sampler.set_template('qwen2_5')
        >>> cond = ModelCondenser(sampler, compression_ratio=4.0)
        >>> compressed = cond(chunks)
    """

    DEFAULT_SYSTEM_PROMPT: str = _DEFAULT_SYSTEM_PROMPT
    DEFAULT_USER_PROMPT_TEMPLATE: str = _DEFAULT_USER_PROMPT_TEMPLATE

    def __init__(
        self,
        sampler: 'Sampler',
        compression_ratio: float = 4.0,
        *,
        sampling_params: Optional['SamplingParams'] = None,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        min_chars: int = 200,
        skip_roles: Sequence[str] = ('system', 'tool', 'assistant'),
        rounds: Optional[Sequence[int]] = None,
        batch_size: int = 8,
        use_base_model: bool = False,
    ):
        if sampler is None:
            raise ValueError('sampler is required')
        if compression_ratio <= 1.0:
            raise ValueError(
                f'compression_ratio must be > 1, got {compression_ratio}')
        if min_chars < 0:
            raise ValueError(f'min_chars must be >= 0, got {min_chars}')
        if batch_size <= 0:
            raise ValueError(f'batch_size must be >= 1, got {batch_size}')

        tpl = user_prompt_template or self.DEFAULT_USER_PROMPT_TEMPLATE
        if '{budget}' not in tpl or '{text}' not in tpl:
            raise ValueError(
                'user_prompt_template must contain both {budget} and {text}')

        self.sampler = sampler
        self.compression_ratio = float(compression_ratio)
        self.sampling_params = sampling_params
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.user_prompt_template = tpl
        self.min_chars = min_chars
        self.skip_roles = tuple(skip_roles)
        self.rounds = set(rounds) if rounds is not None else None
        self.batch_size = batch_size
        self.use_base_model = bool(use_base_model)

    # ------------------------------------------------------------------
    # entry
    # ------------------------------------------------------------------
    def __call__(self, chunks: Chunks, **kwargs) -> Chunks:
        out: List[Chunk] = list(chunks.chunks)
        jobs: List[Tuple[int, Chunk, int]] = []
        for i, c in enumerate(chunks.chunks):
            if not self._should_condense(c):
                continue
            text = c['content']
            budget = max(1, math.ceil(len(text) / self.compression_ratio))
            jobs.append((i, c, budget))

        for start in range(0, len(jobs), self.batch_size):
            batch = jobs[start:start + self.batch_size]
            trajectories = [
                self._build_trajectory(c['content'], b) for _, c, b in batch
            ]
            sp = self._build_sampling_params(max(b for _, _, b in batch))
            sample_kwargs: Dict[str, Any] = {'sampling_params': sp}
            if self.use_base_model:
                sample_kwargs['use_base_model'] = True
            responses = self.sampler.sample(trajectories, **sample_kwargs)
            if len(responses) != len(batch):
                raise RuntimeError(
                    f'sampler returned {len(responses)} responses for '
                    f'{len(batch)} inputs')
            for (i, c, budget), resp in zip(batch, responses):
                raw_text = self._pick_decoded(resp)
                compressed = self._postprocess(raw_text, budget, c['content'])
                out[i] = self._mark_condensed(c, compressed)

        return Chunks(chunks=out)

    # ------------------------------------------------------------------
    # selection policy
    # ------------------------------------------------------------------
    def _should_condense(self, chunk: Chunk) -> bool:
        if chunk.get('type') != 'text':
            return False
        if chunk.get('role') in self.skip_roles:
            return False
        if self.rounds is not None and chunk.get('round') not in self.rounds:
            return False
        content = chunk.get('content')
        if not isinstance(content, str) or not content:
            return False
        if len(content) < self.min_chars:
            return False
        raw = chunk.get('raw') or {}
        if isinstance(raw, dict):
            # Skip chunker-emitted reasoning / tool_call text chunks.
            if raw.get('kind'):
                return False
            # Idempotency — don't re-condense already condensed chunks.
            if raw.get('condensed'):
                return False
        return True

    @staticmethod
    def _mark_condensed(chunk: Chunk, content: str) -> Chunk:
        new: Dict[str, Any] = dict(chunk)
        raw = dict(new.get('raw') or {})
        raw.setdefault('original', new.get('content', ''))
        new['content'] = content
        raw['condensed'] = True
        new['raw'] = raw
        return new  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # prompt construction
    # ------------------------------------------------------------------
    def _build_trajectory(self, text: str, budget: int) -> 'Trajectory':
        # Use str.replace to avoid .format() breaking on braces in text.
        user = (self.user_prompt_template
                .replace('{budget}', str(budget))
                .replace('{text}', text))
        return {  # type: ignore[return-value]
            'messages': [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': user},
            ],
        }

    def _build_sampling_params(self, budget: int) -> 'SamplingParams':
        if self.sampling_params is not None:
            return self.sampling_params
        # Rough heuristic: ~1 token per 2-3 English chars + headroom.
        max_new = max(64, int(budget * 0.8) + 64)
        return _sampling_params_cls()(temperature=0.0, max_tokens=max_new)

    # ------------------------------------------------------------------
    # response parsing & strict-budget clamping
    # ------------------------------------------------------------------
    @staticmethod
    def _pick_decoded(response) -> str:
        seqs = getattr(response, 'sequences', None) or []
        if not seqs:
            return ''
        decoded = getattr(seqs[0], 'decoded', None)
        return decoded or ''

    def _postprocess(self, raw: str, budget: int, original: str) -> str:
        text = _strip_code_fences(raw).strip()
        sections = _parse_markdown_sections(text)
        formatted = _format_sections(sections, fallback=text)
        if formatted and len(formatted) <= budget:
            return formatted
        # Progressive drop on a *copy*: More → Key Facts → Summary. Keep
        # the original ``sections`` intact for the body-only fallback.
        remaining = dict(sections)
        for drop in ('more', 'facts', 'summary'):
            remaining.pop(drop, None)
            reduced = _format_sections(remaining, fallback='')
            if reduced and len(reduced) <= budget:
                return reduced
        # Even "## Summary\n<body>" cannot fit — the header alone eats the
        # budget. Clamp the most informative *body* (no header) so the user
        # still gets meaningful content instead of dangling hash marks.
        for key in ('summary', 'facts', 'more'):
            body = sections.get(key)
            if body:
                clamped = _clamp_to_budget(body, budget)
                if clamped:
                    return clamped
        # No parsable sections at all — clamp the stripped raw text
        # (or the original passage as a last resort).
        return _clamp_to_budget(text or original, budget)


# ---------------------------------------------------------------------------
# helpers (pure functions)
# ---------------------------------------------------------------------------
_SECTION_RE = re.compile(
    r'^[ \t]*#{1,6}[ \t]*(?P<header>summary|key[ \t]*facts?|more)[ \t]*$',
    re.IGNORECASE | re.MULTILINE,
)
_SECTION_KEYS = {
    'summary': 'summary',
    'key fact': 'facts',
    'key facts': 'facts',
    'keyfact': 'facts',
    'keyfacts': 'facts',
    'more': 'more',
}
_HEADER_ORDER: Tuple[Tuple[str, str], ...] = (
    ('summary', _SUMMARY_HEADER),
    ('facts', _FACTS_HEADER),
    ('more', _MORE_HEADER),
)


def _parse_markdown_sections(text: str) -> Dict[str, str]:
    """Extract ``{summary, facts, more}`` sections from ``text``.

    Last-writer wins on duplicate headers (e.g. the model repeats
    ``## Summary`` twice — we keep the later body).
    """
    if not text:
        return {}
    matches = list(_SECTION_RE.finditer(text))
    out: Dict[str, str] = {}
    for i, m in enumerate(matches):
        header = re.sub(r'\s+', ' ', m.group('header').strip().lower())
        key = _SECTION_KEYS.get(header)
        if key is None:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            out[key] = body
    return out


def _format_sections(sections: Dict[str, str], *, fallback: str = '') -> str:
    parts = [
        f'{header}\n{sections[key]}' for key, header in _HEADER_ORDER
        if sections.get(key)
    ]
    if parts:
        return '\n\n'.join(parts)
    return fallback


def _strip_code_fences(text: str) -> str:
    """Unwrap a leading/trailing triple-backtick fence if present."""
    stripped = text.strip()
    m = re.match(r'^```[a-zA-Z]*\s*\n(.*?)\n```\s*$', stripped, re.DOTALL)
    return m.group(1) if m else text


def _clamp_to_budget(text: str, budget: int) -> str:
    """Word-boundary truncate ``text`` to at most ``budget`` chars."""
    if len(text) <= budget:
        return text
    if budget <= 0:
        return ''
    cut = text[:budget]
    sp = cut.rfind(' ')
    trimmed = cut[:sp] if sp >= budget // 2 else cut
    return trimmed.rstrip() or cut
