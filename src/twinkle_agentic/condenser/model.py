# Copyright (c) ModelScope Contributors. All rights reserved.
"""LLM-backed passage condenser.

Pipeline
--------
``Chunks`` → filter eligible chunks → batched ``Sampler.sample(...)`` →
strip code fences → boundary-aware character-budget clamp → ``Chunks``
with ``raw.condensed=True`` (so :meth:`Chunks.to_trajectory` later
wraps them in ``<block_N>``).

The compression prompt asks for up to three markdown sections
(``## Summary / ## Key Facts / ## More``) written in **telegraphic
style** (no articles / copulas / filler) with per-section length
hints. Telegraphic output is ~2–3× denser than natural-prose summaries
and is critical under tight compression ratios. The output is **not**
parsed — sections pass through verbatim. The character budget is a
safety net only; the prompt encourages the model to self-shorten and
drop ``## More`` first, so truncation rarely needs to fire.
"""
from __future__ import annotations

import math
import re
from typing import (TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple)

from twinkle_agentic.condenser.base import Condenser
from twinkle_agentic.data_format import Chunk, Chunks

if TYPE_CHECKING:  # only used for type hints, keep runtime deps minimal
    from twinkle.data_format import SamplingParams, Trajectory  # noqa: F401
    from twinkle.sampler.base import Sampler  # noqa: F401


_SECTION_SCHEMA = (
    'Purpose: produce a compact retrieval index. The reader skims it to'
    ' decide whether — and on what topic — to fetch the full text.'
    ' Every token must carry unique, non-recoverable information.\n\n'
    'Output EXACTLY this skeleton — never rename, merge, or add sections;'
    ' stop immediately after the Topics line:\n\n'
    '## Summary\n'
    '<≤{summary_words} words. Subject + full naming hierarchy'
    ' (family→genus→species; person→role→era; org→function→head).'
    ' Identity and classification ONLY.\n'
    ' PROHIBITED in Summary: any number, rank ("7th largest",'
    ' "most populous", "oldest"), size, area, range, or border fact.'
    ' Every such item must move to Key Facts, no exceptions.>\n\n'
    '## Key Facts\n'
    '<0–{max_bullets} bullets, ≤{bullet_words} words each,'
    ' non-redundant with Summary. Priority:\n'
    ' (1) Verbatim numbers copied from the passage'
    '     ("3287263 km² area", "7516.6 km coastline").\n'
    ' (2) "N <label>" counts when passage enumerates ≥3 same-kind items.\n'
    '     COUNTING RULE: before writing N, re-read the passage and count'
    '     listed entities one by one; write only the verified integer.\n'
    '     LISTING RULE: never name the entities — write'
    '     "6 land-border countries", never "borders: Pakistan, China...".\n'
    ' (3) Short categorical facts not inferable from identity alone.\n'
    ' DISTINCT-FACT RULE: if the passage states two rankings or counts'
    ' with different scopes (e.g. "2nd-most populous country" globally vs.'
    ' "most populous democracy"), emit a separate bullet for each —'
    ' never conflate or drop either one.\n'
    ' Skip the bullet rather than pad. Never restate Summary.>\n\n'
    '## More\n'
    'Topics: <tag>, <tag>, <tag>, <tag>.\n'
    'Each tag is a categorical theme answering "what query would send a'
    ' reader to this source?" (e.g. "demographic scale", "moth taxonomy").'
    ' Never use entity names as tags. Always emit this line.'
)

_STYLE_TELEGRAPHIC = (
    'Telegraphic style — maximize signal per character.\n'
    'Drop: articles (a/an/the), copulas (is/are/was/were),'
    ' prepositions inferable from context, filler phrases'
    ' ("it is notable that", "which is", "there are").\n'
    'Keep: entities, numbers, dates, locations, relations.\n'
    'Compress: colon for "is/has", comma for "and/which",'
    ' "~" for approximations, standard SI units.\n'
    'Never invent facts; copy every number verbatim.'
    ' End on a complete token.'
)

_WORKED_EXAMPLE = (
    'Worked examples — replicate this exact format.'
    ' All outputs end immediately after the Topics line.\n\n'
    'Example 1 (enumeration → counts):\n'
    'Input: "Germany is a Central European country. It shares land'
    ' borders with France, Belgium, Netherlands, Denmark, Poland,'
    ' Czech Republic, Austria, and Switzerland. Its four largest cities'
    ' are Berlin, Hamburg, Munich, and Cologne. Berlin, the capital,'
    ' has about 3.7 million inhabitants."\n'
    'Output:\n'
    '## Summary\n'
    'Germany: Central European country, Berlin capital.\n\n'
    '## Key Facts\n'
    '- 8 land-border countries.\n'
    '- 4 largest cities.\n'
    '- Capital pop.: ~3.7M.\n\n'
    '## More\n'
    'Topics: central-European geography, international borders,'
    ' major cities, capital demographics.\n\n'
    'Example 2 (single-species taxonomy → minimal Key Facts):\n'
    'Input: "Eutrapela is a genus of moth in the Geometridae family.'
    ' It contains only one species, Eutrapela clemataria, the'
    ' curve-toothed geometer moth, found in North America from'
    ' Nova Scotia to Florida, west to Texas and north to Saskatchewan.'
    ' Habitat: deciduous and mixed woodlands."\n'
    'Output:\n'
    '## Summary\n'
    'Eutrapela: Geometridae moth genus, E. clemataria species.\n\n'
    '## Key Facts\n'
    '- 4 range-endpoint regions.\n'
    '- Deciduous + mixed woodland habitat.\n\n'
    '## More\n'
    'Topics: moth taxonomy, species distribution, habitat classification,'
    ' North American biogeography.\n\n'
    'Example 3 (scope-distinct rankings + mixed border types'
    ' — demonstrates COUNTING RULE, LISTING RULE, DISTINCT-FACT RULE):\n'
    'Input: "Brazil is the largest country in South America and the'
    ' fifth-largest in the world. It is the most populous'
    ' Portuguese-speaking country, with 215 million people. Brazil'
    ' shares land borders with Argentina, Bolivia, Colombia, Guyana,'
    ' Paraguay, Peru, Suriname, Uruguay, and Venezuela.'
    ' It has an Atlantic coastline of 7491 km."\n'
    '-- Counting check: Argentina, Bolivia, Colombia, Guyana, Paraguay,'
    ' Peru, Suriname, Uruguay, Venezuela = 9. --\n'
    'Output:\n'
    '## Summary\n'
    'Brazil: South American republic, Brasília capital.\n\n'
    '## Key Facts\n'
    '- Largest in South America; 5th-largest globally.\n'
    '- 215M people; most populous Portuguese-speaking country.\n'
    '- 9 land-border countries.\n'
    '- 7491 km Atlantic coastline.\n\n'
    '## More\n'
    'Topics: South American geography, area rankings,'
    ' population scale, coastal extent.'
)

_LENGTH_CONTRACT = (
    'Length: aim for ~{soft_budget} chars; hard cap {budget} chars.'
    ' Shorter is better — stop once all signal is captured; never pad.'
)

DEFAULT_SYSTEM_PROMPT = '\n\n'.join([
    'You are a precise text compression assistant.',
    _SECTION_SCHEMA,
    _STYLE_TELEGRAPHIC,
])

DEFAULT_USER_PROMPT_TEMPLATE = '\n\n'.join([
    'Compress the passage below per the schema.',
    _WORKED_EXAMPLE,
    _LENGTH_CONTRACT,
    'Passage:\n{text}',
])


# A (chunk_index, chunk, char_budget) triple marking one compression job.
_Job = Tuple[int, Chunk, int]


# ---------------------------------------------------------------------------
# ModelCondenser
# ---------------------------------------------------------------------------
class ModelCondenser(Condenser):
    """Compressor that delegates summarization to an LLM via a :class:`Sampler`.

    Args:
        sampler: Configured :class:`Sampler` with a template set.
        compression_ratio: Target factor (> 1). Output length is clamped
            to ``ceil(len(input) / compression_ratio)`` per chunk.
        sampling_params: Override for per-call sampling; when ``None`` a
            greedy config is derived from the max budget in the batch.
        system_prompt: Override for the system prompt. May contain
            ``{summary_words}``, ``{max_bullets}``, ``{bullet_words}``
            (all substituted per-chunk with budget-scaled word/bullet
            caps).
        user_prompt_template: Override the user prompt. Must contain
            ``{budget}`` and ``{text}``. ``{soft_budget}``,
            ``{summary_words}``, ``{max_bullets}`` and
            ``{bullet_words}`` are optional. Scaling formulas:
            ``soft_budget = int(budget*0.85)``;
            ``summary_words = clamp(budget // 15, 8, 25)``;
            ``max_bullets = clamp(budget // 75, 2, 5)``;
            ``bullet_words = clamp(budget // 25, 6, 12)``.
        min_chars: Pre-filter; chunks shorter than this pass through.
        min_budget_chars: Minimum character budget for any compression.
            When ``ceil(len / compression_ratio)`` falls below this,
            the budget is raised to this floor so short-but-eligible
            passages keep room for all three sections. Default ``250``
            is large enough that ~200-char passages pass through
            almost unclamped, preserving Summary + Key Facts + More;
            for longer passages the ratio still dominates. Pass ``1``
            to disable the floor and enforce strict ratio everywhere.
        template: Optional :class:`Template`. When provided, its
            ``tokenizer.all_special_tokens`` are stripped from every
            decoded response before length-clamping, preventing
            protocol tokens (``<|im_end|>``, ``<|eot_id|>``, ``</s>``,
            ...) from leaking into the compressed output. When
            omitted, falls back to ``sampler.template`` if available.
        skip_roles: Roles whose chunks are never compressed.
        skip_pattern: Optional regex (compiled with ``re.MULTILINE``).
            Any chunk whose ``content`` has a match for this pattern
            is passed through unchanged, regardless of length / ratio.
            Uses :func:`re.search` semantics, so anchor with ``^`` /
            start-of-string if you want boundary-matching only (e.g.
            ``r'^Question:'`` to preserve the question prefix in a
            HotpotQA-style user message). ``None`` disables the filter.
        rounds: Optional set of conversation turn indices to compress.
            ``None`` = no round-based filter; chunks lacking a ``round``
            field are skipped when this filter is active.
        batch_size: Max chunks per sampler call. Partial batches are
            padded with a duplicate of the last trajectory so that
            distributed samplers (DP slice) always receive a full batch.
        use_base_model: When ``True``, forwards ``use_base_model=True``
            to :meth:`Sampler.sample` so compression bypasses any
            currently-synced LoRA adapter — strongly recommended when
            the sampler is also the training policy.

    Compressed chunks are flagged ``raw.condensed=True``; a subsequent
    :meth:`Chunks.to_trajectory` call wraps them in ``<block_N>``.

    Example::

        >>> from twinkle.sampler import vLLMSampler
        >>> sampler = vLLMSampler(model_id='Qwen/Qwen2.5-3B-Instruct',
        ...                       engine_args={'dtype': 'bfloat16'})
        >>> sampler.set_template('qwen2_5')
        >>> cond = ModelCondenser(sampler, compression_ratio=4.0)
        >>> compressed = cond(chunks)
    """

    # Back-compat aliases so external callers can still override at the
    # class level.
    DEFAULT_SYSTEM_PROMPT: str = DEFAULT_SYSTEM_PROMPT
    DEFAULT_USER_PROMPT_TEMPLATE: str = DEFAULT_USER_PROMPT_TEMPLATE

    def __init__(
        self,
        sampler: 'Sampler',
        compression_ratio: float = 4.0,
        *,
        sampling_params: Optional['SamplingParams'] = None,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        min_chars: int = 200,
        min_budget_chars: int = 250,
        template: Optional[Any] = None,
        skip_roles: Sequence[str] = ('system', 'tool', 'assistant'),
        skip_pattern: Optional[str] = None,
        rounds: Optional[Sequence[int]] = None,
        batch_size: int = None,
        use_base_model: bool = False,
    ):
        if sampler is None:
            raise ValueError('sampler is required')
        if compression_ratio <= 1.0:
            raise ValueError(
                f'compression_ratio must be > 1, got {compression_ratio}')
        if min_chars < 0:
            raise ValueError(f'min_chars must be >= 0, got {min_chars}')
        if min_budget_chars < 1:
            raise ValueError(
                f'min_budget_chars must be >= 1, got {min_budget_chars}')
        if batch_size is not None and batch_size <= 0:
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
        self.min_budget_chars = int(min_budget_chars)
        self.template = template
        self.skip_roles = tuple(skip_roles)
        # Pre-compile the skip-regex once; store ``None`` when disabled so
        # ``_should_condense`` can short-circuit without a re-check.
        self.skip_re: Optional[re.Pattern] = (
            re.compile(skip_pattern, re.MULTILINE)
            if skip_pattern else None)
        self.rounds = set(rounds) if rounds is not None else None
        self.batch_size = batch_size
        self.use_base_model = bool(use_base_model)
        self._special_tokens_cache: Optional[Tuple[str, ...]] = None

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------
    def __call__(self, chunks: Chunks, **_kwargs: Any) -> Chunks:
        out: List[Chunk] = list(chunks.chunks)
        jobs = self._collect_jobs(out)
        if not jobs:
            return Chunks(chunks=out)

        batch_size = self.batch_size or len(jobs)
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start:start + batch_size]
            responses = self._sample_batch(batch)
            for (idx, chunk, budget), resp in zip(batch, responses):
                text = self._postprocess(
                    _decoded(resp), budget, chunk['content'])
                out[idx] = _mark_condensed(chunk, text)
        return Chunks(chunks=out)

    # ------------------------------------------------------------------
    # eligibility + job collection
    # ------------------------------------------------------------------
    def _collect_jobs(self, chunks: Sequence[Chunk]) -> List[_Job]:
        jobs: List[_Job] = []
        for i, c in enumerate(chunks):
            if not self._should_condense(c):
                continue
            content = c['content']
            budget = max(
                self.min_budget_chars,
                math.ceil(len(content) / self.compression_ratio))
            jobs.append((i, c, max(1, budget)))
        return jobs

    def _should_condense(self, chunk: Chunk) -> bool:
        if chunk.get('type') != 'text':
            return False
        if chunk.get('role') in self.skip_roles:
            return False
        if self.rounds is not None and chunk.get('round') not in self.rounds:
            return False
        content = chunk.get('content')
        if not isinstance(content, str) or len(content) < self.min_chars:
            return False
        if self.skip_re is not None and self.skip_re.search(content):
            return False
        raw = chunk.get('raw') or {}
        if isinstance(raw, dict):
            # Skip chunker-emitted reasoning / tool_call text chunks.
            if raw.get('kind'):
                return False
            # Idempotent — never re-compress something already compressed.
            if raw.get('condensed'):
                return False
        return True

    # ------------------------------------------------------------------
    # batched sampling
    # ------------------------------------------------------------------
    def _sample_batch(self, batch: Sequence[_Job]) -> List[Any]:
        """Dispatch one batch to the sampler, padded to ``batch_size``.

        Distributed samplers slice inputs across DP workers and can
        mis-behave when the final batch is smaller than ``batch_size``;
        we pad with a duplicate of the last trajectory and trim the
        matching extra responses here.
        """
        trajectories = [
            self._build_trajectory(chunk['content'], budget)
            for _, chunk, budget in batch
        ]
        actual = len(trajectories)
        device_mesh = getattr(self.sampler, 'device_mesh', None)
        min_batch_size = (
            device_mesh.data_world_size if device_mesh is not None else 1)
        if actual < min_batch_size:
            trajectories.extend(
                [trajectories[-1]] * (min_batch_size - actual))

        sp = self._sampling_params_for(max(b for _, _, b in batch))
        kwargs: Dict[str, Any] = {'sampling_params': sp}
        if self.use_base_model:
            kwargs['use_base_model'] = True
        responses = self.sampler.sample(trajectories, **kwargs)
        # Coerce to list (some samplers may return tuples) and drop
        # padding responses so downstream ``zip`` aligns with ``batch``.
        return list(responses)[:actual]

    def _build_trajectory(self, text: str, budget: int) -> 'Trajectory':
        soft_budget = max(1, int(budget * 0.85))
        summary_words = max(8, min(25, budget // 15))
        max_bullets = max(2, min(5, budget // 75))
        bullet_words = max(6, min(12, budget // 25))
        replacements = (
            ('{soft_budget}', str(soft_budget)),
            ('{summary_words}', str(summary_words)),
            ('{max_bullets}', str(max_bullets)),
            ('{bullet_words}', str(bullet_words)),
            ('{budget}', str(budget)),
        )
        system = self.system_prompt
        user = self.user_prompt_template
        for k, v in replacements:
            system = system.replace(k, v)
            user = user.replace(k, v)
        user = user.replace('{text}', text)
        return {  # type: ignore[return-value]
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
        }

    def _sampling_params_for(self, budget: int) -> 'SamplingParams':
        if self.sampling_params is not None:
            return self.sampling_params
        from twinkle.data_format.sampling import SamplingParams
        # Rough heuristic: ~1 token per 2–3 English chars + headroom.
        max_new = max(64, int(budget * 0.8) + 64)
        return SamplingParams(temperature=0.0, max_tokens=max_new)

    # ------------------------------------------------------------------
    # postprocess
    # ------------------------------------------------------------------
    def _postprocess(self, raw: str, budget: int, original: str) -> str:
        """Strip code fences + tokenizer special tokens, clamp to
        ``budget``, guard against degenerate output.

        When the clamp leaves only markdown markers (e.g. ``'##'`` at an
        extreme budget), fall back to clamping the original passage so
        callers never see empty or meaningless markers.
        """
        text = _strip_special_tokens(
            _strip_code_fences(raw), self._get_special_tokens()).strip()
        if not text:
            return _clamp_to_budget(original, budget)
        clamped = _clamp_to_budget(text, budget) if len(text) > budget else text
        if not _has_alnum(clamped):
            return _clamp_to_budget(original, budget)
        return clamped

    def _get_special_tokens(self) -> Tuple[str, ...]:
        """Return protocol tokens to strip from decoded output (cached).

        Resolution order:

        1. ``self.template.tokenizer`` — explicit template passed to
           ``__init__``. Preferred in distributed setups where
           ``sampler.template`` on the driver is a proxy and may be
           ``None``.
        2. ``self.sampler.template.tokenizer`` — best-effort fallback
           for single-process use.
        3. Empty tuple — no stripping (safe no-op).

        Uses ``tokenizer.all_special_tokens`` when available so the
        full eos/bos/pad/unk/sep/cls/mask/additional set is covered
        in one shot; this means ChatML (``<|im_end|>``), Llama
        (``<|eot_id|>``), T5 (``</s>``) etc. are all handled without
        per-model hard-coding.
        """
        if self._special_tokens_cache is not None:
            return self._special_tokens_cache
        tpl = self.template or getattr(self.sampler, 'template', None)
        tokenizer = getattr(tpl, 'tokenizer', None) if tpl is not None else None
        tokens: List[str] = []
        if tokenizer is not None:
            extras = getattr(tokenizer, 'all_special_tokens', None) or []
            if extras:
                tokens.extend(
                    t for t in extras
                    if isinstance(t, str) and t and not t.isspace())
            else:
                for attr in ('eos_token', 'pad_token', 'bos_token'):
                    t = getattr(tokenizer, attr, None)
                    if isinstance(t, str) and t:
                        tokens.append(t)
        # Order-preserving dedupe.
        self._special_tokens_cache = tuple(dict.fromkeys(tokens))
        return self._special_tokens_cache


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------
_CODE_FENCE_RE = re.compile(r'^```[a-zA-Z]*\s*\n(.*?)\n```\s*$', re.DOTALL)
_SENT_PUNCT = ('.', '!', '?', '。', '！', '？')
_WS_TAILS = (' ', '\n', '\t')


def _decoded(response: Any) -> str:
    """Extract the first decoded sequence, or ``''`` on empty/malformed input."""
    seqs = getattr(response, 'sequences', None) or []
    if not seqs:
        return ''
    return getattr(seqs[0], 'decoded', None) or ''


def _mark_condensed(chunk: Chunk, content: str) -> Chunk:
    """Return a shallow copy of ``chunk`` with compressed ``content``
    and ``raw.condensed=True`` (preserving any original content under
    ``raw.original`` so a future :class:`ExtractCondensed` call can
    recover the full text).
    """
    new: Dict[str, Any] = dict(chunk)
    raw = dict(new.get('raw') or {})
    raw.setdefault('original', new.get('content', ''))
    raw['condensed'] = True
    new['content'] = content
    new['raw'] = raw
    return new  # type: ignore[return-value]


def _strip_code_fences(text: str) -> str:
    """Unwrap a leading/trailing triple-backtick fence if present."""
    stripped = text.strip()
    m = _CODE_FENCE_RE.match(stripped)
    return m.group(1) if m else text


def _strip_special_tokens(text: str, tokens: Sequence[str]) -> str:
    """Remove tokenizer special tokens that leaked through decode.

    ``tokens`` is typically ``tokenizer.all_special_tokens`` from the
    template's tokenizer (see :meth:`ModelCondenser._get_special_tokens`).
    Uses literal :meth:`str.replace` rather than a regex so we only
    strip registered protocol markers and never legitimate passage
    content that happens to look like ``<|...|>``.
    """
    for tok in tokens:
        if tok and tok in text:
            text = text.replace(tok, '')
    return text


def _has_alnum(text: str) -> bool:
    """True iff ``text`` contains at least one alphanumeric character.

    Used to detect degenerate clamp outputs like ``'##'`` or ``'- '``
    that are pure markdown markers with no actual words.
    """
    return any(ch.isalnum() for ch in text)


def _clamp_to_budget(text: str, budget: int) -> str:
    """Clamp ``text`` to at most ``budget`` chars on the cleanest boundary.

    Preference order (each candidate must land past ``budget // 2``):

      1. Sentence punctuation (``. ! ? 。 ！ ？``) followed by whitespace
         — either inside the cut, OR at the very end of the cut when
         the next char in the full text is whitespace / EOT. This
         excludes mid-token cuts like the ``.`` in ``1.2`` / ``e.g.``.
      2. Newline — paragraph / bullet boundary.
      3. Plain space — word boundary fallback.
      4. Hard cut when none of the above fire far enough in.
    """
    if budget <= 0:
        return ''
    if len(text) <= budget:
        return text
    cut = text[:budget]
    min_keep = budget // 2

    sent_end = _find_sentence_end(cut, text, budget, min_keep)
    if sent_end >= 0:
        return cut[:sent_end].rstrip()

    nl = cut.rfind('\n')
    if nl >= min_keep:
        return cut[:nl].rstrip()

    sp = cut.rfind(' ')
    if sp >= min_keep:
        return cut[:sp].rstrip()

    return cut.rstrip() or cut


def _find_sentence_end(
        cut: str, text: str, budget: int, min_keep: int) -> int:
    """Position just past a sentence-ending punct, or ``-1`` if none.

    A sentence end is a ``_SENT_PUNCT`` char followed by whitespace. The
    whitespace may be inside ``cut`` OR be the first char after the cut
    (``text[budget]``), so a period at the very end of ``cut`` is
    accepted only when the text continues with whitespace / EOT and
    never mid-token.
    """
    best = -1
    # Case 1: "<punct><ws>" inside cut.
    for punct in _SENT_PUNCT:
        for ws in _WS_TAILS:
            idx = cut.rfind(punct + ws)
            if idx >= min_keep and idx + len(punct) > best:
                best = idx + len(punct)
    # Case 2: "<punct>" at end of cut, next char is ws or EOT.
    next_char = text[budget:budget + 1]
    if next_char == '' or next_char in _WS_TAILS:
        for punct in _SENT_PUNCT:
            if cut.endswith(punct):
                pos = len(cut) - len(punct)
                if pos >= min_keep and pos + len(punct) > best:
                    best = pos + len(punct)
                break
    return best
