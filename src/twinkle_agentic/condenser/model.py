# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

from twinkle_agentic.condenser.base import Condenser
from twinkle_agentic.data_format import Chunk, Chunks

if TYPE_CHECKING:
    from twinkle.data_format import SamplingParams, Trajectory  # noqa: F401
    from twinkle.sampler.base import Sampler  # noqa: F401

_SECTION_SCHEMA = """You are a text compression assistant. A downstream model will read your compressed output to decide whether the detail it needs is inside this block; if yes, it will fetch and read the original passage.

Downstream model workflow:
Read your compressed output -> Decide whether needed info is in this block -> If yes -> Fetch original.

Therefore your compression MUST NOT lose major information from the source.

Output format:

```text
## Summary
Overview plus facts STRONGLY RELATED to the Query, stated explicitly.

## More
A collapsed index; expansion required to see specific information.
```

Rules:
1. Telegraphic style — drop function words ("the", "a", "is", "are", "of", ...); colons and commas mean "is" / "has".
    * Exception: KEEP role-tagging verb+preposition phrases verbatim ("published by X", "written by X", "directed by X", "starring X", "founded by X", "created by X", "composed by X", "produced by X", "based on X", "adapted from X"). Collapsing these to a bare name loses the relation role (author vs publisher vs director) that the downstream question may hinge on.
2. Summary MUST contain the passage's primary topic + 2–4 concrete core facts drawn from the source (entities, numbers, dates, relations). If a Query is given, order Query-relevant facts first, but STILL include other core facts within the budget. A Query is an ORDERING HINT, NOT a filter.
3. Summary MUST NOT be meta-commentary about the Query. Forbidden patterns: "no X mention", "Query info: absent", "passage covers Y only", "does not contain ...", "no relevant info", or summaries that are only abstract category words like "structure/order/usage" with no facts. If the passage is unrelated to the Query, you still summarize the passage normally.
4. More is an INDEX of category keywords, NOT inline data. Enumerate what CAN be recovered from the source (e.g. "birthplace, death place, age"); do NOT paste dates/numbers/names inline. Make sure all category of useful facts are introduced here.
5. Output language MUST match the source language.
6. Do NOT fabricate. Do NOT omit major information. Any fact not in the source MUST NOT appear in your output.

Example:

Source:
```text
Marie Curie (7 Nov 1867 – 4 Jul 1934), born Maria Sklodowska in Warsaw (then Russian Poland); parents were teachers. Barred from Polish universities, she and her sister agreed to take turns funding each other's overseas study.

In 1891 Marie reached Paris and enrolled at the Sorbonne, earning a physics degree (1893) and a mathematics degree (1894), becoming the school's first female physics lecturer. In 1895 she married French physicist Pierre Curie; they spent the rest of their lives on radioactivity research.

In July 1898 she discovered polonium, named after her homeland Poland; in December she and Pierre announced the discovery of radium. She coined "radioactivity" and showed it is an atomic property, not a chemical reaction.

In 1903 she shared the Nobel Prize in Physics with Pierre and Henri Becquerel. In 1911 she alone won the Nobel Prize in Chemistry for polonium and radium. She is the first woman to win a Nobel, and the only person to win Nobels in two different sciences. After Pierre died in a carriage accident in 1906, Marie took his chair and became the first female professor at the Sorbonne.

During World War I she developed mobile X-ray units, called "Petites Curies" in French; about 20 were deployed to the front, examining over 1,000,000 wounded soldiers.

She died of aplastic anaemia from radiation exposure on 4 July 1934 in Passy, Haute-Savoie, France, aged 66. Her notebooks remain highly radioactive, kept in lead boxes; researchers must wear protective gear to consult them.
```

Compressed:
```text
## Summary
Marie Curie: French-Polish physicist/chemist, founder of radioactivity research, first female Sorbonne professor.
- Nobel x2 (Physics + Chemistry); first woman Nobel laureate; only person with Nobels in two sciences.
- Discovered polonium + radium; coined "radioactivity"; proved it is an atomic property.

## More
- birthplace, death place, age, cause of death
- degree years, in-school firsts x2
- element naming origin, collaborators, full timeline
- Nobel year per prize, co-laureates, citation
- device name, deployment scale, patients treated
- notebook radioactivity, storage, access conditions
```

Now begin.
""" # noqa

DEFAULT_SYSTEM_PROMPT = _SECTION_SCHEMA

DEFAULT_USER_PROMPT_TEMPLATE = """\
Downstream model will read your compressed block to decide whether to \
expand it. Compress faithfully: preserve the passage topic + core facts. \
Do NOT invent facts. Do NOT drop major facts. Do NOT write meta-commentary \
about the Query (never write "Query info: absent", "no X mention", etc.); \
if the passage does not address the Query, still summarize the passage.

## Query (ordering hint only — still summarize the whole passage)
{query}

## Target length
Compress AS MUCH AS faithfully possible. HARD CEILING: {budget} chars. \
If core facts fit in far fewer chars, output fewer. \
Never exceed the ceiling.

## Passage
{text}"""

# A (chunk_index, chunk, char_budget) triple marking one compression job.
_Job = Tuple[int, Chunk, int]


# ---------------------------------------------------------------------------
# ModelCondenser
# ---------------------------------------------------------------------------
class ModelCondenser(Condenser):
    """Compressor that delegates summarization to an LLM via a :class:`Sampler`.
    TODO: Experimental feature, wait for testing
    Args:
        sampler: Configured :class:`Sampler` with a template set.
        compression_ratio: Target factor (> 1). Used only to derive a
            soft character budget passed into the prompt and to size
            ``SamplingParams.max_tokens``. Model output is NOT hard
            truncated; a chunk whose decoded output is not strictly
            shorter than the original passage is left unchanged (and
            not flagged ``raw.condensed``).
        sampling_params: Override for per-call sampling; when ``None`` a
            greedy config is derived from the max budget in the batch.
        system_prompt: Override for the system prompt. Used verbatim.
        user_prompt_template: Override the user prompt. Must contain
            ``{budget}`` and ``{text}``. ``{query}`` is optional and is
            replaced with the trajectory's question extracted by the
            ``related_query`` callback (see below); jobs without a
            detected query get a neutral placeholder.
        min_chars: Pre-filter; chunks shorter than this pass through.
        min_budget_chars: Floor for the soft character budget exposed
            to the prompt. When ``ceil(len / compression_ratio)`` falls
            below this, the budget is raised to this floor so short
            passages keep room for all three sections in the model's
            plan. Since the condenser no longer hard-clips output,
            this only influences prompt wording and sampling token
            limits; pass ``1`` to use the raw ratio everywhere.
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
            This flag is purely a compression-skip filter; query
            extraction is the orthogonal job of ``related_query``.
        related_query: Optional ``(chunk) -> Optional[str]`` callback
            that returns the query string carried by ``chunk`` (e.g.
            the user's HotpotQA question), or ``None`` if the chunk
            is not a query carrier. Walked in chunk order; the most
            recently returned non-``None`` query is broadcast to all
            subsequent condense-eligible chunks until the next hit.
            Because :class:`MultiTurnCondenseRollout` may merge
            multiple trajectories into one chunk list, each
            trajectory's question chunk must precede its passages so
            this rolling state correctly partitions queries
            per-trajectory. ``None`` disables query injection (the
            ``{query}`` slot collapses to a neutral placeholder).
        rounds: Optional set of conversation turn indices to compress.
            ``None`` = no round-based filter; chunks lacking a ``round``
            field are skipped when this filter is active.
        batch_size: Max chunks per sampler call. Partial batches are
            padded with a duplicate of the last trajectory so that
            distributed samplers (DP slice) always receive a full batch.
        lora_path: Optional LoRA adapter to use for compression.
            - ``None`` (default): forwards ``use_base_model=True`` to
              :meth:`Sampler.sample` so compression bypasses any
              currently-synced LoRA — strongly recommended when the
              sampler is also the training policy.
            - ``str``: forwards ``adapter_path=lora_path`` so a
              dedicated condenser LoRA (e.g. a ModelScope slug or
              local directory) is loaded and used instead of the base.

    Compressed chunks are flagged ``raw.condensed=True``; a subsequent
    :meth:`Chunks.to_trajectory` call wraps them in ``<block_N>``.

    Example::

        >>> from twinkle.sampler import vLLMSampler
        >>> sampler = vLLMSampler(model_id='Qwen/Qwen2.5-3B-Instruct',
        ...                       engine_args={'dtype': 'bfloat16'})
        >>> sampler.set_template('qwen2_5')
        >>> cond = ModelCondenser(sampler, compression_ratio=2.0)
        >>> compressed = cond(chunks)
    """

    def __init__(
        self,
        sampler: Sampler,
        compression_ratio: float = 2.0,
        *,
        sampling_params: SamplingParams | None = None,
        system_prompt: str | None = None,
        user_prompt_template: str | None = None,
        min_chars: int = 200,
        min_budget_chars: int = 250,
        template: Any | None = None,
        skip_roles: Sequence[str] = ('system', 'tool', 'assistant'),
        skip_pattern: str | None = None,
        related_query: Callable[[Chunk], str | None] | None = None,
        rounds: Sequence[int] | None = None,
        batch_size: int = None,
        lora_path: str | None = None,
    ):
        if sampler is None:
            raise ValueError('sampler is required')
        if compression_ratio <= 1.0:
            raise ValueError(f'compression_ratio must be > 1, got {compression_ratio}')
        if min_chars < 0:
            raise ValueError(f'min_chars must be >= 0, got {min_chars}')
        if min_budget_chars < 1:
            raise ValueError(f'min_budget_chars must be >= 1, got {min_budget_chars}')
        if batch_size is not None and batch_size <= 0:
            raise ValueError(f'batch_size must be >= 1, got {batch_size}')

        tpl = user_prompt_template or DEFAULT_USER_PROMPT_TEMPLATE
        if '{budget}' not in tpl or '{text}' not in tpl:
            raise ValueError('user_prompt_template must contain both {budget} and {text}')

        self.sampler = sampler
        self.compression_ratio = float(compression_ratio)
        self.sampling_params = sampling_params
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.user_prompt_template = tpl
        self.min_chars = min_chars
        self.min_budget_chars = int(min_budget_chars)
        self.template = template
        self.skip_roles = tuple(skip_roles)
        # ``^`` must anchor to start-of-string, not start-of-line: a passage
        # whose body contains a ``Question:`` line would otherwise skip compression.
        self.skip_re: re.Pattern | None = (re.compile(skip_pattern) if skip_pattern else None)
        self.related_query = related_query
        self.rounds = set(rounds) if rounds is not None else None
        self.batch_size = batch_size
        self.lora_path = lora_path if lora_path else None
        self._special_tokens_cache: tuple[str, ...] | None = None

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------
    def __call__(self, chunks: Chunks, **_kwargs: Any) -> Chunks:
        out: list[Chunk] = list(chunks.chunks)
        items = self._collect_jobs(out)
        if not items:
            return Chunks(chunks=out)

        batch_size = self.batch_size or len(items)
        for start in range(0, len(items), batch_size):
            sub = items[start:start + batch_size]
            batch = [job for job, _q in sub]
            queries = [q for _job, q in sub]
            responses = self._sample_batch(batch, queries=queries)
            for (idx, chunk, _budget), resp in zip(batch, responses):
                text = self._postprocess(_decoded(resp), chunk['content'])
                if text is None:
                    continue
                out[idx] = _mark_condensed(chunk, text)
        return Chunks(chunks=out)

    # ------------------------------------------------------------------
    # eligibility + job collection
    # ------------------------------------------------------------------
    def _collect_jobs(
        self,
        chunks: Sequence[Chunk],
    ) -> list[tuple[_Job, str | None]]:
        """Collect compression jobs, tagging each with its trajectory's query.

        Walks ``chunks`` in order and maintains a rolling
        ``current_query`` state driven by the ``related_query``
        callback: every chunk for which the callback returns a
        non-``None`` string updates the state, and every subsequent
        condense-eligible chunk picks up the most recent query.
        Because the chunker emits each trajectory's question chunk
        before its passages, this walk correctly partitions queries
        per-trajectory even when ``MultiTurnCondenseRollout`` merges
        multiple trajectories into a single chunk list — A's
        passages only ever see A's question, B's only B's.
        """
        items: list[tuple[_Job, str | None]] = []
        current_query: str | None = None
        extract = self.related_query
        for i, c in enumerate(chunks):
            content = c.get('content')
            if extract is not None:
                q = extract(c)
                if isinstance(q, str) and q:
                    current_query = q
            if not self._should_condense(c):
                continue
            budget = max(self.min_budget_chars, math.ceil(len(content) / self.compression_ratio))
            if budget >= len(content):
                continue
            items.append(((i, c, max(1, budget)), current_query))
        return items

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
    def _sample_batch(
            self,
            batch: Sequence[_Job],
            *,
            queries: Sequence[str | None] = (),
    ) -> list[Any]:
        """Dispatch one batch to the sampler, padded to ``batch_size``.

        Distributed samplers slice inputs across DP workers and can
        mis-behave when the final batch is smaller than ``batch_size``;
        we pad with a duplicate of the last trajectory and trim the
        matching extra responses here.

        ``queries`` is aligned 1:1 with ``batch``; each per-job query
        is injected into the user prompt's ``{query}`` slot. When
        empty or ``None`` at an index, a neutral placeholder is used.
        """
        qs: list[str | None] = list(queries) if queries else [None] * len(batch)
        if len(qs) != len(batch):
            raise ValueError(f'queries length ({len(qs)}) must match batch length '
                             f'({len(batch)})')
        trajectories = [
            self._build_trajectory(chunk['content'], budget, query=q) for (_, chunk, budget), q in zip(batch, qs)
        ]
        actual = len(trajectories)
        device_mesh = getattr(self.sampler, 'device_mesh', None)
        min_batch_size = (device_mesh.data_world_size if device_mesh is not None else 1)
        if actual < min_batch_size:
            trajectories.extend([trajectories[-1]] * (min_batch_size - actual))

        sp = self._sampling_params_for(max(b for _, _, b in batch))
        kwargs: dict[str, Any] = {'sampling_params': sp}
        if self.lora_path is None:
            kwargs['use_base_model'] = True
        else:
            kwargs['adapter_path'] = self.lora_path
        responses = self.sampler.sample(trajectories, **kwargs)
        # Coerce to list (some samplers may return tuples) and drop
        # padding responses so downstream ``zip`` aligns with ``batch``.
        return list(responses)[:actual]

    def _build_trajectory(
        self,
        text: str,
        budget: int,
        *,
        query: str | None = None,
    ) -> Trajectory:
        system = self.system_prompt
        user = self.user_prompt_template.replace('{budget}', str(budget))
        user = user.replace('{text}', text)
        q_text = (
            query.strip() if isinstance(query, str) and query and query.strip() else
            '(no explicit query; compress by general salience)')
        user = user.replace('{query}', q_text)
        return {  # type: ignore[return-value]
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
        }

    def _sampling_params_for(self, budget: int) -> SamplingParams:
        if self.sampling_params is not None:
            return self.sampling_params
        from twinkle.data_format.sampling import SamplingParams

        # CJK worst case ~2 tokens/char; budget is a soft char ceiling, not output truth.
        max_new = max(512, budget * 3 + 128)
        return SamplingParams(temperature=0.0, max_tokens=max_new)

    # ------------------------------------------------------------------
    # postprocess
    # ------------------------------------------------------------------
    def _postprocess(self, raw: str, original: str) -> str | None:
        """Return compressed text, or ``None`` to signal passthrough.

        ``None`` is returned when the decoded output is empty,
        degenerate (markdown markers only, no alphanumerics), or its
        character length is **not strictly shorter** than ``original``
        — in which case the model failed to produce a useful
        compression and the caller should keep the original passage
        verbatim (no ``<block_N>`` wrap, not marked ``raw.condensed``).
        """
        text = _strip_special_tokens(_strip_code_fences(raw), self._get_special_tokens()).strip()
        if not text or not _has_alnum(text):
            return None
        if len(text) >= len(original):
            return None
        return text

    def _get_special_tokens(self) -> tuple[str, ...]:
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
        tokens: list[str] = []
        if tokenizer is not None:
            extras = getattr(tokenizer, 'all_special_tokens', None) or []
            if extras:
                tokens.extend(t for t in extras if isinstance(t, str) and t and not t.isspace())
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
    new: dict[str, Any] = dict(chunk)
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

    Used to detect degenerate model outputs like ``'##'`` or ``'- '``
    that are pure markdown markers with no actual words.
    """
    return any(ch.isalnum() for ch in text)
