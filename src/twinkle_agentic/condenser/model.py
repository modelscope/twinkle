# Copyright (c) ModelScope Contributors. All rights reserved.
"""LLM-backed passage condenser.

Pipeline
--------
``Chunks`` → filter eligible chunks → batched ``Sampler.sample(...)`` →
strip code fences → length-vs-original guard → ``Chunks`` with
``raw.condensed=True`` (so :meth:`Chunks.to_trajectory` later wraps
them in ``<block_N>``). When the decoded output is empty, degenerate,
or **not strictly shorter than the original passage**, the chunk is
left untouched and is NOT marked ``raw.condensed`` — so downstream
bookkeeping (and the rollout trace) can tell compressed vs.
passthrough chunks apart.

The compression prompt asks for up to three markdown sections
(``## Summary / ## More / ## Key Facts``) written in **telegraphic
style** (no articles / copulas / filler) with per-section length
hints. Telegraphic output is ~2–3× denser than natural-prose summaries
and is critical under tight compression ratios. The output is **not**
parsed — sections pass through verbatim. The character budget the
prompt exposes is a soft target only; we never hard-clip the model
output, we simply discard it (fall back to the original) when it
fails to compress.
"""
from __future__ import annotations

import math
import re
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple)

from twinkle_agentic.condenser.base import Condenser
from twinkle_agentic.data_format import Chunk, Chunks

if TYPE_CHECKING:  # only used for type hints, keep runtime deps minimal
    from twinkle.data_format import SamplingParams, Trajectory  # noqa: F401
    from twinkle.sampler.base import Sampler  # noqa: F401


_SECTION_SCHEMA = """
你是一个文本压缩助手。你的使用场景是针对一大段文字进行压缩，以便后续模型在需要更多信息的时候展开并阅读原始文字。

后续模型工作流程：
阅读你的压缩结果 -> 确定需要的信息是否包含在本block中 -> 是 -> 阅读原文

因此你需要保证你的压缩不会损失原文中的主要信息。

你输出的格式：

```text
## Summary
概述在，以及和Query强相关的事实显式给出

## More
折叠的目录，需要展开才能看到具体信息
```

你需要注意：
1. 使用电报式格式，省略无用文字输出，例如“the”，“always”， “呢”等
2. 概述部分的事实应当和Query强相关，More中的目录应当能体现出其他信息的目录结构，保证模型阅读More后可以了解有哪些信息可以还原
3. 压缩后的语种和压缩前的文本应当相同

例子：

原文：

```text
玛丽·居里（Marie Curie，1867年11月7日—1934年7月4日），原名玛丽亚·斯克沃多夫斯卡，出生于俄属波兰华沙，父母均为教师。因当时波兰女性被禁止接受高等教育，她与姐姐约定轮流资助对方赴海外求学。

1891年，玛丽前往巴黎，入读巴黎大学（索邦大学）。1893年获物理学学士学位，1894年再获数学学士学位，成为该校首位女性物理学讲师。1895年与法国物理学家皮埃尔·居里结婚，两人此后长期共同开展放射性研究。

1898年7月，居里夫人发现新元素钋（Polonium），以其故乡波兰命名；同年12月与皮埃尔共同宣布发现镭（Radium）。她创造了"放射性（radioactivity）"一词，率先证明放射性是原子的固有属性，而非化学反应产物，从根本上重构了人类对物质结构的认识。

1903年，她与皮埃尔·居里及亨利·贝可勒尔共同获得诺贝尔物理学奖，以表彰放射性研究。1911年，她再度单独摘得诺贝尔化学奖，以表彰发现钋与镭。她是史上第一位诺贝尔奖女性得主，也是迄今唯一在两个不同科学领域均获诺贝尔奖的人。1906年皮埃尔因马车事故遇难后，玛丽接任其职位，成为巴黎大学首位女教授。

第一次世界大战期间，居里夫人研发了移动式X射线车，法文称"小居里（Petites Curies）"，共装备约20辆，部署于战场前线。据估计，该装备共为超过100万名伤兵提供了检查服务。

她因长期接触放射性物质导致再生障碍性贫血，于1934年7月4日在法国上萨瓦省帕西逝世，享年66岁。其研究笔记至今仍具高度放射性，存放于铅盒中，研究人员查阅时须穿戴防护服。
```

压缩后：
```text
## Summary
玛丽·居里（Marie Curie）：法籍波兰裔物理/化学家，放射性研究奠基人，巴黎大学首位女教授。
- 诺贝尔奖×2（物理+化学）首位女性得主，唯一双领域得主
- 发现钋+镭；创"放射性"概念；证其为原子固有属性

## More
- 出生地·逝世地·享年·死因
- 学位年份·校内首位记录×2
- 元素命名来源·合作者·完整时间线
- 诺奖各届年份·联颁合作者·颁奖背景
- 装备名·部署规模·救治数量
- 笔记放射性·保存方式·查阅条件
```

现在开始：
"""


DEFAULT_SYSTEM_PROMPT = _SECTION_SCHEMA

DEFAULT_USER_PROMPT_TEMPLATE = (
    '下游模型将基于压缩块回答以下问题。禁止为迎合 Query 而编造原文中不存在的事实。\n\n'
    '禁止编造原文中不存在的信息。\n\n'
    '## Query\n'
    '{query}\n\n'
    '注意：你不需要回答上述问题，你的任务是忠实地压缩\n\n'
    '## 长度目标\n'
    '约 {soft_budget} 字符，上限 {budget}。\n\n'
    '## 原文（Passage）\n'
    '{text}')


# A (chunk_index, chunk, char_budget) triple marking one compression job.
_Job = Tuple[int, Chunk, int]


# ---------------------------------------------------------------------------
# ModelCondenser
# ---------------------------------------------------------------------------
class ModelCondenser(Condenser):
    """Compressor that delegates summarization to an LLM via a :class:`Sampler`.

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
        system_prompt: Override for the system prompt. May contain
            ``{summary_words}``, ``{max_bullets}``, ``{bullet_words}``
            (all substituted per-chunk with budget-scaled word/bullet
            caps).
        user_prompt_template: Override the user prompt. Must contain
            ``{budget}`` and ``{text}``. ``{query}``,
            ``{soft_budget}``, ``{summary_words}``, ``{max_bullets}``
            and ``{bullet_words}`` are optional. ``{query}`` is
            replaced with the trajectory's question extracted by the
            ``related_query`` callback (see below); jobs without a
            detected query get a neutral placeholder. Scaling formulas:
            ``soft_budget = int(budget*0.85)``;
            ``summary_words = clamp(budget // 15, 8, 25)``;
            ``max_bullets = clamp(budget // 75, 2, 5)``;
            ``bullet_words = clamp(budget // 25, 6, 12)``.
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
        related_query: Optional[Callable[[Chunk], Optional[str]]] = None,
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

        tpl = user_prompt_template or DEFAULT_USER_PROMPT_TEMPLATE
        if '{budget}' not in tpl or '{text}' not in tpl:
            raise ValueError(
                'user_prompt_template must contain both {budget} and {text}')

        self.sampler = sampler
        self.compression_ratio = float(compression_ratio)
        self.sampling_params = sampling_params
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
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
        self.related_query = related_query
        self.rounds = set(rounds) if rounds is not None else None
        self.batch_size = batch_size
        self.use_base_model = bool(use_base_model)
        self._special_tokens_cache: Optional[Tuple[str, ...]] = None

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------
    def __call__(self, chunks: Chunks, **_kwargs: Any) -> Chunks:
        out: List[Chunk] = list(chunks.chunks)
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
                text = self._postprocess(
                    _decoded(resp), chunk['content'])
                if text is None:
                    continue
                out[idx] = _mark_condensed(chunk, text)
        return Chunks(chunks=out)

    # ------------------------------------------------------------------
    # eligibility + job collection
    # ------------------------------------------------------------------
    def _collect_jobs(
        self, chunks: Sequence[Chunk],
    ) -> List[Tuple[_Job, Optional[str]]]:
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
        items: List[Tuple[_Job, Optional[str]]] = []
        current_query: Optional[str] = None
        extract = self.related_query
        for i, c in enumerate(chunks):
            content = c.get('content')
            if extract is not None:
                q = extract(c)
                if isinstance(q, str) and q:
                    current_query = q
            if not self._should_condense(c):
                continue
            budget = max(
                self.min_budget_chars,
                math.ceil(len(content) / self.compression_ratio))
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
        queries: Sequence[Optional[str]] = (),
    ) -> List[Any]:
        """Dispatch one batch to the sampler, padded to ``batch_size``.

        Distributed samplers slice inputs across DP workers and can
        mis-behave when the final batch is smaller than ``batch_size``;
        we pad with a duplicate of the last trajectory and trim the
        matching extra responses here.

        ``queries`` is aligned 1:1 with ``batch``; each per-job query
        is injected into the user prompt's ``{query}`` slot. When
        empty or ``None`` at an index, a neutral placeholder is used.
        """
        qs: List[Optional[str]] = list(queries) if queries else [None] * len(batch)
        if len(qs) != len(batch):
            raise ValueError(
                f'queries length ({len(qs)}) must match batch length '
                f'({len(batch)})')
        trajectories = [
            self._build_trajectory(chunk['content'], budget, query=q)
            for (_, chunk, budget), q in zip(batch, qs)
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

    def _build_trajectory(
        self, text: str, budget: int, *, query: Optional[str] = None,
    ) -> 'Trajectory':
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
        # Query broadcast: each job gets its own trajectory's question
        # (collected via ``_collect_jobs`` walking state). Empty/None
        # collapses to a neutral placeholder so the prompt stays
        # well-formed and we never leak another trajectory's query.
        q_text = (
            query.strip()
            if isinstance(query, str) and query and query.strip()
            else '(no explicit query; compress by general salience)')
        user = user.replace('{query}', q_text)
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
    def _postprocess(self, raw: str, original: str) -> Optional[str]:
        """Return compressed text, or ``None`` to signal passthrough.

        ``None`` is returned when the decoded output is empty,
        degenerate (markdown markers only, no alphanumerics), or its
        character length is **not strictly shorter** than ``original``
        — in which case the model failed to produce a useful
        compression and the caller should keep the original passage
        verbatim (no ``<block_N>`` wrap, not marked ``raw.condensed``).
        """
        text = _strip_special_tokens(
            _strip_code_fences(raw), self._get_special_tokens()).strip()
        if not text or not _has_alnum(text):
            return None
        if len(text) >= len(original):
            return None
        return text

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

    Used to detect degenerate model outputs like ``'##'`` or ``'- '``
    that are pure markdown markers with no actual words.
    """
    return any(ch.isalnum() for ch in text)
