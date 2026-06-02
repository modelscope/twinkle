# Copyright (c) ModelScope Contributors. All rights reserved.
"""Hard-example filter using distinct-token CHR (chr_min) + LLM-judged pass@4.

Replaces the legacy IFD = L(A|Q)/L(A) scorer with the ``chr_dist_min_pos`` metric
described in ``results/double_check/distinct_token_chr.py``: for each distinct
asst token id, take the minimum of (cond_lp - asst_lp) across its occurrences,
then report the fraction of distinct tokens whose min-diff is > 0.

Interpretation:
    chr_min HIGH → most distinct tokens benefit from the prompt → easy → drop.
    chr_min LOW  → many distinct tokens degrade under prompt    → hard → keep.

Each kept round is also re-answered ``diagnostic_sample_n`` times (default 4)
and each rollout is graded by an OpenAI-compatible judge against the GT for
both factual correctness AND reasoning/style similarity. The aggregate count
(0..n) is dumped as ``pass4`` alongside the chr_min score.
"""
import math
from typing import Any, Dict, List, Optional, Set, Tuple

from twinkle.preprocessor import Preprocessor
from twinkle.template import Template
from twinkle.utils import get_logger

from .llm_backend import LLMBackend, OpenAIBackend

logger = get_logger(only_local_master=False)

_MIN_RESPONSE_TOKENS = 5
_DEFAULT_CHR_MIN_THRESHOLD = 0.5


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


def _chr_min_distinct(
    cond_lp: List, asst_lp: List,
    cond_ids: List[int], asst_ids: List[int],
    n_prompt: int,
    exclude_ids: Optional[Set[int]] = None,
) -> Optional[float]:
    """Compute chr_dist_min_pos: fraction of distinct A-token ids whose
    per-occurrence min(cond_lp - asst_lp) is strictly positive.

    Mirrors ``aligned_pairs_with_token`` + ``distinct_chr`` from
    ``distinct_token_chr.py`` but operates on raw logprob lists (no JSON I/O).

    If ``exclude_ids`` is given, asst tokens whose id is in this set are
    dropped from the distinct-token statistics (experiment: exclude tokens
    that the asst literally echoes from the prompt vocabulary).
    """
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


def _ifd_family_metrics(
    cond_lp: List, asst_lp: List,
    cond_ids: List[int], asst_ids: List[int],
    n_prompt: int,
) -> Dict[str, Any]:
    """Compute IFD (Cherry-LLM, NAACL'24) and S-IFD (T-SHIRT, NeurIPS'25) for one round.

    Δt   = log P(yt | Q, y<t) - log P(yt | y<t)            (per-token PMI w.r.t. Q)
    IFD  = exp(-mean(Δt))                                  ← all positions, equal weight
    S-IFDk = exp(-mean(Δt over top-k% positions by |Δt|))  ← per-sample top-k% only

    Direction (HIGH = hard, opposite to chr_min):
        IFD/S-IFD ≫ 1 → Q does not reduce response perplexity → hard / informative.
        IFD/S-IFD ≪ 1 → Q strongly reduces perplexity         → easy.

    Returns dict with: n_tokens, mean_delta, ifd, s_ifd_50, s_ifd_75. Empty if invalid.
    """
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


_JUDGE_SYSTEM_PROMPT = (
    '你是一个严格但公平的回答评分员。请基于参考答案 (Ground Truth) 来判断【模型回答】是否合格。\n'
    '综合考量以下三方面，任一项有重大问题即判 FAIL：\n\n'
    '1. 计算/事实正确性：最终结论、数值、关键事实陈述与参考答案是否一致；\n'
    '2. 推理/思路相似度：解题路径、关键步骤、考量维度是否接近参考答案；\n'
    '   对于开放域问题（无明确正确答案），评估回答的风格、立场、考量维度是否与参考答案对齐；\n'
    '3. 完整性：回答没有截断、自然收尾，覆盖问题的所有要点。\n\n'
    '请先用 1-3 句简要说明判断依据，然后在最后一行严格输出：\n'
    '<verdict>PASS</verdict> 或 <verdict>FAIL</verdict>'
)


class IFDFilter(Preprocessor):
    """Filter key rounds by per-distinct-token CHR (chr_min).

    Requires rows pre-annotated by IntentClassifier (user_data.key_rounds).
    For each key round, computes chr_min = chr_dist_min_pos:
      - chr_min >= threshold → easy example → drop from key_rounds
      - chr_min < threshold  → hard example → keep
      - unscored (failed prepare) → kept conservatively

    Rows with all key_rounds removed are discarded entirely.
    Rows without key_rounds are passed through unchanged (or kept if
    ``keep_if_no_key_rounds=True``).

    In addition, each round is re-answered ``diagnostic_sample_n`` times
    (default 4) and each rollout is graded against the GT by an
    OpenAI-compatible judge. The aggregate pass count (``pass4``) and the
    per-rollout judgments are written into the dump alongside ``chr_min``.

    Tokenization MUST go through ``template.encode`` so the prompt/response
    boundary matches the exact byte stream the sampler would emit. Backend
    calls are batched in one shot so distributed samplers can keep every
    DP worker busy (slice_dp dispatch).
    """

    def __init__(
        self,
        backend: LLMBackend = None,
        template: Optional[Template] = None,
        # NEW: chr_min cutoff (replaces ifd_threshold).
        chr_min_threshold: float = _DEFAULT_CHR_MIN_THRESHOLD,
        # DEPRECATED: kept only to surface a warning when old configs pass it.
        # Semantics are INVERTED relative to chr_min so silent translation is
        # unsafe; callers must explicitly switch to chr_min_threshold.
        ifd_threshold: Optional[float] = None,
        keep_if_no_key_rounds: bool = False,
        max_prompt_tokens: int = 1024,
        # Diagnostic sampling: re-answer rounds and grade via judge.
        diagnostic_sample_intents: Optional[List[str]] = None,
        diagnostic_sample_n: int = 4,
        diagnostic_sample_temperature: float = 0.7,
        diagnostic_sample_max_tokens: int = 4096,
        # Pass@4 judge (LLM-as-judge, separate from training backend).
        # Pass either an `API` instance via `judge_api`, or
        # judge_model + judge_base_url + judge_api_key to auto-build OpenAI().
        judge_api=None,
        judge_model: Optional[str] = None,
        judge_base_url: Optional[str] = None,
        judge_api_key: Optional[str] = None,
        judge_client_kwargs: Optional[Dict[str, Any]] = None,
        judge_temperature: float = 0.0,
        judge_max_tokens: int = 512,
        judge_max_rollout_chars: int = 8000,
        judge_max_workers: int = 8,
        enable_pass4_judge: bool = True,
        # Paraphrase mode: replace GT with a model paraphrase produced under GT-injected
        # prompt, then score the paraphrase against the original (no-GT) context.
        # Bypasses filtering; rows pass through unchanged.
        # Accepts False (GT only), True (paraphrase only), or 'both' (dump two files).
        paraphrase_mode='both',
        paraphrase_temperature: float = 0.7,
        paraphrase_max_tokens: int = 4096,
        # Restrict paraphrase to rounds whose intent is in this set (e.g. {'math'}).
        # Empty/None = paraphrase ALL prepared rounds.
        paraphrase_intents: Optional[List[str]] = None,
        # Token budget for the augmented (GT-injected) prompt sent to chat_batch.
        # Must be <= max_model_len - paraphrase_max_tokens to avoid vLLM rejection.
        paraphrase_prompt_budget: int = 4096,
        # Experiment: drop asst tokens whose id appears anywhere in the prompt
        # from chr_min's distinct-token statistics (isolates novel-vocab signal).
        exclude_prompt_echoed_ids: bool = False,
        # Legacy params (used to create OpenAIBackend if backend is None).
        api_endpoint: str = '',
        model: str = 'default',
        # Silently absorbed; kept so existing configs don't break.
        head_k: Optional[int] = None,
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

        if ifd_threshold is not None:
            logger.warning(
                '[IFDFilter] `ifd_threshold` is deprecated; the scorer now produces '
                'chr_min where LOW = hard = keep (semantics inverted vs IFD). '
                f'Ignoring ifd_threshold={ifd_threshold} and using '
                f'chr_min_threshold={chr_min_threshold}. Update your config.')
        self._chr_min_threshold = float(chr_min_threshold)

        self._keep_if_no_key_rounds = keep_if_no_key_rounds
        self._max_prompt_tokens = max_prompt_tokens
        if head_k is not None:
            logger.info(
                f'[IFDFilter] `head_k={head_k}` is ignored: chr_min iterates ALL '
                'A-token positions (no head window).')

        self._diag_sample_intents = set(diagnostic_sample_intents or [])
        self._diag_sample_n = max(1, int(diagnostic_sample_n))
        self._diag_sample_temperature = float(diagnostic_sample_temperature)
        self._diag_sample_max_tokens = int(diagnostic_sample_max_tokens)

        self._judge_api = self._build_judge_api(
            judge_api, judge_model, judge_base_url, judge_api_key, judge_client_kwargs)
        self._judge_temperature = float(judge_temperature)
        self._judge_max_tokens = int(judge_max_tokens)
        self._judge_max_rollout_chars = int(judge_max_rollout_chars)
        self._judge_max_workers = max(1, int(judge_max_workers))
        self._enable_pass4_judge = bool(enable_pass4_judge) and self._judge_api is not None
        if enable_pass4_judge and self._judge_api is None:
            logger.warning(
                '[IFDFilter] enable_pass4_judge=True but no judge_api/judge_model '
                'configured; pass@4 grading is DISABLED. Diagnostic rollouts will '
                'still be sampled and dumped without verdicts.')

        self._paraphrase_mode = 'both' if paraphrase_mode == 'both' else bool(paraphrase_mode)
        self._paraphrase_temperature = float(paraphrase_temperature)
        self._paraphrase_max_tokens = int(paraphrase_max_tokens)
        self._paraphrase_intents = set(paraphrase_intents or [])
        self._paraphrase_prompt_budget = int(paraphrase_prompt_budget)
        self._exclude_prompt_echoed_ids = bool(exclude_prompt_echoed_ids)
        if self._exclude_prompt_echoed_ids:
            logger.info(
                '[IFDFilter] exclude_prompt_echoed_ids=True: chr_min will skip asst '
                'tokens whose id appears in the prompt.')

    @staticmethod
    def _build_judge_api(api, model, base_url, api_key, client_kwargs):
        """Resolve the pass@4 judge API: explicit instance > auto-built OpenAI > None."""
        if api is not None:
            return api
        if not model:
            return None
        try:
            from twinkle_agentic.protocol.openai import OpenAI as OpenAIAPI
            return OpenAIAPI(
                model=model,
                api_key=api_key,
                base_url=base_url,
                client_kwargs=client_kwargs,
            )
        except Exception as e:
            logger.warning(f'[IFDFilter] failed to build pass@4 judge API: {e}')
            return None

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = self.ifd_filter(rows)
        return self.map_row_to_col(rows)

    def _encode_prompt_within_budget(self, context_messages: List[Dict[str, Any]]) -> List[int]:
        """Encode context; drop oldest non-system msgs while over budget, fall back to token-tail."""
        ctx = list(context_messages)
        ids = _to_int_list(self._template.encode({'messages': ctx}, add_generation_prompt=True)['input_ids'])
        budget = self._max_prompt_tokens
        if budget <= 0 or len(ids) <= budget:
            return ids
        has_sys = bool(ctx) and isinstance(ctx[0], dict) and ctx[0].get('role') == 'system'
        body_start = 1 if has_sys else 0
        while len(ctx) - body_start > 1:
            ctx.pop(body_start)
            ids = _to_int_list(self._template.encode({'messages': ctx}, add_generation_prompt=True)['input_ids'])
            if len(ids) <= budget:
                return ids
        # Single message still too long: keep tail tokens, accept minor BPE contamination at start.
        return ids[-budget:]

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

        prompt_ids = self._encode_prompt_within_budget(context_messages)
        # Use raw asst_ids (no chat-template wrapping) so cond/asst paths share
        # byte-equal A-token sequences; otherwise chr_min positions desync.
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

    @staticmethod
    def _lp_to_jsonable(lp_list):
        """Convert a per-position logprobs list into JSON-safe form."""
        out = []
        for lp in lp_list:
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
                    d[str(k)] = {'logprob': float(v.logprob),
                                 'rank': getattr(v, 'rank', None),
                                 'decoded': getattr(v, 'decoded_token', None)}
                elif isinstance(v, dict):
                    d[str(k)] = v
                else:
                    d[str(k)] = repr(v)
            out.append(d)
        return out

    @staticmethod
    def _lookup_intent(row: Dict[str, Any], asst_idx: int) -> Optional[str]:
        """Read IntentClassifier annotation for one assistant turn (handles int/str dict keys)."""
        if not isinstance(row, dict) or asst_idx is None:
            return None
        user_data = row.get('user_data')
        if not isinstance(user_data, dict):
            return None
        intents = user_data.get('intents')
        if not isinstance(intents, dict):
            return None
        v = intents.get(asst_idx)
        if v is None:
            v = intents.get(str(asst_idx))
        return v if isinstance(v, str) else None

    def _collect_diagnostic_samples(
        self,
        rows: List[Dict[str, Any]],
        prepared: Dict[Tuple[int, int], Tuple[List[int], int, List[int]]],
    ) -> Dict[Tuple[int, int], List[Dict[str, str]]]:
        """Re-answer rounds; empty `_diag_sample_intents` means ALL intents (aligned with paraphrase semantics)."""
        if not prepared:
            return {}
        process_all = not self._diag_sample_intents
        # Group by intent to avoid cross-intent ordering issues in DP batching.
        intent_groups: Dict[str, Tuple[List[Tuple[int, int]], List[List[Dict[str, Any]]]]] = {}
        for key in prepared.keys():
            ri, rnd_idx = key
            row = rows[ri] if 0 <= ri < len(rows) else {}
            user_data = row.get('user_data') if isinstance(row, dict) else None
            if not isinstance(user_data, dict):
                continue
            kr = user_data.get('key_rounds')
            if not isinstance(kr, list) or not (0 <= rnd_idx < len(kr)):
                continue
            asst_idx = kr[rnd_idx]
            intent = self._lookup_intent(row, asst_idx)
            if not process_all and intent not in self._diag_sample_intents:
                continue
            messages = row.get('messages') or []
            if not (isinstance(messages, list) and 0 < asst_idx <= len(messages)):
                continue
            group_key = intent or '_unknown'
            if group_key not in intent_groups:
                intent_groups[group_key] = ([], [])
            intent_groups[group_key][0].append(key)
            intent_groups[group_key][1].append(messages[:asst_idx])
        if not intent_groups:
            return {}
        samples_by_key: Dict[Tuple[int, int], List[Dict[str, str]]] = {}
        total_target = 0
        for intent, (keys, ctxs) in intent_groups.items():
            total_target += len(keys)
            try:
                batched = self._backend.chat_batch(
                    ctxs,
                    temperature=self._diag_sample_temperature,
                    max_tokens=self._diag_sample_max_tokens,
                    n=self._diag_sample_n,
                ) or []
            except Exception as e:
                logger.warning(f'[IFDFilter] diagnostic chat_batch failed for intent={intent}: {e}')
                continue
            for key, choices in zip(keys, batched):
                if choices:
                    samples_by_key[key] = choices
        intents_label = 'ALL' if process_all else sorted(self._diag_sample_intents)
        logger.info(
            f'[IFDFilter] diagnostic sampling: re-answered {len(samples_by_key)}/{total_target} rounds '
            f'(intents={intents_label}, n={self._diag_sample_n}) '
            f'in {len(intent_groups)} batched call(s)')
        return samples_by_key

    @staticmethod
    def _extract_text_from_choice(choice: Any) -> str:
        """Pull the visible answer text out of one rollout dict (Message-shaped)."""
        if not isinstance(choice, dict):
            return ''
        parts: List[str] = []
        rc = choice.get('reasoning_content')
        if isinstance(rc, str) and rc.strip():
            parts.append(f'<thinking>\n{rc.strip()}\n</thinking>')
        content = choice.get('content')
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
        if parts:
            return '\n\n'.join(parts)
        return content if isinstance(content, str) else ''

    @staticmethod
    def _gt_text(row: Dict[str, Any], asst_idx: Optional[int]) -> str:
        """Pull the GT assistant text from the original row."""
        if not isinstance(row, dict) or asst_idx is None:
            return ''
        msgs = row.get('messages') or []
        if not (isinstance(msgs, list) and 0 <= asst_idx < len(msgs)):
            return ''
        msg = msgs[asst_idx]
        if not isinstance(msg, dict):
            return ''
        text = msg.get('content', '')
        if isinstance(text, list):
            text = ' '.join(p.get('text', '') for p in text
                            if isinstance(p, dict) and p.get('type') == 'text')
        return text if isinstance(text, str) else ''

    @staticmethod
    def _user_prompt_text(row: Dict[str, Any], asst_idx: Optional[int]) -> str:
        """Concatenate prior turns into a single string for the judge prompt."""
        if not isinstance(row, dict) or asst_idx is None:
            return ''
        msgs = row.get('messages') or []
        if not isinstance(msgs, list):
            return ''
        parts: List[str] = []
        for m in msgs[:asst_idx]:
            if not isinstance(m, dict):
                continue
            role = m.get('role') or 'user'
            content = m.get('content', '')
            if isinstance(content, list):
                content = ' '.join(p.get('text', '') for p in content
                                   if isinstance(p, dict) and p.get('type') == 'text')
            if isinstance(content, str) and content.strip():
                parts.append(f'[{role}] {content.strip()}')
        return '\n\n'.join(parts)

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        """Defensive truncation so the judge prompt fits inside its context window."""
        if not isinstance(text, str) or max_chars <= 0 or len(text) <= max_chars:
            return text
        head = max_chars * 2 // 3
        tail = max_chars - head - 32
        if tail <= 0:
            return text[:max_chars]
        return text[:head] + '\n\n...[truncated]...\n\n' + text[-tail:]

    @staticmethod
    def _parse_verdict(judge_text: str) -> Optional[bool]:
        """Return True if PASS, False if FAIL, None if neither marker found."""
        if not isinstance(judge_text, str):
            return None
        compact = ''.join(judge_text.upper().split())
        has_pass = '<VERDICT>PASS</VERDICT>' in compact
        has_fail = '<VERDICT>FAIL</VERDICT>' in compact
        if has_pass and not has_fail:
            return True
        if has_fail and not has_pass:
            return False
        # Fallback: keyword scan in the tail (last 200 chars, post-compact).
        tail = compact[-200:]
        if 'PASS' in tail and 'FAIL' not in tail:
            return True
        if 'FAIL' in tail and 'PASS' not in tail:
            return False
        return None

    def _judge_one_rollout(
        self,
        user_prompt: str,
        gt_text: str,
        rollout_text: str,
    ) -> Tuple[bool, str]:
        """Single judge call. Returns (passed, raw_judge_text)."""
        from twinkle.data_format.sampling import SamplingParams

        if not rollout_text or not rollout_text.strip():
            return False, '(empty rollout)'
        max_chars = self._judge_max_rollout_chars
        body = (
            f'[问题]\n{self._truncate(user_prompt, max_chars)}\n\n'
            f'[参考答案]\n{self._truncate(gt_text, max_chars)}\n\n'
            f'[模型回答]\n{self._truncate(rollout_text, max_chars)}\n\n'
            '请评分。'
        )
        trajectory = {
            'messages': [
                {'role': 'system', 'content': _JUDGE_SYSTEM_PROMPT},
                {'role': 'user', 'content': body},
            ],
        }
        sp = SamplingParams(
            temperature=self._judge_temperature,
            max_tokens=self._judge_max_tokens,
            num_samples=1,
        )
        try:
            # extra_body forwards `enable_thinking=False` to vLLM/SGLang OpenAI-compatible
            # endpoints so the judge skips chain-of-thought (saves latency + tokens).
            msg = self._judge_api(trajectory, sp, extra_body={'enable_thinking': False})
        except Exception as e:
            return False, f'(judge error: {e})'
        if isinstance(msg, list):
            msg = msg[0] if msg else {}
        text = msg.get('content', '') if isinstance(msg, dict) else str(msg)
        text = text or ''
        verdict = self._parse_verdict(text)
        # Conservative default: ambiguous verdict → FAIL (so we don't inflate pass@4).
        return bool(verdict) if verdict is not None else False, text

    def _judge_pass4(
        self,
        rows: List[Dict[str, Any]],
        prepared: Dict[Tuple[int, int], Tuple[List[int], int, List[int]]],
        samples_by_key: Dict[Tuple[int, int], List[Dict[str, str]]],
    ) -> Dict[Tuple[int, int], Tuple[int, List[Dict[str, Any]]]]:
        """Grade each rollout per round; return {key -> (pass_count, judgments)}."""
        if not self._enable_pass4_judge or not samples_by_key:
            return {}
        from concurrent.futures import ThreadPoolExecutor

        # Build flat work list: (key, rollout_idx, user_prompt, gt_text, rollout_text).
        work: List[Tuple[Tuple[int, int], int, str, str, str]] = []
        for key, choices in samples_by_key.items():
            if not isinstance(choices, list) or not choices:
                continue
            ri, rnd_idx = key
            row = rows[ri] if 0 <= ri < len(rows) else {}
            user_data = row.get('user_data') if isinstance(row, dict) else None
            asst_idx = None
            if isinstance(user_data, dict):
                kr = user_data.get('key_rounds')
                if isinstance(kr, list) and 0 <= rnd_idx < len(kr):
                    asst_idx = kr[rnd_idx]
            gt_text = self._gt_text(row, asst_idx)
            user_prompt = self._user_prompt_text(row, asst_idx)
            for r_i, choice in enumerate(choices):
                rt = self._extract_text_from_choice(choice)
                work.append((key, r_i, user_prompt, gt_text, rt))

        if not work:
            return {}

        def _do(item):
            key, r_i, up, gt, rt = item
            passed, raw = self._judge_one_rollout(up, gt, rt)
            return key, r_i, passed, raw

        with ThreadPoolExecutor(max_workers=self._judge_max_workers) as ex:
            results = list(ex.map(_do, work))

        bucket: Dict[Tuple[int, int], List[Tuple[int, bool, str]]] = {}
        for key, r_i, passed, raw in results:
            bucket.setdefault(key, []).append((r_i, passed, raw))

        out: Dict[Tuple[int, int], Tuple[int, List[Dict[str, Any]]]] = {}
        for key, lst in bucket.items():
            lst.sort(key=lambda x: x[0])
            pass_count = sum(1 for _, p, _ in lst if p)
            per_rollout = [
                {'rollout_idx': r_i, 'passed': bool(p), 'judge_raw': raw}
                for r_i, p, raw in lst
            ]
            out[key] = (pass_count, per_rollout)

        if out:
            avg = sum(p for p, _ in out.values()) / len(out)
            logger.info(
                f'[IFDFilter] pass@4 judging: graded {len(out)} rounds × {self._diag_sample_n} '
                f'rollouts, avg pass@n = {avg:.3f} (judge_temp={self._judge_temperature})')
        return out

    @staticmethod
    def _inject_gt(context_messages: List[Dict[str, Any]], gt_text: str) -> List[Dict[str, Any]]:
        """Append a GT-conditioned instruction so the model paraphrases the standard answer."""
        msgs = [dict(m) if isinstance(m, dict) else m for m in context_messages]
        instr = (
            '以下是这道题的标准答案，仅供参考：\n\n'
            f'<reference_answer>\n{gt_text}\n</reference_answer>\n\n'
            '请基于上面的参考答案，用你自己的语言和推理过程完整回答前面的问题。'
            '直接输出你的回答，不要复述参考答案的原文。'
        )
        if msgs and isinstance(msgs[-1], dict) and msgs[-1].get('role') == 'user':
            last = dict(msgs[-1])
            last['content'] = (last.get('content') or '') + '\n\n' + instr
            msgs[-1] = last
        else:
            msgs.append({'role': 'user', 'content': instr})
        return msgs

    def _truncate_gt_to_budget(self, gt_text: str, n_prompt: int) -> Optional[str]:
        """Truncate GT text so augmented prompt fits within paraphrase_prompt_budget."""
        _INSTR_OVERHEAD = 80  # instruction template tokens (conservative)
        budget = self._paraphrase_prompt_budget - n_prompt - _INSTR_OVERHEAD
        if budget < 50:
            return None
        gt_ids = _to_int_list(self._template.tokenizer(
            gt_text, add_special_tokens=False)['input_ids'])
        if len(gt_ids) <= budget:
            return gt_text
        truncated_ids = gt_ids[:budget]
        return self._template.tokenizer.decode(truncated_ids, skip_special_tokens=False)

    def _paraphrase_rounds(
        self,
        rows: List[Dict[str, Any]],
        prepared: Dict[Tuple[int, int], Tuple[List[int], int, List[int]]],
    ) -> Tuple[Dict[Tuple[int, int], Tuple[List[int], int, List[int]]],
               Dict[Tuple[int, int], str]]:
        """Replace each round's GT with one model paraphrase produced under a GT-injected
        prompt, then re-tokenize cond/asst against the ORIGINAL (no-GT) context so the
        downstream logprob computation reflects pure self-conditional probability."""
        if not prepared:
            return {}, {}
        keys: List[Tuple[int, int]] = []
        augmented_ctxs: List[List[Dict[str, Any]]] = []
        original_ctxs: List[List[Dict[str, Any]]] = []
        for key in prepared.keys():
            ri, rnd_idx = key
            row = rows[ri] if 0 <= ri < len(rows) else {}
            user_data = row.get('user_data') if isinstance(row, dict) else None
            if not isinstance(user_data, dict):
                continue
            kr = user_data.get('key_rounds')
            if not isinstance(kr, list) or not (0 <= rnd_idx < len(kr)):
                continue
            asst_idx = kr[rnd_idx]
            # Gate by intent (e.g. math-only paraphrase) when filter is configured.
            if self._paraphrase_intents and \
                    self._lookup_intent(row, asst_idx) not in self._paraphrase_intents:
                continue
            messages = row.get('messages') or []
            if not (isinstance(messages, list) and 0 < asst_idx <= len(messages)):
                continue
            asst_msg = messages[asst_idx]
            gt_text = asst_msg.get('content') if isinstance(asst_msg, dict) else None
            if isinstance(gt_text, list):
                gt_text = ' '.join(p.get('text', '') for p in gt_text
                                   if isinstance(p, dict) and p.get('type') == 'text')
            if not isinstance(gt_text, str) or not gt_text.strip():
                continue
            # Truncate GT to fit within prompt budget (avoids exceeding max_model_len).
            n_prompt = prepared[key][1]
            gt_text = self._truncate_gt_to_budget(gt_text, n_prompt)
            if gt_text is None:
                continue
            ctx = list(messages[:asst_idx])
            if not ctx:
                continue
            keys.append(key)
            original_ctxs.append(ctx)
            augmented_ctxs.append(self._inject_gt(ctx, gt_text))
        if not keys:
            return {}, {}
        try:
            batched = self._backend.chat_batch(
                augmented_ctxs,
                temperature=self._paraphrase_temperature,
                max_tokens=self._paraphrase_max_tokens,
                n=1,
            ) or []
        except Exception as e:
            logger.warning(f'[IFDFilter] paraphrase chat_batch failed: {e}')
            return {}, {}

        # Start clean: only successfully-paraphrased keys survive. Prevents tail-truncation
        # from chat_batch silently leaving GT entries in the paraphrase dump.
        new_prepared: Dict[Tuple[int, int], Tuple[List[int], int, List[int]]] = {}
        paraphrases: Dict[Tuple[int, int], str] = {}
        for key, ctx, choices in zip(keys, original_ctxs, batched):
            text = None
            if choices:
                choice = choices[0]
                if isinstance(choice, dict):
                    text = choice.get('content')
            if not isinstance(text, str) or not text.strip():
                continue
            prompt_ids = self._encode_prompt_within_budget(ctx)
            asst_ids = _to_int_list(self._template.tokenizer(
                text, add_special_tokens=False)['input_ids'])
            if len(asst_ids) < _MIN_RESPONSE_TOKENS + 1:
                continue
            new_prepared[key] = (prompt_ids + asst_ids, len(prompt_ids), asst_ids)
            paraphrases[key] = text
        logger.info(
            f'[IFDFilter] paraphrase: replaced {len(paraphrases)}/{len(keys)} rounds '
            f'(temp={self._paraphrase_temperature}, max_tokens={self._paraphrase_max_tokens}, '
            f'intents={sorted(self._paraphrase_intents) or "ALL"})')
        return new_prepared, paraphrases

    def _score_and_dump(
        self,
        rows: List[Dict[str, Any]],
        prepared: Dict[Tuple[int, int], Tuple[List[int], int, List[int]]],
        paraphrases_by_key: Dict[Tuple[int, int], str],
        dump_prefix: str,
        samples_by_key: Optional[Dict[Tuple[int, int], List[Dict[str, str]]]] = None,
        pass4_by_key: Optional[Dict[Tuple[int, int], Tuple[int, List[Dict[str, Any]]]]] = None,
    ) -> Dict[Tuple[int, int], float]:
        """Compute chr_min per round and dump records under given prefix."""
        scores: Dict[Tuple[int, int], float] = {}
        ifd_metrics: Dict[Tuple[int, int], Dict[str, Any]] = {}
        if not prepared:
            return scores
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
            exclude_ids = (
                set(int(t) for t in cond_ids[:n_prompt] if t is not None)
                if self._exclude_prompt_echoed_ids else None
            )
            chr_min = _chr_min_distinct(
                cond_lp, asst_lp, cond_ids, asst_ids, n_prompt,
                exclude_ids=exclude_ids,
            )
            if chr_min is not None:
                scores[key] = chr_min
            fam = _ifd_family_metrics(cond_lp, asst_lp, cond_ids, asst_ids, n_prompt)
            if fam:
                ifd_metrics[key] = fam
        self._dump_records(rows, prepared, keys, cond_logprobs, asst_logprobs, scores,
                           samples_by_key or {}, paraphrases_by_key,
                           pass4_by_key or {}, dump_prefix,
                           ifd_metrics_by_key=ifd_metrics)
        return scores

    def _dump_records(self, rows, prepared, keys, cond_logprobs, asst_logprobs, scores,
                      samples_by_key=None, paraphrases_by_key=None, pass4_by_key=None,
                      dump_prefix='chr_min_dump', ifd_metrics_by_key=None):
        """Dump per-round messages + raw logprobs + chr_min + pass@4 for offline diagnosis."""
        try:
            import json, os, time
            dump_path = f'{dump_prefix}_{os.getpid()}_{int(time.time())}.jsonl'
            samples_by_key = samples_by_key or {}
            paraphrases_by_key = paraphrases_by_key or {}
            pass4_by_key = pass4_by_key or {}
            ifd_metrics_by_key = ifd_metrics_by_key or {}
            with open(dump_path, 'w') as fh:
                for key, cond_lp, asst_lp in zip(keys, cond_logprobs, asst_logprobs):
                    ri, rnd_idx = key
                    cond_ids_k, n_prompt_k, asst_ids_k = prepared[key]
                    row = rows[ri] if 0 <= ri < len(rows) else {}
                    user_data = row.get('user_data') if isinstance(row, dict) else None
                    asst_idx = None
                    if isinstance(user_data, dict):
                        kr = user_data.get('key_rounds')
                        if isinstance(kr, list) and 0 <= rnd_idx < len(kr):
                            asst_idx = kr[rnd_idx]
                    p4 = pass4_by_key.get(key)
                    fam = ifd_metrics_by_key.get(key) or {}
                    fh.write(json.dumps({
                        'key': list(key),
                        'asst_idx': asst_idx,
                        'intent': self._lookup_intent(row, asst_idx),
                        'messages': row.get('messages') if isinstance(row, dict) else None,
                        'n_prompt': n_prompt_k,
                        'cond_ids': cond_ids_k,
                        'asst_ids': asst_ids_k,
                        'cond_lp': self._lp_to_jsonable(cond_lp),
                        'asst_lp': self._lp_to_jsonable(asst_lp),
                        'chr_min': scores.get(key),
                        'ifd': fam.get('ifd'),
                        's_ifd_50': fam.get('s_ifd_50'),
                        's_ifd_75': fam.get('s_ifd_75'),
                        'mean_delta': fam.get('mean_delta'),
                        'n_asst_tokens': fam.get('n_tokens'),
                        'pass4': (p4[0] if p4 is not None else None),
                        'pass4_judgments': (p4[1] if p4 is not None else None),
                        'diagnostic_samples': samples_by_key.get(key) or [],
                        'paraphrase': paraphrases_by_key.get(key),
                    }, ensure_ascii=False) + '\n')
            logger.info(f'[IFDFilter] dumped {len(keys)} records to {dump_path}')
        except Exception as e:
            logger.warning(f'[IFDFilter] dump failed: {e}')

    def ifd_filter(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score key rounds by chr_min, drop easy rounds (chr_min ≥ threshold),
        discard rows with none left."""
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

        # Mode dispatch: paraphrase_mode in (False, True, 'both').
        mode = self._paraphrase_mode
        run_gt = mode in (False, 'both')
        run_para = mode in (True, 'both')

        # Diagnostic sampling uses the original (no-GT) prompt and is independent of mode.
        # Run ONCE here so both GT and paraphrase dumps share the same samples (avoids
        # double cost and divergent stochastic outputs across the two dump files).
        samples_by_key = self._collect_diagnostic_samples(rows, prepared)
        # Pass@4 judging is also shared across dumps; run once on the rollouts above.
        pass4_by_key = self._judge_pass4(rows, prepared, samples_by_key)

        paraphrases_by_key: Dict[Tuple[int, int], str] = {}
        prepared_para: Optional[Dict[Tuple[int, int], Tuple[List[int], int, List[int]]]] = None
        if run_para and prepared:
            prepared_para, paraphrases_by_key = self._paraphrase_rounds(rows, prepared)

        scores: Dict[Tuple[int, int], float] = {}
        if run_gt:
            scores = self._score_and_dump(rows, prepared, {},
                                          dump_prefix='chr_min_dump',
                                          samples_by_key=samples_by_key,
                                          pass4_by_key=pass4_by_key)
        if run_para and prepared_para:
            self._score_and_dump(rows, prepared_para, paraphrases_by_key,
                                 dump_prefix='chr_min_paraphrase_dump',
                                 samples_by_key=samples_by_key,
                                 pass4_by_key=pass4_by_key)

        # Any paraphrase variant is diagnostic-only: skip filter, return rows unchanged.
        if run_para:
            return rows

        # Phase 3: apply scores. chr_min LOW = hard = keep.
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
                chr_min = scores.get((ri, rnd_idx))
                # Unscored rounds (failed prepare) are kept conservatively.
                if chr_min is None or chr_min < self._chr_min_threshold:
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
            f'[IFDFilter] removed {n_removed_rounds} easy rounds '
            f'(chr_min ≥ {self._chr_min_threshold}), '
            f'dropped {n_removed_rows} rows, kept {len(out)}/{len(rows)}')
        return out
