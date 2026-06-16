# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pluggable per-round scorer/filter for SFT key rounds.

Architecture:

    ScoreFilter(backend, scorers=[...])
      ├── pre-fetches logprobs once if any scorer requires them
      ├── runs each Scorer in order, collecting ScoreResult per round
      ├── trace dump (per-round JSON, multi_turn-style)
      └── AND aggregation: a round is kept iff every scorer returns passed=True.

Built-in scorers (each is its own class):
    ChrMinScorer      chr_dist_min_pos. LOW = hard = keep.
    SIFDScorer        IFD / S-IFD-50 / S-IFD-75. Default observe-only.
    PassNScorer       Self-rollouts judged by an LLM. extras carry rollouts/verdicts.
    ParaphraseScorer  chr_min over a model paraphrase produced under GT injection.

Decoupling:
    * key_rounds missing/empty → every assistant turn becomes a candidate round.
    * intents=None             → no intent-based gating (all rounds processed).
"""
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Set, Tuple

from twinkle.data_format import pack_value, user_data_get
from twinkle.preprocessor import Preprocessor
from twinkle.template import Template
from twinkle.utils import get_logger
from .llm_backend import LLMBackend
from .utils import _chr_min_distinct, _ifd_family_metrics, _lp_to_jsonable, _pad_batch, _to_int_list

logger = get_logger()

_MIN_RESPONSE_TOKENS = 5


@dataclass
class RoundContext:
    """Per-round payload passed to scorers."""
    row_idx: int
    rnd_idx: int
    asst_idx: int
    row: Dict[str, Any]
    intent: Optional[str]
    messages: List[Dict[str, Any]]
    context_messages: List[Dict[str, Any]]
    cond_ids: List[int]
    n_prompt: int
    asst_ids: List[int]
    asst_text: str
    user_prompt: str
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreResult:
    score: Optional[float] = None
    passed: bool = True
    extras: Dict[str, Any] = field(default_factory=dict)


class Scorer(Protocol):
    name: str
    requires_logprobs: bool

    def score(self, contexts: List[RoundContext]) -> List[ScoreResult]:
        ...


def _user_data_lookup(user_data: Any, key: str) -> Any:
    """Pull a value by key from packed user_data; returns the JSON-decoded value."""
    return user_data_get(user_data, key)


# ============================================================================
# Built-in scorers
# ============================================================================


class ChrMinScorer:
    """chr_dist_min_pos. Dual-threshold: keep samples in [low, high)."""
    name = 'chr_min'
    requires_logprobs = True

    def __init__(self, threshold: float = 0.47):
        self._threshold = float(threshold)

    def score(self, contexts: List[RoundContext]) -> List[ScoreResult]:
        out: List[ScoreResult] = []
        for ctx in contexts:
            cond_lp = ctx.features.get('cond_lp')
            asst_lp = ctx.features.get('asst_lp')
            score = _chr_min_distinct(
                cond_lp,
                asst_lp,
                ctx.cond_ids,
                ctx.asst_ids,
                ctx.n_prompt,
            )
            passed = (score is None) or (score < self._threshold)
            out.append(ScoreResult(
                score=score,
                passed=passed,
                extras={'threshold': self._threshold},
            ))
        return out


class SIFDScorer:
    """IFD / S-IFD-50 / S-IFD-75. Observation-only by default."""
    name = 'sifd'
    requires_logprobs = True

    def __init__(self, ifd_threshold: Optional[float] = None):
        # If set, passed = (ifd >= threshold). HIGH IFD = hard = keep.
        self._ifd_threshold = ifd_threshold

    def score(self, contexts: List[RoundContext]) -> List[ScoreResult]:
        out: List[ScoreResult] = []
        for ctx in contexts:
            cond_lp = ctx.features.get('cond_lp')
            asst_lp = ctx.features.get('asst_lp')
            fam = _ifd_family_metrics(cond_lp, asst_lp, ctx.cond_ids, ctx.asst_ids, ctx.n_prompt)
            score = fam.get('ifd')
            if self._ifd_threshold is None or score is None:
                passed = True
            else:
                passed = score >= self._ifd_threshold
            out.append(ScoreResult(score=score, passed=passed, extras=dict(fam)))
        return out


_JUDGE_SYSTEM_PROMPT = """\
You are a strict but fair answer grader. Judge whether the [Model Answer] is acceptable based on the reference answer (Ground Truth).
Evaluate the following three aspects; if any has a major issue, return FAIL:

1. Computational/factual correctness: whether the final conclusion, numbers, and key factual statements match the reference answer;
2. Reasoning/approach similarity: whether the solution path, key steps, and considered dimensions are close to the reference answer;
   For open-ended questions (no single correct answer), assess whether the style, stance, and considered dimensions align with the reference answer;
3. Completeness: the answer is not truncated, ends naturally, and covers all points of the question.

First give a brief 1-3 sentence justification, then on the last line strictly output:
<verdict>PASS</verdict> or <verdict>FAIL</verdict>""" # noqa


class PassNScorer:
    """Self-rollouts (n × per round) judged by an LLM."""
    name = 'pass_n'
    requires_logprobs = False

    def __init__(
        self,
        backend: LLMBackend,
        judge_api=None,
        judge_model: Optional[str] = None,
        judge_base_url: Optional[str] = None,
        judge_api_key: Optional[str] = None,
        judge_client_kwargs: Optional[Dict[str, Any]] = None,
        n: int = 4,
        min_pass: int = 0,
        sample_temperature: float = 0.7,
        sample_max_tokens: int = 4096,
        judge_temperature: float = 0.0,
        judge_max_tokens: int = 512,
        judge_max_rollout_chars: int = 8000,
        judge_max_workers: int = 8,
    ):
        self._backend = backend
        self._judge_api = self._build_judge_api(judge_api, judge_model, judge_base_url, judge_api_key,
                                                judge_client_kwargs)
        self._n = max(1, int(n))
        self._min_pass = int(min_pass)
        self._sample_temperature = float(sample_temperature)
        self._sample_max_tokens = int(sample_max_tokens)
        self._judge_temperature = float(judge_temperature)
        self._judge_max_tokens = int(judge_max_tokens)
        self._judge_max_rollout_chars = int(judge_max_rollout_chars)
        self._judge_max_workers = max(1, int(judge_max_workers))
        if self._judge_api is None:
            logger.warning('[PassNScorer] no judge_api configured; rollouts will be sampled '
                           'without verdicts (every round trivially passes).')

    @staticmethod
    def _build_judge_api(api, model, base_url, api_key, client_kwargs):
        if api is not None:
            return api
        if not model:
            return None
        from twinkle_agentic.protocol.openai import OpenAI as OpenAIAPI
        return OpenAIAPI(model=model, api_key=api_key, base_url=base_url, client_kwargs=client_kwargs)

    @staticmethod
    def _extract_text_from_choice(choice: Any) -> str:
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
    def _truncate(text: str, max_chars: int) -> str:
        if not isinstance(text, str) or max_chars <= 0 or len(text) <= max_chars:
            return text
        head = max_chars * 2 // 3
        tail = max_chars - head - 32
        if tail <= 0:
            return text[:max_chars]
        return text[:head] + '\n\n...[truncated]...\n\n' + text[-tail:]

    @staticmethod
    def _parse_verdict(judge_text: str) -> Optional[bool]:
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

    def _judge_one(self, user_prompt: str, gt_text: str, rollout_text: str) -> Tuple[bool, str]:
        if self._judge_api is None:
            return True, '(no judge configured)'
        if not rollout_text or not rollout_text.strip():
            return False, '(empty rollout)'
        from twinkle.data_format.sampling import SamplingParams
        body = (f'[问题]\n{self._truncate(user_prompt, self._judge_max_rollout_chars)}\n\n'
                f'[参考答案]\n{self._truncate(gt_text, self._judge_max_rollout_chars)}\n\n'
                f'[模型回答]\n{self._truncate(rollout_text, self._judge_max_rollout_chars)}\n\n'
                '请评分。')
        trajectory = {
            'messages': [
                {
                    'role': 'system',
                    'content': _JUDGE_SYSTEM_PROMPT
                },
                {
                    'role': 'user',
                    'content': body
                },
            ]
        }
        sp = SamplingParams(
            temperature=self._judge_temperature,
            max_tokens=self._judge_max_tokens,
            num_samples=1,
        )
        # extra_body forwards `enable_thinking=False` so the judge skips CoT.
        msg = self._judge_api(trajectory, sp, extra_body={'enable_thinking': False})
        if isinstance(msg, list):
            msg = msg[0] if msg else {}
        text = msg.get('content', '') if isinstance(msg, dict) else str(msg)
        text = text or ''
        verdict = self._parse_verdict(text)
        # Conservative default: ambiguous verdict → FAIL.
        return bool(verdict) if verdict is not None else False, text

    def score(self, contexts: List[RoundContext]) -> List[ScoreResult]:
        if not contexts:
            return []
        ctx_msgs = [ctx.context_messages for ctx in contexts]
        batched = self._backend.chat_batch(
            ctx_msgs,
            temperature=self._sample_temperature,
            max_tokens=self._sample_max_tokens,
            n=self._n,
        ) or []

        while len(batched) < len(contexts):
            batched.append([])

        from concurrent.futures import ThreadPoolExecutor
        work: List[Tuple[int, int, str, str, str]] = []
        for i, (ctx, choices) in enumerate(zip(contexts, batched)):
            if not isinstance(choices, list):
                continue
            for r_i, choice in enumerate(choices):
                rt = self._extract_text_from_choice(choice)
                work.append((i, r_i, ctx.user_prompt, ctx.asst_text, rt))

        verdict_by_round: Dict[int, List[Tuple[int, bool, str]]] = {}
        if work and self._judge_api is not None:

            def _do(item):
                i, r_i, up, gt, rt = item
                ok, raw = self._judge_one(up, gt, rt)
                return i, r_i, ok, raw

            with ThreadPoolExecutor(max_workers=self._judge_max_workers) as ex:
                for i, r_i, ok, raw in ex.map(_do, work):
                    verdict_by_round.setdefault(i, []).append((r_i, ok, raw))

        out: List[ScoreResult] = []
        for i, (ctx, choices) in enumerate(zip(contexts, batched)):
            rollouts = [{
                'rollout_idx': r_i,
                'content': self._extract_text_from_choice(c)
            } for r_i, c in enumerate(choices or [])]
            verdicts = sorted(verdict_by_round.get(i, []), key=lambda x: x[0])
            judgments = [{'rollout_idx': r_i, 'passed': bool(p), 'judge_raw': raw} for r_i, p, raw in verdicts]
            pass_count = sum(1 for _, p, _ in verdicts if p)
            score = (pass_count / self._n) if rollouts else None
            passed = pass_count >= self._min_pass
            out.append(
                ScoreResult(
                    score=score,
                    passed=passed,
                    extras={
                        'pass_count': pass_count,
                        'n_rollouts': len(rollouts),
                        'rollouts': rollouts,
                        'judgments': judgments,
                        'min_pass': self._min_pass,
                    },
                ))

        scored = [r for r in out if r.score is not None]
        if scored:
            avg = sum(r.score for r in scored) / len(scored)
            logger.info(f'[PassNScorer] graded {len(scored)}/{len(out)} rounds × {self._n} '
                        f'rollouts; avg pass-rate = {avg:.3f}')
        return out


class ParaphraseScorer:
    """Generate a model paraphrase under GT injection, then re-score chr_min."""
    name = 'paraphrase'
    # Owns its own logprob fetch on the rewritten asst tokens.
    requires_logprobs = False

    def __init__(
        self,
        backend: LLMBackend,
        template: Template,
        chr_min_threshold: Optional[float] = None,
        prompt_budget: int = 4096,
        sample_temperature: float = 0.7,
        sample_max_tokens: int = 4096,
        max_prompt_tokens: int = 1024,
    ):
        self._backend = backend
        self._template = template
        self._threshold = chr_min_threshold
        self._prompt_budget = int(prompt_budget)
        self._sample_temperature = float(sample_temperature)
        self._sample_max_tokens = int(sample_max_tokens)
        self._max_prompt_tokens = int(max_prompt_tokens)

    @staticmethod
    def _inject_gt(context_messages, gt_text):
        msgs = [dict(m) if isinstance(m, dict) else m for m in context_messages]
        instr = f"""\
Below is the reference answer to this question, for your reference only:

<reference_answer>
{gt_text}
</reference_answer>

Based on the reference answer above, please provide a complete answer to the preceding question in your own words and reasoning. Output your answer directly; do not repeat the reference answer verbatim.""" # noqa
        if msgs and isinstance(msgs[-1], dict) and msgs[-1].get('role') == 'user':
            last = dict(msgs[-1])
            last['content'] = (last.get('content') or '') + '\n\n' + instr
            msgs[-1] = last
        else:
            msgs.append({'role': 'user', 'content': instr})
        return msgs

    def _truncate_gt(self, gt_text: str, n_prompt: int) -> Optional[str]:
        # 80 = conservative instruction-template overhead.
        budget = self._prompt_budget - n_prompt - 80
        if budget < 50:
            return None
        gt_ids = _to_int_list(self._template.tokenizer(gt_text, add_special_tokens=False)['input_ids'])
        if len(gt_ids) <= budget:
            return gt_text
        return self._template.tokenizer.decode(gt_ids[:budget], skip_special_tokens=False)

    def _encode_prompt(self, ctx_msgs):
        ids = _to_int_list(self._template.encode({'messages': list(ctx_msgs)}, add_generation_prompt=True)['input_ids'])
        if self._max_prompt_tokens <= 0 or len(ids) <= self._max_prompt_tokens:
            return ids
        return ids[-self._max_prompt_tokens:]

    def score(self, contexts: List[RoundContext]) -> List[ScoreResult]:
        if not contexts:
            return []

        keys: List[int] = []
        augmented: List[List[Dict[str, Any]]] = []
        for i, ctx in enumerate(contexts):
            gt = self._truncate_gt(ctx.asst_text, ctx.n_prompt)
            if gt is None or not ctx.context_messages:
                continue
            keys.append(i)
            augmented.append(self._inject_gt(ctx.context_messages, gt))

        out: List[ScoreResult] = [
            ScoreResult(score=None, passed=True, extras={'reason': 'paraphrase skipped'}) for _ in contexts
        ]
        if not keys:
            return out

        batched = self._backend.chat_batch(
            augmented,
            temperature=self._sample_temperature,
            max_tokens=self._sample_max_tokens,
            n=1,
        ) or []

        # Re-tokenize against the ORIGINAL (no-GT) context so logprobs reflect
        # pure self-conditional probability of the paraphrase.
        para_data: Dict[int, Tuple[List[int], int, List[int], str]] = {}
        for i, choices in zip(keys, batched):
            text = None
            if choices:
                c0 = choices[0]
                if isinstance(c0, dict):
                    text = c0.get('content')
            if not isinstance(text, str) or not text.strip():
                continue
            ctx = contexts[i]
            prompt_ids = self._encode_prompt(ctx.context_messages)
            asst_ids = _to_int_list(self._template.tokenizer(text, add_special_tokens=False)['input_ids'])
            if len(asst_ids) < _MIN_RESPONSE_TOKENS + 1:
                continue
            cond_ids = prompt_ids + asst_ids
            para_data[i] = (cond_ids, len(prompt_ids), asst_ids, text)

        if not para_data:
            return out

        ordered = list(para_data.keys())
        cond_batch = [para_data[i][0] for i in ordered]
        asst_batch = [para_data[i][2] for i in ordered]
        cond_lps = self._backend.prompt_logprobs_ids(cond_batch)
        asst_lps = self._backend.prompt_logprobs_ids(asst_batch)

        for i, cond_lp, asst_lp in zip(ordered, cond_lps, asst_lps):
            cond_ids, n_prompt, asst_ids, text = para_data[i]
            score = _chr_min_distinct(cond_lp, asst_lp, cond_ids, asst_ids, n_prompt)
            if self._threshold is None or score is None:
                passed = True
            else:
                passed = score < self._threshold
            out[i] = ScoreResult(
                score=score,
                passed=passed,
                extras={
                    'paraphrase_text': text,
                    'n_prompt': n_prompt,
                    'cond_lp': _lp_to_jsonable(cond_lp),
                    'asst_lp': _lp_to_jsonable(asst_lp),
                    'threshold': self._threshold,
                },
            )

        logger.info(f'[ParaphraseScorer] paraphrased + scored {len(para_data)}/'
                    f'{len(contexts)} rounds')
        return out


# ============================================================================
# ScoreFilter (Preprocessor entry point)
# ============================================================================


class ScoreFilter(Preprocessor):
    """Score and filter assistant turns by a pluggable scorer set.

    A round is kept iff every scorer returns ``passed=True``. Rows that lose
    all key rounds are dropped (configurable via ``keep_if_no_key_rounds``).

    Decoupling rules:
        * `key_rounds` missing/empty in `user_data` → every assistant turn
          becomes a candidate round.
        * `intents=None` → no intent-based gating.
    """

    def __init__(
        self,
        template: Template,
        backend: LLMBackend,
        scorers: List[Scorer],
        intents: Optional[Iterable[str]] = None,
        keep_if_no_key_rounds: bool = False,
        drop_row_on_any_fail: bool = True,
        max_prompt_tokens: int = 1024,
        trace_dir: Optional[str] = None,
        trace_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
        success_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ):
        super().__init__()
        if not isinstance(template, Template):
            raise TypeError(f'ScoreFilter requires a `Template` instance, got '
                            f'{type(template).__name__}.')
        self._template = template
        self._backend = backend
        self._scorers = list(scorers)
        self._intents: Optional[Set[str]] = (None if intents is None else set(intents))
        self._keep_if_no_key_rounds = bool(keep_if_no_key_rounds)
        self._drop_row_on_any_fail = bool(drop_row_on_any_fail)
        self._max_prompt_tokens = int(max_prompt_tokens)
        self._trace_dir = trace_dir
        self._trace_callback = trace_callback
        self._success_callback = success_callback
        if self._trace_dir:
            import shutil
            if os.path.exists(self._trace_dir):
                shutil.rmtree(self._trace_dir)
            os.makedirs(self._trace_dir, exist_ok=True)

    def __call__(self, rows):
        rows_list = self.map_col_to_row(rows)
        contexts = self._build_contexts(rows_list)
        dropped: List[Dict[str, Any]] = []
        if contexts:
            score_table = self._score_contexts(contexts)
            self._log_score_summary(contexts, score_table)
            if self._trace_dir:
                self._write_traces(contexts, score_table)
            rows_list, dropped = self._apply_filter(rows_list, contexts, score_table)
        return rows_list, dropped

    def _log_score_summary(self, contexts, score_table):
        for scorer in self._scorers:
            scores = [
                t[scorer.name].score for t in score_table if scorer.name in t and t[scorer.name].score is not None
            ]
            if not scores:
                continue
            n_pass = sum(1 for t in score_table if scorer.name in t and t[scorer.name].passed)
            extras_sample = {}
            for t in score_table:
                if scorer.name in t and t[scorer.name].extras:
                    extras_sample = t[scorer.name].extras
                    break
            extra_keys = [k for k in extras_sample if k != 'threshold']
            extra_stats = ''
            for k in extra_keys:
                vals = [
                    t[scorer.name].extras.get(k) for t in score_table
                    if scorer.name in t and t[scorer.name].extras and t[scorer.name].extras.get(k) is not None
                ]
                if vals and isinstance(vals[0], (int, float)):
                    avg = sum(vals) / len(vals)
                    extra_stats += f', {k}_avg={avg:.4f}'
            logger.info(f'[ScoreFilter/{scorer.name}] n={len(scores)}, '
                        f'mean={sum(scores) / len(scores):.4f}, '
                        f'min={min(scores):.4f}, max={max(scores):.4f}, '
                        f'pass={n_pass}/{len(score_table)}'
                        f'{extra_stats}')

    # ---- scoring (inlined DefaultScoreCalculator) --------------------------

    def _score_contexts(self, contexts: List[RoundContext]) -> List[Dict[str, ScoreResult]]:
        if any(getattr(s, 'requires_logprobs', False) for s in self._scorers):
            self._attach_logprobs(contexts)
        out: List[Dict[str, ScoreResult]] = [dict() for _ in contexts]
        for scorer in self._scorers:
            results = scorer.score(contexts)
            if len(results) != len(contexts):
                raise RuntimeError(f'scorer {scorer.name!r} returned {len(results)} results '
                                   f'for {len(contexts)} contexts')
            for i, r in enumerate(results):
                out[i][scorer.name] = r
        return out

    def _attach_logprobs(self, contexts: List[RoundContext]) -> None:
        cond_batch = [ctx.cond_ids for ctx in contexts]
        asst_batch = [ctx.asst_ids for ctx in contexts]
        floor = self._batch_floor()
        cond_padded, n_cond = _pad_batch(cond_batch, floor)
        asst_padded, n_asst = _pad_batch(asst_batch, floor)
        cond_lps = self._backend.prompt_logprobs_ids(cond_padded)[:n_cond]
        asst_lps = self._backend.prompt_logprobs_ids(asst_padded)[:n_asst]
        for ctx, c, a in zip(contexts, cond_lps, asst_lps):
            ctx.features['cond_lp'] = c
            ctx.features['asst_lp'] = a

    def _batch_floor(self) -> int:
        sampler = getattr(self._backend, '_sampler', None)
        device_mesh = getattr(sampler, 'device_mesh', None)
        return getattr(device_mesh, 'dp_world_size', 1) or 1

    # ---- context construction --------------------------------------------

    def _build_contexts(self, rows: List[Dict[str, Any]]) -> List[RoundContext]:
        out: List[RoundContext] = []
        for ri, row in enumerate(rows):
            messages = row.get('messages') if isinstance(row, dict) else None
            if not isinstance(messages, list):
                continue
            user_data = row.get('user_data') if isinstance(row, dict) else None
            key_rounds = _user_data_lookup(user_data, 'key_rounds')
            if not isinstance(key_rounds, list) or not key_rounds:
                key_rounds = [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get('role') == 'assistant']
            for rnd_idx, asst_idx in enumerate(key_rounds):
                if not isinstance(asst_idx, int):
                    continue
                intent = self._lookup_intent(row, asst_idx)
                if self._intents is not None and intent not in self._intents:
                    continue
                ctx = self._prepare_round(row, messages, ri, rnd_idx, asst_idx, intent)
                if ctx is not None:
                    out.append(ctx)
        return out

    def _prepare_round(
        self,
        row: Dict[str, Any],
        messages: List[Dict[str, Any]],
        ri: int,
        rnd_idx: int,
        asst_idx: int,
        intent: Optional[str],
    ) -> Optional[RoundContext]:
        if not (0 <= asst_idx < len(messages)):
            return None
        asst_msg = messages[asst_idx]
        if not isinstance(asst_msg, dict) or asst_msg.get('role') != 'assistant':
            return None
        asst_text = asst_msg.get('content') or ''
        if isinstance(asst_text, list):
            asst_text = ' '.join(
                p.get('text', '') for p in asst_text if isinstance(p, dict) and p.get('type') == 'text')
        if not asst_text.strip():
            return None
        context_messages = messages[:asst_idx]
        if not context_messages:
            return None
        prompt_ids = self._encode_prompt_within_budget(context_messages)
        # Raw asst_ids (no chat-template wrapping) so cond/asst share byte-equal
        # A-token sequences; otherwise chr_min positions desync.
        asst_ids = _to_int_list(self._template.tokenizer(asst_text, add_special_tokens=False)['input_ids'])
        if len(asst_ids) < _MIN_RESPONSE_TOKENS + 1:
            return None
        return RoundContext(
            row_idx=ri,
            rnd_idx=rnd_idx,
            asst_idx=asst_idx,
            row=row,
            intent=intent,
            messages=messages,
            context_messages=context_messages,
            cond_ids=prompt_ids + asst_ids,
            n_prompt=len(prompt_ids),
            asst_ids=asst_ids,
            asst_text=asst_text,
            user_prompt=self._render_user_prompt(context_messages),
        )

    def _encode_prompt_within_budget(self, ctx_msgs: List[Dict[str, Any]]) -> List[int]:
        ctx = list(ctx_msgs)
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
        # Single message still over budget → keep tail tokens.
        return ids[-budget:]

    @staticmethod
    def _render_user_prompt(ctx_msgs: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for m in ctx_msgs:
            if not isinstance(m, dict):
                continue
            role = m.get('role') or 'user'
            content = m.get('content', '')
            if isinstance(content, list):
                content = ' '.join(
                    p.get('text', '') for p in content if isinstance(p, dict) and p.get('type') == 'text')
            if isinstance(content, str) and content.strip():
                parts.append(f'[{role}] {content.strip()}')
        return '\n\n'.join(parts)

    @staticmethod
    def _lookup_intent(row: Dict[str, Any], asst_idx: int) -> Optional[str]:
        user_data = row.get('user_data') if isinstance(row, dict) else None
        intents = _user_data_lookup(user_data, 'intents')
        if not isinstance(intents, dict):
            return None
        v = intents.get(asst_idx)
        if v is None:
            v = intents.get(str(asst_idx))
        return v if isinstance(v, str) else None

    # ---- trace dump (multi_turn-style) -----------------------------------

    def _write_traces(
        self,
        contexts: List[RoundContext],
        score_table: List[Dict[str, ScoreResult]],
    ) -> None:
        for i, ctx in enumerate(contexts):
            try:
                scores = score_table[i] if i < len(score_table) else {}
                kept = all(r.passed for r in scores.values()) if scores else True
                record = self._build_trace_record(ctx, scores, kept)
                if self._trace_callback is not None and not bool(self._trace_callback(record)):
                    continue
                success = (bool(self._success_callback(record)) if self._success_callback is not None else kept)
                prefix = 'ok' if success else 'fail'
                rid = f'{ctx.row_idx}-{ctx.asst_idx}-{i}-{int(time.time() * 1000)}'
                rid = re.sub(r'[^A-Za-z0-9_\-.]+', '_', rid)[:64]
                path = os.path.join(self._trace_dir, f'{prefix}-{rid}.json')
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(record, f, ensure_ascii=False, indent=2, default=str)
            except Exception as e:
                # Observability must never break filtering; surface the cause.
                logger.warning(f'[ScoreFilter] trace dump failed for row={ctx.row_idx} '
                               f'asst={ctx.asst_idx}: {e}')

    @staticmethod
    def _build_trace_record(
        ctx: RoundContext,
        scores: Dict[str, ScoreResult],
        kept: bool,
    ) -> Dict[str, Any]:
        return {
            'row_idx': ctx.row_idx,
            'rnd_idx': ctx.rnd_idx,
            'asst_idx': ctx.asst_idx,
            'intent': ctx.intent,
            'messages': ctx.messages,
            'n_prompt': ctx.n_prompt,
            'cond_ids': ctx.cond_ids,
            'asst_ids': ctx.asst_ids,
            'features': {
                k: (_lp_to_jsonable(v) if k.endswith('_lp') else v)
                for k, v in ctx.features.items()
            },
            'scores': {
                name: {
                    'score': r.score,
                    'passed': r.passed,
                    'extras': r.extras
                }
                for name, r in scores.items()
            },
            'kept': bool(kept),
        }

    # ---- aggregation & row reassembly ------------------------------------

    def _apply_filter(
        self,
        rows: List[Dict[str, Any]],
        contexts: List[RoundContext],
        score_table: List[Dict[str, ScoreResult]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        per_row: Dict[int, Dict[str, Any]] = {}
        for i, ctx in enumerate(contexts):
            scores = score_table[i] if i < len(score_table) else {}
            passed = all(r.passed for r in scores.values()) if scores else True
            slot = per_row.setdefault(ctx.row_idx, {
                'kept': [],
                'failed': 0,
            })
            if passed:
                slot['kept'].append(ctx.asst_idx)
            else:
                slot['failed'] += 1

        out: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []
        n_removed_rounds = 0
        n_removed_rows = 0
        for ri, row in enumerate(rows):
            user_data = row.get('user_data') if isinstance(row, dict) else None
            kr_val = _user_data_lookup(user_data, 'key_rounds')
            had_key_rounds = isinstance(kr_val, list) and bool(kr_val)
            decision = per_row.get(ri)

            if decision is None:
                # Row produced no contexts (no asst turns or filtered by intent).
                if had_key_rounds and not self._keep_if_no_key_rounds:
                    n_removed_rows += 1
                    dropped.append(dict(row, drop_reason='score_no_context'))
                    continue
                if self._intents is not None and not self._keep_if_no_key_rounds:
                    n_removed_rows += 1
                    dropped.append(dict(row, drop_reason='score_no_context'))
                    continue
                out.append(row)
                continue

            n_removed_rounds += decision['failed']
            kept = decision['kept']
            if had_key_rounds:
                if not kept:
                    n_removed_rows += 1
                    dropped.append(dict(row, drop_reason='score_all_rounds_failed'))
                    continue
                new_row = dict(row)
                # Re-pack key_rounds; keep all other entries as-is (already packed).
                rebuilt = [(k, v) for (k, v) in (user_data or []) if k != 'key_rounds']
                rebuilt.append(('key_rounds', pack_value(list(kept))))
                new_row['user_data'] = rebuilt
                out.append(new_row)
            else:
                if decision['failed'] > 0 and self._drop_row_on_any_fail:
                    n_removed_rows += 1
                    dropped.append(dict(row, drop_reason='score_round_failed'))
                    continue
                out.append(row)

        logger.info(f'[ScoreFilter] removed {n_removed_rounds} rounds, '
                    f'dropped {n_removed_rows} rows, kept {len(out)}/{len(rows)}')
        return out, dropped
