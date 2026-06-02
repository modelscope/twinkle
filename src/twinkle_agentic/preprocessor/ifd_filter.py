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
# Drop positions where asst NLL falls below this floor: token is mechanically forced,
# averaging it pulls both numerator/denominator to noise.
_NLL_NOISE_FLOOR = 0.01
# Skip the first 2 A-token positions: idx 0 has no prior context (lp=None),
# idx 1 is a degenerate constant (~12.32 across all samples) since `<think>`
# always tokenizes the same way; including it injects fixed bias.
_HEAD_SKIP = 2
# Qwen3.5 `<think>` token id; used to detect GT-style thinking prefix so paraphrase
# (which does NOT start with `<think>`) can skip 0 head positions instead of 2.
_THINK_OPEN_ID = 248068


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


def _aligned_head_nlls(
    asst_lp: List, asst_ids: List[int],
    cond_lp: List, cond_ids: List[int],
    n_prompt: int, start: int, end: int,
    floor: float = _NLL_NOISE_FLOOR,
) -> Tuple[Optional[float], Optional[float], int]:
    """Compute (cond_avg_nll, asst_avg_nll, n_kept) over the SAME A-token positions in both paths.

    A position is dropped if either path lacks a logprob, or asst NLL is below `floor`
    (mechanically forced token, no information). Both paths must average over the same
    position set so that the IFD ratio remains meaningful.
    """
    a_n = min(len(asst_lp), len(asst_ids), end)
    c_n = min(len(cond_lp), len(cond_ids))
    cond_vals: List[float] = []
    asst_vals: List[float] = []
    for i in range(start, a_n):
        c_idx = n_prompt + i
        if c_idx >= c_n:
            break
        a_lp = _extract_logprob(asst_lp[i], asst_ids[i])
        c_lp = _extract_logprob(cond_lp[c_idx], cond_ids[c_idx])
        if a_lp is None or c_lp is None:
            continue
        a_nll = -a_lp
        if a_nll < floor:
            continue
        asst_vals.append(a_nll)
        cond_vals.append(-c_lp)
    if len(asst_vals) < _MIN_RESPONSE_TOKENS:
        return None, None, len(asst_vals)
    return sum(cond_vals) / len(cond_vals), sum(asst_vals) / len(asst_vals), len(asst_vals)


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
        head_k: int = 64,
        max_prompt_tokens: int = 1024,
        # Diagnostic sampling: re-answer rounds whose intent is in this set, attach to dump.
        diagnostic_sample_intents: Optional[List[str]] = None,
        diagnostic_sample_n: int = 4,
        diagnostic_sample_temperature: float = 0.7,
        diagnostic_sample_max_tokens: int = 4096,
        # Paraphrase mode: replace GT with a model paraphrase produced under GT-injected
        # prompt, then score the paraphrase against the original (no-GT) context.
        # Bypasses filtering; rows pass through unchanged.
        # Accepts False (GT only), True (paraphrase only), or 'both' (dump two files).
        paraphrase_mode="both",
        paraphrase_temperature: float = 0.7,
        paraphrase_max_tokens: int = 4096,
        # Restrict paraphrase to rounds whose intent is in this set (e.g. {'math'}).
        # Empty/None = paraphrase ALL prepared rounds.
        paraphrase_intents: Optional[List[str]] = None,
        # Token budget for the augmented (GT-injected) prompt sent to chat_batch.
        # Must be <= max_model_len - paraphrase_max_tokens to avoid vLLM rejection.
        paraphrase_prompt_budget: int = 4096,
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
        self._head_k = head_k
        self._max_prompt_tokens = max_prompt_tokens
        self._diag_sample_intents = set(diagnostic_sample_intents or [])
        self._diag_sample_n = max(1, int(diagnostic_sample_n))
        self._diag_sample_temperature = float(diagnostic_sample_temperature)
        self._diag_sample_max_tokens = int(diagnostic_sample_max_tokens)
        self._paraphrase_mode = 'both' if paraphrase_mode == 'both' else bool(paraphrase_mode)
        self._paraphrase_temperature = float(paraphrase_temperature)
        self._paraphrase_max_tokens = int(paraphrase_max_tokens)
        self._paraphrase_intents = set(paraphrase_intents or [])
        self._paraphrase_prompt_budget = int(paraphrase_prompt_budget)

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
    ) -> Dict[Tuple[int, int], float]:
        """Run Phase 2 (cond/asst logprobs + IFD scoring) and dump records under given prefix."""
        scores: Dict[Tuple[int, int], float] = {}
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
        head_k = self._head_k
        head_nlls: Dict[Tuple[int, int], Tuple[float, float, int]] = {}
        for key, cond_lp, asst_lp in zip(keys, cond_logprobs, asst_logprobs):
            cond_ids, n_prompt, asst_ids = prepared[key]
            # GT starts with `<think>` (skip 2 degenerate head positions); paraphrase usually
            # does not, so skip 0 to avoid throwing away the first 2 informative tokens.
            a_start = _HEAD_SKIP if (asst_ids and asst_ids[0] == _THINK_OPEN_ID) else 0
            a_end = (a_start + head_k) if head_k > 0 else len(asst_ids)
            l_a_given_q, l_a, n_kept = _aligned_head_nlls(
                asst_lp, asst_ids, cond_lp, cond_ids, n_prompt, a_start, a_end)
            if l_a_given_q is None or l_a is None or l_a < 1e-8:
                continue
            ifd = l_a_given_q / l_a
            if math.isfinite(ifd):
                scores[key] = ifd
                head_nlls[key] = (l_a_given_q, l_a, n_kept)
        self._dump_records(rows, prepared, keys, cond_logprobs, asst_logprobs, scores,
                           head_nlls, samples_by_key or {}, paraphrases_by_key, dump_prefix)
        return scores

    def _dump_records(self, rows, prepared, keys, cond_logprobs, asst_logprobs, scores,
                      head_nlls=None, samples_by_key=None, paraphrases_by_key=None,
                      dump_prefix='ifd_dump'):
        """TEMP: dump per-round messages + raw logprobs for offline IFD diagnosis."""
        try:
            import json, os, time
            dump_path = f'{dump_prefix}_{os.getpid()}_{int(time.time())}.jsonl'
            head_nlls = head_nlls or {}
            samples_by_key = samples_by_key or {}
            paraphrases_by_key = paraphrases_by_key or {}
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
                    cond_nll_head, asst_nll_head, n_kept_head = (None, None, None)
                    if key in head_nlls:
                        cond_nll_head, asst_nll_head, n_kept_head = head_nlls[key]
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
                        'ifd': scores.get(key),
                        'cond_nll_head': cond_nll_head,
                        'asst_nll_head': asst_nll_head,
                        'n_kept_head': n_kept_head,
                        'diagnostic_samples': samples_by_key.get(key) or [],
                        'paraphrase': paraphrases_by_key.get(key),
                    }, ensure_ascii=False) + '\n')
            logger.info(f'[IFDFilter] dumped {len(keys)} records to {dump_path}')
        except Exception as e:
            logger.warning(f'[IFDFilter] dump failed: {e}')

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

        # Mode dispatch: paraphrase_mode in (False, True, 'both').
        mode = self._paraphrase_mode
        run_gt = mode in (False, 'both')
        run_para = mode in (True, 'both')

        # Diagnostic sampling uses the original (no-GT) prompt and is independent of mode.
        # Run ONCE here so both GT and paraphrase dumps share the same samples (avoids
        # double cost and divergent stochastic outputs across the two dump files).
        samples_by_key = self._collect_diagnostic_samples(rows, prepared)

        paraphrases_by_key: Dict[Tuple[int, int], str] = {}
        prepared_para: Optional[Dict[Tuple[int, int], Tuple[List[int], int, List[int]]]] = None
        if run_para and prepared:
            prepared_para, paraphrases_by_key = self._paraphrase_rounds(rows, prepared)

        scores: Dict[Tuple[int, int], float] = {}
        if run_gt:
            scores = self._score_and_dump(rows, prepared, {}, dump_prefix='ifd_dump',
                                          samples_by_key=samples_by_key)
        if run_para and prepared_para:
            self._score_and_dump(rows, prepared_para, paraphrases_by_key,
                                 dump_prefix='ifd_paraphrase_dump',
                                 samples_by_key=samples_by_key)

        # Any paraphrase variant is diagnostic-only: skip filter, return rows unchanged.
        if run_para:
            return rows

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
