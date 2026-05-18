"""Cold-start SFT dataset builder for the condensed multi-hop QA task.

Pipeline per HotpotQA distractor row:
  1. Build the standard system + user-with-context trajectory using the
     production ``SYSTEM_PROMPT`` and ``_format_context`` from
     ``cookbook/rl/grpo_condensed.py`` so the offline data matches what
     the policy sees at training/inference time.
  2. Run the production ``NativeChunker`` + ``ModelCondenser`` on the
     row to produce ``<block_N>...</block_N>`` compressed text.
  3. **Validation pass** (super-LLM, ``enable_thinking=True``, no oracle,
     no tools): judge whether the question / supporting_facts / GT are
     well-formed against the raw passages; return strict JSON
     ``{"verdict": "ok"|"fix"|"drop", ...}`` with fixed SF + GT when
     applicable. ``drop`` skips the row.
  4. **Oracle rollout pass** via :class:`APIMultiTurnRollout` with a
     trajectory-bound :class:`ExtractCondensed` tool. The oracle hint
     (SF titles + GT) is injected into the system prompt **only for
     the API call**; it is stripped before saving. The model emits
     OpenAI-shape ``tool_calls`` for ``extract_condensed``, the rollout
     dispatches them through :class:`ToolManager` and feeds back the
     pre-compression passage text as a ``tool`` message, looping until
     the model finalises with ``\\boxed{...}`` or hits ``MAX_TURNS``.
  5. Accept iff F1(boxed, used_gt) >= ``F1_ACCEPT_THRESHOLD``. On miss,
     retry once with a higher temperature.
  6. Convert OpenAI-shape ``tool_calls`` into the textual
     ``<tool_call><function=extract_condensed><parameter=blocks>N</parameter></function></tool_call>``
     format consumed by the training chat template (mirrors
     ``grpo_condensed.SYSTEM_PROMPT`` L232-239), restore the clean
     system prompt, and emit one JSONL line.

Run::

    python cookbook/rl/make_condensed_sft.py \\
        --output hotpotqa_sft_coldstart.jsonl \\
        --model <super-llm> --api-key $KEY --base-url $URL \\
        --total 9000 --easy 1500 --medium 3000 --hard 4500 \\
        --concurrency 16 --seed 42 \\
        --condenser-model-id ms://Qwen/Qwen3.5-4B \\
        --condenser-lora ms://twinkle-kit/Qwen3.5-4B-Condenser
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from twinkle.data_format.sampling import SamplingParams
from twinkle.sampler import vLLMSampler
from twinkle.template import Qwen3_5Template
from twinkle_agentic.chunker.native import NativeChunker
from twinkle_agentic.condenser import ModelCondenser
from twinkle_agentic.data_format import Chunks
from twinkle_agentic.protocol.openai import OpenAI
from twinkle_agentic.reward.f1 import _extract_final_answer, _f1_score
from twinkle_agentic.rollout import APIMultiTurnRollout
from twinkle_agentic.tools.extract_condensed import ExtractCondensed
from twinkle_agentic.tools.tool_manager import ToolManager


# --------------------------------------------------------------------------
# Constants mirrored from grpo_condensed.py so the SFT data matches the
# runtime contract byte-for-byte. Re-import would pull the whole training
# module; copying these few strings keeps the builder standalone.
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a careful multi-hop QA assistant.

## Context Format (Mixed)
The context you receive is a **mix of two forms**:

1. **Compressed blocks** — long passages wrapped in `<block_N>...</block_N>`, \
displayed as a Markdown digest in **telegraphic style** (no \
articles / "is" / "are"; colons and commas mean "is" / "has") \
with two sections:
   - **Summary**: overview plus facts strongly related to the question, stated explicitly.
   - **More**: a collapsed INDEX of category keywords hinting at extra details hidden in the full text (call `extract_condensed` to see them).
   Reading example: `India: 7th largest by area. Borders: Pakistan, \
China.` means "India is the 7th largest country by area and \
shares borders with Pakistan and China."
2. **Raw passages** — short passages shown inline as plain text (`Title: \
body`) **without** any `<block_N>` wrapping. These are already the full \
text; nothing is hidden.

Only the `<block_N>`-wrapped blocks are compressed and can be expanded. \
Block ids `N` are 1-based and assigned in the order compressed blocks \
appear in the context, so they are always contiguous (`<block_1>`, \
`<block_2>`, `<block_3>`, ...). Raw passages have no block id and cannot \
be extracted — they are already complete.

## Workflow

### Phase 1 — Scan and Decide
Step 1: Read each compressed block's Summary, and read raw \
passages directly, to get an overview.
Step 2: For compressed blocks, check the More keywords to judge whether \
hidden details are needed.
Step 3: Decide which compressed blocks to expand, then call \
`extract_condensed` with their block ids. Raw passages need no extraction.

### Phase 2 — Reason and Answer
After the tool returns the full text, continue stepping through the evidence:
Step N:   From block X (or the raw passage titled "..."), I learn that [fact A].
Step N+1: From block Y, I need to call `extract_condensed` to get more information, because this block is related to...
Step N+2: Combining these, the answer is ...
\\boxed{answer}

You may call `extract_condensed` several times to expand more blocks if the information is not enough, only answer the question if you are sure about the facts.
The `blocks` parameter accepts **exactly one integer** per call (e.g. `3`); lists are rejected. Expand additional blocks by issuing separate `extract_condensed` calls, one per block. Only pass ids that actually appear as `<block_N>` in the context, and do **not** request the same block twice — its text is already in the conversation after the first expansion.

## Tool Call Format
<tool_call>
<function=extract_condensed>
<parameter=blocks>
3
</parameter>
</function>
</tool_call>

## Output Format
End your final response with \\boxed{answer}, e.g. \\boxed{Delhi}.
Keep the boxed text short: a name, entity, date, or "yes"/"no".
Answers not inside \\boxed{} will not be scored."""


# Oracle suffix appended ONLY for API generation; stripped before save.
_ORACLE_HINT_TEMPLATE = (
    '\n\n## Oracle hint (PRIVATE — do NOT quote verbatim)\n'
    'The following supporting-fact titles and ground-truth answer are '
    'provided to make your final answer reliable. Use them as a signpost '
    'while you reason from the context; your final `\\boxed{{...}}` MUST '
    'paraphrase the ground truth using evidence from the blocks (after '
    'expanding compressed blocks when needed), not just echo it.\n'
    'Supporting facts (titles): {sf}\n'
    'Ground truth: {gt}\n'
    'You MUST still call `extract_condensed` on EVERY compressed block '
    'whose Summary or More keywords touch any supporting-fact title, even '
    'if the Summary already seems to state the answer — the compressed '
    'Summary occasionally loses pronoun referents or attribution and the '
    'raw passage is the authoritative source.'
)


VALIDATION_SYSTEM = (
    'You are a HotpotQA annotation auditor. Read the raw passages, the '
    'question, the supplied supporting-fact titles and the supplied '
    'ground-truth answer. Decide whether this row is usable for training '
    'a multi-hop QA model.\n\n'
    'Pathologies to catch (drop or fix):\n'
    '  - question template leakage: the question literally contains the '
    'answer, references a passage id, or is malformed;\n'
    '  - subject/answer mismatch: the GT does not actually answer the '
    'question given the passages (e.g. the question asks about an event '
    'X but GT is from a sibling event Y);\n'
    '  - GT entity not present in any passage AND not directly inferable '
    'by a 2-hop bridge from the passages;\n'
    '  - supporting-fact titles obviously incomplete for a 2-hop question.\n'
    '\n'
    'Return STRICT JSON ONLY (no markdown fence, no preamble) with this '
    'exact shape:\n'
    '  {"verdict": "ok"|"fix"|"drop", "reason": "<short>", '
    '"fixed_supporting_facts": ["<title>", ...], '
    '"fixed_ground_truth": "<short answer>"}\n'
    'Use verdict "ok" when the supplied SF + GT are correct (then '
    '"fixed_supporting_facts" and "fixed_ground_truth" MAY be empty). '
    'Use verdict "fix" when the question is answerable but SF or GT are '
    'wrong/incomplete -- fill the fixed fields with the corrected values, '
    'titles drawn verbatim from the passage titles below. Use verdict '
    '"drop" when the question itself is invalid or unanswerable from the '
    'given passages.'
)


VALIDATION_USER_TEMPLATE = (
    'Question: {question}\n'
    '\n'
    'Supplied supporting-fact titles: {sf}\n'
    'Supplied ground truth: {gt}\n'
    '\n'
    'Passage titles (verbatim):\n{titles}\n'
    '\n'
    'Passages (raw, uncompressed):\n\n{passages}'
)


# JSON Schema for the OpenAI API; the in-process ExtractCondensed tool's
# tool_info() emits a free-form description that the OpenAI SDK rejects.
EXTRACT_CONDENSED_TOOL: Dict[str, Any] = {
    'type': 'function',
    'function': {
        'name': 'extract_condensed',
        'description': (
            'Recover the full, uncompressed text of ONE previously '
            'condensed passage, identified by its <block_N> tag. Use '
            'this tool whenever you need to re-read the original detail '
            'of a compressed block. Each call expands exactly one block; '
            'issue separate calls for additional blocks, and do not '
            'request the same block twice.'),
        'parameters': {
            'type': 'object',
            'properties': {
                'blocks': {
                    'type': 'integer',
                    'description': (
                        'The 1-indexed block number N appearing inside '
                        '<block_N>...</block_N>. Exactly one block per '
                        'call (e.g. 3); lists are rejected.'),
                },
            },
            'required': ['blocks'],
        },
    },
}


F1_ACCEPT_THRESHOLD: float = 0.5
ROLLOUT_MAX_TURNS: int = 8
ROLLOUT_MAX_TOKENS: int = 2048
VALIDATION_MAX_TOKENS: int = 1024
ROLLOUT_TEMPERATURE_LADDER: Tuple[float, ...] = (0.4, 0.7)


# --------------------------------------------------------------------------
# Trajectory + chunk helpers (mirror HotpotQAProcessor + production prompt).
# --------------------------------------------------------------------------
def _format_passage(title: str, sentences: Any) -> str:
    if isinstance(sentences, list):
        body = ' '.join(s.strip() for s in sentences if s and s.strip())
    else:
        body = str(sentences).strip()
    return f'{title}: {body}'


def _format_context(titles: List[str], sentences_list: List[Any]) -> str:
    return '\n\n'.join(
        _format_passage(t, s) for t, s in zip(titles, sentences_list))


def _build_initial_trajectory(row: Dict[str, Any]) -> Dict[str, Any]:
    """Build the pre-compression trajectory dict the chunker expects."""
    ctx = row.get('context') or {}
    titles = list(ctx.get('title') or [])
    sentences_list = list(ctx.get('sentences') or [])
    user_msg = (
        f"Question: {row['question']}\n\n"
        f"Context:\n\n{_format_context(titles, sentences_list)}")
    return {
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_msg},
        ],
    }


def _extract_question_from_chunk(chunk):
    content = chunk.get('content')
    if chunk.get('type') != 'text' or not isinstance(content, str):
        return None
    m = re.search(r'\AQuestion:\s*(.+)', content)
    return m.group(1).strip() if m else None


# --------------------------------------------------------------------------
# Per-batch compression (re-use MultiTurnCondenseRollout's batching trick:
# merge all per-row chunks into ONE Chunks so the sampler sees a packed batch).
# --------------------------------------------------------------------------
def compress_rows(
    rows: List[Dict[str, Any]],
    chunker: NativeChunker,
    condenser: ModelCondenser,
) -> List[Tuple[Dict[str, Any], Chunks]]:
    """Return ``[(compressed_trajectory_dict, per_row_Chunks), ...]``.

    ``compressed_trajectory_dict`` already has ``<block_N>...</block_N>``
    wrapping in its user message (see :meth:`Chunks.to_trajectory`).
    ``per_row_Chunks`` carries ``raw.original`` snapshots so
    :class:`ExtractCondensed` can return the pre-compression text.
    """
    if not rows:
        return []
    initial = [_build_initial_trajectory(r) for r in rows]
    per_row_chunks = [chunker(t) for t in initial]
    merged_list: List[Any] = []
    boundaries: List[int] = []
    for ck in per_row_chunks:
        merged_list.extend(ck.chunks)
        boundaries.append(len(merged_list))
    merged = condenser(Chunks(chunks=merged_list))
    out: List[Tuple[Dict[str, Any], Chunks]] = []
    start = 0
    for end in boundaries:
        slc = Chunks(chunks=list(merged.chunks[start:end]))
        out.append((slc.to_trajectory(), slc))
        start = end
    return out


# --------------------------------------------------------------------------
# Stage 1: validation pass.
# --------------------------------------------------------------------------
_JSON_FENCE_RE = re.compile(r'```(?:json)?\s*\n(.*?)\n```', re.DOTALL)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON parse: strip fence, then locate first ``{...}`` block."""
    if not text:
        return None
    candidate = text.strip()
    m = _JSON_FENCE_RE.search(candidate)
    if m:
        candidate = m.group(1).strip()
    depth = 0
    start = -1
    for i, ch in enumerate(candidate):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start != -1:
                blob = candidate[start:i + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    start = -1
                    continue
    return None


def validate_row(
    api: OpenAI, row: Dict[str, Any], original_gt: List[str], sf_titles: List[str],
) -> Optional[Dict[str, Any]]:
    """Return parsed JSON verdict, or ``None`` on unrecoverable parse failure."""
    ctx = row.get('context') or {}
    titles = list(ctx.get('title') or [])
    sentences_list = list(ctx.get('sentences') or [])
    passages = _format_context(titles, sentences_list)
    user = VALIDATION_USER_TEMPLATE.format(
        question=row['question'],
        sf=json.dumps(sf_titles, ensure_ascii=False),
        gt=json.dumps(original_gt, ensure_ascii=False),
        titles='\n'.join(f'- {t}' for t in titles),
        passages=passages,
    )
    trajectory = {
        'messages': [
            {'role': 'system', 'content': VALIDATION_SYSTEM},
            {'role': 'user', 'content': user},
        ],
    }
    sp = SamplingParams(
        temperature=0.0, max_tokens=VALIDATION_MAX_TOKENS, num_samples=1)
    for attempt in range(2):
        try:
            reply = api(
                trajectory, sp, extra_body={'enable_thinking': True})
        except Exception as exc:
            sys.stderr.write(f'[validate] row={row.get("id")} attempt={attempt} api error: {exc}\n')
            return None
        content = reply.get('content') or ''
        parsed = _extract_json_object(content)
        if parsed and parsed.get('verdict') in ('ok', 'fix', 'drop'):
            return parsed
    return None


def resolve_validation(
    verdict: Dict[str, Any], original_gt: List[str], sf_titles: List[str],
) -> Tuple[List[str], List[str]]:
    """Pick the SF + GT list to use downstream based on verdict."""
    v = verdict.get('verdict')
    if v == 'fix':
        fixed_gt = verdict.get('fixed_ground_truth') or ''
        fixed_sf = verdict.get('fixed_supporting_facts') or []
        gt_list: List[str] = []
        if isinstance(fixed_gt, list):
            gt_list = [str(x).strip() for x in fixed_gt if str(x).strip()]
        elif isinstance(fixed_gt, str) and fixed_gt.strip():
            gt_list = [fixed_gt.strip()]
        if not gt_list:
            gt_list = original_gt
        sf_list = (
            [str(x).strip() for x in fixed_sf if str(x).strip()]
            if isinstance(fixed_sf, list) else sf_titles)
        if not sf_list:
            sf_list = sf_titles
        return gt_list, sf_list
    return original_gt, sf_titles


# --------------------------------------------------------------------------
# Stage 2 prep: build oracle trajectory + per-trajectory ToolManager.
# --------------------------------------------------------------------------
def _oracle_system_prompt(sf_titles: List[str], gt_list: List[str]) -> str:
    sf_render = ', '.join(repr(t) for t in sf_titles) if sf_titles else '(none)'
    gt_render = ' | '.join(gt_list) if gt_list else '(unknown)'
    return SYSTEM_PROMPT + _ORACLE_HINT_TEMPLATE.format(
        sf=sf_render, gt=gt_render)


def _build_oracle_trajectory(
    compressed_traj: Dict[str, Any],
    sf_titles: List[str],
    gt_list: List[str],
) -> Dict[str, Any]:
    """Replace the system message with the oracle-suffixed variant and
    attach the JSON-schema tools field consumed by the OpenAI API."""
    oracle_sp = _oracle_system_prompt(sf_titles, gt_list)
    out_messages: List[Dict[str, Any]] = []
    sys_inserted = False
    for m in compressed_traj.get('messages') or []:
        if m.get('role') == 'system' and not sys_inserted:
            out_messages.append({'role': 'system', 'content': oracle_sp})
            sys_inserted = True
        else:
            out_messages.append(dict(m))
    if not sys_inserted:
        out_messages.insert(0, {'role': 'system', 'content': oracle_sp})
    return {
        'messages': out_messages,
        'tools': [EXTRACT_CONDENSED_TOOL],
    }


def _make_tool_manager(chunks: Chunks) -> ToolManager:
    """One ToolManager + ExtractCondensed per trajectory; the tool keeps
    a ``_already_expanded`` set, so reusing across trials would lie to
    the model on retry."""
    tm = ToolManager()
    tm.register(ExtractCondensed(chunks))
    return tm


# --------------------------------------------------------------------------
# Stage 3 + 4: F1 acceptance + conversion to training-runtime format.
# --------------------------------------------------------------------------
def boxed_f1(boxed: str, gt_list: List[str]) -> float:
    if not boxed or not gt_list:
        return 0.0
    return max(_f1_score(boxed, g)[0] for g in gt_list)


def _last_assistant_text(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get('role') == 'assistant' and isinstance(m.get('content'), str):
            return m['content']
    return ''


def _format_tool_call_text(blocks: int) -> str:
    return (
        '<tool_call>\n'
        '<function=extract_condensed>\n'
        '<parameter=blocks>\n'
        f'{blocks}\n'
        '</parameter>\n'
        '</function>\n'
        '</tool_call>'
    )


def convert_to_runtime_messages(
    api_messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """OpenAI tool_calls -> textual <tool_call> format consumed by the
    training chat template. The first system message has its oracle
    suffix stripped (we just replace it with the clean SYSTEM_PROMPT).
    """
    out: List[Dict[str, Any]] = []
    sys_done = False
    for m in api_messages:
        role = m.get('role')
        if role == 'system' and not sys_done:
            out.append({'role': 'system', 'content': SYSTEM_PROMPT})
            sys_done = True
            continue
        if role == 'assistant':
            content = m.get('content') or ''
            tool_calls = m.get('tool_calls') or []
            if tool_calls:
                pieces = [content.rstrip()] if content else []
                for tc in tool_calls:
                    fn = tc.get('function') or {}
                    args_raw = fn.get('arguments')
                    try:
                        args = (
                            json.loads(args_raw) if isinstance(args_raw, str)
                            else (args_raw or {}))
                    except json.JSONDecodeError:
                        args = {}
                    blocks_val = args.get('blocks', args.get('block'))
                    try:
                        n = int(blocks_val)
                    except (TypeError, ValueError):
                        continue
                    pieces.append(_format_tool_call_text(n))
                text = '\n\n'.join(p for p in pieces if p)
                out.append({'role': 'assistant', 'content': text})
            else:
                out.append({'role': 'assistant', 'content': content})
            continue
        if role == 'tool':
            out.append({'role': 'tool', 'content': m.get('content') or ''})
            continue
        out.append({k: v for k, v in m.items() if k in ('role', 'content')})
    return out


def trajectory_achieved_ratio(chunks: Chunks) -> float:
    total_src = 0
    total_cmp = 0
    for c in chunks.chunks:
        if c.get('type') != 'text':
            continue
        raw = c.get('raw')
        if not (isinstance(raw, dict) and raw.get('condensed')):
            continue
        original = raw.get('original')
        compressed = c.get('content')
        if isinstance(original, str) and isinstance(compressed, str):
            total_src += len(original)
            total_cmp += len(compressed)
    return round(total_cmp / total_src, 4) if total_src else 0.0


def build_record(
    row: Dict[str, Any],
    runtime_messages: List[Dict[str, Any]],
    chunks: Chunks,
    verdict: Dict[str, Any],
    original_gt: List[str],
    used_gt: List[str],
    used_sf: List[str],
    boxed: str,
    f1: float,
    num_tool_calls: int,
) -> Dict[str, Any]:
    ctx = row.get('context') or {}
    titles = list(ctx.get('title') or [])
    sentences_list = list(ctx.get('sentences') or [])
    raw_passages = [
        {
            'title': t,
            'sentences': list(s) if isinstance(s, list) else [str(s)],
        }
        for t, s in zip(titles, sentences_list)
    ]
    sf_full = row.get('supporting_facts') or {}
    return {
        'id': row['id'],
        'level': row.get('level'),
        'type': row.get('type'),
        'messages': runtime_messages,
        'tools': [EXTRACT_CONDENSED_TOOL],
        'meta': {
            'num_tool_calls': num_tool_calls,
            'achieved_ratio': trajectory_achieved_ratio(chunks),
            'validation_verdict': verdict.get('verdict'),
            'validation_reason': verdict.get('reason'),
            'original_question': row.get('question'),
            'original_answer': row.get('answer'),
            'original_gt': original_gt,
            'used_gt': used_gt,
            'used_supporting_facts': used_sf,
            'original_supporting_facts': {
                'title': list(sf_full.get('title') or []),
                'sent_id': list(sf_full.get('sent_id') or []),
            },
            'original_passages': raw_passages,
            'f1': round(f1, 4),
            'boxed': boxed,
        },
    }


# --------------------------------------------------------------------------
# Per-batch pipeline orchestration.
# --------------------------------------------------------------------------
def _extract_original_gt_sf(row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    answers = row.get('answers')
    if isinstance(answers, list) and answers:
        original_gt = [str(a).strip() for a in answers if str(a).strip()]
    else:
        original_gt = [(row.get('answer', '') or '').strip()]
    original_gt = [g for g in original_gt if g]
    sf = row.get('supporting_facts') or {}
    sf_titles = list(dict.fromkeys(t for t in (sf.get('title') or []) if t))
    return original_gt, sf_titles


def _validate_in_parallel(
    api: OpenAI, batch: List[Dict[str, Any]], pool: ThreadPoolExecutor,
) -> Tuple[List[Optional[Dict[str, Any]]], List[Tuple[List[str], List[str]]]]:
    """Run ``validate_row`` for every row in parallel (one OpenAI call each)."""
    futures = []
    payloads: List[Tuple[List[str], List[str]]] = []
    for row in batch:
        original_gt, sf_titles = _extract_original_gt_sf(row)
        payloads.append((original_gt, sf_titles))
        futures.append(pool.submit(
            validate_row, api, row, original_gt, sf_titles))
    verdicts: List[Optional[Dict[str, Any]]] = [f.result() for f in futures]
    return verdicts, payloads


def _num_tool_calls(messages: List[Dict[str, Any]]) -> int:
    return sum(
        len(m.get('tool_calls') or [])
        for m in messages if m.get('role') == 'assistant')


def process_batch(
    api: OpenAI,
    rollout: APIMultiTurnRollout,
    batch: List[Dict[str, Any]],
    chunker: NativeChunker,
    condenser: ModelCondenser,
    validation_pool: ThreadPoolExecutor,
) -> List[Dict[str, Any]]:
    """Validate -> compress -> rollout (T-ladder) -> accept. Returns the
    list of accepted JSONL records for the batch."""
    if not batch:
        return []
    # 1. Validation in parallel.
    verdicts, payloads = _validate_in_parallel(api, batch, validation_pool)

    survivors_meta: List[Dict[str, Any]] = []
    for row, verdict, (original_gt, sf_titles) in zip(batch, verdicts, payloads):
        if verdict is None or verdict.get('verdict') == 'drop':
            continue
        if not original_gt:
            continue
        used_gt, used_sf = resolve_validation(verdict, original_gt, sf_titles)
        if not used_gt:
            continue
        survivors_meta.append({
            'row': row, 'verdict': verdict,
            'original_gt': original_gt,
            'used_gt': used_gt, 'used_sf': used_sf,
        })
    if not survivors_meta:
        return []

    # 2. Compress survivors (one packed batch through ModelCondenser).
    survivor_rows = [m['row'] for m in survivors_meta]
    try:
        compressed = compress_rows(survivor_rows, chunker, condenser)
    except Exception as exc:
        sys.stderr.write(f'[compress] batch crashed: {exc}\n')
        return []

    # 3. Build oracle trajectories + per-trajectory ToolManagers.
    trajs: List[Dict[str, Any]] = []
    chunks_list: List[Chunks] = []
    for meta, (compressed_traj, chunks) in zip(survivors_meta, compressed):
        trajs.append(_build_oracle_trajectory(
            compressed_traj, meta['used_sf'], meta['used_gt']))
        chunks_list.append(chunks)

    # 4. Temperature ladder. Each rung gets fresh ExtractCondensed tools so
    #    a retry does not see the previous attempt's already-expanded set.
    accepted: List[Dict[str, Any]] = []
    pending_idx = list(range(len(trajs)))
    for temperature in ROLLOUT_TEMPERATURE_LADDER:
        if not pending_idx:
            break
        sp = SamplingParams(
            temperature=temperature, max_tokens=ROLLOUT_MAX_TOKENS, num_samples=1)
        run_trajs = [trajs[i] for i in pending_idx]
        run_tms = [_make_tool_manager(chunks_list[i]) for i in pending_idx]
        try:
            outs = rollout(
                run_trajs, tool_manager=run_tms, sampling_params=sp)
        except Exception as exc:
            sys.stderr.write(f'[rollout] batch crashed at T={temperature}: {exc}\n')
            return accepted
        next_pending: List[int] = []
        for local_pos, traj_idx in enumerate(pending_idx):
            out_traj = outs[local_pos]
            if out_traj.get('stop_reason') == 'api_error':
                continue  # hard-drop API failures, do not retry
            messages = out_traj.get('messages') or []
            boxed = _extract_final_answer(_last_assistant_text(messages))
            meta = survivors_meta[traj_idx]
            f1 = boxed_f1(boxed, meta['used_gt'])
            if f1 >= F1_ACCEPT_THRESHOLD:
                runtime_messages = convert_to_runtime_messages(messages)
                accepted.append(build_record(
                    row=meta['row'],
                    runtime_messages=runtime_messages,
                    chunks=chunks_list[traj_idx],
                    verdict=meta['verdict'],
                    original_gt=meta['original_gt'],
                    used_gt=meta['used_gt'],
                    used_sf=meta['used_sf'],
                    boxed=boxed, f1=f1,
                    num_tool_calls=_num_tool_calls(messages)))
            else:
                next_pending.append(traj_idx)
        pending_idx = next_pending
    return accepted


# --------------------------------------------------------------------------
# Stratified sampling + resume.
# --------------------------------------------------------------------------
LEVELS: Tuple[str, str, str] = ('easy', 'medium', 'hard')


def stratified_sample(
    ds, per_level: Dict[str, int], seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    buckets: Dict[str, List[int]] = {lv: [] for lv in LEVELS}
    for i, lv in enumerate(ds['level']):
        if lv in buckets:
            buckets[lv].append(i)
    picked: List[int] = []
    for lv in LEVELS:
        need = per_level[lv]
        pool = buckets[lv]
        if len(pool) < need:
            raise RuntimeError(
                f'level={lv} has only {len(pool)} rows, need {need}')
        picked.extend(rng.sample(pool, need))
    rng.shuffle(picked)
    return [ds[int(i)] for i in picked]


def load_done_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = obj.get('id')
            if rid:
                done.add(rid)
    return done


def apply_reannotation_overlay(
    rows: List[Dict[str, Any]], path: str,
) -> List[Dict[str, Any]]:
    """Drop verdict=drop ids; overlay ``question_fixed`` and multi-form ``answers``.

    The validation stage in ``process_batch`` still runs on every survivor
    because the audit ran on a different HF subset (fullwiki) than this
    builder's default (distractor) and passage contexts differ.
    """
    overrides: Dict[str, Dict[str, Any]] = {}
    drop_ids: set = set()
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = obj.get('id')
            if not rid:
                continue
            if obj.get('verdict') == 'drop':
                drop_ids.add(rid)
            else:
                overrides[rid] = obj
    out: List[Dict[str, Any]] = []
    overridden = 0
    for row in rows:
        rid = row.get('id')
        if rid in drop_ids:
            continue
        ov = overrides.get(rid)
        if ov is not None:
            row = dict(row)
            qfix = (ov.get('question_fixed') or '').strip()
            if qfix:
                row['question'] = qfix
            ans = [str(a).strip() for a in (ov.get('answers') or []) if str(a).strip()]
            if ans:
                row['answers'] = ans
            overridden += 1
        out.append(row)
    sys.stderr.write(
        f'[REANNOTATED] {path}: {len(rows)} -> {len(out)} rows '
        f'(dropped={len(drop_ids)}, overridden={overridden})\n')
    return out


# --------------------------------------------------------------------------
# CLI + main loop.
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True,
                        help='Super-LLM model name (OpenAI-protocol).')
    parser.add_argument('--api-key', default=os.environ.get('OPENAI_API_KEY'))
    parser.add_argument('--base-url', default=os.environ.get('OPENAI_BASE_URL'))
    parser.add_argument('--total', type=int, default=12000)
    parser.add_argument('--easy', type=int, default=2000)
    parser.add_argument('--medium', type=int, default=4000)
    parser.add_argument('--hard', type=int, default=6000)
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reannotated', default=os.environ.get('REANNOTATED_FILE', ''),
                        help='Path to wrong_ids_reannotated.jsonl. Drops verdict=drop ids and overlays question_fixed + multi-form answers. Validation stage still runs because the audit was on a different HF subset.')
    parser.add_argument('--hf-subset', default='distractor')
    parser.add_argument('--hf-split', default='train')
    parser.add_argument('--condenser-model-id',
                        default=os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3.5-4B'))
    parser.add_argument('--condenser-lora',
                        default='ms://twinkle-kit/Qwen3.5-4B-Condenser')
    parser.add_argument('--chunk-size', type=int, default=1024)
    parser.add_argument('--hotpotqa-max-length', type=int, default=64000)
    parser.add_argument('--compress-batch-size', type=int, default=32,
                        help='How many rows to feed to ModelCondenser at once.')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.8)
    return parser.parse_args()


def build_condenser(args: argparse.Namespace) -> Tuple[NativeChunker, ModelCondenser]:
    sampler = vLLMSampler(
        model_id=args.condenser_model_id,
        engine_args={
            'gpu_memory_utilization': args.gpu_memory_utilization,
            'max_model_len': max(8192, args.hotpotqa_max_length),
            'max_lora_rank': 32,
            'enable_lora': True,
            'max_loras': 2,
        },
    )
    sampler.set_template(
        'Qwen3_5Template', model_id=args.condenser_model_id,
        enable_thinking=False, max_length=args.hotpotqa_max_length)
    rollout_template = Qwen3_5Template(
        args.condenser_model_id, max_length=args.hotpotqa_max_length,
        enable_thinking=False)
    chunker = NativeChunker(
        chunk_size=args.chunk_size,
        passage_boundary_re=r'(?<=\n\n)',
    )
    condenser = ModelCondenser(
        sampler=sampler,
        compression_ratio=2.0,
        sampling_params=SamplingParams(
            max_tokens=1024, num_samples=1, temperature=0.4, top_p=0.9),
        min_chars=200,
        template=rollout_template,
        lora_path=args.condenser_lora or None,
        skip_pattern=r'^Question:',
        related_query=_extract_question_from_chunk,
    )
    return chunker, condenser


def main() -> None:
    args = parse_args()
    if args.easy + args.medium + args.hard != args.total:
        raise ValueError(
            f'--easy + --medium + --hard ({args.easy + args.medium + args.hard}) '
            f'must equal --total ({args.total})')
    per_level = {'easy': args.easy, 'medium': args.medium, 'hard': args.hard}

    sys.stderr.write(
        f'Loading hotpotqa/hotpot_qa:{args.hf_subset}:{args.hf_split}...\n')
    ds = load_dataset(
        'hotpotqa/hotpot_qa', args.hf_subset, split=args.hf_split)

    rows = stratified_sample(ds, per_level=per_level, seed=args.seed)
    if args.reannotated.strip():
        rows = apply_reannotation_overlay(rows, args.reannotated.strip())
    done = load_done_ids(args.output)
    sys.stderr.write(f'Resume: {len(done)} rows already emitted.\n')
    pending = [r for r in rows if r['id'] not in done]
    sys.stderr.write(f'Pending: {len(pending)} / {len(rows)}\n')

    chunker, condenser = build_condenser(args)
    api = OpenAI(
        model=args.model, api_key=args.api_key, base_url=args.base_url)

    # APIMultiTurnRollout itself owns the per-trajectory thread pool. The
    # validation phase runs on a separate pool of equal size; both phases
    # are network-bound so we never need more threads than ``concurrency``.
    rollout = APIMultiTurnRollout(
        api=api,
        tool_manager=ToolManager(),  # placeholder; per-call list overrides
        sampling_params=SamplingParams(
            temperature=ROLLOUT_TEMPERATURE_LADDER[0],
            max_tokens=ROLLOUT_MAX_TOKENS, num_samples=1),
        max_turns=ROLLOUT_MAX_TURNS,
        concurrency=args.concurrency,
        extra_body={'enable_thinking': False},
    )

    write_lock = threading.Lock()
    out_fh = open(args.output, 'a', encoding='utf-8')
    accepted_total = 0
    seen_total = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as validation_pool:
        try:
            for start in range(0, len(pending), args.compress_batch_size):
                batch = pending[start:start + args.compress_batch_size]
                seen_total += len(batch)
                try:
                    records = process_batch(
                        api, rollout, batch, chunker, condenser,
                        validation_pool)
                except Exception as exc:
                    sys.stderr.write(
                        f'[batch {start}-{start + len(batch)}] crashed: {exc}\n')
                    continue
                with write_lock:
                    for record in records:
                        out_fh.write(
                            json.dumps(record, ensure_ascii=False) + '\n')
                    out_fh.flush()
                accepted_total += len(records)
                sys.stderr.write(
                    f'[progress] seen={seen_total}/{len(pending)} '
                    f'accepted={accepted_total} '
                    f'(+{len(records)} from this batch)\n')
        finally:
            out_fh.close()

    sys.stderr.write(
        f'Done. accepted={accepted_total} total_pending={len(pending)}\n')


if __name__ == '__main__':
    main()
