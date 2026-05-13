"""Offline SFT dataset builder for the compression task: one sample per HotpotQA passage.

Pipeline per item:
  1. Pick HotpotQA rows stratified by ``level`` (easy / medium / hard).
  2. For every passage in ``context`` call a super-LLM via the OpenAI protocol
     to produce a telegraphic Summary/More markdown under a 0.5 hard ceiling.
  3. Emit one JSONL sample per passage with the standard single-turn chat shape:
     ``messages = [system = CONDENSER_SYSTEM, user = CONDENSER_USER(...), assistant = compressed]``.
  4. Resume by row_id: any row already represented in the output is skipped.

Run:
    python make_condenser_dataset.py \\
        --model gpt-4o --api-key $OPENAI_API_KEY \\
        --base-url https://api.openai.com/v1 \\
        --output hotpotqa_condenser_sft.jsonl --concurrency 16
"""
import argparse
import json
import os
import re
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from twinkle.data_format.sampling import SamplingParams
from twinkle_agentic.protocol.openai import OpenAI


# English port of src/twinkle_agentic/condenser/model.py ``_SECTION_SCHEMA``.
CONDENSER_SYSTEM = """You are a text compression assistant. A downstream model will read your compressed output to decide whether the detail it needs is inside this block; if yes, it will fetch and read the original passage.

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
"""

CONDENSER_USER = (
    'Downstream model will read your compressed block to decide whether to '
    'expand it. Compress faithfully: preserve the passage topic + core facts. '
    'Do NOT invent facts. Do NOT drop major facts. Do NOT write meta-commentary '
    'about the Query (never write "Query info: absent", "no X mention", etc.); '
    'if the passage does not address the Query, still summarize the passage.\n\n'
    '## Query (ordering hint only — still summarize the whole passage)\n{query}\n\n'
    '## Target length\n'
    'Compress AS MUCH AS faithfully possible. HARD CEILING: {budget} chars '
    '(~50% of the source). If core facts fit in far fewer chars, output fewer. '
    'Never exceed the ceiling.\n\n'
    '## Passage\n{text}')


# Deferred: kept for future trajectory-assembly script; currently unused.
# RUNTIME_SYSTEM = """You are a careful multi-hop QA assistant.
#
# ## Context Format (Mixed)
# The context you receive is a **mix of two forms**:
#
# 1. **Compressed blocks** — long passages wrapped in `<block_N>...</block_N>`, displayed as a Markdown digest in **telegraphic style** (no articles / "is" / "are"; colons and commas mean "is" / "has") with up to three sections:
#    - **Summary**: one short phrase (<= 15 words), NOT a full sentence
#    - **Key Facts**: up to 4 short bullets (each <= 10 words)
#    - **More**: 5-8 comma-separated keywords hinting at details hidden in the full text
# 2. **Raw passages** — short passages shown inline as plain text (e.g. `[K] Title: ...`) **without** any `<block_N>` wrapping.
#
# Only the `<block_N>`-wrapped blocks are compressed and can be expanded.
#
# ## Workflow
#
# ### Phase 1 - Scan and Decide
# Step 1: Read each compressed block's Summary, and read raw passages directly.
# Step 2: Check the More keywords for compressed blocks to judge whether hidden details are needed.
# Step 3: Decide which compressed blocks to expand, then call `extract_condensed` with their block ids.
#
# ### Phase 2 - Reason and Answer
# After the tool returns, continue stepping through the evidence and emit \\boxed{answer}.
#
# The `blocks` parameter accepts **exactly one integer** per call. Expand additional blocks by issuing separate `extract_condensed` calls, one per block. Do not request the same block twice.
#
# ## Output Format
# End your final response with \\boxed{answer}. Keep the boxed text short (a name, entity, date, or yes/no)."""
#
#
# EXTRACT_CONDENSED_TOOL: Dict[str, Any] = {
#     'type': 'function',
#     'function': {
#         'name': 'extract_condensed',
#         'description': (
#             'Recover the full, uncompressed text of ONE previously condensed '
#             'passage, identified by its <block_N> tag. Each call expands '
#             'exactly one block; issue separate calls for additional blocks, '
#             'and do not request the same block twice.'),
#         'parameters': {
#             'type': 'object',
#             'properties': {
#                 'blocks': {
#                     'type': 'integer',
#                     'description': (
#                         'The 1-indexed block number N appearing inside '
#                         '<block_N>...</block_N>. Exactly one block per call.'),
#                 },
#             },
#             'required': ['blocks'],
#         },
#     },
# }


RATIO_CEILING: float = 0.5
LEVELS: Tuple[str, str, str] = ('easy', 'medium', 'hard')


def _strip_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith('```'):
        return text
    first_nl = text.find('\n')
    last_fence = text.rfind('```')
    if first_nl == -1 or last_fence <= first_nl:
        return text
    return text[first_nl + 1:last_fence].strip()


_META_MARKERS = (
    'query info', 'no mention', 'not mention', 'not contain',
    'does not contain', 'does not address', 'no relevant',
    'passage covers', 'passage only', 'only covers', 'only provides',
    ': absent', 'info absent',
)

_SUMMARY_RE = re.compile(
    r'##\s*Summary\s*\n(.+?)(?:\n##\s*More|\Z)', re.DOTALL)


def _validate_compressed(compressed: str, budget: int) -> Optional[str]:
    """Return error reason, or ``None`` if ``compressed`` passes all gates."""
    if len(compressed) > int(budget * 1.15):
        return f'over-budget: {len(compressed)} > {int(budget * 1.15)}'
    m = _SUMMARY_RE.search(compressed)
    if not m:
        return 'missing ## Summary section'
    summary = m.group(1).strip()
    if not summary:
        return 'empty Summary'
    low = summary.lower()
    for marker in _META_MARKERS:
        if marker in low:
            return f'Summary contains meta-commentary: {marker!r}'
    # Concrete-fact signal: digit, ASCII/CJK colon, or multi-letter capitalized token.
    if not re.search(r'[\d:\uff1a]', summary) and not re.search(
            r'[A-Z][a-z]{2,}', summary):
        return 'Summary lacks concrete facts (no digit / colon / proper noun)'
    return None


def compress_passage(
    api: OpenAI, model: str, question: str, title: str, sentences: List[str],
) -> Optional[Tuple[str, str, str]]:
    """Compress one passage; return ``(original, compressed, user_prompt)`` or ``None``."""
    original = ' '.join(s.strip() for s in sentences if s and s.strip())
    if not original:
        return None
    passage_with_title = f'{title}: {original}'
    # Short passage: no meaningful compression signal, skip SFT sample.
    if len(passage_with_title) < 200:
        return None
    budget = max(160, int(len(passage_with_title) * RATIO_CEILING))
    user = CONDENSER_USER.format(
        query=question, budget=budget, text=passage_with_title)
    trajectory = {
        'messages': [
            {'role': 'system', 'content': CONDENSER_SYSTEM},
            {'role': 'user', 'content': user},
        ]
    }
    # ~2 chars/token + 16-token safety; keeps hard cap biting at the API layer.
    sp = SamplingParams(
        temperature=0.3,
        max_tokens=max(128, int(budget * 0.6) + 16))

    last_err: Optional[str] = None
    for attempt in range(2):
        try:
            reply = api(trajectory, sp, extra_body={'enable_thinking': True})
        except Exception as exc:
            sys.stderr.write(f'[compress] {title!r}: {exc}\n')
            return None
        content = reply.get('content') or ''
        compressed = _strip_fence(content).strip()
        if not compressed:
            last_err = 'empty response'
            continue
        if len(compressed) >= len(original):
            last_err = 'no compression (output >= source)'
            break
        err = _validate_compressed(compressed, budget)
        if err is None:
            return (original, compressed, user)
        last_err = err
        if attempt == 0:
            sys.stderr.write(f'[compress retry] {title!r}: {err}\n')
    sys.stderr.write(f'[compress drop] {title!r}: {last_err}\n')
    return None


# Deferred: QA-trajectory dataset builder, kept for future use, currently unused.
# def _gold_block_ids(supporting_facts: Dict[str, Any], titles: List[str]) -> List[int]:
#     gold_titles = set(supporting_facts.get('title') or [])
#     return sorted({i + 1 for i, t in enumerate(titles) if t in gold_titles})
#
#
# def build_trajectory(
#     row: Dict[str, Any], compressed: List[Tuple[str, str, str]],
#     gold_ids: List[int],
# ) -> Dict[str, Any]:
#     """Assemble the full SFT trajectory message list."""
#     lines = []
#     for i, (title, _orig, comp) in enumerate(compressed, start=1):
#         lines.append(f'<block_{i}>\n# {title}\n{comp}\n</block_{i}>')
#     context_block = '\n\n'.join(lines)
#     user_content = (
#         f'Question: {row["question"]}\n\nContext:\n\n{context_block}')
#
#     messages: List[Dict[str, Any]] = [
#         {'role': 'system', 'content': RUNTIME_SYSTEM},
#         {'role': 'user', 'content': user_content},
#     ]
#
#     bid_to_orig = {i + 1: orig for i, (_t, orig, _c) in enumerate(compressed)}
#     gold_titles_joined = ', '.join(
#         compressed[bid - 1][0] for bid in gold_ids if 1 <= bid <= len(compressed))
#
#     for turn_idx, bid in enumerate(gold_ids):
#         if turn_idx == 0:
#             reasoning = (
#                 f'Step 1: Scan the compressed blocks. Blocks covering '
#                 f'{gold_titles_joined} look directly relevant to the question.\n'
#                 f'Step 2: I will expand block {bid} first to read its full text.')
#         else:
#             reasoning = (
#                 f'I still need the full text of block {bid} to confirm the '
#                 f'remaining evidence. Expanding it now.')
#         tc_id = f'call_{turn_idx + 1}'
#         messages.append({
#             'role': 'assistant',
#             'content': reasoning,
#             'tool_calls': [{
#                 'id': tc_id,
#                 'type': 'function',
#                 'function': {
#                     'name': 'extract_condensed',
#                     'arguments': json.dumps({'blocks': bid}),
#                 },
#             }],
#         })
#         messages.append({
#             'role': 'tool',
#             'tool_call_id': tc_id,
#             'content': bid_to_orig[bid],
#         })
#
#     answer = (row.get('answer') or '').strip()
#     final_reasoning = (
#         f'Combining the expanded passages ({gold_titles_joined}), the '
#         f'evidence points to a single answer.\n\\boxed{{{answer}}}')
#     messages.append({'role': 'assistant', 'content': final_reasoning})
#
#     total_src = sum(len(o) for _t, o, _c in compressed) or 1
#     total_cmp = sum(len(c) for _t, _o, c in compressed)
#     achieved_ratio = round(total_cmp / total_src, 4)
#
#     return {
#         'id': row['id'],
#         'level': row.get('level'),
#         'type': row.get('type'),
#         'achieved_ratio': achieved_ratio,
#         'answer': answer,
#         'messages': messages,
#         'tools': [EXTRACT_CONDENSED_TOOL],
#     }


def process_row(
    api: OpenAI, model: str, row: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build per-passage SFT samples; returns [] if the row is unusable."""
    context = row.get('context') or {}
    titles = list(context.get('title') or [])
    sentences_list = list(context.get('sentences') or [])
    if not titles or len(titles) != len(sentences_list):
        return []

    row_id = row['id']
    question = row['question']
    level = row.get('level')
    row_type = row.get('type')
    samples: List[Dict[str, Any]] = []
    for idx, (title, sents) in enumerate(zip(titles, sentences_list)):
        result = compress_passage(api, model, question, title, sents)
        if result is None:
            continue
        original, compressed, user_prompt = result
        samples.append({
            'id': f'{row_id}__{idx}',
            'row_id': row_id,
            'level': level,
            'type': row_type,
            'title': title,
            'original_len': len(original),
            'compressed_len': len(compressed),
            'achieved_ratio': round(len(compressed) / len(original), 4),
            'messages': [
                {'role': 'system', 'content': CONDENSER_SYSTEM},
                {'role': 'user', 'content': user_prompt},
                {'role': 'assistant', 'content': compressed},
            ],
        })
    return samples


def stratified_sample(
    ds, per_level: int, seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    buckets: Dict[str, List[int]] = {lv: [] for lv in LEVELS}
    for i, lv in enumerate(ds['level']):
        if lv in buckets:
            buckets[lv].append(i)
    picked: List[int] = []
    for lv in LEVELS:
        pool = buckets[lv]
        if len(pool) < per_level:
            raise RuntimeError(
                f'level={lv} has only {len(pool)} rows, need {per_level}')
        picked.extend(rng.sample(pool, per_level))
    rng.shuffle(picked)
    return [ds[int(i)] for i in picked]


def load_done_row_ids(path: str) -> set:
    """Collect row_ids already emitted so we can resume by row."""
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = obj.get('row_id')
            if rid:
                done.add(rid)
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True,
                        help='API model name, e.g. gpt-4o or qwen-max')
    parser.add_argument('--api-key', default=os.environ.get('OPENAI_API_KEY'))
    parser.add_argument('--base-url', default=os.environ.get('OPENAI_BASE_URL'))
    parser.add_argument('--total', type=int, default=9000)
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hf-subset', default='distractor')
    parser.add_argument('--hf-split', default='train')
    args = parser.parse_args()

    if args.total % len(LEVELS) != 0:
        raise ValueError(
            f'--total must be divisible by {len(LEVELS)} (levels), '
            f'got {args.total}')
    per_level = args.total // len(LEVELS)

    sys.stderr.write(
        f'Loading hotpotqa/hotpot_qa:{args.hf_subset}:{args.hf_split}...\n')
    ds = load_dataset(
        'hotpotqa/hotpot_qa', args.hf_subset, split=args.hf_split)

    rows = stratified_sample(ds, per_level=per_level, seed=args.seed)

    done = load_done_row_ids(args.output)
    sys.stderr.write(f'Resume: {len(done)} rows already emitted, skipping.\n')
    pending = [row for row in rows if row['id'] not in done]
    sys.stderr.write(f'Pending: {len(pending)} / {len(rows)}\n')

    api = OpenAI(
        model=args.model, api_key=args.api_key, base_url=args.base_url)

    write_lock = threading.Lock()
    out_fh = open(args.output, 'a', encoding='utf-8')
    rows_done = 0
    samples_emitted = 0
    failed_rows = 0
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {
                ex.submit(process_row, api, args.model, row): row['id']
                for row in pending
            }
            for fut in as_completed(futures):
                rid = futures[fut]
                try:
                    samples = fut.result()
                except Exception as exc:
                    sys.stderr.write(f'[row {rid}] crashed: {exc}\n')
                    failed_rows += 1
                    continue
                if not samples:
                    failed_rows += 1
                    continue
                with write_lock:
                    for s in samples:
                        out_fh.write(
                            json.dumps(s, ensure_ascii=False) + '\n')
                    out_fh.flush()
                rows_done += 1
                samples_emitted += len(samples)
                if rows_done % 100 == 0:
                    sys.stderr.write(
                        f'[progress] rows={rows_done} '
                        f'samples={samples_emitted} failed={failed_rows}\n')
    finally:
        out_fh.close()

    sys.stderr.write(
        f'Done. rows={rows_done}, samples={samples_emitted}, '
        f'failed_rows={failed_rows}, total_rows={len(pending)}\n')


if __name__ == '__main__':
    main()
