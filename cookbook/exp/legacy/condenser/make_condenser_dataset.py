import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Dict, Iterator, List, Optional, Set

from tqdm import tqdm

from twinkle.data_format.sampling import SamplingParams
from twinkle_agentic.protocol.openai import OpenAI


# ═══════════════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════════════

QUERY_GEN_SYSTEM = """\
You are a query designer. Given a source passage, enumerate distinct information \
queries a reader might ask of it. Each query must steer toward a meaningfully \
DIFFERENT compression of the same source — different facets, not rephrasings of \
the same need.

Category hints (not exhaustive — combine or invent as fits the source):
- Interface extraction (code): class / method signatures, parameter and return types
- Functional summary: what the passage accomplishes at a high level
- Error & pitfall analysis: bugs, anti-patterns, failure modes, edge cases
- Experience distillation: lessons learned, best practices, do's and don'ts
- Skill extraction (knowledge-as-skill): WHAT this passage lets you do, HOW to \
apply it as reusable steps, WHEN to invoke it (trigger conditions / use cases)
- Abstract analysis: design patterns, architectural decisions, trade-offs
- Information summary: key facts, entities, numbers, relationships
- Dependency & context: prerequisites, imports, environment, related modules

Rules:
1. SHAPE — each query is one short imperative or interrogative sentence (e.g. \
"List all public method signatures with parameter and return types", "What race \
conditions does this code contain?").
2. DISTINCT — reject any pair whose answers would substantially overlap; \
rephrasings of the same information need do NOT count as separate queries.
3. SKILL FOR KNOWLEDGE — when the source reads as tutorial / experience / \
how-to / domain knowledge, ALWAYS include exactly one skill-style query asking \
what the reader can accomplish with it and how to apply it (phrased in the \
source language).
4. ANSWERABLE — skip queries the source cannot actually answer, and skip \
trivial queries that would just reproduce the source verbatim.
5. SCALE — short / single-purpose → 1; medium → 2; rich / multi-topic → 3–4. \
Do not pad.
6. LANGUAGE — query language MUST match the source language.
7. OUTPUT — a single JSON array of strings; no preamble, no code fences, \
nothing else.\
"""

QUERY_GEN_USER = 'Analyze the following text and return a JSON array of queries.\n\n{text}'

COMPRESS_SYSTEM = """\
You are a compression assistant. For the (query, source) pair, emit a Markdown \
answer with TWO sections, designed to pair with the `extract_compressed` tool: \
the reader absorbs `## Summary` directly, then calls `extract_compressed` \
on any topic-key listed under `## More` to recover its \
fuller content.

  `## Summary`               — extreme-density text the reader reads directly.
  `## More` — a topic index whose keys are valid arguments \
to `extract_compressed` for recovering material not captured inline.

Together the two sections must form a COMPLETE, NON-DISTORTING inventory of the \
source for the query — nothing essential lost, nothing implied that the source \
does not support. NO preamble, NO meta-commentary, NO code fences wrapping the \
whole output.

Output skeleton:

## Summary
Topic: <what the source is about + scope, one line>
<dense body answering the query>

## More
- <topic-key>: <one-line hint of what is revealed when expanded>
- ...

Format selection for the inline body (pick the MOST COMPACT form per query, mix \
when helpful):
- Interface / signature → code notation directly: `func(a:int)->str`
- Factual / entity → telegraphic prose; drop function words; ":" for "is", "," \
for "has"
- Skill / how-to / usage → lead with `Use when: <trigger>`; numbered telegraphic \
steps `1.do X 2.then Y`; close with `Output: <result>` when relevant
- Procedural → numbered short steps
- Analytical / design → hierarchical bullets with abbreviations

`## Summary` rules:
1. TOPIC LINE — line 1 is ALWAYS `Topic: <subject — scope>`, even when the \
query is narrow. Anchors both the reader and the tool.
2. DENSITY — every token in the body carries query-relevant signal; cut filler.
3. PRIMARY-COMPLETE — never silently drop a fact essential to answering the \
query. Anything cut for length MUST appear as a key under \
`## More`.
4. NON-MISLEADING — phrasing must not let the reader infer anything the source \
does not support; partial truths that mislead are worse than honest omissions \
flagged in the index.
5. SELF-CONTAINED — the reader can act on the answer without re-opening the source.
6. FAITHFUL — only content the source supports; no fabrication, no extrapolation.
7. LANGUAGE — match the source language.
8. NO outer code fences around the whole answer; no meta-commentary.

`## More` rules (MANDATORY — this section is never omitted):
1. FORMAT — each bullet is `- <topic-key>: <one-line hint>`:
   • topic-key — short, unambiguous, grounded in source vocabulary so the \
`extract_compressed` tool can locate the aspect (e.g. `decorators`, \
`error handling`, `pitfalls`).
   • hint — tells WHAT the reader gains by expanding (concrete numbers, code \
listings, secondary cases, edge details, related context, …); do NOT restate \
the inline answer.
2. CRITERION — each bullet names an aspect that EXISTS in the source but is \
NOT fully captured inline. Material that genuinely fits inline without \
distortion MUST NOT be duplicated here.
3. FAITHFUL — hints must be grounded in the source; never speculate or invent.
4. ORDER — by relevance to the query, then by importance.
5. EMPTY CASE — if the source is so short / single-purpose that everything \
fits inline, write a single line `- (none)`.

Examples:

Query: List all public method signatures with parameter and return types
Source: (a Python HTTP client class with retry decorator, structured logging, \
and request helpers)
## Summary
Topic: Python HTTP client class — public surface of retried request helpers.
retry_request(url:str, max_retries:int=3, timeout:float=10.0) -> Response
fetch_json(endpoint:str, params:dict|None=None) -> dict
post_data(endpoint:str, payload:dict, headers:dict|None=None) -> Response

## More
- decorators: @retry config — exponential backoff (base=2.0, max=60s)
- logging: structured per-request logs with request_id and latency_ms
- private helpers: _build_headers, _parse_error — not in public surface
───
Query: What can this passage help you accomplish, and how to use it?
Source: (a tutorial on configuring Linux cgroups v2 caps for a systemd service)
## Summary
Topic: Linux cgroups v2 — per-service CPU / memory caps via systemd slice units.
Use when: needing per-service CPU/memory caps on systemd hosts.
1.create slice unit /etc/systemd/system/<name>.slice with CPUQuota=, MemoryMax=
2.attach service via Slice=<name>.slice in [Service]
3.systemctl daemon-reload + restart service
4.verify: systemctl status <svc> shows Tasks/CPU/Memory inside slice
Output: hard caps enforced by kernel cgroup v2.

## More
- pitfalls: cgroup v1/v2 mode detection, MemorySwapMax behavior on OOM
- delegation: Delegate=yes for nested controllers in container managers
- examples: nginx and postgres slice templates with concrete numeric caps
- diagnostics: systemd-cgls / systemd-cgtop walkthrough
───
Query: 总结这段代码的错误和改进经验
Source: (一段有 race condition 和未关闭资源的 Go 代码)
## Summary
Topic: Go HTTP fetch 循环 — 并发写共享 map + 未关闭响应体导致的稳定性缺陷。
1.race: 并发写 map 未锁 → sync.RWMutex 或 sync.Map
2.泄漏: resp.Body 未 Close → 请求后立即 defer resp.Body.Close()
3.吞错: err 未检查 → 每处 err!=nil 必处理或上抛

## More
- (none)

Now begin.\
"""

COMPRESS_USER = '## Query\n{query}\n\n## Source\n{text}'

# Short system prompt embedded in emitted SFT samples — the long COMPRESS_SYSTEM
# is for data generation only; training samples carry only the binding contract.
COMPRESS_SYSTEM_TRAIN = """\
You are a compression assistant. For the (query, source) pair, emit a Markdown \
answer with TWO sections, designed to pair with the `extract_compressed` tool: \
the reader absorbs `## Summary` directly, then calls `extract_compressed` \
on any topic-key listed under `## More` to recover its \
fuller content.

Output skeleton:

## Summary
Topic: <subject — scope, one line>
<dense body answering the query>

## More
- <topic-key>: <one-line hint of what is revealed when expanded>
- ...

Rules:
1. Line 1 of `## Summary` is ALWAYS `Topic: ...`.
2. Body is maximally dense; every token carries query-relevant signal.
3. Never silently drop a fact — anything cut for length MUST appear as a key \
under `## More` (do not duplicate inline material here).
4. No fabrication, no extrapolation, no misleading partial truths.
5. Match the source language. No outer code fences, no meta-commentary.\
"""

# Fixed queries — used directly (no Phase-1 LLM generation) for a proportion of items.
FIXED_QUERY_NEED = (
    'What problem does this passage address, and what skill or method is needed? '
    'Topic must name the specific pattern, never generic labels. '
    'Compress into a retrieval-friendly need description.')
FIXED_QUERY_SKILL = (
    'Extract the reusable skill: trigger conditions, key steps, and expected output. '
    'Topic names the method/pattern; format as "Use when: ...", numbered steps, '
    '"Output: ...". Compress into a standardized procedure for retrieval.')
FIXED_QUERIES = [FIXED_QUERY_NEED, FIXED_QUERY_SKILL]
FIXED_QUERY_RATIO = 0.3


# ═══════════════════════════════════════════════════════════════════════════════
# Core logic
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_json_array(text: str) -> Optional[List[str]]:
    """Best-effort extraction of a JSON string array from LLM output."""
    text = text.strip()
    # Try direct parse first
    if text.startswith('['):
        try:
            arr = json.loads(text)
            if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                return arr
        except json.JSONDecodeError:
            pass
    # Fallback: find first [...] block
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group())
            if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                return arr
        except json.JSONDecodeError:
            pass
    return None


def generate_queries(api: OpenAI, text: str) -> List[str]:
    """Phase 1: ask the LLM what queries can be asked about ``text``."""
    trajectory = {
        'messages': [
            {'role': 'system', 'content': QUERY_GEN_SYSTEM},
            {'role': 'user', 'content': QUERY_GEN_USER.format(text=text)},
        ]
    }
    sp = SamplingParams(temperature=0.7, max_tokens=1024)
    for attempt in range(2):
        try:
            reply = api(trajectory, sp, extra_body={'enable_thinking': True})
        except Exception as exc:
            sys.stderr.write(f'[query_gen] error: {exc}\n')
            return []
        content = reply.get('content') or ''
        queries = _extract_json_array(content)
        if queries:
            return queries
        if attempt == 0:
            sys.stderr.write('[query_gen] retry: failed to parse JSON array\n')
    return []


def compress_for_query(api: OpenAI, text: str, query: str,
                       thinking_budget: int = 1024) -> Optional[str]:
    """Phase 2: compress ``text`` w.r.t. ``query``. Returns compressed content or None."""
    trajectory = {
        'messages': [
            {'role': 'system', 'content': COMPRESS_SYSTEM},
            {'role': 'user', 'content': COMPRESS_USER.format(query=query, text=text)},
        ]
    }
    sp = SamplingParams(temperature=0.3, max_tokens=16384)
    for attempt in range(2):
        try:
            reply = api(trajectory, sp, extra_body={
                'enable_thinking': False,
                'thinking_budget': thinking_budget,
            })
        except Exception as exc:
            sys.stderr.write(f'[compress] error: {exc}\n')
            return None
        content = (reply.get('content') or '').strip()
        if not content:
            if attempt == 0:
                sys.stderr.write('[compress] retry: empty response\n')
            continue
        # Strip whole-answer code fence if present.
        m = re.match(r'^```[a-zA-Z]*\n(.*?)\n```\s*$', content, re.DOTALL)
        if m:
            content = m.group(1).strip()
        if not (re.search(r'(?im)^##\s*Summary\b', content)
                and re.search(r'(?im)^##\s*More\b', content)):
            if attempt == 0:
                sys.stderr.write('[compress] retry: missing required sections\n')
            continue
        return content
    return None


def _query_hash(query: str) -> str:
    """Stable short hash of a query string — embedded in sample id for resume."""
    return hashlib.md5(query.strip().encode('utf-8')).hexdigest()[:8]


def process_item(
    api: OpenAI,
    item: Dict[str, Any],
    done_sample_ids: Optional[Set[str]] = None,
    thinking_budget: int = 1024,
    fixed_query_ratio: float = FIXED_QUERY_RATIO,
) -> List[Dict[str, Any]]:
    """Run both phases on one dataset item. Returns list of SFT samples.

    Input rows come from ``dataset.py`` (single assistant message) or
    ``dataset_think.py`` (user query + assistant with reasoning_content).
    For thinking-data rows, ``FIXED_QUERY_NEED`` is applied to the query
    and ``FIXED_QUERY_SKILL`` to the CoT, skipping Phase-1 generation.

    ``done_sample_ids`` (full sample ids already on disk for this item)
    lets resume skip queries that were already emitted, keyed by query
    content hash so a phase-1 reorder still resolves correctly.
    """
    done = done_sample_ids or set()
    messages = item.get('messages') or []

    # Detect thinking-data: user message + assistant with reasoning_content
    user_query = ''
    cot_text = ''
    assistant_text = ''
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get('role', '')
        if role == 'user' and not user_query:
            user_query = (m.get('content') or '').strip()
        elif role == 'assistant':
            cot_text = (m.get('reasoning_content') or '').strip()
            assistant_text = (m.get('content') or '').strip()
            break

    item_id = item.get('id')
    if not item_id:
        return []
    source = item.get('source', 'unknown')

    # Thinking-data path: compress query and CoT separately with fixed queries
    if user_query and cot_text:
        pairs = [(user_query, FIXED_QUERY_NEED), (cot_text, FIXED_QUERY_SKILL)]
        samples: List[Dict[str, Any]] = []
        for text, query in pairs:
            if len(text) < 100:
                continue
            sample_id = f'{item_id}__{_query_hash(query)}'
            if sample_id in done:
                continue
            compressed = compress_for_query(api, text, query, thinking_budget=thinking_budget)
            if not compressed:
                continue
            sft_messages = [
                {'role': 'system', 'content': COMPRESS_SYSTEM_TRAIN},
                {'role': 'user', 'content': COMPRESS_USER.format(query=query, text=text)},
                {'role': 'assistant', 'content': compressed},
            ]
            samples.append({
                'id': sample_id,
                'source': source,
                'query': query,
                'original_len': len(text),
                'compressed_len': len(compressed),
                'original_tokens': 0,
                'compressed_tokens': 0,
                'messages': sft_messages,
                '__src': text,
                '__cmp': compressed,
            })
        return samples

    # Plain-data path: single assistant message
    text = assistant_text
    if not text or len(text) < 100:
        return []

    queries = generate_queries(api, text)
    if not queries:
        return []
    queries = queries[:2]

    # Mix in fixed queries for a proportion of items
    if random.random() < fixed_query_ratio:
        queries = list(FIXED_QUERIES)

    samples: List[Dict[str, Any]] = []
    for query in queries:
        sample_id = f'{item_id}__{_query_hash(query)}'
        if sample_id in done:
            continue
        compressed = compress_for_query(api, text, query, thinking_budget=thinking_budget)
        if not compressed:
            continue
        sft_messages = [
            {'role': 'system', 'content': COMPRESS_SYSTEM_TRAIN},
            {'role': 'user', 'content': COMPRESS_USER.format(query=query, text=text)},
            {'role': 'assistant', 'content': compressed},
        ]
        samples.append({
            'id': sample_id,
            'source': source,
            'query': query,
            'original_len': len(text),
            'compressed_len': len(compressed),
            'original_tokens': 0,
            'compressed_tokens': 0,
            'messages': sft_messages,
            # Stashed for sparse tokenization on main thread; popped before write.
            '__src': text,
            '__cmp': compressed,
        })
    return samples


def process_failure(
    api: OpenAI,
    item: Dict[str, Any],
    thinking_budget: int = 1024,
) -> List[Dict[str, Any]]:
    """Re-compress a single failure record (id, query, text already pinned).

    Used by ``--failures`` mode: query and source passage are taken verbatim
    from the original failure entry, so Phase-1 generation is skipped and the
    output id matches the original sample id.
    """
    sid = item.get('id') or ''
    query = (item.get('query') or '').strip()
    text = (item.get('text') or '').strip()
    if not sid or not query or not text:
        return []
    compressed = compress_for_query(api, text, query, thinking_budget=thinking_budget)
    if not compressed:
        return []
    sft_messages = [
        {'role': 'system', 'content': COMPRESS_SYSTEM_TRAIN},
        {'role': 'user', 'content': COMPRESS_USER.format(query=query, text=text)},
        {'role': 'assistant', 'content': compressed},
    ]
    return [{
        'id': sid,
        'source': item.get('source', 'failure_regen'),
        'query': query,
        'original_len': len(text),
        'compressed_len': len(compressed),
        'original_tokens': 0,
        'compressed_tokens': 0,
        'messages': sft_messages,
        '__src': text,
        '__cmp': compressed,
    }]


# ═══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════

def iter_input(path: str) -> Iterator[Dict[str, Any]]:
    """Stream JSONL dataset row-by-row (no full-file load)."""
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def iter_dataset_py(total: Optional[int], load_from_cache_file: bool) -> Iterator[Dict[str, Any]]:
    """Stream rows directly from ``dataset.py::get_dataset`` without any JSONL hop."""
    # Lazy import: dataset.py triggers HF / ModelScope downloads at module load.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cookbook.exp.condenser.dataset import get_dataset
    hf = get_dataset(total=total, load_from_cache_file=load_from_cache_file)
    sys.stderr.write(f'Loaded dataset.py::get_dataset: {len(hf)} rows\n')
    for row in hf:
        yield row


def iter_dataset_think_py(total: Optional[int], load_from_cache_file: bool) -> Iterator[Dict[str, Any]]:
    """Stream rows from ``dataset_think.py::get_dataset`` (query + CoT data)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataset_think import get_dataset
    hf = get_dataset(total=total, load_from_cache_file=load_from_cache_file)
    sys.stderr.write(f'Loaded dataset_think.py::get_dataset: {len(hf)} rows\n')
    for row in hf:
        yield row


def iter_failures(path: str, skip_ids: Optional[Set[str]] = None) -> Iterator[Dict[str, Any]]:
    """Stream records from a ``failures.jsonl`` for re-compression.

    Each input record carries a full sample id, the original query, and a
    user message whose body embeds the source passage after a ``## Passage``
    or ``## Source`` header. The yielded item is shaped for ``process_failure``
    (id, source, query, text). Items whose id is in ``skip_ids`` are skipped.
    """
    skip = skip_ids or set()
    n_total = n_skipped = n_yielded = n_bad = 0
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            sid = obj.get('id') or ''
            if not sid:
                n_bad += 1
                continue
            if sid in skip:
                n_skipped += 1
                continue
            query = (obj.get('query') or '').strip()
            user_content = ''
            for m in obj.get('messages') or []:
                if isinstance(m, dict) and m.get('role') == 'user':
                    user_content = m.get('content') or ''
                    break
            text = ''
            for sep in ('## Passage\n', '## Source\n'):
                if sep in user_content:
                    text = user_content.split(sep, 1)[1].strip()
                    break
            if not query or not text:
                sys.stderr.write(f'[failures] skip {sid}: missing query/passage\n')
                n_bad += 1
                continue
            n_yielded += 1
            yield {
                'id': sid,
                'source': obj.get('source', 'failure_regen'),
                'query': query,
                'text': text,
            }
    sys.stderr.write(
        f'[failures] total={n_total} yielded={n_yielded} '
        f'resume_skipped={n_skipped} malformed={n_bad}\n')


def load_done_sample_ids(path: str) -> Set[str]:
    """Collect already-written full sample ids (``base__hash``) for resume."""
    if not os.path.exists(path):
        return set()
    done: Set[str] = set()
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = obj.get('id', '')
            if sid:
                done.add(sid)
    return done


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Two-phase query-diverse condenser dataset builder.')
    parser.add_argument('--input', default=None,
                        help='Optional JSONL override; default uses dataset.py::get_dataset')
    parser.add_argument('--output', required=True,
                        help='Output JSONL file for SFT samples')
    parser.add_argument('--total', type=int, default=0,
                        help='Total input rows for proportional scaling in dataset.py (0 = base sizes)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable load_from_cache_file when calling dataset.py::get_dataset')
    parser.add_argument('--model', required=True,
                        help='API model name')
    parser.add_argument('--api-key', default=os.environ.get('OPENAI_API_KEY'))
    parser.add_argument('--base-url', default=os.environ.get('OPENAI_BASE_URL'))
    parser.add_argument('--concurrency', type=int, default=32,
                        help='Number of parallel workers')
    parser.add_argument('--limit', type=int, default=0,
                        help='Max items to process (0 = all)')
    parser.add_argument('--thinking-budget', type=int, default=1024,
                        help='Max thinking tokens for phase-2 compress (shorter = faster, cheaper)')
    parser.add_argument('--tokenizer', default='Qwen/Qwen3.5-4B',
                        help='HF/ModelScope tokenizer id for sparse token-ratio probe')
    parser.add_argument('--tokenize-every', type=int, default=1000,
                        help='Tokenize one sample every N writes; others get tokens=0')
    parser.add_argument('--fixed-query-ratio', type=float, default=FIXED_QUERY_RATIO,
                        help='Proportion of plain-data items using fixed queries instead of LLM-generated ones')
    parser.add_argument('--source', choices=['think', 'plain', 'both'], default='think',
                        help='Data source: think=dataset_think.py (query+CoT), plain=dataset.py, both=chain both')
    parser.add_argument('--failures', default=None,
                        help='Path to a failures.jsonl; when set, re-generate compressions for every record '
                             'using its original (query, passage) pair and ignore --input/--source.')
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    done_sample_ids = load_done_sample_ids(args.output)
    # Group done sample ids by base item id so each worker only sees its slice.
    done_per_item: Dict[str, Set[str]] = {}
    for sid in done_sample_ids:
        if '__' in sid:
            base = sid.rsplit('__', 1)[0]
            done_per_item.setdefault(base, set()).add(sid)
    sys.stderr.write(
        f'Resume: {len(done_sample_ids)} samples on disk across '
        f'{len(done_per_item)} items.\n')

    api = OpenAI(model=args.model, api_key=args.api_key, base_url=args.base_url)

    from modelscope import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    def iter_pending() -> Iterator[Dict[str, Any]]:
        if args.failures:
            source_iter = iter_failures(args.failures, done_sample_ids)
        elif args.input:
            source_iter = iter_input(args.input)
        else:
            import itertools
            sources = []
            if args.source in ('plain', 'both'):
                sources.append(iter_dataset_py(
                    total=args.total or None,
                    load_from_cache_file=not args.no_cache,
                ))
            if args.source in ('think', 'both'):
                sources.append(iter_dataset_think_py(
                    total=args.total or None,
                    load_from_cache_file=not args.no_cache,
                ))
            source_iter = itertools.chain(*sources)
        emitted = 0
        for it in source_iter:
            iid = it.get('id')
            if not iid:
                sys.stderr.write('[skip] row missing "id" field\n')
                continue
            if args.limit > 0 and emitted >= args.limit:
                return
            yield it
            emitted += 1

    write_lock = threading.Lock()
    out_fh = open(args.output, 'a', encoding='utf-8')
    items_done = 0
    items_failed = 0
    samples_emitted = 0
    pbar = tqdm(desc='condense', unit='item', dynamic_ncols=True)

    items_iter = iter_pending()
    in_flight: Dict[Any, str] = {}
    # Sliding window: keep ~2x concurrency tasks queued so the pool never starves.
    window = max(args.concurrency * 2, args.concurrency + 4)

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            exhausted = False
            while True:
                while not exhausted and len(in_flight) < window:
                    try:
                        it = next(items_iter)
                    except StopIteration:
                        exhausted = True
                        break
                    iid = it['id']
                    if args.failures:
                        fut = ex.submit(
                            process_failure, api, it, args.thinking_budget,
                        )
                    else:
                        fut = ex.submit(
                            process_item, api, it, done_per_item.get(iid),
                            args.thinking_budget, args.fixed_query_ratio,
                        )
                    in_flight[fut] = iid
                if not in_flight:
                    break
                done, _ = wait(list(in_flight.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    iid = in_flight.pop(fut)
                    try:
                        samples = fut.result()
                    except Exception as exc:
                        sys.stderr.write(f'[item {iid}] crashed: {exc}\n')
                        items_failed += 1
                        pbar.update(1)
                        continue
                    if not samples:
                        items_failed += 1
                        pbar.update(1)
                        continue
                    with write_lock:
                        for s in samples:
                            src = s.pop('__src', '')
                            cmp = s.pop('__cmp', '')
                            samples_emitted += 1
                            if (samples_emitted - 1) % args.tokenize_every == 0:
                                s['original_tokens'] = len(tokenizer(src).input_ids)
                                s['compressed_tokens'] = len(tokenizer(cmp).input_ids)
                            out_fh.write(json.dumps(s, ensure_ascii=False) + '\n')
                        out_fh.flush()
                    items_done += 1
                    pbar.set_postfix(
                        done=items_done, failed=items_failed,
                        samples=samples_emitted, refresh=False,
                    )
                    pbar.update(1)
    finally:
        out_fh.close()
        pbar.close()

    sys.stderr.write(
        f'Done. items_done={items_done}, samples={samples_emitted}, '
        f'failed={items_failed}\n')


if __name__ == '__main__':
    main()
