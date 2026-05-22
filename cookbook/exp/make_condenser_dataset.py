"""Two-phase query-diverse condenser dataset builder.

Pipeline per item (from dataset.py output: {id, source, messages}):
  Phase 1 — Query Generation:
      Ask the LLM: "Given this text, what distinct information queries can be asked?"
      System prompt hints categories (interface extraction, error summary, abstract
      analysis, information summary, experience/skill extraction, etc.).
      The LLM returns a JSON list of query strings.

  Phase 2 — Query-Specific Compression:
      For each (text, query) pair, call the LLM to produce a maximally dense
      compression tailored to that query. No fixed compression ratio; the goal
      is maximum information density with continuous characters.

Output: one JSONL row per (text, query) pair:
    {id, source, original_len, compressed_len, query, messages: [system, user, assistant]}

Run:
    python make_condenser_dataset.py \
        --input condenser_input.jsonl \
        --output condenser_sft.jsonl \
        --model qwen3-235b-a22b \
        --base-url http://localhost:8000/v1 \
        --concurrency 32
"""
import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from twinkle.data_format.sampling import SamplingParams
from twinkle_agentic.protocol.openai import OpenAI


# ═══════════════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════════════

QUERY_GEN_SYSTEM = """\
You are a query designer. Given a piece of text, enumerate distinct "information \
queries" that a reader might ask about it. Each query represents a DIFFERENT \
perspective or information need that would lead to a DIFFERENT compression of the \
same source.

Category hints (not exhaustive — invent more if appropriate):
- Interface extraction: class names, method signatures, input/output types
- Functional summary: what does this code/text accomplish at a high level
- Error & pitfall analysis: bugs, anti-patterns, failure modes, edge cases
- Experience distillation: lessons learned, best practices, do's and don'ts
- Skill extraction: reusable step-by-step procedures or techniques
- Abstract analysis: design patterns, architectural decisions, trade-offs
- Information summary: key facts, entities, numbers, relationships
- Dependency & context: prerequisites, imports, environment, related modules

Rules:
1. Each query must be a short imperative sentence (e.g. "List all public method \
signatures with parameter types and return types").
2. Queries must be MUTUALLY DISTINCT — different queries should lead to different \
compressions.
3. Skip trivial queries that would just reproduce the source verbatim.
4. Output a JSON array of strings, nothing else.
5. Generate 1–4 queries depending on text richness. Simple texts get 1; rich texts get up to 3.
6. Query language MUST match the source language.\
"""

QUERY_GEN_USER = "Analyze the following text and return a JSON array of queries.\n\n{text}"

COMPRESS_SYSTEM = """\
You are a text compression assistant. Compress the source text to answer the \
given query with maximum information density.

Format selection — pick the MOST COMPACT representation for the query type:
- Interface/signature queries → use code notation directly (e.g. `func(a:int)->str`)
- Factual/entity queries → telegraphic prose: drop function words, colons = "is", commas = "has"
- Procedural/skill queries → numbered short steps (1.xxx 2.xxx)
- Analytical/design queries → hierarchical bullets with abbreviations
Mix formats within one output if different parts benefit from different styles.

Rules:
1. Maximally DENSE — every token must carry query-relevant information.
2. Preserve ALL facts relevant to the query — no fabrication, no omission.
3. SELF-CONTAINED — reader understands without seeing the original.
4. Output language MUST match source language.
5. Do NOT wrap in markdown fences or add meta-commentary.
6. No fixed length — be as short as faithfully possible.

Examples:

Query: List all public method signatures with parameter and return types
Source: (a Python class with retry decorator, logging, and HTTP request methods)
Compressed:
retry_request(url:str, max_retries:int=3, timeout:float=10.0) -> Response
fetch_json(endpoint:str, params:dict|None=None) -> dict
post_data(endpoint:str, payload:dict, headers:dict|None=None) -> Response
───
Query: Summarize key facts of this context
Source: (a biography paragraph about Alan Turing)
Compressed:
Alan Turing: British mathematician/logician, father of CS + AI
- Turing machine (1936): universal computation model
- Enigma codebreaker, WWII Bletchley Park
- Turing test (1950): machine intelligence criterion
- Death 1954, cyanide, aged 41; royal pardon 2013
───
Query: 总结这段代码的错误和改进经验
Source: (一段有 race condition 和未关闭资源的 Go 代码)
Compressed:
1. race condition: 并发写 map 未加锁 → 改用 sync.RWMutex 或 sync.Map
2. 资源泄漏: resp.Body 未 defer Close → 请求后立即 defer resp.Body.Close()
3. 错误吞没: err 赋值后未检查 → 每次 err != nil 必须处理或上抛

Now begin.\
"""

COMPRESS_USER = "## Query\n{query}\n\n## Source\n{text}"


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


def compress_for_query(api: OpenAI, text: str, query: str) -> Optional[str]:
    """Phase 2: compress ``text`` with respect to a specific ``query``."""
    trajectory = {
        'messages': [
            {'role': 'system', 'content': COMPRESS_SYSTEM},
            {'role': 'user', 'content': COMPRESS_USER.format(query=query, text=text)},
        ]
    }
    # Allow generous tokens — no fixed ratio; let the model decide length.
    sp = SamplingParams(temperature=0.3, max_tokens=2048)
    for attempt in range(2):
        try:
            reply = api(trajectory, sp, extra_body={'enable_thinking': True})
        except Exception as exc:
            sys.stderr.write(f'[compress] error: {exc}\n')
            return None
        content = (reply.get('content') or '').strip()
        if not content:
            if attempt == 0:
                sys.stderr.write('[compress] retry: empty response\n')
            continue
        # Strip markdown fences if model wraps output
        if content.startswith('```'):
            first_nl = content.find('\n')
            last_fence = content.rfind('```')
            if first_nl != -1 and last_fence > first_nl:
                content = content[first_nl + 1:last_fence].strip()
        return content
    return None


def process_item(
    api: OpenAI, item: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run both phases on one dataset item. Returns list of SFT samples."""
    # Extract raw text from messages (concatenate all message contents)
    messages = item.get('messages') or []
    text_parts = [m['content'] for m in messages if m.get('content')]
    text = '\n\n'.join(text_parts).strip()
    if not text or len(text) < 100:
        return []

    item_id = item['id']
    source = item.get('source', 'unknown')

    # Phase 1: generate queries
    queries = generate_queries(api, text)
    if not queries:
        return []

    # Phase 2: compress for each query
    samples: List[Dict[str, Any]] = []
    for q_idx, query in enumerate(queries):
        compressed = compress_for_query(api, text, query)
        if not compressed:
            continue
        # Build SFT sample: system + user + assistant
        sft_messages = [
            {'role': 'system', 'content': COMPRESS_SYSTEM},
            {'role': 'user', 'content': COMPRESS_USER.format(query=query, text=text)},
            {'role': 'assistant', 'content': compressed},
        ]
        samples.append({
            'id': f'{item_id}__q{q_idx}',
            'source': source,
            'query': query,
            'original_len': len(text),
            'compressed_len': len(compressed),
            'messages': sft_messages,
        })
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_input(path: str) -> List[Dict[str, Any]]:
    """Load JSONL dataset (output of dataset.py)."""
    items: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def load_done_ids(path: str) -> set:
    """Collect item ids already processed for resume support."""
    if not os.path.exists(path):
        return set()
    done: set = set()
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Extract base item id (strip __qN suffix)
            sample_id = obj.get('id', '')
            base_id = re.sub(r'__q\d+$', '', sample_id)
            if base_id:
                done.add(base_id)
    return done


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Two-phase query-diverse condenser dataset builder.')
    parser.add_argument('--input', required=True,
                        help='Input JSONL file (output of dataset.py)')
    parser.add_argument('--output', required=True,
                        help='Output JSONL file for SFT samples')
    parser.add_argument('--model', required=True,
                        help='API model name')
    parser.add_argument('--api-key', default=os.environ.get('OPENAI_API_KEY'))
    parser.add_argument('--base-url', default=os.environ.get('OPENAI_BASE_URL'))
    parser.add_argument('--concurrency', type=int, default=16,
                        help='Number of parallel workers')
    parser.add_argument('--limit', type=int, default=0,
                        help='Max items to process (0 = all)')
    args = parser.parse_args()

    # Load input
    sys.stderr.write(f'Loading input from {args.input}...\n')
    items = load_input(args.input)
    sys.stderr.write(f'Loaded {len(items)} items.\n')

    # Resume
    done_ids = load_done_ids(args.output)
    pending = [it for it in items if it['id'] not in done_ids]
    sys.stderr.write(f'Resume: {len(done_ids)} already done, {len(pending)} pending.\n')

    if args.limit > 0:
        pending = pending[:args.limit]
        sys.stderr.write(f'Limited to {len(pending)} items.\n')

    # API client
    api = OpenAI(model=args.model, api_key=args.api_key, base_url=args.base_url)

    # Process with thread pool
    write_lock = threading.Lock()
    out_fh = open(args.output, 'a', encoding='utf-8')
    items_done = 0
    samples_emitted = 0
    items_failed = 0

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {
                ex.submit(process_item, api, item): item['id']
                for item in pending
            }
            for fut in as_completed(futures):
                item_id = futures[fut]
                try:
                    samples = fut.result()
                except Exception as exc:
                    sys.stderr.write(f'[item {item_id}] crashed: {exc}\n')
                    items_failed += 1
                    continue
                if not samples:
                    items_failed += 1
                    continue
                with write_lock:
                    for s in samples:
                        out_fh.write(json.dumps(s, ensure_ascii=False) + '\n')
                    out_fh.flush()
                items_done += 1
                samples_emitted += len(samples)
                if items_done % 50 == 0:
                    sys.stderr.write(
                        f'[progress] items={items_done} '
                        f'samples={samples_emitted} failed={items_failed}\n')
    finally:
        out_fh.close()

    sys.stderr.write(
        f'Done. items={items_done}, samples={samples_emitted}, '
        f'failed={items_failed}, total_pending={len(pending)}\n')


if __name__ == '__main__':
    main()
