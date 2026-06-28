"""RAG-index corpus loader — abstract reasoning skills + textbook-style methods.

Distinct from training-time ``dataset_think.py``. Optimizes for **abstraction
density**, not raw coverage: every row should encode a transferable method,
theorem, or solution pattern that downstream queries can retrieve as a
"use-when-X-do-Y" recipe.

Single-table design (``thinking_traces``); EMBED_QUERY_COT condense step in
``build_thinking_rag_index`` homogenizes thinking-style and textbook-style
content into the same retrieval form, so dual-table is unnecessary. The
``source`` field carries the original dataset name for eval-time
domain-bucket diagnostics.

Output schema matches ``dataset_think.get_dataset()``: ``{id, source, messages}``
with ``messages[1].reasoning_content`` carrying the CoT.

Mix (≈3.6M rows base, 10 datasets):
    Math thinking      23% — OpenMathReasoning + OpenR1-Math-220k + s1K-1.1
    Code thinking      19% — OpenCodeReasoning-2 + codeforces-cots
    Cross-domain R1    39% — Bespoke-Stratos + dolphin-r1 + reasoning-v1-20m
                              + natural_reasoning
    Textbook synth     17% — cosmopedia v1 (auto_math_text, chunked by H2)
    Olympiad solutions <1% — Omni-MATH

Dropped: camel-ai/{physics,chemistry,biology} (zip-only, no parquet/jsonl) and
swift/stack-exchange-paired (dataset_infos.json/data layout mismatch); the
textbook-density gap is covered by a larger cosmopedia slice.

Textbook processors synthesize a question from the chapter heading and place
the explanatory body into the ``cot`` field — embedding+condense reads
``query | cot`` so the textbook prose becomes a retrievable method.

Field extraction is defensive: each processor tries multiple plausible column
names and silently drops rows that miss a usable signal. Inspect
``dropped_index.jsonl`` after the first run to verify field-name guesses.
"""
import re
from typing import Any, Dict, List, Optional

from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor import Preprocessor

from dataset_think import _THINK_RE, _hash_id, _register, ToMessagesProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Sky-T1 / Bespoke-Stratos custom markers (used in place of <think>).
_BOT_RE = re.compile(
    r'<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>', re.DOTALL)
_BOS_RE = re.compile(
    r'<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>', re.DOTALL)

# H2 heading split for cosmopedia-style markdown chunks.
_H2_RE = re.compile(r'^##\s+(.+?)\s*$', re.MULTILINE)


def _split_think(text: str) -> tuple:
    """Return ``(cot, response)``; cot empty if no ``<think>`` block found."""
    if not text:
        return '', ''
    m = _THINK_RE.search(text)
    if not m:
        return '', text.strip()
    return m.group(1).strip(), text[m.end():].strip()


def _split_sky_t1(text: str) -> tuple:
    """Return ``(cot, response)`` for Sky-T1 / Bespoke-Stratos marker format."""
    if not text:
        return '', ''
    bot = _BOT_RE.search(text)
    bos = _BOS_RE.search(text)
    cot = bot.group(1).strip() if bot else ''
    sol = bos.group(1).strip() if bos else ''
    return cot, sol


def _from_messages(messages: Any) -> tuple:
    """Pull (first_user, first_assistant) from OpenAI/ShareGPT-style list."""
    if not isinstance(messages, list):
        return '', ''
    query, assistant = '', ''
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get('role') or msg.get('from') or ''
        content = msg.get('content') or msg.get('value') or ''
        if not isinstance(content, str):
            continue
        if role in ('user', 'human') and not query:
            query = content.strip()
        elif role in ('assistant', 'gpt') and not assistant:
            assistant = content.strip()
            break
    return query, assistant


def _chunk_by_h2(text: str, min_chars: int = 200, max_chars: int = 6000):
    """Split markdown text on ``## `` headings; yield ``(title, body)`` pairs."""
    if not text:
        return
    matches = list(_H2_RE.finditer(text))
    if not matches:
        head = text.strip()[:80].splitlines()[0] if text.strip() else ''
        body = text.strip()
        if head and min_chars <= len(body) <= max_chars:
            yield head, body
        return
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if min_chars <= len(body) <= max_chars and title:
            yield title, body


# ===========================================================================
# Math thinking
# ===========================================================================

OPEN_MATH_REASONING_REPO = 'ms://AI-ModelScope/OpenMathReasoning'


class OpenMathReasoningProcessor(Preprocessor):
    """OpenMathReasoning → ``{id, source, query, cot, response}``.

    Schema: ``problem``, ``generated_solution`` (R1 trace with ``<think>``),
    ``expected_answer``. The ``cot`` *split* (not column) is the long-CoT
    portion — TIR/genselect/additional_problems sit in sibling splits and
    are filtered at load time, not row-level.
    """

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            query = (row.get('problem') or row.get('question') or '').strip()
            assistant = (row.get('generated_solution') or row.get('solution')
                         or row.get('output') or '').strip()
            if not query or not assistant:
                continue
            cot, response = _split_think(assistant)
            if not cot:
                continue
            if not response:
                response = (row.get('expected_answer') or row.get('answer') or '').strip()
            if not response:
                continue
            out.append({
                'id': _hash_id('open_math_reasoning', f'{query}\n{response}'),
                'source': 'OpenMathReasoning',
                'query': query,
                'cot': cot,
                'response': response,
            })
        return self.map_row_to_col(out)


OPEN_R1_MATH_REPO = 'ms://open-r1/OpenR1-Math-220k'


class OpenR1MathProcessor(Preprocessor):
    """OpenR1-Math-220k → ``{id, source, query, cot, response}``.

    Schema: ``problem``, ``solution``, ``answer``, ``generations`` (list of
    R1 traces), ``correctness_math_verify`` (parallel bool list). Pick the
    first generation whose math-verify passed; fall back to ``solution``.
    """

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            query = (row.get('problem') or row.get('question') or '').strip()
            if not query:
                continue
            assistant = ''
            gens = row.get('generations')
            verifies = row.get('correctness_math_verify')
            if isinstance(gens, list):
                if isinstance(verifies, list) and len(verifies) == len(gens):
                    for g, v in zip(gens, verifies):
                        if v and isinstance(g, str) and g.strip():
                            assistant = g.strip()
                            break
                if not assistant:
                    for g in gens:
                        if isinstance(g, str) and g.strip():
                            assistant = g.strip()
                            break
            if not assistant:
                assistant = (row.get('solution') or '').strip()
            if not assistant:
                continue
            cot, response = _split_think(assistant)
            if not cot:
                continue
            if not response:
                response = (row.get('answer') or '').strip()
            if not response:
                continue
            out.append({
                'id': _hash_id('open_r1_math', f'{query}\n{response}'),
                'source': 'OpenR1-Math-220k',
                'query': query,
                'cot': cot,
                'response': response,
            })
        return self.map_row_to_col(out)


S1K_REPO = 'ms://simplescaling/s1K-1.1'


class S1KProcessor(Preprocessor):
    """s1K-1.1 → ``{id, source, query, cot, response}``.

    Schema: ``question`` + ``deepseek_thinking_trajectory`` (or
    ``thinking_trajectories`` legacy) + ``deepseek_attempt`` (final answer).
    Hand-curated peak-abstraction set, kept whole.
    """

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            query = (row.get('question') or row.get('problem') or '').strip()
            thinking = (row.get('deepseek_thinking_trajectory')
                        or row.get('thinking_trajectories')
                        or row.get('thinking') or '')
            if isinstance(thinking, list):
                thinking = '\n\n'.join(t for t in thinking if isinstance(t, str))
            cot = (thinking or '').strip()
            response = (row.get('deepseek_attempt') or row.get('attempt')
                        or row.get('answer') or row.get('solution') or '').strip()
            if not query or not cot or not response:
                continue
            out.append({
                'id': _hash_id('s1k', f'{query}\n{response}'),
                'source': 's1K-1.1',
                'query': query,
                'cot': cot,
                'response': response,
            })
        return self.map_row_to_col(out)


# ===========================================================================
# Code thinking
# ===========================================================================

OPEN_CODE_REASONING_REPO = 'ms://nv-community/OpenCodeReasoning-2'


class OpenCodeReasoning2Processor(Preprocessor):
    """OpenCodeReasoning-2 → ``{id, source, query, cot, response}``.

    Schema: ``input``/``problem``, plus per-model R1-style trace columns
    (``r1_generation``, ``qwq_generation``, etc.). Prefer the ``r1`` trace;
    fall back to ``solution``.
    """

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            query = (row.get('input') or row.get('problem')
                     or row.get('question') or '').strip()
            # OCR-2 'python' split ships dirty rows where question is literally '-';
            # the real prompt is buried in r1_generation and not recoverable here.
            if not query or query == '-':
                continue
            assistant = (row.get('r1_generation') or row.get('reasoning_content')
                         or row.get('solution') or row.get('output') or '').strip()
            if not assistant:
                continue
            cot, response = _split_think(assistant)
            if not cot:
                continue
            if not response:
                response = (row.get('expected_solution') or row.get('answer') or '').strip()
            if not response:
                continue
            out.append({
                'id': _hash_id('opencode_reasoning2', f'{query}\n{response}'),
                'source': 'OpenCodeReasoning-2',
                'query': query,
                'cot': cot,
                'response': response,
            })
        return self.map_row_to_col(out)


CODEFORCES_COTS_REPO = 'ms://open-r1/codeforces-cots'


class CodeforcesCotsProcessor(Preprocessor):
    """codeforces-cots → ``{id, source, query, cot, response}``.

    Schema: ``description``/``problem``, ``generation``/``solution`` (R1
    trace with ``<think>`` + final code). Algorithmic patterns at high
    abstraction density.
    """

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            query = (row.get('description') or row.get('problem')
                     or row.get('input') or row.get('question') or '').strip()
            assistant = (row.get('generation') or row.get('solution')
                         or row.get('output') or '').strip()
            if not query or not assistant:
                continue
            cot, response = _split_think(assistant)
            if not cot or not response:
                continue
            out.append({
                'id': _hash_id('codeforces_cots', f'{query}\n{response}'),
                'source': 'codeforces-cots',
                'query': query,
                'cot': cot,
                'response': response,
            })
        return self.map_row_to_col(out)


# ===========================================================================
# Cross-domain R1
# ===========================================================================

BESPOKE_STRATOS_REPO = 'ms://bespokelabs/Bespoke-Stratos-17k'


class BespokeStratosProcessor(Preprocessor):
    """Bespoke-Stratos-17k → ``{id, source, query, cot, response}``.

    Schema: ``conversations`` (ShareGPT). Assistant content uses Sky-T1
    markers ``<|begin_of_thought|>...<|end_of_thought|>`` then
    ``<|begin_of_solution|>...<|end_of_solution|>``.
    """

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            query, assistant = _from_messages(
                row.get('conversations') or row.get('messages'))
            if not query or not assistant:
                continue
            cot, response = _split_sky_t1(assistant)
            if not cot:
                cot, response = _split_think(assistant)
            if not cot or not response:
                continue
            out.append({
                'id': _hash_id('bespoke_stratos', f'{query}\n{response}'),
                'source': 'Bespoke-Stratos-17k',
                'query': query,
                'cot': cot,
                'response': response,
            })
        return self.map_row_to_col(out)


DOLPHIN_R1_REPO = 'ms://AI-ModelScope/dolphin-r1'


class DolphinR1Processor(Preprocessor):
    """dolphin-r1 → ``{id, source, query, cot, response}``.

    Schema (reasoning-deepseek subset): ``messages=[system, user]`` (no
    assistant turn) + flat ``reasoning`` (CoT) + ``answer`` (final response)
    + ``model``. Pull the user turn as query, ``reasoning``/``answer`` as
    cot/response. Fallback to embedded ``<think>`` for legacy rows.
    """

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            msgs = row.get('messages') or row.get('conversations')
            query = ''
            if isinstance(msgs, list):
                for msg in msgs:
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get('role') or msg.get('from') or ''
                    content = msg.get('content') or msg.get('value') or ''
                    if role in ('user', 'human') and isinstance(content, str):
                        query = content.strip()
            cot = (row.get('reasoning') or row.get('reasoning_content') or '').strip()
            response = (row.get('answer') or '').strip()
            if (not cot or not response) and isinstance(msgs, list):
                _, assistant = _from_messages(msgs)
                if assistant:
                    c2, r2 = _split_think(assistant)
                    if c2:
                        cot = cot or c2
                        response = response or r2 or assistant
            if not query or not cot or not response:
                continue
            out.append({
                'id': _hash_id('dolphin_r1', f'{query}\n{response}'),
                'source': 'dolphin-r1',
                'query': query,
                'cot': cot,
                'response': response,
            })
        return self.map_row_to_col(out)


GLAIVE_REASONING_REPO = 'ms://glaiveai/reasoning-v1-20m'


class GlaiveReasoningProcessor(Preprocessor):
    """reasoning-v1-20m → ``{id, source, query, cot, response}``.

    Schema: ``prompt``, ``response`` (R1 trace with ``<think>`` + answer).
    Largest cross-domain corpus in the mix; downsample aggressively.
    """

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            query = (row.get('prompt') or row.get('question')
                     or row.get('input') or '').strip()
            assistant = (row.get('response') or row.get('output')
                         or row.get('answer') or '').strip()
            if not query or not assistant:
                continue
            cot, response = _split_think(assistant)
            if not cot or not response:
                continue
            out.append({
                'id': _hash_id('glaive_reasoning', f'{query}\n{response}'),
                'source': 'reasoning-v1-20m',
                'query': query,
                'cot': cot,
                'response': response,
            })
        return self.map_row_to_col(out)


NATURAL_REASONING_REPO = 'ms://facebook/natural_reasoning'


class NaturalReasoningProcessor(Preprocessor):
    """natural_reasoning → ``{id, source, query, cot, response}``.

    Schema: ``question`` + ``reference_answer`` + ``responses=[{response_model,
    response}]``. The ``response`` field itself is the step-by-step CoT
    (``## Step 1...## Step 2...``); there is no separate ``reasoning`` key.
    Map ``responses[i].response`` → cot, ``reference_answer`` → response.
    Rows with empty ``reference_answer`` (~18% per README) are dropped.
    """

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            query = (row.get('question') or '').strip()
            if not query:
                continue
            cot = ''
            responses = row.get('responses')
            if isinstance(responses, list):
                for r in responses:
                    if not isinstance(r, dict):
                        continue
                    txt = (r.get('response') or r.get('reasoning')
                           or r.get('thinking') or r.get('answer') or '').strip()
                    if txt:
                        cot = txt
                        break
            if not cot:
                cot = (row.get('reasoning') or row.get('thinking')
                       or row.get('response') or '').strip()
            response = (row.get('reference_answer') or row.get('answer') or '').strip()
            if not cot or not response:
                continue
            out.append({
                'id': _hash_id('natural_reasoning', f'{query}\n{response}'),
                'source': 'natural_reasoning',
                'query': query,
                'cot': cot,
                'response': response,
            })
        return self.map_row_to_col(out)


# ===========================================================================
# Textbook-style — synthesize query from chapter heading; body → cot
# ===========================================================================

COSMOPEDIA_REPO = 'ms://HuggingFaceTB/cosmopedia'

class CosmopediaProcessor(Preprocessor):
    """cosmopedia v1 → ``{id, source, query, cot, response}``.

    Schema: ``prompt`` (writing instruction), ``text`` (full chapter body),
    ``format``/``audience``/``seed_data``. The subset is selected at load
    time (``subset_name='auto_math_text'`` — densest math-textbook slice);
    H2 chunking inside each row yields synthetic queries
    (``Explain {heading}``) with the body placed into ``cot``.
    """

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            text = (row.get('text') or row.get('content') or '').strip()
            if not text:
                continue
            for title, body in _chunk_by_h2(text):
                # Heading-only "Explain: X" was 1-2 tokens and impossible to align
                # with full-section cot. Promote the section's lead paragraph into
                # the query so anchor carries real semantic content.
                parts = body.split('\n\n', 1)
                first_para = parts[0].strip()
                rest = parts[1].strip() if len(parts) > 1 else ''
                if len(first_para) < 256 or len(rest) < 256:
                    continue
                query = f'{title}\n\n{first_para}' if title else first_para
                out.append({
                    'id': _hash_id('cosmopedia', f'{title}\n{first_para[:200]}'),
                    'source': 'cosmopedia-v1',
                    'query': query,
                    'cot': rest,
                    'response': '',
                })
        return self.map_row_to_col(out)


OMNI_MATH_REPO = 'ms://AI-ModelScope/Omni-MATH'


class OmniMathProcessor(Preprocessor):
    """Omni-MATH → ``{id, source, query, cot, response}``.

    Schema: ``problem``, ``solution`` (full proof), ``answer``, ``domain``,
    ``difficulty``. Olympiad-grade derivations — solution body → cot,
    answer → response.
    """

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            query = (row.get('problem') or row.get('question') or '').strip()
            solution = (row.get('solution') or '').strip()
            answer = (row.get('answer') or row.get('expected_answer') or '').strip()
            if not query or not solution:
                continue
            out.append({
                'id': _hash_id('omni_math', f'{query}\n{solution[:200]}'),
                'source': 'Omni-MATH',
                'query': query,
                'cot': solution,
                'response': answer,
            })
        return self.map_row_to_col(out)


# ===========================================================================
# Mix configuration — base sizes target ≈3.6M total rows
# ===========================================================================

_BASE_SIZES = {
    'open_math_reasoning': 600_000,
    'open_r1_math': 220_000,
    's1k': 1_000,
    'opencode_reasoning2': 500_000,
    'codeforces_cots': 200_000,
    'bespoke_stratos': 17_000,
    'dolphin_r1': 400_000,
    'glaive_reasoning': 800_000,
    'natural_reasoning': 200_000,
    'cosmopedia': 700_000,
    'omni_math': 4_000,
}


def _scaled_sizes(total: Optional[int]) -> Dict[str, int]:
    if total is None or total <= 0:
        return dict(_BASE_SIZES)
    scale = total / sum(_BASE_SIZES.values())
    return {k: max(1, int(round(v * scale))) for k, v in _BASE_SIZES.items()}


def _build_dataset(total: Optional[int] = None,
                   load_from_cache_file: bool = True) -> Dataset:
    sizes = _scaled_sizes(total)
    dataset = Dataset()

    _register(dataset, OpenMathReasoningProcessor,
              DatasetMeta(dataset_id=OPEN_MATH_REASONING_REPO, split='cot',
                          data_slice=range(sizes['open_math_reasoning'])),
              load_from_cache_file=load_from_cache_file)

    _register(dataset, OpenR1MathProcessor,
              DatasetMeta(dataset_id=OPEN_R1_MATH_REPO, split='train',
                          data_slice=range(sizes['open_r1_math'])),
              load_from_cache_file=load_from_cache_file)

    _register(dataset, S1KProcessor,
              DatasetMeta(dataset_id=S1K_REPO, split='train'),
              load_from_cache_file=load_from_cache_file)

    _register(dataset, OpenCodeReasoning2Processor,
              DatasetMeta(dataset_id=OPEN_CODE_REASONING_REPO,
                          subset_name='train', split='python',
                          data_slice=range(sizes['opencode_reasoning2'])),
              load_from_cache_file=load_from_cache_file)

    _register(dataset, CodeforcesCotsProcessor,
              DatasetMeta(dataset_id=CODEFORCES_COTS_REPO,
                          subset_name='solutions_w_editorials_decontaminated',
                          split='train',
                          data_slice=range(sizes['codeforces_cots'])),
              load_from_cache_file=load_from_cache_file)

    _register(dataset, BespokeStratosProcessor,
              DatasetMeta(dataset_id=BESPOKE_STRATOS_REPO, split='train'),
              load_from_cache_file=load_from_cache_file)

    _register(dataset, DolphinR1Processor,
              DatasetMeta(dataset_id=DOLPHIN_R1_REPO,
                          subset_name='reasoning-deepseek', split='train',
                          data_slice=range(sizes['dolphin_r1'])),
              load_from_cache_file=load_from_cache_file)

    _register(dataset, GlaiveReasoningProcessor,
              DatasetMeta(dataset_id=GLAIVE_REASONING_REPO, split='train',
                          data_slice=range(sizes['glaive_reasoning'])),
              load_from_cache_file=load_from_cache_file)

    _register(dataset, NaturalReasoningProcessor,
              DatasetMeta(dataset_id=NATURAL_REASONING_REPO, split='train',
                          data_slice=range(sizes['natural_reasoning'])),
              load_from_cache_file=load_from_cache_file)

    _register(dataset, CosmopediaProcessor,
              DatasetMeta(dataset_id=COSMOPEDIA_REPO,
                          subset_name='auto_math_text', split='train',
                          data_slice=range(sizes['cosmopedia'])),
              load_from_cache_file=load_from_cache_file)

    _register(dataset, OmniMathProcessor,
              DatasetMeta(dataset_id=OMNI_MATH_REPO, split='test'),
              load_from_cache_file=load_from_cache_file)

    dataset.mix_dataset(False)
    # Mix is concatenated in registration order; shuffle so the streaming
    # consumer sees all sources interleaved instead of 600k OpenMathReasoning
    # rows before it ever reaches code/textbook splits.
    dataset.dataset = dataset.dataset.shuffle(seed=42)
    return dataset


def get_dataset(total: Optional[int] = None,
                dropped_log: Optional[str] = None,
                load_from_cache_file: bool = True) -> Dataset:
    """Build, convert to messages, and quality-filter the RAG-index corpus.

    Mirrors ``dataset_think.get_dataset``: identical signature + output
    schema so ``build_thinking_rag_index`` consumes both modules unchanged.
    """
    from twinkle_agentic.preprocessor import (
        DeadLoopFilter,
        FixUnicodeFilter,
        HardFilter,
        MessageSanityFilter,
        QualityPreprocessor,
        RefuseFilter,
        RemoveRepeatSentencesFilter,
        TokenNumFilter,
        TokenSoupFilter,
    )

    dataset = _build_dataset(total=total, load_from_cache_file=load_from_cache_file)
    # Drop trivially-short queries (e.g. one-line math problems, OmniMath stubs)
    # before message conversion — anchor side needs enough tokens to embed meaningfully.
    dataset.dataset = dataset.dataset.filter(
        lambda x: len((x.get('query') or '').strip()) >= 100,
        num_proc=32, load_from_cache_file=load_from_cache_file)
    dataset.map(ToMessagesProcessor(), remove_columns=['query', 'cot', 'response'],
                load_from_cache_file=load_from_cache_file)
    qp = QualityPreprocessor(
        pipeline=[
            HardFilter(),
            RefuseFilter(),
            DeadLoopFilter(),
            TokenSoupFilter(),
            MessageSanityFilter(min_turns=1, max_msg_chars=200000),
            FixUnicodeFilter(),
            RemoveRepeatSentencesFilter(),
            TokenNumFilter(max_num=32768),
        ],
        dropped_log_path=dropped_log or '',
    )
    dataset.map(qp, batched=True, num_proc=32, load_from_cache_file=load_from_cache_file)
    return dataset


if __name__ == '__main__':
    import os
    dropped_log = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'dropped_index.jsonl')
    if os.path.exists(dropped_log):
        os.remove(dropped_log)
    dataset = get_dataset(load_from_cache_file=False)
    print(len(dataset))
