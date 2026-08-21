# Copyright (c) ModelScope Contributors. All rights reserved.
"""RSI step 1 — read a raw data source, parallel-preprocess it with the
twinkle_agentic preprocessor, and write the surviving subset to disk.

Usage
-----
    python -m twinkle_agentic.rsi.rsi_prepare \
        --input  /path/to/raw.jsonl \
        --output output/rsi/subset.jsonl \
        --num-proc 4

``--input`` accepts a local ``.jsonl``/``.parquet`` path or an ``ms://`` dataset
id; the raw schema is intentionally not pinned here (decided per data source at
test time). Every row must expose a ``messages`` list — the preprocessor keys
off it. Adapt other schemas in :func:`load_source` before the pipeline runs.

Pipeline
--------
Core steps (no external deps, always on), each using the filter's OWN default
thresholds (no thresholds invented here):

    MessageNormalizer -> MessageSanityFilter -> RefuseFilter -> DeadLoopFilter
    -> TokenSoupFilter -> HardFilter

Optional steps, off by default (enabling needs extra packages):
    RSI_USE_LANG=1        LanguageFilter            (langid, degrades to heuristic)
    RSI_USE_DATAJUICER=1  FixUnicode/RemoveRepeat/SpecialChars/TokenNum (data_juicer[, modelscope])
    RSI_USE_PII=1         PIIPresidioFilter         (presidio-analyzer/anonymizer)

``DedupFilter`` is NOT part of the parallel pipeline: its docstring requires it
to see the whole dataset in one call, so it runs once after the parallel pass.
"""
import argparse
import os

from twinkle.dataset import Dataset, DatasetMeta
from twinkle.utils import get_logger
from twinkle_agentic.preprocessor import (DeadLoopFilter, DedupFilter, HardFilter, MessageNormalizer,
                                          MessageSanityFilter, QualityPreprocessor, RefuseFilter, TokenSoupFilter,
                                          merge_dropped_shards, run_quality_pipeline, truncate_dropped_logs)

logger = get_logger()


def _env_flag(name: str, default: str = '0') -> bool:
    return os.environ.get(name, default).strip().lower() in ('1', 'true', 'yes', 'on')


def build_pipeline():
    """Return the ordered list of preprocessor steps for the parallel pass.

    DedupFilter is deliberately excluded (see module docstring); it is applied
    separately on the full materialized dataset.
    """
    # RSI_NORMALIZE_TOOL_CALLS=0 for pure code data (e.g. MBPP): the bracket-DSL
    # parser is a marker-less fallback matching ``[name(``, which is also what a
    # python list comprehension or a call-indexed subscript looks like, so the
    # rewrite silently deletes real code from the assistant turn.
    steps = [
        MessageNormalizer(normalize_tool_calls=_env_flag('RSI_NORMALIZE_TOOL_CALLS', '1')),
        MessageSanityFilter(),   # role order / tool-id matching / content integrity / sensitive words
        RefuseFilter(),          # drop assistant self-referential refusals
        DeadLoopFilter(),        # drop degenerate / stuck (hesitation, cascade, ngram repeat)
        TokenSoupFilter(),       # drop garbled text (replacement/control/private-use chars, script chaos)
        # min_assistant_chars_2turn=0: a single-turn valid tool call (e.g. `[Func(x=1)]`)
        # is only tens of chars; HardFilter's default 80-char floor wrongly drops it
        # as a "shallow_reply". Zeroing the floor keeps these tool-call rows (Rule 3
        # still removes genuinely empty assistants). Overridable via env.
        HardFilter(min_assistant_chars_2turn=int(os.environ.get('RSI_MIN_ASST_CHARS_2TURN', 0))),
    ]
    if _env_flag('RSI_USE_LANG'):
        from twinkle_agentic.preprocessor import LanguageFilter
        steps.append(LanguageFilter())
    if _env_flag('RSI_USE_DATAJUICER'):
        from twinkle_agentic.preprocessor import (FixUnicodeFilter, RemoveRepeatSentencesFilter, SpecialCharsFilter,
                                                  TokenNumFilter)
        steps += [FixUnicodeFilter(), RemoveRepeatSentencesFilter(), SpecialCharsFilter(), TokenNumFilter()]
    if _env_flag('RSI_USE_PII'):
        from twinkle_agentic.preprocessor import PIIPresidioFilter
        steps.append(PIIPresidioFilter())
    return steps


# ShareGPT `from` value -> standard message role. ToolACE uses
# system/user/assistant/tool; other ShareGPT variants use human/gpt/observation.
_ROLE_MAP = {
    'system': 'system',
    'user': 'user', 'human': 'user',
    'assistant': 'assistant', 'gpt': 'assistant', 'bot': 'assistant',
    'tool': 'tool', 'observation': 'tool', 'function': 'tool',
    'function_call': 'assistant', 'function_response': 'tool', 'tool_response': 'tool',
}


def _row_to_messages(row: dict) -> dict:
    """Map one ShareGPT ``conversations`` row to a ``messages`` row.

    Only ``from``->``role`` and ``value``->``content`` are rewritten; the tool
    call embedded in an assistant turn is left as-is in ``content`` (ToolACE keeps
    it as a bracket-DSL string) and parsed later in rsi_refine/rsi_rl. Turns whose
    ``from`` is unknown are dropped so no invalid role reaches the pipeline.
    """
    messages = []
    for turn in (row.get('conversations') or []):
        if not isinstance(turn, dict):
            continue
        role = _ROLE_MAP.get(str(turn.get('from', '')).lower())
        if role is None:
            continue
        messages.append({'role': role, 'content': turn.get('value', '') or ''})
    return {'messages': messages, 'id': row.get('id', '')}


def load_source(input_path: str, num_proc: int = 4) -> Dataset:
    """Load the raw source into a twinkle Dataset.

    A local path is loaded by extension (jsonl->json, parquet, csv...); anything
    else is treated as a hub id (e.g. ``ms://org/name``). Rows are passed through
    unchanged except for one adaptation: ShareGPT-style rows (a ``conversations``
    list of ``{"from", "value"}`` turns, e.g. ToolACE) are mapped to a standard
    ``messages`` list, because the whole preprocessor keys off ``messages``. Rows
    that already carry ``messages`` are left untouched.
    """
    ds = Dataset(DatasetMeta(dataset_id=input_path))
    cols = ds.dataset.column_names
    if 'messages' not in cols and 'conversations' in cols:
        logger.info('[rsi_prepare] ShareGPT `conversations` detected -> mapping to `messages`')
        # Materialize + convert in Python then rebuild: twinkle's Dataset.map forces
        # batched=True and wraps the fn as a Preprocessor, which does not fit a plain
        # per-row schema rewrite. The source is small enough to hold in memory.
        rows = [_row_to_messages(r) for r in ds.dataset.to_list()]
        ds = Dataset(DatasetMeta(data=rows))
    return ds


def main():
    parser = argparse.ArgumentParser(description='RSI step 1: preprocess a raw source into a clean subset.')
    parser.add_argument('--input', required=True, help='Local .jsonl/.parquet path or an ms:// dataset id.')
    parser.add_argument('--output', default='output/rsi/subset.jsonl', help='Where to write the surviving subset.')
    parser.add_argument('--num-proc', type=int, default=int(os.environ.get('RSI_NUM_PROC', '4')),
                        help='Parallel workers for the preprocessor map pass.')
    parser.add_argument('--dropped-log', default='', help='Optional JSONL of dropped-row metadata (empty=off).')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)

    pipeline = build_pipeline()
    step_names = [type(s).__name__ for s in pipeline]
    logger.info(f'[rsi_prepare] pipeline: {" -> ".join(step_names)} + DedupFilter(global)')

    dataset = load_source(args.input, num_proc=args.num_proc)
    n_in = len(dataset.dataset)
    logger.info(f'[rsi_prepare] loaded {n_in} rows from {args.input}')

    # 'mark' mode + run_quality_pipeline is the ghost-proof parallel path:
    # map returns equal-length columns flagged _keep, then a single filter removes.
    if args.dropped_log:
        truncate_dropped_logs(args.dropped_log)
    qp = QualityPreprocessor(pipeline, dropped_log_path=args.dropped_log, drop_mode='mark')
    run_quality_pipeline(dataset, qp, num_proc=args.num_proc)
    if args.dropped_log:
        merge_dropped_shards(args.dropped_log)

    n_after_pipeline = len(dataset.dataset)
    logger.info(f'[rsi_prepare] after parallel pipeline: {n_in} -> {n_after_pipeline}')

    # Global longest-wins dedup — must see the whole dataset at once.
    rows = dataset.dataset.to_list()
    kept, dropped = DedupFilter()(rows)
    logger.info(f'[rsi_prepare] after global dedup: {n_after_pipeline} -> {len(kept)} '
                f'(dropped {len(dropped)} duplicates)')

    out = Dataset(DatasetMeta(data=kept))
    out.save_as(args.output)
    logger.info(f'[rsi_prepare] wrote {len(kept)} rows -> {args.output}')


if __name__ == '__main__':
    main()
