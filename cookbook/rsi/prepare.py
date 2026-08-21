# Copyright (c) ModelScope Contributors. All rights reserved.
"""Clean a raw dataset into seeds for the RSI challengers.

Reads a raw source, runs it through the ``twinkle_agentic.preprocessor``
pipeline in parallel, and writes the surviving rows. What comes out is seed
material: ``cookbook/rsi/code/challenge.py`` and the agentic challenger take it
as a pool to draw inspiration from, so anything junk in here becomes junk the
challenger imitates.

Usage
-----
    # multi-turn tool-calling data (ToolACE and friends)
    python cookbook/rsi/prepare.py --input ms://... --output output/rsi/agentic_seeds.jsonl

    # pure code data (MBPP and friends)
    python cookbook/rsi/prepare.py --input mbpp.jsonl --output output/rsi/code_seeds.jsonl \
        --no-normalize-tool-calls

``--input`` accepts a local ``.jsonl``/``.parquet`` path or an ``ms://`` dataset
id. Every row must expose a ``messages`` list -- the preprocessor keys off it;
ShareGPT ``conversations`` rows are adapted automatically.

Pipeline
--------
Core steps, always on, each using the filter's OWN default thresholds (nothing
invented here):

    MessageNormalizer -> MessageSanityFilter -> RefuseFilter -> DeadLoopFilter
    -> TokenSoupFilter -> HardFilter

Optional steps, off unless asked for (each needs extra packages):
    --use-lang        LanguageFilter        (langid, degrades to a heuristic)
    --use-datajuicer  FixUnicode / RemoveRepeat / SpecialChars / TokenNum
    --use-pii         PIIPresidioFilter     (presidio-analyzer/anonymizer)

``DedupFilter`` is not part of the parallel pipeline: it has to see the whole
dataset in one call, so it runs once afterwards.
"""
import argparse
import os

from twinkle.dataset import Dataset, DatasetMeta
from twinkle.utils import get_logger
from twinkle_agentic.preprocessor import (DeadLoopFilter, DedupFilter, HardFilter, MessageNormalizer,
                                          MessageSanityFilter, QualityPreprocessor, RefuseFilter, TokenSoupFilter,
                                          merge_dropped_shards, run_quality_pipeline, truncate_dropped_logs)

logger = get_logger()

# ShareGPT `from` value -> standard message role. ToolACE uses
# system/user/assistant/tool; other ShareGPT variants use human/gpt/observation.
_ROLE_MAP = {
    'system': 'system',
    'user': 'user', 'human': 'user',
    'assistant': 'assistant', 'gpt': 'assistant', 'bot': 'assistant',
    'tool': 'tool', 'observation': 'tool', 'function': 'tool',
    'function_call': 'assistant', 'function_response': 'tool', 'tool_response': 'tool',
}


def build_pipeline(args):
    """The ordered steps for the parallel pass (dedup is applied separately)."""
    steps = [
        MessageNormalizer(normalize_tool_calls=args.normalize_tool_calls),
        MessageSanityFilter(),   # role order / tool-id matching / content integrity / sensitive words
        RefuseFilter(),          # drop assistant self-referential refusals
        DeadLoopFilter(),        # drop degenerate / stuck (hesitation, cascade, ngram repeat)
        TokenSoupFilter(),       # drop garbled text (replacement/control/private-use chars, script chaos)
        HardFilter(min_assistant_chars_2turn=args.min_assistant_chars_2turn),
    ]
    if args.use_lang:
        from twinkle_agentic.preprocessor import LanguageFilter
        steps.append(LanguageFilter())
    if args.use_datajuicer:
        from twinkle_agentic.preprocessor import (FixUnicodeFilter, RemoveRepeatSentencesFilter, SpecialCharsFilter,
                                                  TokenNumFilter)
        steps += [FixUnicodeFilter(), RemoveRepeatSentencesFilter(), SpecialCharsFilter(), TokenNumFilter()]
    if args.use_pii:
        from twinkle_agentic.preprocessor import PIIPresidioFilter
        steps.append(PIIPresidioFilter())
    return steps


def _row_to_messages(row: dict) -> dict:
    """Map one ShareGPT ``conversations`` row to a ``messages`` row.

    Only ``from``->``role`` and ``value``->``content`` are rewritten; a tool call
    embedded in an assistant turn is left as-is in ``content`` (ToolACE keeps it
    as a bracket-DSL string). Turns whose ``from`` is unknown are dropped so no
    invalid role reaches the pipeline.
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


def load_source(input_path: str) -> Dataset:
    """Load the raw source into a twinkle Dataset.

    A local path is loaded by extension (jsonl->json, parquet, csv...); anything
    else is treated as a hub id (e.g. ``ms://org/name``). Rows pass through
    unchanged except for the ShareGPT adaptation above.
    """
    ds = Dataset(DatasetMeta(dataset_id=input_path))
    cols = ds.dataset.column_names
    if 'messages' not in cols and 'conversations' in cols:
        logger.info('[prepare] ShareGPT `conversations` detected -> mapping to `messages`')
        # Materialize + convert in Python then rebuild: twinkle's Dataset.map forces
        # batched=True and wraps the fn as a Preprocessor, which does not fit a plain
        # per-row schema rewrite. The source is small enough to hold in memory.
        rows = [_row_to_messages(r) for r in ds.dataset.to_list()]
        ds = Dataset(DatasetMeta(data=rows))
    return ds


def parse_args():
    p = argparse.ArgumentParser(description='Clean a raw source into RSI seed material.',
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', required=True, help='Local .jsonl/.parquet path or an ms:// dataset id.')
    p.add_argument('--output', default='output/rsi/subset.jsonl', help='Where to write the surviving rows.')
    p.add_argument('--num-proc', type=int, default=4, help='Parallel workers for the map pass.')
    p.add_argument('--dropped-log', default='', help='Optional JSONL of dropped-row metadata (empty=off).')

    # On for tool-calling data. Off for pure code (e.g. MBPP): the bracket-DSL
    # parser is a marker-less fallback matching ``[name(``, which is also what a
    # list comprehension or a call-indexed subscript looks like, so the rewrite
    # silently deletes real code from the assistant turn.
    p.add_argument('--no-normalize-tool-calls', dest='normalize_tool_calls',
                   action='store_false', help='pure code data: leave assistant text alone')
    p.set_defaults(normalize_tool_calls=True)
    # 0, not HardFilter's own 80-char floor: a single-turn valid tool call
    # (e.g. `[Func(x=1)]`) is only tens of chars and would be dropped as a
    # "shallow_reply". Rule 3 still removes genuinely empty assistant turns.
    p.add_argument('--min-assistant-chars-2turn', type=int, default=0)

    p.add_argument('--use-lang', action='store_true', help='LanguageFilter (needs langid)')
    p.add_argument('--use-datajuicer', action='store_true', help='data_juicer-based filters')
    p.add_argument('--use-pii', action='store_true', help='PIIPresidioFilter (needs presidio)')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)

    pipeline = build_pipeline(args)
    logger.info(f'[prepare] pipeline: {" -> ".join(type(s).__name__ for s in pipeline)} '
                f'+ DedupFilter(global)')

    dataset = load_source(args.input)
    n_in = len(dataset.dataset)
    logger.info(f'[prepare] loaded {n_in} rows from {args.input}')

    # 'mark' mode + run_quality_pipeline is the ghost-proof parallel path: map
    # returns equal-length columns flagged _keep, then a single filter removes.
    if args.dropped_log:
        truncate_dropped_logs(args.dropped_log)
    qp = QualityPreprocessor(pipeline, dropped_log_path=args.dropped_log, drop_mode='mark')
    run_quality_pipeline(dataset, qp, num_proc=args.num_proc)
    if args.dropped_log:
        merge_dropped_shards(args.dropped_log)

    n_after_pipeline = len(dataset.dataset)
    logger.info(f'[prepare] after parallel pipeline: {n_in} -> {n_after_pipeline}')

    # Global longest-wins dedup -- must see the whole dataset at once.
    rows = dataset.dataset.to_list()
    kept, dropped = DedupFilter()(rows)
    logger.info(f'[prepare] after global dedup: {n_after_pipeline} -> {len(kept)} '
                f'(dropped {len(dropped)} duplicates)')

    out = Dataset(DatasetMeta(data=kept))
    out.save_as(args.output)
    logger.info(f'[prepare] wrote {len(kept)} rows -> {args.output}')


if __name__ == '__main__':
    main()
