"""Standalone dataset-processing demo: run the full agentic QualityPreprocessor
over a slice of the raw OpenClaw CSV, then persist the cleaned + scored + tagged
trajectories with ``Dataset.save_as`` for inspection.

This is NOT a training script — no template/encode/pack. The point is to see how
the preprocessor behaves end-to-end and to keep ALL enrichment:

- **tags / scores in ``user_data``** (AUDIT A5 envelope): per-round & trajectory
  scores (``TrajectoryScorer``), safety (``SafetyScorer``), intent key-rounds
  (``IntentClassifier``), structural-noise ratio (``StructuralNoiseTagger``),
  and provenance/lineage (``ProvenanceStamp``).
- **important fields preserved**: ``id``/``source``/``model_id``/``messages``
  are never dropped; ``MessageNormalizer`` now passes through
  ``reasoning_content``/``thinking`` (AUDIT P3).

Pipeline follows the tag-then-filter architecture: mappers annotate, a final
read-only ``TrajectoryOutcomeFilter`` drops on the tags (no DAG, linear order).

Run:
    CSV_PATH=/mnt/data/yzhao/tastelikefeet/bc/20260531.csv \
    USE_RUBRIC=1 SELECT_FRAC=0.1 DATASET_TOTAL=2000 \
    python cookbook/exp/data_pipeline/process_and_save.py

With gating, rubric LLM cost ~ ``SELECT_FRAC * N_kept`` (not ``N_kept``). Ingest
more rows in pass 1; only the global top fraction gets rubric in pass 2.
"""
import json
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List

from twinkle.dataset import Dataset
from twinkle.dataset.base import DatasetMeta
from twinkle.utils import get_logger
from twinkle_agentic.preprocessor import (DeadLoopFilter, HardFilter,
                                          IntentClassifier, LanguageFilter,
                                          MessageNormalizer,
                                          MessageSanityFilter, ModelFilter,
                                          ProvenanceStamp, QualityPreprocessor,
                                          RefuseFilter, SafetyScorer,
                                          SpecialCharsFilter,
                                          StructuralNoiseTagger,
                                          TokenSoupFilter,
                                          TrajectoryOutcomeFilter,
                                          TrajectoryScorer,
                                          ValueSelector,
                                          merge_dropped_shards,
                                          run_quality_pipeline,
                                          select_top_for_rubric,
                                          truncate_dropped_logs)
from twinkle_agentic.preprocessor import label_schema as L

logger = get_logger()

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH = os.environ.get('CSV_PATH', '/mnt/data/yzhao/tastelikefeet/bc/20260531.csv')
# Default 2000: pass-1 (filter + value_score) scales with N; with USE_RUBRIC=1 and
# SELECT_FRAC=0.1, rubric cost stays ~10% of survivors (similar to old 200×full rubric).
DATASET_TOTAL = int(os.environ.get('DATASET_TOTAL', 2000))
MAP_NUM_PROC = int(os.environ.get('MAP_NUM_PROC', 8))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', './output/data_pipeline')
_OUTPUT_BASENAME = os.environ.get(
    'OUTPUT_BASENAME',
    f'processed_{Path(CSV_PATH).stem}_{DATASET_TOTAL}.jsonl',
)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, _OUTPUT_BASENAME)
DROPPED_PATH = os.path.join(OUTPUT_DIR, 'dropped.jsonl')
PIPELINE_VERSION = os.environ.get('PIPELINE_VERSION', 'audit-v1')
# Set to keep only trajectories above this fused score. None -> keep all (inspect scores only).
MIN_TRAJ_SCORE = os.environ.get('MIN_TRAJ_SCORE')
MIN_TRAJ_SCORE = float(MIN_TRAJ_SCORE) if MIN_TRAJ_SCORE else None
# TrajectoryScorer runs hard-only by default (deterministic, fast). Set USE_RUBRIC=1
# to attach a RubricVerifier so per-segment scores get real LLM semantic signal
# (needs LLM_BACKUP_* / OPENAI_API_KEY; much slower). Without it every clean
# trajectory tends to collapse to level 4 because only hard checks discriminate.
USE_RUBRIC = os.environ.get('USE_RUBRIC', '') not in ('', '0', 'false', 'False')
TRAJ_FUSION = os.environ.get('TRAJ_FUSION', 'hard_soft_blend')
TRAJ_HARD_CEIL_SKIP = os.environ.get('TRAJ_HARD_CEIL_SKIP')
TRAJ_HARD_CEIL_SKIP = float(TRAJ_HARD_CEIL_SKIP) if TRAJ_HARD_CEIL_SKIP else (0.92 if USE_RUBRIC else None)
TRAJ_SCORER_WORKERS = int(os.environ.get('TRAJ_SCORER_WORKERS', '2' if USE_RUBRIC else '1'))
# Rubric stabilization policy (reduces per-call score jitter for template-like
# intents). 'skeleton' = half-fixed core + generated tail (flexible, DEFAULT),
# 'fixed' = fully fixed per intent (max stability), 'off' = pure generation.
TRAJ_RUBRIC_MODE = os.environ.get('TRAJ_RUBRIC_MODE', 'skeleton').strip().lower()
# Active-learning pre-selection: only the global top SELECT_FRAC by value_score
# gets an (expensive) rubric pass; the rest are hard-only. 1.0 = label everyone
# (disables gating). Only takes effect with USE_RUBRIC=1.
SELECT_FRAC = float(os.environ.get('SELECT_FRAC', '0.1'))
SELECT_MIN = int(os.environ.get('SELECT_MIN', '0'))
SELECT_MAX = os.environ.get('SELECT_MAX')
SELECT_MAX = int(SELECT_MAX) if SELECT_MAX else None
# Persist the full per-segment rubric diagnosis (per-criterion verdict + reason +
# fix + raw teacher output) for rubric-scored rows — the SFT corpus to distill a
# PRM / error-checker LoRA later. Defaults ON when rubric labeling is enabled
# (one extra teacher call per scored segment). Set PERSIST_DIAGNOSIS=0 to skip.
PERSIST_DIAGNOSIS = os.environ.get(
    'PERSIST_DIAGNOSIS', '1' if USE_RUBRIC else '0') not in ('', '0', 'false', 'False')


# ── CSV ingestion (custom format: `ts,model,req_id,messages_json`) ─────────────
def _canonicalize_tool_call(tc: Any) -> Dict[str, Any]:
    """Coerce a raw tool_call into a fixed-schema dict for stable Arrow inference."""
    tc = tc if isinstance(tc, dict) else {}
    fn = tc.get('function') if isinstance(tc.get('function'), dict) else {}
    args = fn.get('arguments')
    if isinstance(args, dict):
        args_str = json.dumps(args, ensure_ascii=False)
    elif isinstance(args, str) and args.strip():
        try:
            decoded = json.loads(args)
        except json.JSONDecodeError:
            decoded = {}
        args_str = json.dumps(decoded if isinstance(decoded, dict) else {}, ensure_ascii=False)
    else:
        args_str = '{}'
    return {
        'id': str(tc.get('id') or ''),
        'type': str(tc.get('type') or 'function'),
        'function': {'name': str(fn.get('name') or ''), 'arguments': args_str},
    }


def _stream_csv_rows(csv_path: str, max_rows: int = 0) -> Iterator[Dict[str, Any]]:
    """Stream the custom CSV. First 3 fields are scalar; the rest of the line is a
    JSON array of chat messages (may contain commas) — split on the first 3 commas.

    ``reasoning_content`` is folded into a ``<think>...</think>`` prefix so it
    survives as visible content and is later re-exposed by MessageNormalizer (P3).
    """
    emitted = 0
    with open(csv_path, 'rb') as f:
        for raw in f:
            try:
                line = raw.decode('utf-8').rstrip('\n').rstrip('\r')
            except UnicodeDecodeError:
                continue
            if not line:
                continue
            parts = line.split(',', 3)
            if len(parts) < 4:
                continue
            ts, model, req_id, msgs_raw = parts
            try:
                raw_msgs = json.loads(msgs_raw)
            except json.JSONDecodeError:
                continue
            messages: List[Dict[str, Any]] = []
            for m in raw_msgs:
                role = m.get('role', '')
                content = m.get('content')
                if isinstance(content, list):
                    content = ''.join(p.get('text', '') for p in content
                                      if isinstance(p, dict) and p.get('type') == 'text')
                if content is None:
                    content = ''
                if not isinstance(content, str):
                    continue
                raw_tcs = m.get('tool_calls') if role == 'assistant' else None
                tc_list = [_canonicalize_tool_call(tc) for tc in raw_tcs] if raw_tcs else []
                if role == 'assistant':
                    if not content and not tc_list:
                        continue
                    if m.get('reasoning_content'):
                        content = f"<think>{m['reasoning_content']}</think>{content}"
                elif role != 'tool' and not content:
                    continue
                messages.append({
                    'role': role,
                    'content': content,
                    'tool_calls': json.dumps(tc_list, ensure_ascii=False) if tc_list else '',
                    'tool_call_id': str(m.get('tool_call_id') or '') if role == 'tool' else '',
                })
            if not messages:
                continue
            yield {
                'id': f'csv__{ts}__{req_id}',
                'source': Path(csv_path).stem,
                'model_id': model,
                'messages': messages,
                'user_data': [],
                # Pre-declare the only top-level column a downstream step adds
                # (IntentClassifier). Without it the ingest schema lacks `intent`,
                # and HF datasets.map(num_proc>1) infers per-shard features from
                # the FIRST finished writer; shards that added `intent` later get
                # that column dropped on concat -> rows with intent/value/prov all
                # None (observed as 140/327 at 500 rows x num_proc=32). Declaring
                # it up front keeps the Arrow schema identical across shards.
                'intent': None,
            }
            emitted += 1
            if max_rows and emitted >= max_rows:
                break


def _build_trajectory_scorer() -> TrajectoryScorer:
    """Hard-only by default; attach an LLM RubricVerifier when USE_RUBRIC is set."""
    rubric_verifier = None
    if USE_RUBRIC:
        from twinkle_agentic.verifier import (RubricVerifier,
                                              default_intent_base_rubrics,
                                              default_intent_fixed_rubrics)
        # Intent-aware rubric stabilization: half-fixed skeleton (default) keeps
        # the generator flexible while anchoring a shared core; 'fixed' drops
        # generation entirely for tool_call/code/math; 'off' = pure generation.
        intent_base = intent_fixed = None
        if TRAJ_RUBRIC_MODE == 'skeleton':
            intent_base = default_intent_base_rubrics()
        elif TRAJ_RUBRIC_MODE == 'fixed':
            intent_fixed = default_intent_fixed_rubrics()
        rubric_verifier = RubricVerifier(
            max_votes=5,
            max_votes_long=3,
            min_votes_long=2,
            long_margin_threshold=0.18,
            # Re-sample top-band segments so "looks perfect" isn't a lucky draw.
            min_votes_high=3,
            high_score_threshold=0.85,
            intent_base_rubrics=intent_base,
            intent_rubrics=intent_fixed,
        )
    return TrajectoryScorer(
        rubric_verifier=rubric_verifier,
        fusion=TRAJ_FUSION,
        hard_ceil_skip=TRAJ_HARD_CEIL_SKIP,
        scorer_workers=TRAJ_SCORER_WORKERS,
        reconcile_max_messages=int(os.environ.get('TRAJ_RECONCILE_MAX_MSGS', '80')),
        # Persist the full rubric diagnosis (verdict+reason+fix+raw) for scored
        # segments — the SFT corpus for a distilled PRM/checker LoRA.
        persist_diagnosis=PERSIST_DIAGNOSIS,
    )


def build_pipeline_pass1() -> QualityPreprocessor:
    """Pass 1: clean + tag + cheap value scoring (NO LLM rubric).

    Everything here is deterministic/parallel-safe. It ends by stamping a
    ``value_score`` on every surviving row so the driver can then pick the global
    top fraction for the (expensive) rubric pass. When rubric labeling is off,
    the whole pipeline is a single pass and ValueSelector is skipped.
    """
    steps = [
        # 0) lineage first, so even dropped rows carry provenance in the log.
        ProvenanceStamp(source=Path(CSV_PATH).stem, pipeline_version=PIPELINE_VERSION),
        # 1) canonicalize message schema (heartbeat strip, tool-call normalize,
        #    reasoning passthrough — P3), then structural / content filters.
        MessageNormalizer(),
        ModelFilter(),
        LanguageFilter(allowed=('en', 'zh')),
        # Shallow-chat round cap is 40; agent traces capped at 20 logical rounds
        # (min user/assistant counts) for pipeline experiments — raise for prod.
        HardFilter(min_user_chars_cjk=14, min_user_chars=24, max_rounds=40,
                   agent_max_rounds=20),
        RefuseFilter(),
        DeadLoopFilter(),
        MessageSanityFilter(),
        SpecialCharsFilter(max_ratio=0.6),
        TokenSoupFilter(max_chars=8000),
        # 2) taggers (never drop): intent key-rounds, structural noise ratio.
        IntentClassifier(),
        StructuralNoiseTagger(),
    ]
    if USE_RUBRIC and SELECT_FRAC < 1.0:
        # Active-learning pre-selection: cheap value_score for top-fraction gating.
        steps.append(ValueSelector())
    if not USE_RUBRIC:
        # Single-pass mode: fold scoring + safety + outcome filter in here.
        # No selection happened, so safety scores every row (gated=None).
        steps += _pass2_tail(gated=False)
    # drop_mode='mark': map returns equal-length columns (dropped rows flagged),
    # and run_quality_pipeline materializes the removal via Dataset.filter — so a
    # partially filtered batch can never leave ghost rows (no remove_columns hack).
    return QualityPreprocessor(pipeline=steps, dropped_log_path=DROPPED_PATH,
                               drop_mode='mark')


def _pass2_tail(gated: bool) -> list:
    """Scoring + safety + outcome filter (the LLM-touching tail).

    When ``gated`` is True (two-pass active-learning mode) both the rubric scorer
    and the safety scorer only spend an LLM call on rows pre-selected by
    ValueSelector (``selected_for_rubric``); everyone else is hard/neutral only.
    """
    gate = L.KEY_SELECTED_FOR_RUBRIC if gated else None
    return [
        _build_trajectory_scorer(),   # hard-only, or LLM rubric when USE_RUBRIC=1
        # Fixed safety rubric; gated post-selection so the LLM safety pass runs
        # only on selected rows (neutral-safe otherwise).
        SafetyScorer(gate_label=gate),
        # read-only outcome filter: drops on the tags above (D6). Enabled only
        # when MIN_TRAJ_SCORE is set, else we keep everything to inspect.
        TrajectoryOutcomeFilter(
            min_traj_score=MIN_TRAJ_SCORE if MIN_TRAJ_SCORE is not None else 0.0,
            min_safety_score=None,
            drop_unsafe_flag=False,
        ),
    ]


def build_pipeline_pass2(gated: bool = True) -> QualityPreprocessor:
    """Pass 2: rubric + safety (both gated on ``selected_for_rubric``) + filter."""
    return QualityPreprocessor(pipeline=_pass2_tail(gated=gated),
                               dropped_log_path=DROPPED_PATH, drop_mode='mark')


def _print_sample(dataset: Dataset, n: int = 3) -> None:
    """Show the enrichment kept on a few rows so you can eyeball tags/scores."""
    hf = dataset.dataset
    show = min(n, len(hf))
    logger.info(f'── sample of {show} processed rows (tags/scores in user_data) ──')
    for i in range(show):
        row = hf[i]
        logger.info(
            f"[{row.get('id')}] model={row.get('model_id')} n_msgs={len(row.get('messages') or [])}\n"
            f"    intent      = {row.get('intent')}\n"
            f"    traj_score  = {L.get_label(row, L.KEY_TRAJ_SCORE)}  "
            f"level={L.get_label(row, L.KEY_TRAJ_LEVEL)}  "
            f"conf={L.get_label(row, L.KEY_TRAJ_CONFIDENCE)}\n"
            f"    round_scores= {L.get_label(row, L.KEY_ROUND_SCORES)}\n"
            f"    safety      = {L.get_label(row, L.KEY_SAFETY_SCORE)} "
            f"(unsafe={L.get_label(row, L.KEY_SAFETY_UNSAFE)})\n"
            f"    noise_ratio = {L.get_label(row, 'structural_noise_ratio')}\n"
            f"    provenance  = {L.get_label(row, L.KEY_PROVENANCE)}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f'Loading up to {DATASET_TOTAL} rows from {CSV_PATH}')

    meta = DatasetMeta(
        dataset_id=Path(CSV_PATH).stem,
        data=partial(_stream_csv_rows, csv_path=CSV_PATH, max_rows=DATASET_TOTAL),
    )
    dataset = Dataset(meta)
    logger.info(f'Ingested {len(dataset.dataset)} rows.')

    truncate_dropped_logs(DROPPED_PATH)
    # Pass 1: clean + tag (+ value_score when gating). run_quality_pipeline runs
    # the pipeline as map(equal-length columns, dropped rows flagged) + a single
    # Dataset.filter — map never changes row count, so no ghost rows can appear
    # (unlike a filtering map, which needs remove_columns and still risks
    # partial-batch ghosting). num_proc parallelizes across shards.
    run_quality_pipeline(dataset, build_pipeline_pass1(),
                         num_proc=MAP_NUM_PROC, load_from_cache_file=False)
    merge_dropped_shards(DROPPED_PATH)
    logger.info(f'After pass 1 (clean+tag): {len(dataset.dataset)} rows kept.')

    if USE_RUBRIC:
        gating = SELECT_FRAC < 1.0
        if gating:
            # Global top-fraction: pick the most valuable rows for the LLM pass.
            _, n_sel = select_top_for_rubric(
                dataset, select_frac=SELECT_FRAC,
                min_select=SELECT_MIN, max_select=SELECT_MAX)
            logger.info(f'Value-gated rubric: {n_sel} rows selected for LLM labeling '
                        f'(frac={SELECT_FRAC}).')
        # Pass 2: rubric + safety (both gated when a selection ran) + outcome filter.
        run_quality_pipeline(dataset, build_pipeline_pass2(gated=gating),
                             num_proc=MAP_NUM_PROC, load_from_cache_file=False)
        merge_dropped_shards(DROPPED_PATH)

    logger.info(f'After pipeline: {len(dataset.dataset)} rows kept.')
    _print_sample(dataset)

    dataset.save_as(OUTPUT_PATH, format='jsonl')
    logger.info(f'Saved processed dataset -> {OUTPUT_PATH}')
    logger.info(f'Dropped rows log -> {DROPPED_PATH}')


if __name__ == '__main__':
    main()
