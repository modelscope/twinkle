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
    DATASET_TOTAL=200 python cookbook/exp/data_pipeline/process_and_save.py
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
                                          TrajectoryScorer)
from twinkle_agentic.preprocessor import label_schema as L

logger = get_logger()

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH = os.environ.get('CSV_PATH', '/mnt/data/yzhao/tastelikefeet/bc/20260531.csv')
DATASET_TOTAL = int(os.environ.get('DATASET_TOTAL', 200))
MAP_NUM_PROC = int(os.environ.get('MAP_NUM_PROC', 8))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', './output/data_pipeline')
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'processed_20260531_200.jsonl')
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
            }
            emitted += 1
            if max_rows and emitted >= max_rows:
                break


def _build_trajectory_scorer() -> TrajectoryScorer:
    """Hard-only by default; attach an LLM RubricVerifier when USE_RUBRIC is set."""
    rubric_verifier = None
    if USE_RUBRIC:
        from twinkle_agentic.verifier import RubricVerifier
        # No sampler -> scores via the llm_backup teacher path (LLM_BACKUP_*/OPENAI_*).
        rubric_verifier = RubricVerifier()
    return TrajectoryScorer(rubric_verifier=rubric_verifier)


def build_pipeline() -> QualityPreprocessor:
    """Full tag-then-filter pipeline. Mappers annotate; the tail filter drops on tags."""
    return QualityPreprocessor(
        pipeline=[
            # 0) lineage first, so even dropped rows carry provenance in the log.
            ProvenanceStamp(source=Path(CSV_PATH).stem, pipeline_version=PIPELINE_VERSION),
            # 1) canonicalize message schema (heartbeat strip, tool-call normalize,
            #    reasoning passthrough — P3), then structural / content filters.
            MessageNormalizer(),
            ModelFilter(),
            LanguageFilter(allowed=('en', 'zh')),
            # Shallow-chat round cap is 40; agent traces get a far higher ceiling
            # (agent_max_rounds) so long tool-calling loops — the highest-value
            # distillation data — are not clipped.
            HardFilter(min_user_chars_cjk=14, min_user_chars=24, max_rounds=40,
                       agent_max_rounds=200),
            RefuseFilter(),
            DeadLoopFilter(),
            MessageSanityFilter(),
            SpecialCharsFilter(max_ratio=0.6),
            TokenSoupFilter(max_chars=8000),
            # 2) taggers (never drop): intent key-rounds, structural noise ratio,
            #    per-round + trajectory scores, safety score. All write user_data.
            IntentClassifier(),
            StructuralNoiseTagger(),
            _build_trajectory_scorer(),   # hard-only, or LLM rubric when USE_RUBRIC=1
            SafetyScorer(),       # fixed safety rubric; no sampler -> neutral score, still tagged
            # 3) read-only outcome filter: drops on the tags above (D6). Enabled
            #    only when MIN_TRAJ_SCORE is set, else we keep everything to inspect.
            TrajectoryOutcomeFilter(
                min_traj_score=MIN_TRAJ_SCORE if MIN_TRAJ_SCORE is not None else 0.0,
                min_safety_score=None,
                drop_unsafe_flag=False,
            ),
        ],
        dropped_log_path=DROPPED_PATH,
    )


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

    pipeline = build_pipeline()
    # Dataset.map runs the QualityPreprocessor over HF batches (batched=True is
    # forced internally). num_proc parallelizes across shards.
    dataset.map(pipeline, num_proc=MAP_NUM_PROC, load_from_cache_file=False)

    logger.info(f'After pipeline: {len(dataset.dataset)} rows kept.')
    _print_sample(dataset)

    dataset.save_as(OUTPUT_PATH, format='jsonl')
    logger.info(f'Saved processed dataset -> {OUTPUT_PATH}')
    logger.info(f'Dropped rows log -> {DROPPED_PATH}')


if __name__ == '__main__':
    main()
