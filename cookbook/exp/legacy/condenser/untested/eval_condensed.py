"""Evaluation: native (full ctx) vs condensed (chunk → condense → extract_condensed tool).

Reuses the training-time data shape and prompt so the comparison is apples-to-apples.

Launch:
    # native baseline (full HotpotQA context, no compression, no tool)
    python cookbook/exp/eval_condensed.py --mode native \\
        --dataset /path/to/hotpot_dev_fullwiki.jsonl

    # condensed (chunk → condense via Qwen3.5-4B-Condenser → extract_condensed tool)
    python cookbook/exp/eval_condensed.py --mode condensed \\
        --dataset /path/to/hotpot_dev_fullwiki.jsonl

Outputs (under --out_dir / <mode>_<run_id>/):
    predictions.jsonl   one row per sample with pred / gold / f1 / em / token-counts / tool-calls
    summary.json        aggregate metrics
"""
import argparse
import json
import os
import re
import time
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.data_format import Message, SamplingParams, Trajectory
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.sampler import vLLMSampler
from twinkle.template import Qwen3_5Template
from twinkle_agentic.chunker.native import NativeChunker
from twinkle_agentic.condenser import ModelCondenser
from twinkle_agentic.reward.f1 import _f1_score
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout
from twinkle_agentic.rollout.multi_turn_condense import MultiTurnCondenseRollout
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle.preprocessor.base import Preprocessor

# Reuse training assets so eval and train share data shape + condensed prompt.
from cookbook.exp.legacy.grpo_condensed import (
    SYSTEM_PROMPT as CONDENSED_SYSTEM_PROMPT,
    HotpotQAProcessor,
    _BOXED_RE,
    _last_assistant_text,
)


class MuSiQueProcessor(Preprocessor):
    """MuSiQue-Ans → Trajectory adapter.

    MuSiQue native schema (per row):
        id, question, paragraphs=[{idx, title, paragraph_text, is_supporting}], answer,
        answer_aliases=[...], answerable, question_decomposition=[...]

    Maps to the same Trajectory(messages, user_data) shape that
    :class:`HotpotQAProcessor` produces, so downstream rollout code is
    schema-agnostic. ``ground_truth`` carries answer + answer_aliases.
    """

    def __init__(self, system: str):
        self.system = system

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out = [self.preprocess(r) for r in rows]
        out = [r for r in out if r is not None]
        return self.map_row_to_col(out)

    @staticmethod
    def _format_context(paragraphs: List[Dict[str, Any]]) -> str:
        lines = []
        for p in paragraphs or []:
            title = (p.get('title') or '').strip()
            body = (p.get('paragraph_text') or '').strip()
            if not body:
                continue
            lines.append(f'{title}: {body}' if title else body)
        return '\n\n'.join(lines)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Trajectory]:
        if row.get('answerable') is False:
            return None
        question = (row.get('question') or '').strip()
        if not question:
            return None
        gold_main = (row.get('answer') or '').strip()
        aliases = row.get('answer_aliases') or []
        gold = [g for g in dict.fromkeys([gold_main] + list(aliases)) if g]
        if not gold:
            return None
        paragraphs = row.get('paragraphs') or []
        context_block = self._format_context(paragraphs)
        user_msg = f'Question: {question}\n\nContext:\n\n{context_block}'
        messages = [
            Message(role='system', content=self.system),
            Message(role='user', content=user_msg),
        ]
        sf_titles = list(dict.fromkeys(
            (p.get('title') or '').strip()
            for p in paragraphs
            if p.get('is_supporting') and (p.get('title') or '').strip()))
        user_data = [('ground_truth', g) for g in gold] + [('sf_title', t) for t in sf_titles]
        return Trajectory(messages=messages, user_data=user_data)

logger = get_logger()

NATIVE_SYSTEM_PROMPT = """You are a careful multi-hop QA assistant.

The user message contains a Question and a Context. Read both, reason step by step,
then commit to a final answer.

## Output Format
End your final response with \\boxed{answer}.
Keep the boxed text short: a name, entity, date, or "yes"/"no".
Answers not inside \\boxed{} will not be scored."""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['native', 'condensed'], required=True)
    p.add_argument('--dataset', required=True,
                   help='Eval set jsonl. HotpotQA or MuSiQue-Ans schema (see --dataset_format).')
    p.add_argument('--dataset_format', choices=['hotpotqa', 'musique'], default='musique',
                   help='Schema of --dataset. MuSiQue-Ans (default) is harder multi-hop and OOD vs training.')
    p.add_argument('--model_id', default='ms://Qwen/Qwen3.5-4B')
    p.add_argument('--lora_path', default=None,
                   help='Optional LoRA adapter on top of model_id (e.g. trained QA LoRA).')
    p.add_argument('--condenser_lora', default='ms://twinkle-kit/Qwen3.5-4B-Condenser')
    p.add_argument('--limit', type=int, default=500)
    p.add_argument('--num_gpus', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_model_len', type=int, default=32768)
    p.add_argument('--max_new_tokens', type=int, default=2048)
    p.add_argument('--max_turns', type=int, default=4)
    p.add_argument('--max_trajectory_tokens', type=int, default=8192)
    p.add_argument('--chunk_size', type=int, default=1024)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--out_dir', default='eval_out')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def build_dataset(path: str, dataset_format: str, model_id: str,
                  max_length: int, limit: int, system: str) -> Dataset:
    """Load eval JSONL and produce Trajectory rows tagged with ground_truth user_data."""
    ds = Dataset()
    ds.add_dataset(DatasetMeta(path))
    if limit > 0 and len(ds) > limit:
        ds = ds.select(range(limit))
    ds.set_template(
        'Qwen3_5Template', model_id=model_id, max_length=max_length,
        truncation_strategy='delete', enable_thinking=False)
    if dataset_format == 'musique':
        # MuSiQue-Ans cols (drop everything; we keep only the produced messages/user_data)
        cols = ['id', 'question', 'paragraphs', 'answer', 'answer_aliases',
                'answerable', 'question_decomposition']
        ds.map(MuSiQueProcessor(system=system), remove_columns=cols)
    else:
        cols = ['id', 'question', 'question_fixed', 'answers', 'original_answer',
                'type', 'level', 'verdict', 'reasoning', 'supporting_facts', 'context']
        ds.map(HotpotQAProcessor(system=system), remove_columns=cols)
    return ds


def extract_boxed(text: str) -> Optional[str]:
    """Pull the inner text of the LAST `\\boxed{...}` marker, brace-balanced enough for short answers."""
    if not text:
        return None
    matches = _BOXED_RE.findall(text)
    if not matches:
        return None
    last = matches[-1]
    return last[len(r'\boxed{'):-1].strip()


def best_f1_em(pred: str, golds: List[str]) -> Dict[str, float]:
    """Max-over-references SQuAD-style F1 / EM, reusing the training reward's normalizer."""
    if not golds:
        return {'f1': 0.0, 'em': 0.0}
    if not pred:
        return {'f1': 0.0, 'em': 0.0}
    best_f1, best_em = 0.0, 0.0
    for g in golds:
        f1, em = _f1_score(pred, g)
        if f1 > best_f1:
            best_f1 = f1
        if em > best_em:
            best_em = em
    return {'f1': best_f1, 'em': best_em}


def _user_text(traj_or_msg) -> str:
    """Concat all text parts of the first user message — used to count original context tokens."""
    msgs = traj_or_msg if isinstance(traj_or_msg, list) else (traj_or_msg.get('messages') or [])
    for m in msgs:
        role = m.get('role') if isinstance(m, dict) else getattr(m, 'role', None)
        if role != 'user':
            continue
        content = m.get('content') if isinstance(m, dict) else getattr(m, 'content', None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return ''.join(p.get('text') or '' for p in content if isinstance(p, dict) and p.get('type') == 'text')
        return ''
    return ''


def _count_tool_calls(traj: Dict[str, Any]) -> int:
    return sum(len(m.get('tool_calls') or [])
               for m in (traj.get('messages') or []) if m.get('role') == 'assistant')


def main():
    args = parse_args()
    run_id = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    out_dir = os.path.join(args.out_dir, f'{args.mode}_{run_id}')
    os.makedirs(out_dir, exist_ok=True)

    device_groups = [DeviceGroup(name='sampler', ranks=list(range(args.num_gpus)), device_type='GPU')]
    sampler_mesh = DeviceMesh.from_sizes(world_size=args.num_gpus, dp_size=args.num_gpus)
    twinkle.initialize(mode='ray', nproc_per_node=args.num_gpus,
                       groups=device_groups, lazy_collect=False)

    system = CONDENSED_SYSTEM_PROMPT if args.mode == 'condensed' else NATIVE_SYSTEM_PROMPT
    ds = build_dataset(args.dataset, args.dataset_format, args.model_id,
                       args.max_model_len, args.limit, system)
    logger.info('Eval dataset: %d rows from %s (mode=%s, format=%s)',
                len(ds), args.dataset, args.mode, args.dataset_format)

    sampler = vLLMSampler(
        model_id=args.model_id,
        engine_args={
            'gpu_memory_utilization': 0.85, 'max_model_len': args.max_model_len,
            'max_lora_rank': 32, 'enable_lora': True,
            'enable_tower_connector_lora': True, 'max_loras': 5,
            'seed': args.seed,
        },
        device_mesh=sampler_mesh, remote_group='sampler')
    sampler.set_template('Qwen3_5Template', model_id=args.model_id,
                         enable_thinking=False, max_length=args.max_model_len)
    template = Qwen3_5Template(args.model_id, max_length=args.max_model_len, enable_thinking=False)

    # stop=['</tool_call>'] only matters for condensed mode where the model issues tool calls
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens, num_samples=1,
        temperature=args.temperature, top_p=0.95,
        stop=['</tool_call>'] if args.mode == 'condensed' else None,
    )

    if args.mode == 'condensed':
        chunker = NativeChunker(chunk_size=args.chunk_size, passage_boundary_re=r'(?<=\n\n)')
        # Chunk-level extraction of the question line; \A anchor avoids matching "Question:" inside passages.
        _q_re = re.compile(r'\AQuestion:\s*(.+)')

        def _q_from_chunk(chunk):
            c = chunk.get('content')
            if chunk.get('type') != 'text' or not isinstance(c, str):
                return None
            m = _q_re.search(c)
            return m.group(1).strip() if m else None

        condenser = ModelCondenser(
            sampler=sampler, compression_ratio=2.0,
            sampling_params=SamplingParams(max_tokens=1024, num_samples=1,
                                           temperature=0.4, top_p=0.9),
            min_chars=200, template=template,
            lora_path=args.condenser_lora, skip_pattern=r'^Question:',
            related_query=_q_from_chunk,
        )
        rollout = MultiTurnCondenseRollout(
            sampler=sampler, template=template, tool_manager=ToolManager(),
            chunker=chunker, condenser=condenser,
            sampling_params=sampling_params,
            max_turns=args.max_turns, max_trajectory_tokens=args.max_trajectory_tokens,
        )
    else:
        # max_turns=1, no tools: reduces to single-turn QA over the full original context
        rollout = MultiTurnRollout(
            sampler=sampler, template=template, tool_manager=ToolManager(),
            sampling_params=sampling_params,
            max_turns=1, max_trajectory_tokens=args.max_trajectory_tokens,
        )

    dataloader = DataLoader(dataset=ds, batch_size=args.batch_size,
                            min_batch_size=1, shuffle=False)

    pred_path = os.path.join(out_dir, 'predictions.jsonl')
    pf = open(pred_path, 'w', encoding='utf-8')

    agg = Counter()
    sums = {'f1': 0.0, 'em': 0.0,
            'prompt_tok': 0, 'comp_tok': 0, 'orig_ctx_tok': 0,
            'turns': 0, 'tool_calls': 0}
    t0 = time.time()

    for batch in dataloader:
        trajs = rollout(batch)

        for src, traj in zip(batch, trajs):
            text = _last_assistant_text(traj) or ''
            pred = extract_boxed(text) or ''
            golds = [v for k, v in (src.user_data or []) if k == 'ground_truth' and v]

            scores = best_f1_em(pred, golds)
            ids = traj.get('input_ids') or []
            comp_tok = sum(1 for l in (traj.get('labels') or []) if l != -100)
            prompt_tok = max(0, len(ids) - comp_tok)
            tool_calls = _count_tool_calls(traj)

            # Original (uncondensed) context size — feed only the user msg, not the system prompt,
            # so the compression ratio stays comparable across modes.
            orig_user = _user_text(src.messages)
            orig_ctx_tok = len(template.tokenizer.encode(orig_user)) if orig_user else 0

            agg['n'] += 1
            agg['no_box'] += int(_BOXED_RE.search(text) is None)
            agg['tool_use'] += int(tool_calls > 0)
            sums['f1'] += scores['f1']
            sums['em'] += scores['em']
            sums['prompt_tok'] += prompt_tok
            sums['comp_tok'] += comp_tok
            sums['orig_ctx_tok'] += orig_ctx_tok
            sums['turns'] += int(traj.get('turns') or 1)
            sums['tool_calls'] += tool_calls

            pf.write(json.dumps({
                'pred': pred,
                'gold': golds,
                'f1': scores['f1'],
                'em': scores['em'],
                'prompt_tok': prompt_tok,
                'comp_tok': comp_tok,
                'orig_ctx_tok': orig_ctx_tok,
                'tool_calls': tool_calls,
                'turns': int(traj.get('turns') or 1),
                'no_boxed': _BOXED_RE.search(text) is None,
                'response': text,
            }, ensure_ascii=False) + '\n')

        logger.info('[eval] %d / %d processed', agg['n'], len(ds))

    pf.close()
    wall = time.time() - t0
    n = max(1, agg['n'])
    summary = {
        'mode': args.mode,
        'dataset_format': args.dataset_format,
        'model_id': args.model_id,
        'lora_path': args.lora_path,
        'condenser_lora': args.condenser_lora if args.mode == 'condensed' else None,
        'dataset': args.dataset,
        'n_samples': agg['n'],
        # quality
        'f1': sums['f1'] / n,
        'em': sums['em'] / n,
        'no_boxed_rate': agg['no_box'] / n,
        # cost
        'avg_prompt_tokens': sums['prompt_tok'] / n,
        'avg_completion_tokens': sums['comp_tok'] / n,
        'avg_orig_context_tokens': sums['orig_ctx_tok'] / n,
        'compression_ratio': (sums['prompt_tok'] / sums['orig_ctx_tok']
                              if sums['orig_ctx_tok'] else None),
        # tool / multi-turn behavior
        'avg_turns': sums['turns'] / n,
        'avg_tool_calls': sums['tool_calls'] / n,
        'tool_use_rate': agg['tool_use'] / n,
        # wall
        'wall_time_sec': wall,
        'samples_per_sec': agg['n'] / wall if wall > 0 else 0.0,
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info('Done. Output: %s', out_dir)
    logger.info('Summary: %s', json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
