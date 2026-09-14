"""Benchmark: streaming sampling benefit for GRPO + reward loop (local ray mode).

Runs in the same local process two sampling paths over the *same* prompts and
weights, then compares them:

- Path A (batch baseline): ``sampler.sample`` returns the whole batch, then all
  RewardItems are submitted at once; rewards are collected when sampling ends.
- Path B (streaming): one concurrent remote ``sample([input])`` call per
  sequence; as each sequence finishes its RewardItem is submitted immediately,
  so reward computation overlaps with the remaining sequences' generation
  (local-mode equivalent of ``stream_sample_to_data_plane`` per-sample events).

Correctness is verified on two levels before / during the sweep:
- Level 1 (deterministic): greedy + fixed seed, both paths must produce
  identical tokens / logprobs / rewards on the same inputs.
- Level 2 (semantic): under random sampling, no dropped / duplicated / shuffled
  items, reward function determinism, and group-mean-zero advantages.

Outputs (JSONL timeline + CSV per-step summary) go to ``BENCH_OUT_DIR``.

Environment knobs: TWINKLE_MODEL_ID / TWINKLE_DATASET_ID / TWINKLE_MODEL_GPUS /
TWINKLE_SAMPLER_GPUS / TWINKLE_LEARNING_RATE / TWINKLE_ADAPTER_NAME /
TWINKLE_REWARD_NUM_WORKERS / TWINKLE_REWARD_DELAY_MS / BENCH_RUNS (all|smoke) /
BENCH_OUT_DIR.
"""
from __future__ import annotations

import asyncio
import csv
import itertools
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams, user_data_get
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.metric import CompletionRewardMetric
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.preprocessor.base import Preprocessor
from twinkle.reward import GSM8KAccuracyReward
from twinkle.reward_loop import (AsyncRewardPipeline, RewardItem, RewardResult,
                                 register)
from twinkle.reward_loop.reward_manager import RewardManagerBase
from twinkle.sampler import vLLMSampler

logger = get_logger()

# ---------------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------------
MODEL_ID = os.environ.get('TWINKLE_MODEL_ID', 'ms://Qwen/Qwen3.5-4B')
DATASET_ID = os.environ.get('TWINKLE_DATASET_ID', 'ms://modelscope/gsm8k')
# 奖励模型（judge）可独立于训练模型配置；默认跟随训练模型。
REWARD_MODEL_ID = os.environ.get('TWINKLE_REWARD_MODEL_ID', MODEL_ID)
# Qwen3.5/3.6 are multimodal (vision tower); other models use the plain chat
# template. Override explicitly with TWINKLE_TEMPLATE_CLS when needed.
_IS_MULTIMODAL_QWEN = 'Qwen3.5' in MODEL_ID or 'Qwen3.6' in MODEL_ID
TEMPLATE_CLS = os.environ.get(
    'TWINKLE_TEMPLATE_CLS',
    'Qwen3_5Template' if _IS_MULTIMODAL_QWEN else 'Template',
)
_IS_MULTIMODAL_REWARD = 'Qwen3.5' in REWARD_MODEL_ID or 'Qwen3.6' in REWARD_MODEL_ID
REWARD_TEMPLATE_CLS = os.environ.get(
    'TWINKLE_REWARD_TEMPLATE_CLS',
    'Qwen3_5Template' if _IS_MULTIMODAL_REWARD else 'Template',
)

MODEL_GPUS = int(os.environ.get('TWINKLE_MODEL_GPUS', '1'))
SAMPLER_GPUS = int(os.environ.get('TWINKLE_SAMPLER_GPUS', '1'))
REWARD_GPUS = int(os.environ.get('TWINKLE_REWARD_GPUS', '1'))
BENCH_RM = os.environ.get('BENCH_RM', '0') == '1'
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS + (REWARD_GPUS if BENCH_RM else 0)

LEARNING_RATE = float(os.environ.get('TWINKLE_LEARNING_RATE', '1e-5'))
ADAPTER_NAME = os.environ.get('TWINKLE_ADAPTER_NAME', 'bench-streaming-grpo')
REWARD_NUM_WORKERS = int(os.environ.get('TWINKLE_REWARD_NUM_WORKERS', '2'))
REWARD_DELAY_MS = float(os.environ.get('TWINKLE_REWARD_DELAY_MS', '0'))
# Absolute default: repo-root/results, independent of the working directory.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
BENCH_OUT_DIR = os.environ.get('BENCH_OUT_DIR', os.path.join(_REPO_ROOT, 'results'))
TIMELINE_PATH = os.path.join(BENCH_OUT_DIR, 'bench_timeline.jsonl')
SUMMARY_PATH = os.path.join(BENCH_OUT_DIR, 'bench_summary.csv')
LEVEL1_TEXTS_PATH = os.path.join(BENCH_OUT_DIR, 'level1_texts.json')
BENCH_RUNS = os.environ.get('BENCH_RUNS', 'all')
# 'engine' = one remote call (vLLMSampler.sample_sequences_to_queue): all
# sequences scheduled concurrently in the sampler actor (vLLM keeps batching),
# completion events stream back through a Ray queue.
# 'legacy' = N concurrent per-input remote calls (serialized by the actor).
BENCH_PATH_B_STREAM = os.environ.get('BENCH_PATH_B_STREAM', 'engine')
# RM 模式：奖励提交粒度（whole=整批 / mini=每 K 条一批 / per-item=逐条）。
BENCH_SUBMIT_GRANULARITY = os.environ.get('BENCH_SUBMIT_GRANULARITY', 'per-item')
MINI_SUBMIT_SIZE = int(os.environ.get('BENCH_MINI_BATCH_SIZE', '2'))

BASE_STEPS = 6
SWEEP_STEPS = 4

# Backlog must cover one streaming step's in-flight per-sequence handles.
MAX_TOTAL_PER_STEP = 4 * 8  # batch=4, gen=8
REWARD_BACKLOG = MAX_TOTAL_PER_STEP + 2


def build_runs() -> List[Dict[str, Any]]:
    """Run matrix: one base config plus single-variable sweep points.

    ``BENCH_RUNS`` accepts 'all', 'smoke', or a comma-separated run-name list
    (e.g. 'base,gen8,d1000') for targeted reruns.
    """
    runs = [
        dict(name='base', batch=4, gen=4, max_tokens=1024, delay_ms=0, steps=BASE_STEPS),
        dict(name='gen2', batch=4, gen=2, max_tokens=1024, delay_ms=0, steps=SWEEP_STEPS),
        dict(name='gen8', batch=4, gen=8, max_tokens=1024, delay_ms=0, steps=SWEEP_STEPS),
        dict(name='tok512', batch=4, gen=4, max_tokens=512, delay_ms=0, steps=SWEEP_STEPS),
        dict(name='tok2048', batch=4, gen=4, max_tokens=2048, delay_ms=0, steps=SWEEP_STEPS),
        dict(name='d200', batch=4, gen=4, max_tokens=1024, delay_ms=200, steps=SWEEP_STEPS),
        dict(name='d1000', batch=4, gen=4, max_tokens=1024, delay_ms=1000, steps=SWEEP_STEPS),
    ]
    if BENCH_RM:
        # RM 场景专用矩阵：路径 × 提交粒度（小规模，每步 4 条序列）。
        base = dict(batch=2, gen=2, max_tokens=BENCH_RM_MAX_TOKENS, delay_ms=0, steps=3)
        return [
            dict(name='rm-whole', path='A', granularity='whole', **base),
            dict(name='rm-b-whole', path='B', granularity='whole', **base),
            dict(name='rm-b-mini', path='B', granularity='mini', **base),
            dict(name='rm-b-per', path='B', granularity='per-item', **base),
        ]
    if BENCH_RUNS == 'smoke':
        return [dict(name='smoke', batch=1, gen=2, max_tokens=64, delay_ms=0, steps=1)]
    if BENCH_RUNS == 'all':
        return runs
    names = [r['name'] for r in runs]
    selected = [name.strip() for name in BENCH_RUNS.split(',') if name.strip()]
    unknown = [name for name in selected if name not in names]
    if unknown or not selected:
        raise ValueError(
            f"BENCH_RUNS must be 'all', 'smoke', or a comma-separated subset of "
            f"{names}; got {BENCH_RUNS!r}")
    return [r for r in runs if r['name'] in selected]


# ---------------------------------------------------------------------------
# Timeline instrumentation (thread-safe; reward workers append from threads)
# ---------------------------------------------------------------------------
class Timeline:
    """Monotonic-clock event log shared by the main thread and reward workers."""

    def __init__(self) -> None:
        self._events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def record(self, run: str, path: str, kind: str, step: Optional[int] = None,
               idx: Optional[int] = None, item_id: Optional[str] = None,
               value: Optional[float] = None) -> None:
        event = {
            'ts': time.perf_counter(), 'run': run, 'path': path, 'kind': kind,
            'step': step, 'idx': idx, 'item_id': item_id, 'value': value,
        }
        with self._lock:
            self._events.append(event)

    def finalize(self) -> None:
        """Patch reward-worker events (which don't know run/path) from item_id.

        item_id format: ``{run}/{path}/step-{step}/sample-{idx}``.
        """
        for event in self._events:
            if event['run'] == '__run__' and event['item_id']:
                parts = event['item_id'].split('/')
                if len(parts) >= 4:
                    event['run'] = parts[0]
                    event['path'] = parts[1]
                    event['step'] = int(parts[2].split('-')[1]) if parts[2].startswith('step-') else None
                    event['idx'] = int(parts[3].split('-')[1]) if parts[3].startswith('sample-') else None

    def dump(self, path: str) -> None:
        self.finalize()
        with open(path, 'w', encoding='utf-8') as fh:
            for event in self._events:
                fh.write(json.dumps(event, ensure_ascii=False) + '\n')


TIMELINE = Timeline()

# Current run's reward delay (ms), read by gsm8k_score running in worker threads.
_current_delay_ms: float = REWARD_DELAY_MS
_current_delay_lock = threading.Lock()


def set_reward_delay(delay_ms: float) -> None:
    global _current_delay_ms
    with _current_delay_lock:
        _current_delay_ms = delay_ms


def get_reward_delay() -> float:
    with _current_delay_lock:
        return _current_delay_ms


# ---------------------------------------------------------------------------
# Dataset + reward scoring (same contract as minimal_grpo_local.py)
# ---------------------------------------------------------------------------
def create_dataset() -> Dataset:
    """GSM8K dataset adapted from its message-format rows.

    Local jsonl files (``TWINKLE_DATASET_ID`` pointing at a file) are loaded
    through the in-memory path (``DatasetMeta(data=rows)``) so loading never
    touches the modelscope hub loader — guarantees offline operation even on
    hosts where modelscope's loader is unavailable or network-restricted.

    Otherwise (``ms://...``) rows come from the hub and are already
    ``messages`` (user question + assistant reference solution ending with
    ``#### <n>``). ``_MessagesGSM8KPreprocessor`` extracts the ground truth
    into ``user_data``, drops the reference message so it never leaks into
    the sampled prompt, and prepends the ``\\boxed{}`` system prompt. Rows
    stay trajectories (not ``encode()``-d).
    """
    dataset = Dataset(DatasetMeta(DATASET_ID, subset_name='main', split='train'))
    dataset.map(_MessagesGSM8KPreprocessor(system='Put the final answer within \\boxed{}.'))
    dataset.set_template(TEMPLATE_CLS, model_id=MODEL_ID, max_length=400)
    return dataset


class _MessagesGSM8KPreprocessor(Preprocessor):
    """Adapt message-format GSM8K rows to (messages, user_data) trajectories."""

    def __init__(self, system: str = None):
        self.system = system

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        return self.map_row_to_col(rows)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        messages = list(row.get('messages') or [])
        ground_truth = ''
        kept = []
        for msg in messages:
            if msg.get('role') == 'assistant':
                # Reference solution embedded in messages; use for GT, drop
                # from the prompt so it never leaks into sampling.
                if not ground_truth:
                    ground_truth = _extract_ground_truth_from_answer(msg.get('content', ''))
            else:
                kept.append(msg)
        if not ground_truth:
            # modelscope rows carry the reference solution in a separate
            # 'gold_answer' column instead of an assistant message.
            ground_truth = _extract_ground_truth_from_answer(row.get('gold_answer', ''))
        if self.system:
            kept = [{'role': 'system', 'content': self.system}] + kept
        return {'messages': kept, 'user_data': [('ground_truth', ground_truth)]}


def _extract_predicted_answer(completion: str) -> str:
    """Extract the model's answer: \\boxed{} > #### > last number.

    ``GSM8KAccuracyReward.extract_answer`` only recognizes \\boxed{} and ####;
    models without the boxed instruction emit plain text like
    ``**Final Answer:** 72 clips.``, so fall back to the last number.
    """
    predicted = GSM8KAccuracyReward.extract_answer(completion)
    if predicted:
        return predicted
    tail = completion[-200:] if len(completion) > 200 else completion
    numbers = re.findall(r'-?\d+(?:[.,]\d+)?', tail)
    return numbers[-1].replace(',', '') if numbers else ''


def _numerically_equal(predicted: str, ground_truth: str) -> bool:
    try:
        return abs(float(predicted) - float(str(ground_truth).strip())) < 1e-5
    except (ValueError, OverflowError):
        return predicted == str(ground_truth).strip()


def _score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict):
    """Pure scalar reward: 1.0 iff the extracted answer matches ground truth."""
    prompt = extra_info.get('prompt') if isinstance(extra_info, dict) else {}
    prompt = dict(prompt) if isinstance(prompt, dict) else {}
    messages = list(prompt.get('messages') or [])
    messages.append({'role': 'assistant', 'content': solution_str})
    trajectory = {**prompt, 'messages': messages}
    user_data = list(trajectory.get('user_data') or [])
    if user_data_get(user_data, 'ground_truth', None) in (None, ''):
        user_data.append(('ground_truth', str(ground_truth)))
    trajectory['user_data'] = user_data
    reward = GSM8KAccuracyReward()([trajectory])[0]
    if reward == 0.0 and str(ground_truth).strip():
        predicted = _extract_predicted_answer(solution_str)
        if predicted and _numerically_equal(predicted, str(ground_truth).strip()):
            reward = 1.0
    return reward, {'data_source': data_source}


def gsm8k_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict):
    """reward_loop worker entry: optional artificial delay + timeline events.

    Workers don't know run/path; Timeline.finalize() patches them from item_id.
    """
    item_id = extra_info.get('_bench_item_id') if isinstance(extra_info, dict) else None
    TIMELINE.record('__run__', '__path__', 'reward_start', item_id=item_id)
    delay_s = get_reward_delay() / 1000.0
    if delay_s > 0:
        time.sleep(delay_s)
    score, meta = _score(data_source, solution_str, ground_truth, extra_info)
    TIMELINE.record('__run__', '__path__', 'reward_end', item_id=item_id)
    return score, meta


# ---------------------------------------------------------------------------
# RM（奖励模型）模式：生成式 judge 打分（独立 GPU 批级调用）
# ---------------------------------------------------------------------------
# 边界验证（长尾生成 + 昂贵 judge）：扩大判词长度让 judge 先推理再判定，
# 单条延迟升至秒级；RM 采样 max_tokens 由 BENCH_RM_MAX_TOKENS 控制。
_JUDGE_MAX_TOKENS = int(os.environ.get('TWINKLE_REWARD_JUDGE_MAX_TOKENS', '8'))
BENCH_RM_MAX_TOKENS = int(os.environ.get('BENCH_RM_MAX_TOKENS', '512'))
_JUDGE_SYSTEM = ('You are a strict math answer verifier. Reason briefly about '
                 'whether the model answer matches the ground truth, then end '
                 'your response with exactly one word on the last line: Correct '
                 'or Incorrect.')
_JUDGE_PARAMS = SamplingParams(max_tokens=_JUDGE_MAX_TOKENS, num_samples=1,
                               temperature=0.0)


_JUDGE_MAX_SOLUTION_CHARS = 3000


def judge_prompt_for(item: RewardItem) -> Dict[str, Any]:
    """Verification trajectory (dict with ``messages``): question + model
    answer + ground truth — the shape ``sampler.sample`` expects.

    Long model answers are truncated to their tail (the final answer region),
    so judge prompts stay within the judge engine's context window.
    """
    prompt = item.extra_info.get('prompt') if isinstance(item.extra_info, dict) else {}
    question = ''
    for msg in reversed(list((prompt or {}).get('messages') or [])):
        if msg.get('role') == 'user':
            question = msg.get('content', '')
            break
    solution = item.solution_str or ''
    if len(solution) > _JUDGE_MAX_SOLUTION_CHARS:
        solution = '...[truncated, showing tail]...\n' + solution[-_JUDGE_MAX_SOLUTION_CHARS:]
    return {'messages': [
        {'role': 'system', 'content': _JUDGE_SYSTEM},
        {'role': 'user', 'content':
            f'Question: {question}\n\nModel answer: {solution}\n\n'
            f'Ground truth: {item.ground_truth}\n\nIs the model answer correct?'},
    ]}


def parse_judge_verdict(text: str) -> Optional[float]:
    if re.search(r'\bcorrect\b', text, re.IGNORECASE):
        return 1.0
    if re.search(r'\bincorrect\b', text, re.IGNORECASE):
        return 0.0
    return None


@register('batch_judge')
class BatchJudgeRewardManager(RewardManagerBase):
    """Reward manager that scores a whole chunk with one judge engine call.

    Items in a chunk are packed into one ``judge_sampler.sample(prompts)``
    call so the judge's vLLM batches them (real RM batching), then each
    verdict is parsed per item. Chunk size = reward submission granularity
    (whole / mini / per-item), so the granularity experiment controls exactly
    how many trajectories the judge sees per engine call.
    """

    def __init__(self, compute_score=None, judge_sampler=None, **kwargs):
        super().__init__(compute_score=compute_score, **kwargs)
        self.judge_sampler = judge_sampler

    async def run_batch(self, items):
        if not items:
            return []
        prompts = [judge_prompt_for(item) for item in items]
        responses = await asyncio.to_thread(
            self.judge_sampler.sample, prompts, _JUDGE_PARAMS, '')
        results = []
        for item, response in zip(items, responses):
            text = response.sequences[0].decoded or ''
            score = parse_judge_verdict(text)
            if score is None:
                score = 0.0
                logger.warning(f'[judge] unparsed verdict {text[:80]!r} for {item.item_id}')
            results.append(RewardResult(
                item.item_id, score, {'judge_verdict': text.strip()[:60]}))
        return results


_ANS_RE = re.compile(r'####\s*(-?\d+(?:[.,]\d+)?)')


def _extract_ground_truth_from_answer(answer: Any) -> str:
    """Extract the final numeric answer from a raw GSM8K row's ``answer`` field.

    Raw rows (no GSM8KProcessor) carry the full solution in ``answer`` ending
    with ``#### <number>`` (huggingface-style); fall back to the last number.
    """
    if not answer:
        return ''
    text = str(answer)
    match = _ANS_RE.search(text)
    if match:
        return match.group(1).replace(',', '')
    numbers = re.findall(r'-?\d+(?:[.,]\d+)?', text)
    return numbers[-1].replace(',', '') if numbers else ''


def _ground_truth(prompt: Dict[str, Any]) -> str:
    gt = user_data_get(prompt.get('user_data'), 'ground_truth', '')
    if gt in (None, ''):
        gt = _extract_ground_truth_from_answer(prompt.get('answer', ''))
    return str(gt)


def make_reward_items(run: str, path: str, prompts: List[Dict[str, Any]],
                      sequences: List[Any], step: int, num_generations: int) -> List[RewardItem]:
    """One RewardItem per completed sequence, in prompt-major index order.

    ``sequences`` is a list of (index, SampledSequence) aligned to the expanded
    prompt copies (index // num_generations -> prompt).
    """
    items: List[RewardItem] = []
    for idx, sequence in sequences:
        prompt = prompts[idx // num_generations]
        item_id = f'{run}/{path}/step-{step}/sample-{idx}'
        items.append(RewardItem(
            item_id=item_id,
            data_source='gsm8k',
            solution_str=sequence.decoded or '',
            ground_truth=_ground_truth(prompt),
            extra_info={'prompt': prompt, '_bench_item_id': item_id},
        ))
    return items


# ---------------------------------------------------------------------------
# Sampling paths
# ---------------------------------------------------------------------------
def _expand(prompts: List[Dict[str, Any]], num_generations: int) -> List[Dict[str, Any]]:
    return [prompt for prompt in prompts for _ in range(num_generations)]


def _seq_tokens(sequence) -> List[int]:
    return list(sequence.tokens)


def _seq_logprobs(sequence) -> List[float]:
    return [logprob[0][1] for logprob in sequence.logprobs]


def _seq_input_feature(sequence):
    return sequence.new_input_feature


def _collect_payload(sequence) -> tuple:
    return (_seq_input_feature(sequence), _seq_logprobs(sequence), len(sequence.tokens))


def sample_batch(sampler, prompts: List[Dict[str, Any]], params: SamplingParams,
                 num_generations: int) -> List[Any]:
    """Path A: single batch call; returns one SampledSequence per copy."""
    responses = sampler.sample(_expand(prompts, num_generations), params, ADAPTER_NAME)
    return [resp.sequences[0] for resp in responses]


def sample_stream(sampler, prompts: List[Dict[str, Any]], params: SamplingParams,
                  num_generations: int):
    """Path B: per-sequence completion events with engine-level concurrency.

    'engine' (default): one remote call
    (``vLLMSampler.sample_sequences_to_queue``) schedules ALL sequences in the
    sampler actor's event loop — vLLM keeps batching the whole batch, so
    t_sample stays at the batch level — and streams ``(index, SampleResponse)``
    events back through a Ray queue in completion order (local-mode counterpart
    of the server's ``stream_sample_to_data_plane``).

    'legacy': N concurrent per-input remote calls, serialized by the actor
    (~N x single-sequence time; kept for comparison).
    """
    expanded = _expand(prompts, num_generations)
    if BENCH_PATH_B_STREAM == 'legacy':
        with ThreadPoolExecutor(max_workers=len(expanded)) as pool:
            futures = {pool.submit(sampler.sample, [traj], params, ADAPTER_NAME): idx
                       for idx, traj in enumerate(expanded)}
            for future in as_completed(futures):
                idx = futures[future]
                response = future.result()[0]
                yield idx, response.sequences[0]
        return
    if BENCH_PATH_B_STREAM != 'engine':
        raise ValueError(f"BENCH_PATH_B_STREAM must be 'engine' or 'legacy', got {BENCH_PATH_B_STREAM!r}")

    import queue as stdlib_queue
    import ray
    from ray.util.queue import Queue
    queue = Queue()
    with ThreadPoolExecutor(max_workers=1) as pool:
        remote = pool.submit(
            sampler.sample_sequences_to_queue, queue, expanded, params, ADAPTER_NAME)
        try:
            expected = len(expanded)
            received = 0
            while True:
                try:
                    idx, response = queue.get(timeout=1.0)
                except stdlib_queue.Empty:
                    if remote.done():
                        # Engine side finished (or failed) without a sentinel.
                        remote.result()  # re-raises engine-side errors
                        raise RuntimeError(
                            f'stream ended early: {received}/{expected} events, no sentinel')
                    continue
                if idx is None:
                    break
                received += 1
                yield idx, response.sequences[0]
        finally:
            # Drain complete; propagate any engine-side error.
            remote.result()


# ---------------------------------------------------------------------------
# Training step (same as minimal_grpo_local.py)
# ---------------------------------------------------------------------------
def train_batch(*, model, advantage_fn, metrics, input_data, old_logps,
                completion_lengths, rewards, num_generations, micro_batch_size=2) -> None:
    advantages = advantage_fn(rewards, num_generations=num_generations, scale='group').tolist()
    metrics.accumulate(completion_lengths=completion_lengths, rewards={'total': rewards})
    total = len(input_data)
    for mb_start in range(0, total, micro_batch_size):
        mb_end = min(mb_start + micro_batch_size, total)
        model.forward_backward(
            inputs=input_data[mb_start:mb_end],
            old_logps=old_logps[mb_start:mb_end],
            advantages=advantages[mb_start:mb_end],
            micro_batch_size=micro_batch_size,
        )
        model.clip_grad_and_step()
    log_dict = metrics.calculate()
    log_dict.update(model.calculate_metric(is_training=True))
    return advantages, log_dict


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------
def check_semantics(run: str, path: str, step: int, prompts, items, results,
                    rewards, advantages, num_generations) -> Dict[str, Any]:
    """Level-2 structural checks; returns a dict of pass/fail booleans."""
    total = len(prompts) * num_generations
    checks: Dict[str, Any] = {}
    checks['item_count'] = len(items) == total
    ids = [item.item_id for item in items]
    checks['no_duplicate_ids'] = len(set(ids)) == len(ids)
    checks['results_aligned'] = len(results) == len(items) and {r.item_id for r in results} == set(ids)
    checks['reward_count'] = len(rewards) == total
    checks['advantage_count'] = len(advantages) == total
    # Group-mean-zero: GRPOAdvantage normalizes each group of num_generations.
    groups_ok = True
    for g in range(len(prompts)):
        group = advantages[g * num_generations:(g + 1) * num_generations]
        if abs(sum(group)) > 1e-4:
            groups_ok = False
    checks['advantage_group_mean_zero'] = groups_ok
    # Reward-function determinism: re-score the first item without the pipeline.
    if items:
        first = items[0]
        direct, _ = _score('gsm8k', first.solution_str, first.ground_truth, first.extra_info)
        checks['reward_deterministic'] = abs(direct - results[0].reward_score) < 1e-9
    ok = all(checks.values())
    checks['all_ok'] = ok
    TIMELINE.record(run, path, 'semantic_checks', step=step,
                    value=1.0 if ok else 0.0, item_id=json.dumps(checks))
    if not ok:
        logger.warning(f'[semantic checks failed] run={run} path={path} step={step}: {checks}')
    return checks


def run_level1(sampler, pipeline, prompts, num_generations, max_tokens, run='level1'):
    """Determinism: greedy + fixed seed, both paths on identical inputs, no
    training in between. Returns a summary dict of per-check results."""
    params = SamplingParams(max_tokens=max_tokens, num_samples=1, logprobs=1,
                            temperature=0.0, seed=0)
    checks: Dict[str, Any] = {'run': run}
    path_a = list(sample_batch(sampler, prompts, params, num_generations))
    # Streaming yields in completion order; sort by input index for comparison.
    streamed = sorted(sample_stream(sampler, prompts, params, num_generations), key=lambda p: p[0])
    path_b = [seq for _, seq in streamed]
    checks['sample_count_equal'] = len(path_a) == len(path_b) == len(prompts) * num_generations
    n = min(len(path_a), len(path_b))

    max_token_diff = 0
    token_mismatches = 0
    max_logprob_diff = 0.0
    logprob_mismatches = 0
    decoded_mismatches = 0
    text_pairs = []
    for i in range(n):
        ta, tb = _seq_tokens(path_a[i]), _seq_tokens(path_b[i])
        first_diff = None
        if ta != tb:
            token_mismatches += 1
            shorter = min(len(ta), len(tb))
            for j in range(shorter):
                if ta[j] != tb[j]:
                    first_diff = j
                    break
            if first_diff is None:
                first_diff = shorter  # prefix identical, lengths differ
            max_token_diff = max(max_token_diff, abs(len(ta) - len(tb)))
        text_pairs.append({
            'idx': i,
            'tokens_identical': ta == tb,
            'first_diff_pos': first_diff,
            'len_a': len(ta),
            'len_b': len(tb),
            'a_text': path_a[i].decoded or '',
            'b_text': path_b[i].decoded or '',
        })
        la, lb = _seq_logprobs(path_a[i]), _seq_logprobs(path_b[i])
        if la != lb:
            logprob_mismatches += 1
            shorter = min(len(la), len(lb))
            if shorter:
                diffs = [abs(x - y) for x, y in zip(la[:shorter], lb[:shorter])]
                max_logprob_diff = max(max_logprob_diff, max(diffs))
            max_logprob_diff = max(max_logprob_diff, abs(len(la) - len(lb)))
        if (path_a[i].decoded or '') != (path_b[i].decoded or ''):
            decoded_mismatches += 1
    checks['tokens_identical'] = token_mismatches == 0
    checks['token_mismatches'] = token_mismatches
    checks['max_token_len_diff'] = max_token_diff
    checks['logprobs_identical'] = logprob_mismatches == 0
    checks['logprob_mismatches'] = logprob_mismatches
    checks['max_logprob_diff'] = max_logprob_diff
    checks['decoded_identical'] = decoded_mismatches == 0
    checks['decoded_mismatches'] = decoded_mismatches

    # Reward path must also agree deterministically through the pipeline.
    def _scored_items(sequences, path_label):
        seqs = [(i, s) for i, s in enumerate(sequences)]
        return make_reward_items(run, path_label, prompts, seqs, 0, num_generations)
    items_a = _scored_items(path_a, 'A')
    items_b = _scored_items(path_b, 'B')
    results_a = pipeline.collect(pipeline.submit(items_a))
    results_b = pipeline.collect(pipeline.submit(items_b))
    rewards_a = [r.reward_score for r in results_a]
    rewards_b = [r.reward_score for r in results_b]
    checks['reward_count_equal'] = len(rewards_a) == len(rewards_b) == n
    checks['rewards_identical'] = rewards_a == rewards_b
    if rewards_a == rewards_b and rewards_a:
        adv_a = GRPOAdvantage()(rewards_a, num_generations=num_generations, scale='group').tolist()
        adv_b = GRPOAdvantage()(rewards_b, num_generations=num_generations, scale='group').tolist()
        checks['advantages_identical'] = adv_a == adv_b
    else:
        checks['advantages_identical'] = False
    checks['all_ok'] = all(
        v is True for k, v in checks.items() if k not in ('run',) and isinstance(v, bool))
    TIMELINE.record(run, 'both', 'level1_checks', value=1.0 if checks['all_ok'] else 0.0,
                    item_id=json.dumps(checks))
    logger.info(f'[Level-1 determinism] {json.dumps(checks, ensure_ascii=False)}')

    # Dump per-pair texts so divergent pairs can be inspected by hand.
    _write_level1_texts(prompts, text_pairs, rewards_a, rewards_b, num_generations, checks)
    return checks


def _write_level1_texts(prompts, text_pairs, rewards_a, rewards_b, num_generations, checks) -> None:
    """Write A/B text pairs, rewards and first divergence position per index."""
    for i, pair in enumerate(text_pairs):
        prompt = prompts[i // num_generations]
        pair['ground_truth'] = _ground_truth(prompt)
        pair['reward_a'] = rewards_a[i] if i < len(rewards_a) else None
        pair['reward_b'] = rewards_b[i] if i < len(rewards_b) else None
    record = {
        'num_pairs': len(text_pairs),
        'num_generations': num_generations,
        'level1_checks': {k: v for k, v in checks.items() if k != 'run'},
        'pairs': text_pairs,
    }
    with open(LEVEL1_TEXTS_PATH, 'w', encoding='utf-8') as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    logger.info(f'[Level-1] per-pair texts written to {LEVEL1_TEXTS_PATH}')


# ---------------------------------------------------------------------------
# Per-path step loop
# ---------------------------------------------------------------------------
def run_path(run_cfg: Dict[str, Any], path: str, sampler, pipeline, model,
             advantage_fn, metrics, batches: List[List[Dict[str, Any]]],
             sync_weights) -> List[Dict[str, Any]]:
    """Run one path over the given batches; returns per-step summary rows."""
    run = run_cfg['name']
    num_generations = run_cfg['gen']
    # 提交粒度：RM 模式由 run 矩阵决定；普通模式 B 保持逐条（流式语义）。
    granularity = run_cfg.get('granularity') or (
        'per-item' if path == 'B' else 'whole')
    params = SamplingParams(max_tokens=run_cfg['max_tokens'], num_samples=1,
                            logprobs=1, temperature=1.0, top_p=0.95)
    rows: List[Dict[str, Any]] = []
    metrics.reset()
    for step, prompts in enumerate(batches):
        TIMELINE.record(run, path, 'step_start', step=step)
        t_step0 = time.perf_counter()
        sync_weights()
        sampler.reset_prefix_cache()

        if path == 'A':
            t0 = time.perf_counter()
            TIMELINE.record(run, path, 'sample_start', step=step)
            sequences = sample_batch(sampler, prompts, params, num_generations)
            t_sample = time.perf_counter() - t0
            TIMELINE.record(run, path, 'sample_end', step=step, value=t_sample)
            seqs = [(i, s) for i, s in enumerate(sequences)]
        else:
            # Streaming: submit rewards as sequences complete; the submission
            # granularity controls how many rewards ride in one handle
            # (per-item = immediate, mini = every K, whole = after sampling).
            t0 = time.perf_counter()
            TIMELINE.record(run, path, 'sample_start', step=step)
            seqs, items, handles = [], [], []
            pending: List[RewardItem] = []
            submit_threshold = 1 if granularity == 'per-item' else (
                MINI_SUBMIT_SIZE if granularity == 'mini' else len(prompts) * num_generations)

            def _flush_pending():
                if not pending:
                    return
                if len(pending) < submit_threshold:
                    return
                _items, pending[:] = pending[:], []
                handles.append(pipeline.submit(_items))

            t_submit = 0.0
            for idx, sequence in sample_stream(sampler, prompts, params, num_generations):
                TIMELINE.record(run, path, 'sample_done', step=step, idx=idx)
                seqs.append((idx, sequence))
                item = make_reward_items(run, path, prompts, [(idx, sequence)],
                                         step, num_generations)[0]
                items.append(item)
                pending.append(item)
                t1 = time.perf_counter()
                _flush_pending()
                t_submit += time.perf_counter() - t1
            if pending:
                handles.append(pipeline.submit(pending))
            t_sample = time.perf_counter() - t0
            TIMELINE.record(run, path, 'sample_end', step=step, value=t_sample)

        # Submit remaining rewards (A: one batch call after sampling).
        if path == 'A':
            t0 = time.perf_counter()
            items = make_reward_items(run, path, prompts, seqs, step, num_generations)
            TIMELINE.record(run, path, 'submit_start', step=step)
            handle = pipeline.submit(items)
            TIMELINE.record(run, path, 'submit_end', step=step)
            t_submit = time.perf_counter() - t0
            handles = [handle]

        # Collect all rewards for this step (single-buffer schedule).
        t0 = time.perf_counter()
        TIMELINE.record(run, path, 'collect_start', step=step)
        results = []
        for handle in handles:
            results.extend(pipeline.collect(handle))
        t_collect = time.perf_counter() - t0
        TIMELINE.record(run, path, 'collect_end', step=step, value=t_collect)
        by_id = {r.item_id: r for r in results}
        rewards = [by_id[item.item_id].reward_score for item in items]

        # Train.
        t0 = time.perf_counter()
        TIMELINE.record(run, path, 'train_start', step=step)
        payloads = [_collect_payload(seq) for _, seq in seqs]
        advantages, log_dict = train_batch(
            model=model, advantage_fn=advantage_fn, metrics=metrics,
            input_data=[p[0] for p in payloads], old_logps=[p[1] for p in payloads],
            completion_lengths=[p[2] for p in payloads], rewards=rewards,
            num_generations=num_generations,
        )
        t_train = time.perf_counter() - t0
        TIMELINE.record(run, path, 'train_end', step=step, value=t_train)

        checks = check_semantics(run, path, step, prompts, items, results, rewards,
                                 advantages, num_generations)

        t_total = time.perf_counter() - t_step0
        TIMELINE.record(run, path, 'step_end', step=step, value=t_total)
        row = dict(run=run, path=path, step=step, batch=len(prompts),
                   gen=num_generations, max_tokens=run_cfg['max_tokens'],
                   delay_ms=run_cfg['delay_ms'],
                   t_sample=t_sample, t_submit=t_submit, t_collect=t_collect,
                   t_train=t_train, t_total=t_total,
                   seq_per_s=(len(prompts) * num_generations) / t_total,
                   semantic_ok=checks['all_ok'])
        rows.append(row)
        logger.info(f"[{run}/{path}] step={step} t_sample={t_sample:.2f}s "
                    f"t_submit={t_submit:.2f}s t_collect={t_collect:.2f}s "
                    f"t_train={t_train:.2f}s t_total={t_total:.2f}s "
                    f"reward_mean={sum(rewards) / len(rewards):.4f} {log_dict}")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    runs = build_runs()
    os.makedirs(BENCH_OUT_DIR, exist_ok=True)

    _sampler_start = MODEL_GPUS
    _reward_start = MODEL_GPUS + SAMPLER_GPUS
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(_sampler_start, _reward_start)),
                    device_type='GPU'),
    ]
    if BENCH_RM:
        device_groups.append(DeviceGroup(
            name='reward', ranks=list(range(_reward_start, NUM_GPUS)), device_type='GPU'))
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups,
                       lazy_collect=False)

    lora_config = LoraConfig(
        target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
            'in_proj_qkv', 'in_proj_z', 'in_proj_a', 'in_proj_b', 'out_proj',
        ],
        r=32, lora_alpha=64, lora_dropout=0.05,
    )
    model = TransformersModel(model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model')
    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=1)
    model.set_optimizer('AdamW', lr=LEARNING_RATE)
    model.set_lr_scheduler('CosineAnnealingLR', T_max=200, eta_min=0)
    model.set_loss('GRPOLoss', epsilon=0.2)
    model.set_processor(InputProcessor)
    model.set_template(TEMPLATE_CLS, model_id=MODEL_ID)

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 4496,
            'max_lora_rank': 32,
            'enable_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(TEMPLATE_CLS, model_id=MODEL_ID)

    judge_sampler = None
    if BENCH_RM:
        # 独立 GPU 上的 judge 引擎（冻结权重，不参与训练更新）。
        reward_mesh = DeviceMesh.from_sizes(world_size=REWARD_GPUS, dp_size=REWARD_GPUS)
        judge_sampler = vLLMSampler(
            model_id=REWARD_MODEL_ID,
            engine_args={
                'gpu_memory_utilization': 0.8,
                'max_model_len': 8192,
            },
            device_mesh=reward_mesh,
            remote_group='reward',
        )
        judge_sampler.set_template(REWARD_TEMPLATE_CLS, model_id=REWARD_MODEL_ID)

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)
    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()
    if BENCH_RM:
        pipeline = AsyncRewardPipeline(
            num_workers=REWARD_NUM_WORKERS,
            mode='async',
            backlog=REWARD_BACKLOG,
            worker_kwargs={
                'compute_score': gsm8k_score,
                'manager_name': 'batch_judge',
                'reward_kwargs': {'judge_sampler': judge_sampler},
            },
        )
    else:
        pipeline = AsyncRewardPipeline(
            num_workers=REWARD_NUM_WORKERS,
            mode='async',
            backlog=REWARD_BACKLOG,
            worker_kwargs={'compute_score': gsm8k_score},
        )

    def sync_weights():
        ckpt_manager.sync_weights(merge_and_sync=False)

    logger.info(get_device_placement())
    if BENCH_RM:
        logger.info(f'[bench] ** RM MODE ENABLED ** judge={REWARD_MODEL_ID} '
                    f'reward_gpus={REWARD_GPUS} granularity-matrix='
                    f'{BENCH_SUBMIT_GRANULARITY}')
    else:
        logger.info(f'[bench] RM mode disabled (add BENCH_RM=1 for reward-model runs)')
    logger.info(f'[bench] outputs -> {BENCH_OUT_DIR}')
    summary_rows: List[Dict[str, Any]] = []
    try:
        if not BENCH_RM:
            first = runs[0]
            base_batch = max(1, first['batch'])
            # Level-1 determinism check on one batch, before any training.
            # instance_id keeps Ray actor names unique: remote_class derives the
            # actor name from the caller's source line, so two DataLoaders created
            # on the same line would collide with ActorAlreadyExistsError.
            dataloader = DataLoader(
                dataset=create_dataset, batch_size=base_batch, min_batch_size=base_batch,
                device_mesh=model_mesh, remote_group='model', instance_id='bench-level1',
            )
            one_batch = next(iter(dataloader))
            prompts = list(one_batch) if isinstance(one_batch, list) else [one_batch]
            # Diagnostic: confirm the row schema and that ground truths resolve.
            row0 = prompts[0]
            preview = json.dumps(row0, ensure_ascii=False, default=str)
            logger.info(f'[level1 row0] keys={sorted(row0.keys())} '
                        f'user_data={row0.get("user_data")!r} preview={preview[:300]!r}')
            for i, p in enumerate(prompts[:3]):
                logger.info(f'[level1 prompt {i}] gt={_ground_truth(p)!r} '
                            f'answer={str(p.get("answer", ""))[:60]!r}')
            sync_weights()
            sampler.reset_prefix_cache()
            level1 = run_level1(sampler, pipeline, prompts, first['gen'], first['max_tokens'])
            summary_rows.append(dict(run='level1', path='both', step=-1, batch=len(prompts),
                                     gen=first['gen'], max_tokens=first['max_tokens'],
                                     delay_ms=0, t_sample=0.0, t_submit=0.0, t_collect=0.0,
                                     t_train=0.0, t_total=0.0, seq_per_s=0.0,
                                     semantic_ok=level1['all_ok']))
            # Persist level-1 results even if a sweep run fails afterwards.
            flush_outputs(summary_rows)

        for run_cfg in runs:
            run = run_cfg['name']
            set_reward_delay(run_cfg['delay_ms'])
            dataloader = DataLoader(
                dataset=create_dataset, batch_size=run_cfg['batch'],
                min_batch_size=run_cfg['batch'],
                device_mesh=model_mesh, remote_group='model',
                instance_id=f'bench-{run}',
            )
            batches = list(itertools.islice(iter(dataloader), run_cfg['steps']))
            batches = [list(b) if isinstance(b, list) else [b] for b in batches]
            logger.info(f'[{run}] materialized {len(batches)} batches x {run_cfg["batch"]} prompts '
                        f'(gen={run_cfg["gen"]}, max_tokens={run_cfg["max_tokens"]}, '
                        f'delay={run_cfg["delay_ms"]}ms, granularity={run_cfg.get("granularity", "-")})')
            paths = [run_cfg['path']] if run_cfg.get('path') else ('A', 'B')
            for path in paths:
                summary_rows.extend(run_path(
                    run_cfg, path, sampler, pipeline, model, advantage_fn, metrics,
                    batches, sync_weights))
            # Persist after every run so a crash keeps all completed runs.
            flush_outputs(summary_rows)
    except BaseException:
        logger.exception('[bench] run failed; flushing partial results before re-raising')
        flush_outputs(summary_rows)
        raise
    finally:
        pipeline.close()

    logger.info(f'[bench] final outputs: {TIMELINE_PATH}, {SUMMARY_PATH}')

    # Compact console summary: per-run-path means over steps.
    logger.info('=== benchmark summary (per-run path means) ===')
    means: Dict[str, Dict[str, float]] = {}
    for row in summary_rows:
        if row['step'] < 0:
            continue
        key = f"{row['run']}/{row['path']}"
        bucket = means.setdefault(key, {k: 0.0 for k in
                                        ('t_sample', 't_submit', 't_collect', 't_train',
                                         't_total', 'seq_per_s', 'reward_head_start',
                                         'reward_tail_after_sample', 'n')})
        for k in ('t_sample', 't_submit', 't_collect', 't_train', 't_total',
                  'seq_per_s', 'reward_head_start', 'reward_tail_after_sample'):
            if isinstance(row.get(k), (int, float)):
                bucket[k] += row[k]
        bucket['n'] += 1
    for key, bucket in means.items():
        n = bucket.pop('n')
        if n:
            means[key] = {k: v / n for k, v in bucket.items()}
            logger.info(f"{key}: " + ' '.join(f'{k}={v:.3f}' for k, v in means[key].items()))


def flush_outputs(summary_rows: List[Dict[str, Any]]) -> None:
    """Write timeline JSONL + summary CSV with data collected so far.

    Safe to call repeatedly (after each run) and from the failure handler:
    ``Timeline.dump`` patches reward-worker events in place, and the CSV is
    fully rewritten from ``summary_rows`` each time.
    """
    TIMELINE.dump(TIMELINE_PATH)
    overlap = _aggregate_reward_overlap()
    with open(SUMMARY_PATH, 'w', newline='', encoding='utf-8') as fh:
        fieldnames = ['run', 'path', 'step', 'batch', 'gen', 'max_tokens', 'delay_ms',
                      't_sample', 't_submit', 't_collect', 't_train', 't_total',
                      'seq_per_s', 'reward_head_start', 'reward_tail_after_sample',
                      'semantic_ok']
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            key = (row['run'], row['path'], row['step'])
            if key in overlap:
                row['reward_head_start'] = overlap[key][0]
                row['reward_tail_after_sample'] = overlap[key][1]
            else:
                row['reward_head_start'] = ''
                row['reward_tail_after_sample'] = ''
            writer.writerow(row)
    logger.info(f'[bench] flushed {len(summary_rows)} summary rows, '
                f'{len(TIMELINE._events)} timeline events -> {BENCH_OUT_DIR}')


def _aggregate_reward_overlap() -> Dict[tuple, tuple]:
    """Per (run, path, step) -> (reward_head_start, reward_tail_after_sample).

    reward_head_start: time from sampling start to the first reward computation.
    reward_tail_after_sample: time from sampling end to the last reward ready.
    Both in seconds; for streaming both shrink as rewards overlap with sampling.
    """
    starts: Dict[tuple, float] = {}
    ends: Dict[tuple, float] = {}
    rewards: Dict[tuple, List[float]] = {}
    for event in TIMELINE._events:
        key = (event['run'], event['path'], event['step'])
        if key[0] == '__run__' or key[0] == 'level1':
            continue
        if event['kind'] == 'sample_start':
            starts.setdefault(key, event['ts'])
        elif event['kind'] == 'sample_end':
            ends.setdefault(key, event['ts'])
        elif event['kind'] == 'reward_start':
            rewards.setdefault(key, []).append(event['ts'])
        elif event['kind'] == 'reward_end':
            rewards.setdefault(key, []).append(event['ts'])
    result: Dict[tuple, tuple] = {}
    for key in set(starts) & set(rewards):
        r_ts = sorted(rewards[key])
        head = (r_ts[0] - starts[key]) if len(r_ts) >= 2 else 0.0
        tail = (r_ts[-1] - ends.get(key, r_ts[-1])) if len(r_ts) >= 2 else 0.0
        result[key] = (head, tail)
    return result


if __name__ == '__main__':
    main()
