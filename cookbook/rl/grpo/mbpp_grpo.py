"""GRPO training script for MBPP (code generation with assert-verified reward).

Same structure as kodcode_grpo.py, but MBPP's tests are bare asserts that call
the function by name (``assert min_cost(...) == 8``), so the generated code,
``test_setup_code`` and the asserts are concatenated into a single file and
executed -- no ``from solution import`` layout is needed. That judging path was
checked against all 974 reference solutions and passes 974/974.

The problem statement does not name the function, and the asserts do, so the
asserts are shown in the prompt (the standard MBPP setup used by OpenCompass /
EvalPlus). Without them the function name is unguessable and every sample fails
for reasons unrelated to coding ability.

Measured difficulty of the full 974-problem set under Qwen3-4B (8 samples each,
see output/mbpp/measure_mbpp_difficulty.py): 21.97% all-wrong, 56.67% all-right,
21.36% mixed. Only the mixed ones carry a GRPO gradient; the full set is used
here as requested.
"""
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from peft import LoraConfig

import swanlab
import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.cli import CLI
from twinkle.data_format import Message, SamplingParams, Trajectory
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.metric import CompletionRewardMetric
from twinkle.model import TransformersModel
from twinkle.preprocessor import Preprocessor
from twinkle.processor import InputProcessor
from twinkle.reward.base import Reward
from twinkle.sampler import vLLMSampler

logger = get_logger()
args = CLI.from_args()

swanlab.init(project='twinkle')

# ========== Configuration ==========
MODEL_ID = args.model.model_id or 'ms://Qwen/Qwen3-4B'
USE_MEGATRON = args.model.strategy != 'native_fsdp'

MODEL_GPUS = args.infra.model_gpus or 4
SAMPLER_GPUS = args.infra.sampler_gpus or 4
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = args.rl.num_generations or 8
MAX_NEW_TOKENS = args.sampling.max_tokens or 2048
LEARNING_RATE = args.optimizer.learning_rate or 1e-5
MAX_STEPS = args.training.max_steps or 1000
BATCH_SIZE = args.training.batch_size or 8
MINI_BATCH_SIZE = args.training.mini_batch_size or 8
MICRO_BATCH_SIZE = args.training.micro_batch_size or 2
GRADIENT_ACCUMULATION_STEPS = args.training.gradient_accumulation_steps or 1
SAVE_STEPS = args.training.save_steps or 200
LORA_RANK = args.lora.lora_r or 16

JUDGE_WORKERS = int(os.environ.get('JUDGE_WORKERS', max(24, min(96, (os.cpu_count() or 24) // 2))))
TEST_TIMEOUT = int(os.environ.get('TEST_TIMEOUT', 30))

SYSTEM_PROMPT = ('You are an expert Python programmer. Write a complete, self-contained '
                 'solution in a single ```python code block. Do not include tests.')

_FENCE_RE = re.compile(r'```(?:python|py)?\s*\n(.*?)```', re.S)


# ========== Text handling ==========
def extract_code(text: str) -> str:
    """Take the last fenced block; fall back to the whole body when unfenced."""
    idx = (text or '').rfind('</think>')
    body = text[idx + len('</think>'):] if idx >= 0 else (text or '')
    blocks = _FENCE_RE.findall(body)
    return (blocks[-1] if blocks else body).strip()


# ========== Sandbox ==========
def run_asserts(code: str, setup: str, asserts: List[str], timeout: int = TEST_TIMEOUT) -> bool:
    """True when every assert passes.

    MBPP asserts call the function by name, so code + setup + asserts run as a
    single file. Uses start_new_session + killpg so a forking solution cannot
    leave stray processes, and caps the child at 2GB of address space.
    """
    if not code.strip():
        return False
    parts = [code]
    if (setup or '').strip():
        parts.append(setup)
    parts.extend(asserts)
    script = '\n\n'.join(parts) + '\n'
    tmp = tempfile.mkdtemp(prefix='mbpp_')
    try:
        with open(os.path.join(tmp, '_run.py'), 'w', encoding='utf-8') as f:
            f.write(script)
        env = dict(os.environ, MPLBACKEND='Agg', PYTHONHASHSEED='0', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
        env.pop('CUDA_VISIBLE_DEVICES', None)

        def _limit():
            resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))

        proc = subprocess.Popen([sys.executable, '_run.py'], cwd=tmp, env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True, preexec_fn=_limit)
        try:
            proc.communicate(timeout=timeout)
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.communicate(timeout=5)
            except Exception:
                pass
            return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ========== Reward ==========
class MbppAssertReward(Reward):
    """1.0 when the generated code satisfies every assert of its problem.

    The asserts travel per-sample through ``user_data`` (``mbpp_payload``), so
    each trajectory is judged against its own tests. Judging runs in a thread
    pool because every verdict is a separate subprocess.
    """

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        jobs: List[Tuple[int, str, Dict[str, Any]]] = []
        rewards = [0.0] * len(trajectories)
        for i, traj in enumerate(trajectories):
            payload = None
            for item in traj.get('user_data') or []:
                if item[0] == 'mbpp_payload':
                    payload = item[1]
                    break
            if payload is None:
                continue
            completion = ''
            for msg in reversed(traj.get('messages', []) or []):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '') or ''
                    break
            jobs.append((i, extract_code(completion), payload))

        if not jobs:
            return rewards
        # Same (task, code) judged once: identical completions are common.
        uniq: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for _, code, payload in jobs:
            uniq.setdefault((payload['task_id'], code), payload)
        todo = list(uniq)
        with ThreadPoolExecutor(max_workers=max(1, min(JUDGE_WORKERS, len(todo)))) as ex:
            verdicts = dict(zip(todo, ex.map(
                lambda k: run_asserts(k[1], uniq[k]['setup'], uniq[k]['asserts']), todo)))
        for i, code, payload in jobs:
            rewards[i] = 1.0 if verdicts.get((payload['task_id'], code)) else 0.0
        return rewards


class MbppFormatReward(Reward):
    """1.0 when the completion contains a parseable python code block."""

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for traj in trajectories:
            completion = ''
            for msg in reversed(traj.get('messages', []) or []):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '') or ''
                    break
            rewards.append(1.0 if extract_code(completion).strip() else 0.0)
        return rewards


# ========== Dataset ==========
# The problem statement never names the function while the asserts do, so the
# asserts go into the prompt (standard MBPP setup). Without them the name is
# unguessable and every sample fails regardless of coding ability.
_TEST_HINT = '\n\nYour code should satisfy these tests:\n```python\n{tests}\n```'


def _asserts(row: Dict[str, Any]) -> List[str]:
    tl = row.get('test_list')
    if tl is None:
        return []
    return list(tl) if not isinstance(tl, str) else json.loads(tl)


class MbppProcessor(Preprocessor):
    """MBPP row -> prompt-only Trajectory carrying its asserts."""

    def __init__(self, system=SYSTEM_PROMPT):
        self.system = system

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        return self.map_row_to_col(rows)

    def preprocess(self, row) -> Trajectory:
        asserts = _asserts(row)
        question = (row.get('text') or '') + _TEST_HINT.format(tests='\n'.join(asserts))
        payload = {
            'task_id': str(row.get('task_id') or ''),
            'setup': row.get('test_setup_code') or '',
            'asserts': asserts,
        }
        return Trajectory(
            messages=[
                Message(role='system', content=self.system),
                Message(role='user', content=question),
            ],
            user_data=[('mbpp_payload', payload)],
        )


def create_mbpp_dataset():
    # opencompass/mbpp ships bare jsonl with no HF subset config, so loading it
    # by dataset id raises KeyError('default'); download the file and read it
    # as a local jsonl instead.
    from modelscope.hub.file_download import dataset_file_download
    path = dataset_file_download(dataset_id='opencompass/mbpp', file_path='mbpp.jsonl')
    dataset = Dataset()
    dataset.add_dataset(DatasetMeta(path, split='train'))
    dataset.set_template('Template', model_id=MODEL_ID, max_length=4096,
                         truncation_strategy='delete', enable_thinking=True)
    dataset.map(MbppProcessor())
    dataset.encode(add_generation_prompt=True)
    return dataset


def compute_rewards(
    trajectories: List[Dict[str, Any]],
) -> Tuple[List[float], List[float], List[float]]:
    pass_rewards = MbppAssertReward()(trajectories)
    format_rewards = MbppFormatReward()(trajectories)
    total_rewards = [p + f for p, f in zip(pass_rewards, format_rewards)]
    return total_rewards, format_rewards, pass_rewards


# ========== Main ==========
def main():
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU'),
    ]

    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    lora_config = LoraConfig(
        target_modules='all-linear',
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.05,
    )

    if USE_MEGATRON:
        from twinkle.model.megatron import MegatronModel
        model = MegatronModel(
            model_id=MODEL_ID,
            device_mesh=model_mesh,
            remote_group='model',
            mixed_precision='bf16',
            variable_seq_lengths=True,
        )
    else:
        model = TransformersModel(
            model_id=MODEL_ID,
            device_mesh=model_mesh,
            remote_group='model',
        )

    model.add_adapter_to_model('default', lora_config,
                               gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    if USE_MEGATRON:
        model.set_optimizer('default', lr=LEARNING_RATE)
        model.set_lr_scheduler('default', lr_decay_steps=MAX_STEPS, max_lr=LEARNING_RATE)
    else:
        model.set_optimizer('AdamW', lr=LEARNING_RATE)
        model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)

    model.set_loss('GRPOLoss', epsilon=0.2)
    model.set_processor(InputProcessor, padding_free=True)
    model.set_template('Template', model_id=MODEL_ID, enable_thinking=True)

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 8192,
            'max_lora_rank': 32,
            'enable_lora': True,
            'enable_tower_connector_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Template', model_id=MODEL_ID, enable_thinking=True)

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)

    GLOBAL_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    dataloader = DataLoader(
        dataset=create_mbpp_dataset,
        batch_size=GLOBAL_BATCH_SIZE,
        min_batch_size=GLOBAL_BATCH_SIZE,
        device_mesh=model_mesh,
        remote_group='model',
    )

    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()
    sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1,
                                     temperature=1.0, top_p=0.95)

    optim_step = 0
    logger.info(f'Starting MBPP GRPO (974 problems, judge workers {JUDGE_WORKERS})')
    logger.info(get_device_placement())

    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break

        metrics.reset()
        expand_prompts = []
        for prompt in batch:
            expand_prompts.extend([prompt] * NUM_GENERATIONS)

        ckpt_manager.sync_weights(merge_and_sync=False)
        sampler.reset_prefix_cache()

        sample_responses = sampler.sample(expand_prompts, sampling_params)

        all_input_data: List[Dict[str, Any]] = []
        all_old_logps: List[List[float]] = []
        all_completion_lengths: List[int] = []

        for sample_response in sample_responses:
            for sequence in sample_response.sequences:
                all_input_data.append(sequence.new_input_feature)
                all_old_logps.append([logprob[0][1] for logprob in sequence.logprobs])
                all_completion_lengths.append(len(sequence.tokens))

        total_rewards, format_rewards, pass_rewards = compute_rewards(all_input_data)

        metrics.accumulate(
            completion_lengths=all_completion_lengths,
            rewards={
                'total': total_rewards,
                'format': format_rewards,
                'pass': pass_rewards,
            },
        )

        advantages = advantage_fn(total_rewards, num_generations=NUM_GENERATIONS,
                                  scale='group').tolist()

        total_completions = len(all_input_data)
        for mb_start in range(0, total_completions, MINI_BATCH_SIZE):
            mb_end = min(mb_start + MINI_BATCH_SIZE, total_completions)
            model.forward_backward(
                inputs=all_input_data[mb_start:mb_end],
                old_logps=all_old_logps[mb_start:mb_end],
                advantages=advantages[mb_start:mb_end],
                micro_batch_size=MICRO_BATCH_SIZE,
            )
            model.clip_grad_and_step()
            optim_step += 1

            if optim_step >= MAX_STEPS:
                break
            if optim_step % SAVE_STEPS == 0:
                model.save(f'mbpp-grpo-checkpoint-{optim_step}')

        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True))
        swanlab.log(log_dict)
        metrics.reset()
        logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('mbpp-grpo-final')


if __name__ == '__main__':
    main()
