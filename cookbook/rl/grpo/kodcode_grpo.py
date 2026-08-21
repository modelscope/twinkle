"""GRPO training script for KodCode-V1 (code generation with pytest-verified reward).

Same structure as short_math_grpo.py, but the reward runs the dataset's own
pytest suite against the generated code instead of comparing a final number.

Difficulty is filtered by KodCode's own ``gpt_pass_percentage`` so that the
sampled group is unlikely to collapse (all-correct or all-wrong within a group
gives a zero GRPO advantage and therefore no gradient).

Sandbox judging follows .temp/human_e18/e18_kodcode.py (``run_tests``): the
submitted code is written to ``solution.py`` and the official test to
``test_solution.py``, then pytest runs in a subprocess with a timeout and a 2GB
address-space limit. That logic is inlined here rather than imported, because
Ray deserializes the dataset builder and the reward inside worker processes that
do not share this driver's ``sys.path``.
"""
import ast as _ast
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
MAX_NEW_TOKENS = args.sampling.max_tokens or 4096
LEARNING_RATE = args.optimizer.learning_rate or 1e-5
MAX_STEPS = args.training.max_steps or 1000
BATCH_SIZE = args.training.batch_size or 8
MINI_BATCH_SIZE = args.training.mini_batch_size or 8
MICRO_BATCH_SIZE = args.training.micro_batch_size or 2
GRADIENT_ACCUMULATION_STEPS = args.training.gradient_accumulation_steps or 1
SAVE_STEPS = args.training.save_steps or 1000
LORA_RANK = args.lora.lora_r or 16

# Keep only problems the teacher solved sometimes but not always: a group whose
# 8 samples are all right or all wrong contributes no advantage.
KOD_MIN_PASS_PCT = float(os.environ.get('KOD_MIN_PASS_PCT', 0.2))
KOD_MAX_PASS_PCT = float(os.environ.get('KOD_MAX_PASS_PCT', 0.8))
# Judging is a subprocess and runs while the GPUs idle, so keep it wide.
JUDGE_WORKERS = int(os.environ.get('JUDGE_WORKERS', max(24, min(96, (os.cpu_count() or 24) // 2))))

SYSTEM_PROMPT = ('You are an expert Python programmer. Write a complete, self-contained '
                 'solution in a single ```python code block. Do not include tests.')

TEST_TIMEOUT = int(os.environ.get('TEST_TIMEOUT', 60))

_FENCE_RE = re.compile(r'```(?:python|py)?\s*\n(.*?)```', re.S)
_SPECIAL_TOKEN_RE = re.compile(r'<\|[^|]+\|>')


# ========== Text handling (same as e18_kodcode) ==========
def after_think(text: str) -> str:
    """Keep only what follows </think>; return the text unchanged if unclosed."""
    idx = text.rfind('</think>')
    return text[idx + len('</think>'):] if idx >= 0 else text


def clean_text(decoded: Optional[str]) -> str:
    return _SPECIAL_TOKEN_RE.sub('', decoded or '').strip()


def extract_code(text: str) -> str:
    """Take the last fenced block; fall back to the whole body when unfenced.

    The last one, not the first: models often draft a version before the final
    one, and the last block is their conclusion.
    """
    body = after_think(text)
    blocks = _FENCE_RE.findall(body)
    if blocks:
        return blocks[-1].strip()
    return body.strip()


# ========== Sandbox (same contract as e18_kodcode.run_tests) ==========
# Assertion vs exception must be told apart via ``reprcrash.message``: pytest
# rewrites assertions, so the summary reads "E  assert -1 == 3" and the string
# "AssertionError" never appears -- matching on it misclassifies every failed
# assertion as an exception.
_RUNNER = r"""
import sys, pytest


class _Collect:
    def __init__(self):
        self.n_tests = self.n_fail = self.n_err = 0

    @staticmethod
    def _is_assertion(report):
        crash = getattr(getattr(report, 'longrepr', None), 'reprcrash', None)
        msg = getattr(crash, 'message', '') or ''
        return msg.startswith('assert') or msg.startswith('AssertionError')

    def pytest_runtest_logreport(self, report):
        if report.when == 'call':
            self.n_tests += 1
            if report.failed:
                if self._is_assertion(report):
                    self.n_fail += 1
                else:
                    self.n_err += 1
        elif report.failed:
            self.n_err += 1


c = _Collect()
rc = pytest.main(['-q', '--no-header', '-p', 'no:cacheprovider',
                  '--tb=short', 'test_solution.py'], plugins=[c])
print('__KOD__', c.n_tests, c.n_fail, c.n_err)
sys.exit(0 if int(rc) == 0 else 1)
"""


def run_tests(code: str, payload: Dict[str, Any], timeout: int = TEST_TIMEOUT) -> Dict[str, Any]:
    """Run the submitted code (solution.py) against the official test in a subprocess.

    The code goes into its own ``solution.py`` because KodCode tests grab the
    function under test via ``from solution import X``.
    """
    if not code.strip():
        return {'passed': False, 'kind': 'no_code', 'error': 'no parseable code block'}
    entry = payload.get('entry_point') or ''
    if entry and entry not in code:
        return {'passed': False, 'kind': 'no_entry',
                'error': f'function {entry} is not defined in the submitted code'}
    tmp = tempfile.mkdtemp(prefix='kod_')
    try:
        with open(os.path.join(tmp, 'solution.py'), 'w', encoding='utf-8') as f:
            f.write(code)
        with open(os.path.join(tmp, 'test_solution.py'), 'w', encoding='utf-8') as f:
            f.write(payload['test'])
        with open(os.path.join(tmp, '_run.py'), 'w', encoding='utf-8') as f:
            f.write(_RUNNER)
        env = dict(os.environ, MPLBACKEND='Agg', PYTHONHASHSEED='0', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
        env.pop('CUDA_VISIBLE_DEVICES', None)

        # start_new_session + killpg on timeout: pytest can fork, and a bare
        # kill would leave grandchildren running. RLIMIT_AS caps the child at
        # 2GB so a runaway solution cannot take the host down.
        def _limit():
            resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))

        proc = subprocess.Popen([sys.executable, '_run.py'], cwd=tmp, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                errors='replace', start_new_session=True, preexec_fn=_limit)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.communicate(timeout=5)
            except Exception:
                pass
            return {'passed': False, 'kind': 'timeout',
                    'error': f'the tests did not finish within {timeout}s'}
        n_tests = n_fail = n_err = 0
        for line in (stdout or '').splitlines():
            if line.startswith('__KOD__'):
                _, a, b, c = line.split()
                n_tests, n_fail, n_err = int(a), int(b), int(c)
        if returncode == 0 and n_tests > 0:
            return {'passed': True, 'kind': 'pass', 'error': ''}
        kind = 'assertion' if n_fail else ('exception' if n_err else 'import_or_syntax')
        return {'passed': False, 'kind': kind, 'error': ''}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ========== Row helpers (same as e18_kodcode) ==========
def _entry_point(row: Dict[str, Any]) -> str:
    """Function under test, from test_info; else from ``from solution import X``."""
    ti = row.get('test_info')
    if ti is not None:
        try:
            items = list(ti) if not isinstance(ti, str) else _ast.literal_eval(ti)
            for it in items:
                name = (it or {}).get('function_name')
                if name:
                    return str(name)
        except Exception:
            pass
    m = re.search(r'from\s+solution\s+import\s+([A-Za-z_]\w*)', row.get('test') or '')
    return m.group(1) if m else ''


def _code_prompt(row: Dict[str, Any]) -> str:
    """The function signature, used to pin the entry point for the model."""
    ti = row.get('test_info')
    if ti is not None:
        try:
            items = list(ti) if not isinstance(ti, str) else _ast.literal_eval(ti)
            for it in items:
                decl = (it or {}).get('function_declaration')
                if decl:
                    return str(decl)
        except Exception:
            pass
    return ''


def _usable(row: Dict[str, Any]) -> bool:
    """Minimum bar to enter the pool.

    The test must import from ``solution``: 11.7% of rows call bare function
    names, which can never resolve under this sandbox layout, so keeping them
    would permanently depress the reward for reasons unrelated to the model.
    """
    test = row.get('test') or ''
    if 'def test_' not in test:
        return False
    if not re.search(r'from\s+solution\s+import|import\s+solution\b', test):
        return False
    return bool((row.get('solution') or '').strip()) and bool(_entry_point(row))


# ========== Reward ==========
class KodCodePytestReward(Reward):
    """1.0 when the generated code passes the problem's own pytest suite.

    The suite is carried per-sample through ``user_data`` (``kod_payload``), so
    each trajectory is judged against its own tests. Judging runs in a thread
    pool because every verdict is a separate subprocess.
    """

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        jobs: List[Tuple[int, str, Dict[str, Any]]] = []
        rewards = [0.0] * len(trajectories)
        for i, traj in enumerate(trajectories):
            payload = None
            for item in traj.get('user_data') or []:
                if item[0] == 'kod_payload':
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
        # Same (code, test) pair judged once: identical completions are common.
        uniq: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for _, code, payload in jobs:
            uniq.setdefault((payload['task_id'], code), payload)
        todo = list(uniq)
        with ThreadPoolExecutor(max_workers=max(1, min(JUDGE_WORKERS, len(todo)))) as ex:
            verdicts = dict(zip(todo, ex.map(lambda k: run_tests(k[1], uniq[k]), todo)))
        for i, code, payload in jobs:
            v = verdicts.get((payload['task_id'], code))
            rewards[i] = 1.0 if (v and v['passed']) else 0.0
        return rewards


class KodCodeFormatReward(Reward):
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
# Only 8% of KodCode questions name the function under test, but the tests grab
# it via ``from solution import <name>``. Append the signature or nearly every
# sample scores 0 regardless of how good the answer is.
_SIG_HINT = '\n\nYou should write self-contained code starting with:\n```\n{decl}\n```'


class KodCodeProcessor(Preprocessor):
    """KodCode row -> prompt-only Trajectory carrying its pytest suite."""

    def __init__(self, system=SYSTEM_PROMPT):
        self.system = system

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        return self.map_row_to_col(rows)

    def preprocess(self, row) -> Trajectory:
        question = row.get('question') or ''
        decl = _code_prompt(row)
        if decl and decl.strip() not in question:
            question = question + _SIG_HINT.format(decl=decl.strip())
        payload = {
            'task_id': str(row.get('question_id') or ''),
            'entry_point': _entry_point(row),
            'test': row.get('test') or '',
        }
        return Trajectory(
            messages=[
                Message(role='system', content=self.system),
                Message(role='user', content=question),
            ],
            user_data=[('kod_payload', payload)],
        )


def create_kodcode_dataset():
    dataset = Dataset()
    dataset.add_dataset(DatasetMeta('ms://AI-ModelScope/KodCode-V1', split='train'))
    # Filter before templating: the full set is 73747 rows.
    dataset.filter(lambda r: KOD_MIN_PASS_PCT <= float(r.get('gpt_pass_percentage') or 0.0)
                   <= KOD_MAX_PASS_PCT)
    # Tests must import from ``solution``; the 11.7% that call bare names can
    # never pass in this sandbox layout and would only drag the reward down.
    dataset.filter(_usable)
    dataset.set_template('Template', model_id=MODEL_ID, max_length=4096,
                         truncation_strategy='delete', enable_thinking=True)
    dataset.map(KodCodeProcessor())
    dataset.encode(add_generation_prompt=True)
    return dataset


def compute_rewards(
    trajectories: List[Dict[str, Any]],
) -> Tuple[List[float], List[float], List[float]]:
    pass_rewards = KodCodePytestReward()(trajectories)
    format_rewards = KodCodeFormatReward()(trajectories)
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
        dataset=create_kodcode_dataset,
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
    logger.info(f'Starting KodCode GRPO (pass_pct window '
                f'[{KOD_MIN_PASS_PCT}, {KOD_MAX_PASS_PCT}], judge workers {JUDGE_WORKERS})')
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
                model.save(f'kodcode-grpo-checkpoint-{optim_step}')

        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True))
        swanlab.log(log_dict)
        metrics.reset()
        logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('kodcode-grpo-final')


if __name__ == '__main__':
    main()
