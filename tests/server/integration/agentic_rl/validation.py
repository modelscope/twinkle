"""Controlled real AgentENV validation for client-orchestrated multi-turn GRPO."""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import os
import sys
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from peft import LoraConfig

from twinkle import init_twinkle_client
from twinkle.advantage import GRPOAdvantage
from twinkle.data_format import SamplingParams
from twinkle.template import Qwen3_5Template
from twinkle_client.rollout import ClientMultiTurnRollout

_EVALUATION_TASKS = [
    {
        'id': 'count-even',
        'solution': 'def count_even(values):\n    return sum(value % 2 == 1 for value in values)\n',
        'smoke': 'from count_even import count_even\nassert count_even([1, 2, 4, 5]) == 2\n',
        'prompt': 'The workspace has count_even.py and smoke_test.py. Run the test, fix the odd/even boundary bug, and rerun it.',
        'filename': 'count_even.py',
        'cases': [([], 0), ([2, 4], 2), ([1, 3], 0), ([-2, 0, 7], 2)],
    },
    {
        'id': 'last-negative',
        'solution': 'def last_negative(values):\n    negatives = [value for value in values if value < 0]\n    return negatives[0] if negatives else None\n',
        'smoke': 'from last_negative import last_negative\nassert last_negative([-1, -2, 3]) == -2\n',
        'prompt': 'The workspace has last_negative.py and smoke_test.py. Run the test, return the last negative correctly, and rerun it.',
        'filename': 'last_negative.py',
        'cases': [([-1, -2, 3], -2), ([1, 2], None), ([-4], -4), ([0, -5, -6], -6)],
    },
]


@dataclass(frozen=True)
class ValidationConfig:
    backend: str
    phase: str
    server_url: str
    model_id: str
    template_model_id: str
    agentenv_template: str
    result_dir: str
    max_groups: int = 4
    active_groups: int = 2
    seed: int = 7
    eval_repeats: int = 2
    min_success_improvement: float | None = None
    cancel_stage: str = 'sample'


def _load_recipe():
    path = Path(__file__).parents[4] / 'cookbook/client/async_rl/multi_turn_grpo.py'
    spec = importlib.util.spec_from_file_location('agentic_rl_validation_recipe', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def learning_gate(*, updates: int, baseline: float, final: float, minimum_improvement: float) -> bool:
    return updates >= 2 and final - baseline > minimum_improvement


def pipeline_evidence_gate(evidence: dict) -> bool:
    required = (
        'two_updates', 'parameter_change', 'multi_turn_feedback', 'token_alignment', 'distinct_snapshots',
        'new_policy_requests', 'sync_async_equivalent', 'independent_episodes', 'environment_isolation',
        'cleanup_confirmed',
    )
    return all(evidence.get(key) is True for key in required)


def cancel_evidence_gate(evidence: dict) -> bool:
    return (
        evidence.get('stage_reached') is True
        and evidence.get('cancelled') is True
        and not evidence.get('post_cancel_operations')
        and not evidence.get('cleanup_issues')
    )


def result_payload(config: ValidationConfig, *, mode: str = 'real', **values) -> dict:
    return {
        'mode': mode,
        'backend': config.backend,
        'phase': config.phase,
        'config': {
            key: value
            for key, value in asdict(config).items()
            if key not in {'server_url', 'result_dir'}
        },
        **values,
    }


def write_result(config: ValidationConfig, **values) -> Path:
    destination = Path(config.result_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f'{config.backend}-{config.phase}.json'
    path.write_text(json.dumps(result_payload(config, **values), indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return path


def _set_task(recipe, task: dict) -> None:
    recipe._INITIAL_SOLUTION = task['solution']
    recipe._SMOKE_TEST = task['smoke']
    recipe._TASK_PROMPT = task['prompt']
    recipe._HIDDEN_CASES = task['cases']
    recipe._INITIAL_TASK_FILE = task['filename']


def _create_task_attempt(recipe, task, on_created):
    original_solution = recipe._INITIAL_SOLUTION
    original_smoke = recipe._SMOKE_TEST
    original_prompt = recipe._TASK_PROMPT
    original_cases = recipe._HIDDEN_CASES
    _set_task(recipe, task)
    try:
        env = recipe._TaskAgentEnv(
            template=recipe.AGENTENV_TEMPLATE,
            api_url=os.environ.get('AGENTENV_API_URL'),
        )
        on_created(env)
        env.reset()
        for path, content in [(task['filename'], task['solution']), ('smoke_test.py', task['smoke'])]:
            result = env.step('write_file', {'path': path, 'content': content})
            env.ensure_idle()
            if result.info.get('error'):
                raise recipe._TaskInfrastructureError('setup', env.sandbox_id, str(result.info['error']))
        manager = recipe.ToolManager([
            recipe.EnvTool(env, name) for name in ('run_command', 'read_file', 'write_file')
        ])
        return recipe._Attempt(env, manager, {
            'messages': [{'role': 'user', 'content': task['prompt']}],
            'tools': manager.tool_infos(),
        })
    finally:
        recipe._INITIAL_SOLUTION = original_solution
        recipe._SMOKE_TEST = original_smoke
        recipe._TASK_PROMPT = original_prompt
        recipe._HIDDEN_CASES = original_cases


def _training_evidence(rolled: list[dict]) -> dict[str, bool]:
    multi_turn = False
    aligned = True
    for trajectory in rolled:
        messages = trajectory.get('messages') or []
        for index, message in enumerate(messages):
            if message.get('role') == 'tool' and any(
                    later.get('role') == 'assistant' for later in messages[index + 1:]):
                multi_turn = True
        input_ids = list(trajectory.get('input_ids') or [])
        labels = list(trajectory.get('labels') or [])
        masks = list(trajectory.get('completion_mask') or [])
        response_ids = list(trajectory.get('response_token_ids') or [])
        old_logps = list(trajectory.get('rollout_logprobs') or [])
        if not (len(input_ids) == len(labels) == len(masks) and response_ids and len(response_ids) == len(old_logps)):
            aligned = False
        if not all(math.isfinite(value) for value in old_logps):
            aligned = False
        policy_positions = sum(label != -100 and bool(mask) for label, mask in zip(labels, masks, strict=False))
        if policy_positions != len(old_logps):
            aligned = False
    return {'multi_turn_feedback': multi_turn, 'token_alignment': aligned}


def _metrics_prove_updates(metrics: list[dict]) -> bool:
    nonzero = []
    for metric in metrics:
        for key, value in metric.items():
            if 'grad_norm' in key and isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
                nonzero.append(value)
    return len(nonzero) >= 2


class _RecordingSampler:

    def __init__(self, sampler):
        self._sampler = sampler
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0
        self.uris = []

    def __getattr__(self, name):
        return getattr(self._sampler, name)

    def sample(self, *args, **kwargs):
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            self.uris.append(kwargs.get('adapter_uri'))
        try:
            return self._sampler.sample(*args, **kwargs)
        finally:
            with self._lock:
                self._active -= 1


async def _build_runtime(config, recipe):
    client = init_twinkle_client(
        base_url=config.server_url,
        api_key=os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY_TOKEN'),
    )
    stop_event = asyncio.Event()
    try:
        model = client.model(config.model_id)
        sampler = client.sampler(config.model_id)
        await recipe._await_operation(
            model.add_adapter_to_model,
            recipe.ADAPTER_NAME,
            LoraConfig(target_modules='all-linear', r=8, lora_alpha=32),
            stop_event=stop_event,
        )
        await recipe._await_operation(model.set_loss, 'GRPOLoss', epsilon=0.2, beta=0.0, stop_event=stop_event)
        await recipe._await_operation(model.set_optimizer, 'AdamW', lr=2e-5, stop_event=stop_event)
        await recipe._await_operation(model.set_processor, 'InputProcessor', stop_event=stop_event)
        await recipe._await_operation(
            model.set_template, 'Qwen3_5Template', model_id=config.template_model_id, stop_event=stop_event)
        await recipe._await_operation(
            sampler.set_template, 'Qwen3_5Template', model_id=config.template_model_id, stop_event=stop_event)
        template = Qwen3_5Template(model_id=config.template_model_id, max_length=8192, enable_thinking=False)
        template.truncation_strategy = 'delete'
        rollout = ClientMultiTurnRollout(
            sampler,
            template,
            sampling_params=SamplingParams(max_tokens=512, temperature=1.0, top_p=0.95, logprobs=1),
            max_turns=recipe.MAX_TURNS,
        )
        return client, model, sampler, rollout, stop_event
    except BaseException:
        client.close()
        raise


async def _initial_policy(recipe, model, stop_event, snapshots):
    initial = await recipe._await_operation(
        model.save, 'policy-0', save_optimizer=False, stop_event=stop_event, on_result=snapshots.append)
    return recipe._checkpoint_path(initial)


async def _evaluate(recipe, rollout, policy_uri, environments, stop_event, config):
    rewards = []
    evaluations = []
    for task_index, task in enumerate(_EVALUATION_TASKS):
        for repeat in range(config.eval_repeats):
            seed = config.seed + task_index * 1000 + repeat
            original_create = recipe._create_attempt
            recipe._create_attempt = lambda on_created, current=task: _create_task_attempt(recipe, current, on_created)
            try:
                params = replace(rollout.sampling_params, seed=seed)
                _, group_rewards = await recipe._rollout_group(
                    rollout, policy_uri, environments, stop_event, sampling_params=params)
            finally:
                recipe._create_attempt = original_create
            rewards.extend(group_rewards)
            evaluations.append({'task_id': task['id'], 'repeat': repeat, 'seed': seed, 'attempts': len(group_rewards)})
    return rewards, evaluations


async def _sync_async_evidence(recipe, rollout, policy_uri, environments, stop_event, config):
    task = _EVALUATION_TASKS[0]
    attempts = []
    for _ in range(2):
        attempts.append(await recipe._await_operation(
            lambda: _create_task_attempt(recipe, task, environments.append), stop_event=stop_event))
    params = replace(rollout.sampling_params, seed=config.seed + 50000)
    common = {
        'adapter_name': recipe.ADAPTER_NAME,
        'adapter_uri': policy_uri,
        'sampling_params': params,
    }
    sync = rollout([attempts[0].trajectory], tool_manager=[attempts[0].manager], **common)[0]
    async_result = await rollout.acall([attempts[1].trajectory], tool_manager=[attempts[1].manager], **common)
    async_result = async_result[0]
    keys = ('input_ids', 'labels', 'completion_mask', 'rollout_logprobs', 'messages')
    equivalent = all(sync.get(key) == async_result.get(key) for key in keys)
    sandbox_ids = {attempt.env.sandbox_id for attempt in attempts}
    return {
        'sync_async_equivalent': equivalent,
        'environment_isolation': len(sandbox_ids) == 2 and None not in sandbox_ids,
        'sync_stop_reason': sync.get('stop_reason'),
        'async_stop_reason': async_result.get('stop_reason'),
    }


async def run_validation(config: ValidationConfig) -> dict:
    if config.phase == 'learning' and config.min_success_improvement is None:
        raise ValueError('learning validation requires a frozen --min-success-improvement threshold')
    os.environ['AGENTENV_TEMPLATE'] = config.agentenv_template
    recipe = _load_recipe()
    client, model, sampler, rollout, stop_event = await _build_runtime(config, recipe)
    recording_sampler = _RecordingSampler(sampler)
    rollout.sampler = recording_sampler
    rollout._core.response_callback._sampler = recording_sampler
    environments, snapshots = [], []
    active_tasks = set()
    values = {'learning_gate': None}
    try:
        policy_uri = await _initial_policy(recipe, model, stop_event, snapshots)
        initial_uri = policy_uri
        baseline_rewards = baseline_evaluations = None
        if config.phase == 'learning':
            baseline_rewards, baseline_evaluations = await _evaluate(
                recipe, rollout, policy_uri, environments, stop_event, config)
        advantage_fn = GRPOAdvantage()
        metrics = []
        all_rolled = []

        async def train_groups():
            nonlocal policy_uri
            updates = 0
            next_group = 0
            tasks = set()
            while next_group < config.max_groups or tasks:
                while next_group < config.max_groups and len(tasks) < config.active_groups and not stop_event.is_set():
                    task = asyncio.create_task(recipe._rollout_group(rollout, policy_uri, environments, stop_event))
                    tasks.add(task)
                    active_tasks.add(task)
                    next_group += 1
                if not tasks:
                    break
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    active_tasks.discard(task)
                    rolled, rewards = task.result()
                    all_rolled.extend(rolled)
                    recipe._ensure_running(stop_event)
                    updated = await recipe._train_group(
                        model, rolled, rewards, advantage_fn, stop_event, on_metrics=metrics.append)
                    if updated:
                        updates += 1
                        saved = await recipe._await_operation(
                            model.save,
                            f'policy-{updates}',
                            save_optimizer=False,
                            stop_event=stop_event,
                            on_result=snapshots.append,
                        )
                        policy_uri = recipe._checkpoint_path(saved)
            return updates

        updates = await train_groups()
        sync_async = await _sync_async_evidence(recipe, rollout, policy_uri, environments, stop_event, config)
        trajectory = _training_evidence(all_rolled)
        snapshot_uris = {recipe._checkpoint_path(snapshot) for snapshot in snapshots}
        evidence = {
            'two_updates': updates >= 2,
            'parameter_change': _metrics_prove_updates(metrics),
            **trajectory,
            'distinct_snapshots': len(snapshot_uris) >= 3 and policy_uri != initial_uri,
            'new_policy_requests': policy_uri in recording_sampler.uris,
            'independent_episodes': recording_sampler.max_active >= 2,
            **sync_async,
        }
        values.update({
            'updates': updates,
            'final_policy_uri': policy_uri,
            'snapshot_uris': sorted(snapshot_uris),
            'policy_requests': recording_sampler.uris,
            'metrics': metrics,
            'evidence': evidence,
        })
        if config.phase == 'learning':
            final_rewards, final_evaluations = await _evaluate(
                recipe, rollout, policy_uri, environments, stop_event, config)
            baseline = sum(reward == 1.0 for reward in baseline_rewards) / len(baseline_rewards)
            final = sum(reward == 1.0 for reward in final_rewards) / len(final_rewards)
            same_plan = baseline_evaluations == final_evaluations
            held_out = all(item['task_id'] != 'smallest-positive' for item in final_evaluations)
            values.update({
                'baseline_rewards': baseline_rewards,
                'final_rewards': final_rewards,
                'baseline_success_rate': baseline,
                'final_success_rate': final,
                'evaluation_plan': final_evaluations,
                'learning_gate': (
                    same_plan and held_out and learning_gate(
                        updates=updates,
                        baseline=baseline,
                        final=final,
                        minimum_improvement=config.min_success_improvement,
                    )),
            })
    except BaseException:
        stop_event.set()
        raise
    finally:
        cleanup_issues = await recipe._drain_cleanup(client, environments, active_tasks, stop_event)
        values['cleanup_issues'] = cleanup_issues
        evidence = values.get('evidence', {})
        evidence['cleanup_confirmed'] = not cleanup_issues
        values['evidence'] = evidence
        values['pipeline_gate'] = pipeline_evidence_gate(evidence)
    return values


class _CancelRecorder:

    def __init__(self, stage):
        self.stage = stage
        self.started = threading.Event()
        self.release = threading.Event()
        self.operations = []
        self._triggered = False

    def block(self, operation):
        self.operations.append(operation)
        if operation == self.stage and not self._triggered:
            self._triggered = True
            self.started.set()
            if not self.release.wait(timeout=300):
                raise TimeoutError(f'cancel validation did not release {operation}')


class _CancelModel:

    def __init__(self, model, recorder):
        self._model = model
        self._recorder = recorder

    def __getattr__(self, name):
        return getattr(self._model, name)

    def forward_backward(self, *args, **kwargs):
        self._recorder.block('backward')
        return self._model.forward_backward(*args, **kwargs)

    def clip_grad_and_step(self, *args, **kwargs):
        self._recorder.block('step')
        return self._model.clip_grad_and_step(*args, **kwargs)

    def save(self, name, *args, **kwargs):
        if name != 'policy-0':
            self._recorder.block('save')
        return self._model.save(name, *args, **kwargs)


async def run_cancel_validation(config: ValidationConfig) -> dict:
    os.environ['AGENTENV_TEMPLATE'] = config.agentenv_template
    recipe = _load_recipe()
    client, model, sampler, rollout, stop_event = await _build_runtime(config, recipe)
    recorder = _CancelRecorder(config.cancel_stage)
    environments, snapshots = [], []
    active_tasks = set()
    values = {}
    original_sample = sampler.sample
    original_step = recipe._TaskAgentEnv.step
    original_verify = recipe._TaskAgentEnv.verify

    def sample(*args, **kwargs):
        recorder.block('sample')
        return original_sample(*args, **kwargs)

    def step(self, tool_name, arguments=None):
        recorder.block('tool')
        return original_step(self, tool_name, arguments)

    def verify(self):
        recorder.block('verify')
        return original_verify(self)

    sampler.sample = sample
    recipe._TaskAgentEnv.step = step
    recipe._TaskAgentEnv.verify = verify
    model = _CancelModel(model, recorder)
    try:
        policy_uri = await _initial_policy(recipe, model, stop_event, snapshots)

        async def operation():
            rolled, rewards = await recipe._rollout_group(rollout, policy_uri, environments, stop_event)
            if config.cancel_stage in {'backward', 'step', 'save'}:
                updated = await recipe._train_group(model, rolled, rewards, GRPOAdvantage(), stop_event)
                if updated and config.cancel_stage == 'save':
                    await recipe._await_operation(
                        model.save,
                        'policy-1',
                        save_optimizer=False,
                        stop_event=stop_event,
                        on_result=snapshots.append,
                    )

        task = asyncio.create_task(operation())
        active_tasks.add(task)
        await asyncio.wait_for(asyncio.to_thread(recorder.started.wait), timeout=300)
        cancel_index = len(recorder.operations)
        task.cancel()
        recorder.release.set()
        cancelled = False
        try:
            await task
        except asyncio.CancelledError:
            cancelled = True
        values = {
            'stage_reached': True,
            'cancelled': cancelled,
            'post_cancel_operations': recorder.operations[cancel_index:],
            'operations': recorder.operations,
        }
    finally:
        cleanup_issues = await recipe._drain_cleanup(client, environments, active_tasks, stop_event)
        values['cleanup_issues'] = cleanup_issues
        values['pipeline_gate'] = cancel_evidence_gate(values)
        values['learning_gate'] = None
    return values


def parse_args(argv=None) -> ValidationConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['transformers', 'megatron'], required=True)
    parser.add_argument('--phase', choices=['pipeline', 'learning', 'cancel'], required=True)
    parser.add_argument('--server-url', default='http://127.0.0.1:8000')
    parser.add_argument('--model-id', default='ms://Qwen/Qwen3.5-4B')
    parser.add_argument('--template-model-id', default=None)
    parser.add_argument('--agentenv-template', required=True)
    parser.add_argument('--result-dir', required=True)
    parser.add_argument('--max-groups', type=int, default=4)
    parser.add_argument('--active-groups', type=int, default=2)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--eval-repeats', type=int, default=2)
    parser.add_argument('--min-success-improvement', type=float, default=None)
    parser.add_argument(
        '--cancel-stage', choices=['sample', 'tool', 'verify', 'backward', 'step', 'save'], default='sample')
    args = parser.parse_args(argv)
    if args.max_groups < 2 or args.active_groups < 1 or args.eval_repeats < 1:
        parser.error('max-groups must be >=2 and active-groups/eval-repeats must be positive')
    return ValidationConfig(
        backend=args.backend,
        phase=args.phase,
        server_url=args.server_url,
        model_id=args.model_id,
        template_model_id=args.template_model_id or args.model_id,
        agentenv_template=args.agentenv_template,
        result_dir=args.result_dir,
        max_groups=args.max_groups,
        active_groups=args.active_groups,
        seed=args.seed,
        eval_repeats=args.eval_repeats,
        min_success_improvement=args.min_success_improvement,
        cancel_stage=args.cancel_stage,
    )


def main(argv=None) -> int:
    config = parse_args(argv)
    try:
        if config.phase == 'cancel':
            values = asyncio.run(run_cancel_validation(config))
        else:
            values = asyncio.run(run_validation(config))
        status = 'completed' if values['pipeline_gate'] and values.get('learning_gate') is not False else 'failed'
        path = write_result(config, status=status, **values)
        print(path)
        return 0 if status == 'completed' else 1
    except BaseException as error:
        path = write_result(
            config,
            status='blocked',
            error={'type': type(error).__name__, 'message': str(error)},
            pipeline_gate=False,
            learning_gate=False if config.phase == 'learning' else None,
        )
        print(path, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
