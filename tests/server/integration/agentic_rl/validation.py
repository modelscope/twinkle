"""Real multi-turn GRPO acceptance with AgentENV or explicitly scoped OpenEnv functional coverage."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import json
import math
import os
import sys
import threading
import time
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
        'filename': 'count_even.py',
        'function': 'count_even',
        'solution': 'def count_even(values):\n    return sum(value % 2 == 1 for value in values)\n',
        'smoke_test': 'from count_even import count_even\nassert count_even([1, 2, 4, 5]) == 2\n',
        'prompt': 'The workspace has count_even.py and smoke_test.py. Run the test, fix the odd/even boundary bug, and rerun it.',
        'hidden_cases': (([], 0), ([2, 4], 2), ([1, 3], 0), ([-2, 0, 7], 2)),
    },
    {
        'id': 'last-negative',
        'filename': 'last_negative.py',
        'function': 'last_negative',
        'solution': 'def last_negative(values):\n    negatives = [value for value in values if value < 0]\n    return negatives[0] if negatives else None\n',
        'smoke_test': 'from last_negative import last_negative\nassert last_negative([-1, -2, 3]) == -2\n',
        'prompt': 'The workspace has last_negative.py and smoke_test.py. Run the test, return the last negative correctly, and rerun it.',
        'hidden_cases': (([-1, -2, 3], -2), ([1, 2], None), ([-4], -4), ([0, -5, -6], -6)),
    },
]
_STAGE_TIMEOUT = 300.0


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
    slow_tool_seconds: float = 20.0
    environment_backend: str = 'agentenv'
    openenv_url: str | None = None
    openenv_name: str = 'coding_env'
    gradient_accumulation_steps: int = 2


def _load_recipe():
    path = Path(__file__).parents[4] / 'cookbook/client/async_rl/multi_turn_grpo.py'
    spec = importlib.util.spec_from_file_location('agentic_rl_validation_recipe', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_recipe(config):
    if config.environment_backend == 'agentenv':
        os.environ['AGENTENV_TEMPLATE'] = config.agentenv_template
    recipe = _load_recipe()
    if config.environment_backend == 'openenv':
        from tests.server.integration.agentic_rl.openenv_adapter import OpenEnvTaskEnv

        def create(task, on_created):
            env = OpenEnvTaskEnv(task, base_url=config.openenv_url, env_name=config.openenv_name)
            on_created(env)
            env.reset()
            for path, content in [(task.filename, task.solution), ('smoke_test.py', task.smoke_test)]:
                env.step('write_file', {'path': path, 'content': content})
                env.ensure_idle()
            manager = recipe.ToolManager(recipe.EnvTool.from_schemas(env, env.tools()))
            return recipe._Attempt(env, manager, {
                'messages': [{'role': 'user', 'content': task.prompt +
                              '\nAll task files are virtual files under /workspace. Use read_file/write_file with '
                              'relative file names or /workspace paths. Run tests with python smoke_test.py, '
                              'which loads the current task source automatically. Shell commands, OS imports, '
                              'and importing workspace modules with python -c are not available. '
                              'After the test passes, give a short final answer without repeated reruns.'}],
                'tools': manager.tool_infos(),
            })

        recipe._TaskAgentEnv = OpenEnvTaskEnv
        recipe._create_attempt = create
    return recipe


def learning_gate(*, updates: int, baseline: float, final: float, minimum_improvement: float) -> bool:
    return updates >= 2 and final - baseline > minimum_improvement


def pipeline_evidence_gate(evidence: dict) -> bool:
    required = (
        'training_update', 'parameter_change', 'multi_turn_feedback', 'token_alignment', 'distinct_snapshots',
        'new_policy_requests', 'sync_async_equivalent', 'slow_tool_isolated', 'training_overlap',
        'environment_isolation', 'cleanup_confirmed',
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
        'environment_backend': config.environment_backend,
        'acceptance_scope': ('openenv_functional' if config.environment_backend == 'openenv' else 'agentenv'),
        'agentenv_security_gate': None,
        'config': {
            key: value
            for key, value in asdict(config).items()
            if key not in {'server_url', 'result_dir', 'openenv_url'}
        },
        **values,
    }


def write_result(config: ValidationConfig, **values) -> Path:
    destination = Path(config.result_dir)
    destination.mkdir(parents=True, exist_ok=True)
    suffix = f'-{config.cancel_stage}' if config.phase == 'cancel' else ''
    path = destination / f'{config.backend}-{config.environment_backend}-{config.phase}{suffix}.json'
    path.write_text(json.dumps(result_payload(config, **values), indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return path


def _code_task(recipe, item: dict):
    return recipe.CodeTask(**{key: value for key, value in item.items() if key != 'id'})


def _supervised_label_count(value) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        return sum(_supervised_label_count(item) for item in value)
    return int(value != -100)


def _rollout_batch_evidence(rolled: list[dict], rewards: list[float]) -> dict:
    return {
        'rewards': rewards,
        'completion_tokens': [sum(_numbers(row.get('completion_mask'))) for row in rolled],
        'supervised_labels': [_supervised_label_count(row.get('labels')) for row in rolled],
        'rollout_logprobs': [len(row.get('rollout_logprobs') or []) for row in rolled],
    }


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


def _numbers(value) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return _numbers(value.get('logps'))
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in _numbers(item)]
    return []


def parameters_changed(before: list[float] | None, after: list[float] | None, *, tolerance: float = 1e-6) -> bool:
    """Same probe, same token positions: any log-prob shift beyond tolerance proves the adapter moved."""
    if not before or not after or len(before) != len(after):
        return False
    if not all(math.isfinite(value) for value in before + after):
        return False
    return max(abs(left - right) for left, right in zip(before, after, strict=True)) > tolerance


def slow_tool_isolated(spans: list[dict]) -> bool:
    """A peer generation must start and finish while the injected slow tool batch is still running."""
    for slow in (span for span in spans if span['kind'] == 'tool' and span.get('slow')):
        if any(span['kind'] == 'sample' and span['start'] >= slow['start'] and span['end'] <= slow['end']
               for span in spans):
            return True
    return False


def training_overlaps_rollout(spans: list[dict]) -> bool:
    training = [span for span in spans if span['kind'] == 'train']
    rollout = [span for span in spans if span['kind'] in {'sample', 'tool'}]
    return any(item['start'] < train['end'] and train['start'] < item['end'] for train in training for item in rollout)


class _Timeline:

    def __init__(self):
        self._lock = threading.Lock()
        self._origin = time.monotonic()
        self.spans: list[dict] = []

    @contextlib.contextmanager
    def span(self, kind: str, **labels):
        start = time.monotonic() - self._origin
        try:
            yield
        finally:
            end = time.monotonic() - self._origin
            with self._lock:
                self.spans.append({'kind': kind, 'start': start, 'end': end, **labels})


class _RecordingSampler:

    def __init__(self, sampler, timeline: _Timeline):
        self._sampler = sampler
        self._timeline = timeline
        self._lock = threading.Lock()
        self.uris = []

    def __getattr__(self, name):
        return getattr(self._sampler, name)

    def sample(self, *args, **kwargs):
        with self._lock:
            self.uris.append(kwargs.get('adapter_uri'))
        with self._timeline.span('sample'):
            return self._sampler.sample(*args, **kwargs)


class _TimedModel:

    def __init__(self, model, timeline: _Timeline):
        self._model = model
        self._timeline = timeline

    def __getattr__(self, name):
        return getattr(self._model, name)

    def forward_backward(self, *args, **kwargs):
        with self._timeline.span('train', operation='backward'):
            return self._model.forward_backward(*args, **kwargs)

    def clip_grad_and_step(self, *args, **kwargs):
        with self._timeline.span('train', operation='step'):
            return self._model.clip_grad_and_step(*args, **kwargs)


def _timed_tool_manager(base, timeline: _Timeline, slow_seconds: float):
    """Time every rollout tool batch and hold the first one for ``slow_seconds``."""
    lock = threading.Lock()
    pending = {'slow': slow_seconds > 0}

    class TimedToolManager(base):

        def call_many(self, tool_calls):
            with lock:
                slow, pending['slow'] = pending['slow'], False
            with timeline.span('tool', manager=id(self), slow=slow):
                if slow:
                    time.sleep(slow_seconds)
                return super().call_many(tool_calls)

    return TimedToolManager


async def _build_runtime(config, recipe):
    stop_event = asyncio.Event()
    clients = []
    try:
        client = await recipe._await_operation(
            init_twinkle_client,
            base_url=config.server_url,
            api_key=os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY_TOKEN'),
            stop_event=stop_event,
            on_result=clients.append,
        )
        model = await recipe._await_operation(client.model, config.model_id, stop_event=stop_event)
        sampler = await recipe._await_operation(client.sampler, config.model_id, stop_event=stop_event)
        await recipe._await_operation(
            model.add_adapter_to_model,
            recipe.ADAPTER_NAME,
            LoraConfig(target_modules='all-linear', r=8, lora_alpha=32),
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            stop_event=stop_event,
        )
        await recipe._await_operation(model.set_loss, 'GRPOLoss', epsilon=0.2, beta=0.0, stop_event=stop_event)
        # Adam is accepted by both Transformers and Megatron training backends.
        await recipe._await_operation(model.set_optimizer, 'Adam', lr=2e-5, stop_event=stop_event)
        await recipe._await_operation(model.set_processor, 'InputProcessor', stop_event=stop_event)
        await recipe._await_operation(
            model.set_template, 'Qwen3_5Template', model_id=config.template_model_id, stop_event=stop_event)
        await recipe._await_operation(
            sampler.set_template, 'Qwen3_5Template', model_id=config.template_model_id, stop_event=stop_event)
        template = await recipe._await_operation(
            Qwen3_5Template,
            model_id=config.template_model_id,
            max_length=8192,
            enable_thinking=False,
            stop_event=stop_event,
        )
        template.truncation_strategy = 'delete'
        rollout = ClientMultiTurnRollout(
            sampler,
            template,
            sampling_params=SamplingParams(max_tokens=512, temperature=1.0, top_p=0.95, logprobs=1),
            max_turns=recipe.MAX_TURNS,
        )
        return clients, model, sampler, rollout, stop_event
    except BaseException:
        await recipe._drain_cleanup(clients, [], set(), stop_event, propagate_cancel=False)
        raise


async def _initial_policy(recipe, model, stop_event, snapshots):
    initial = await recipe._await_operation(
        model.save, 'policy-0', save_optimizer=False, stop_event=stop_event, on_result=snapshots.append)
    return recipe._checkpoint_path(initial)


async def _evaluate_group(recipe, rollout, policy_uri, environments, stop_event, task, seeds):
    """Score one held-out task; each attempt gets its own seed so a group is not one sample repeated."""
    attempts = []
    for _ in seeds:
        recipe._ensure_running(stop_event)
        await recipe._await_operation(
            recipe._create_attempt, task, environments.append, stop_event=stop_event, on_result=attempts.append)
    results = await asyncio.gather(
        *[
            rollout.acall(
                [attempt.trajectory],
                tool_manager=[attempt.manager],
                adapter_name=recipe.ADAPTER_NAME,
                adapter_uri=policy_uri,
                sampling_params=replace(rollout.sampling_params, seed=seed),
            ) for attempt, seed in zip(attempts, seeds, strict=True)
        ],
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, BaseException):
            raise result
    recipe._ensure_running(stop_event)
    recipe._validate_training_rows([result[0] for result in results])
    rewards = []
    for attempt in attempts:
        recipe._ensure_running(stop_event)
        rewards.append(await recipe._await_operation(attempt.env.verify, stop_event=stop_event))
    return rewards


async def _evaluate(recipe, rollout, policy_uri, environments, stop_event, config):
    rewards = []
    evaluations = []
    for task_index, item in enumerate(_EVALUATION_TASKS):
        task = _code_task(recipe, item)
        for repeat in range(config.eval_repeats):
            first = config.seed + task_index * 1000 + repeat * recipe.NUM_GENERATIONS
            seeds = list(range(first, first + recipe.NUM_GENERATIONS))
            group_rewards = await _evaluate_group(recipe, rollout, policy_uri, environments, stop_event, task, seeds)
            rewards.extend(group_rewards)
            evaluations.append({'task_id': item['id'], 'repeat': repeat, 'seeds': seeds, 'attempts': len(group_rewards)})
    return rewards, evaluations


async def _sync_async_evidence(recipe, rollout, policy_uri, environments, stop_event, config):
    task = _code_task(recipe, _EVALUATION_TASKS[0])
    attempts = []
    for _ in range(2):
        await recipe._await_operation(
            recipe._create_attempt, task, environments.append, stop_event=stop_event, on_result=attempts.append)
    # Fixed seeds do not make two independent sampled requests identical.
    params = replace(rollout.sampling_params, seed=config.seed + 50000, temperature=0.0)
    common = {
        'adapter_name': recipe.ADAPTER_NAME,
        'adapter_uri': policy_uri,
        'sampling_params': params,
    }
    sync = rollout([attempts[0].trajectory], tool_manager=[attempts[0].manager], **common)[0]
    async_result = await rollout.acall([attempts[1].trajectory], tool_manager=[attempts[1].manager], **common)
    async_result = async_result[0]
    recipe._validate_training_rows([sync, async_result])
    keys = ('input_ids', 'labels', 'completion_mask', 'rollout_logprobs', 'messages')
    key_equality = {key: sync.get(key) == async_result.get(key) for key in keys}
    sync_logps = sync.get('rollout_logprobs') or []
    async_logps = async_result.get('rollout_logprobs') or []
    logprob_aligned = bool(sync_logps) and len(sync_logps) == len(async_logps)
    logprob_max_delta = (
        max(abs(left - right) for left, right in zip(sync_logps, async_logps, strict=True))
        if logprob_aligned else None
    )
    # Batching can change floating-point logprobs even when both APIs produce the same trajectory.
    equivalent = all(key_equality[key] for key in keys if key != 'rollout_logprobs') and logprob_aligned
    sandbox_ids = {attempt.env.sandbox_id for attempt in attempts}
    return {
        'sync_async_equivalent': equivalent,
        'sync_async_key_equality': key_equality,
        'sync_async_logprob_aligned': logprob_aligned,
        'sync_async_logprob_max_delta': logprob_max_delta,
        'environment_isolation': len(sandbox_ids) == 2 and None not in sandbox_ids,
        'sync_stop_reason': sync.get('stop_reason'),
        'async_stop_reason': async_result.get('stop_reason'),
    }


async def _probe_logps(recipe, model, trajectories, stop_event) -> list[float]:
    response = await recipe._await_operation(
        model.forward_only, inputs=trajectories, compute_loss=False, stop_event=stop_event)
    result = response['result'] if isinstance(response, dict) else response.result
    return _numbers(result)


async def run_validation(config: ValidationConfig) -> dict:
    if config.phase == 'learning' and config.min_success_improvement is None:
        raise ValueError('learning validation requires a frozen --min-success-improvement threshold')
    recipe = _prepare_recipe(config)
    timeline = _Timeline()
    recipe.ToolManager = _timed_tool_manager(recipe.ToolManager, timeline, config.slow_tool_seconds)
    clients, model, sampler, rollout, stop_event = await _build_runtime(config, recipe)
    recording_sampler = _RecordingSampler(sampler, timeline)
    rollout.sampler = recording_sampler
    rollout._core.response_callback._sampler = recording_sampler
    model = _TimedModel(model, timeline)
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
        probe = {}
        incomplete_rollout_groups = 0
        rollout_batches = []

        async def train_groups():
            nonlocal policy_uri, incomplete_rollout_groups
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
                    try:
                        rolled, rewards = task.result()
                    except recipe._IncompleteRolloutError:
                        # Model-limited groups are not training data. Other failures remain hard errors.
                        incomplete_rollout_groups += 1
                        continue
                    all_rolled.extend(rolled)
                    rollout_batches.append(_rollout_batch_evidence(rolled, rewards))
                    recipe._ensure_running(stop_event)
                    if 'trajectories' not in probe:
                        probe['trajectories'] = rolled
                        probe['before'] = await _probe_logps(recipe, model, rolled, stop_event)
                        probe['latest'] = probe['before']
                        probe['updates'] = []
                    updated = False
                    # Finish all accumulation micro-steps before counting an optimizer update and saving.
                    for _ in range(config.gradient_accumulation_steps):
                        updated = await recipe._train_group(
                            model, rolled, rewards, advantage_fn, stop_event, on_metrics=metrics.append)
                        if not updated:
                            break
                    if updated:
                        after = await _probe_logps(recipe, model, probe['trajectories'], stop_event)
                        changed = parameters_changed(probe['latest'], after)
                        probe['updates'].append({'changed': changed, 'before': probe['latest'], 'after': after})
                        probe['latest'] = after
                        if not changed:
                            continue
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
        probe_after = await _probe_logps(recipe, model, probe['trajectories'], stop_event) if probe else None
        sync_async = await _sync_async_evidence(recipe, rollout, policy_uri, environments, stop_event, config)
        trajectory = _training_evidence(all_rolled)
        snapshot_uris = {recipe._checkpoint_path(snapshot) for snapshot in snapshots}
        spans = list(timeline.spans)
        evidence = {
            'training_update': updates >= 1,
            # Keep the stricter count as diagnostics for learning runs, not as a functional pipeline gate.
            'two_updates': updates >= 2,
            'parameter_change': parameters_changed(probe.get('before'), probe_after),
            **trajectory,
            'distinct_snapshots': len(snapshot_uris) >= 2 and policy_uri != initial_uri,
            'new_policy_requests': policy_uri in recording_sampler.uris,
            'slow_tool_isolated': slow_tool_isolated(spans),
            'training_overlap': training_overlaps_rollout(spans),
            **sync_async,
        }
        values.update({
            'updates': updates,
            'incomplete_rollout_groups': incomplete_rollout_groups,
            'rollout_batches': rollout_batches,
            'final_policy_uri': policy_uri,
            'snapshot_uris': sorted(snapshot_uris),
            'policy_requests': recording_sampler.uris,
            'metrics': metrics,
            'parameter_probe': {
                'before': probe.get('before'), 'after': probe_after, 'updates': probe.get('updates', [])
            },
            'timeline': spans,
            'evidence': evidence,
        })
        if config.phase == 'learning':
            final_rewards, final_evaluations = await _evaluate(
                recipe, rollout, policy_uri, environments, stop_event, config)
            baseline = sum(reward == 1.0 for reward in baseline_rewards) / len(baseline_rewards)
            final = sum(reward == 1.0 for reward in final_rewards) / len(final_rewards)
            same_plan = baseline_evaluations == final_evaluations
            held_out = all(item['filename'] != recipe.TASK.filename for item in _EVALUATION_TASKS)
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
        cleanup_issues = await recipe._drain_cleanup(clients, environments, active_tasks, stop_event)
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
        self.post_cancel_operations = []
        self._lock = threading.Lock()
        self._cancelled = False
        self._triggered = False

    def mark_cancelled(self):
        with self._lock:
            self._cancelled = True

    def block(self, operation):
        with self._lock:
            self.operations.append(operation)
            if self._cancelled:
                self.post_cancel_operations.append(operation)
            trigger = operation == self.stage and not self._triggered
            if trigger:
                self._triggered = True
        if trigger:
            self.started.set()
            if not self.release.wait(timeout=_STAGE_TIMEOUT):
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


def _cancel_tool_manager(base, recorder: _CancelRecorder):
    """Record rollout tool batches only; environment setup writes go through ``Env.step`` directly."""

    class CancelToolManager(base):

        def call_many(self, tool_calls):
            recorder.block('tool')
            return super().call_many(tool_calls)

    return CancelToolManager


async def _wait_for_stage(recorder: _CancelRecorder, task: asyncio.Task, timeout: float) -> bool:
    """Wait on the event loop (no helper thread) until the stage blocks, the task ends, or time runs out."""
    deadline = time.monotonic() + timeout
    while not recorder.started.is_set():
        if task.done() or time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0.05)
    return True


async def run_cancel_validation(config: ValidationConfig) -> dict:
    recipe = _prepare_recipe(config)
    recorder = _CancelRecorder(config.cancel_stage)
    recipe.ToolManager = _cancel_tool_manager(recipe.ToolManager, recorder)
    clients, model, sampler, rollout, stop_event = await _build_runtime(config, recipe)
    environments, snapshots = [], []
    active_tasks = set()
    values = {}
    original_sample = sampler.sample
    original_verify = recipe._TaskAgentEnv.verify

    def sample(*args, **kwargs):
        recorder.block('sample')
        return original_sample(*args, **kwargs)

    def verify(self):
        recorder.block('verify')
        return original_verify(self)

    sampler.sample = sample
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
        stage_reached = await _wait_for_stage(recorder, task, _STAGE_TIMEOUT)
        recorder.mark_cancelled()
        if not task.done():
            task.cancel()
            # Deliver cancellation before releasing in-flight work, without a measurement blind spot.
            await asyncio.sleep(0)
        recorder.release.set()
        cancelled = False
        try:
            await task
        except asyncio.CancelledError:
            cancelled = True
        active_tasks.discard(task)
        values = {
            'stage_reached': stage_reached,
            'cancelled': cancelled,
            'post_cancel_operations': list(recorder.post_cancel_operations),
            'operations': list(recorder.operations),
        }
    finally:
        recorder.release.set()
        cleanup_issues = await recipe._drain_cleanup(clients, environments, active_tasks, stop_event)
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
    parser.add_argument('--environment-backend', choices=['agentenv', 'openenv'], default='agentenv')
    parser.add_argument('--agentenv-template', default=os.environ.get('AGENTENV_TEMPLATE'))
    parser.add_argument('--openenv-url', default=os.environ.get('OPENENV_BASE_URL'))
    parser.add_argument('--openenv-name', default='coding_env')
    parser.add_argument('--result-dir', required=True)
    parser.add_argument('--max-groups', type=int, default=4)
    parser.add_argument('--active-groups', type=int, default=2)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--eval-repeats', type=int, default=2)
    parser.add_argument('--min-success-improvement', type=float, default=None)
    parser.add_argument(
        '--cancel-stage', choices=['sample', 'tool', 'verify', 'backward', 'step', 'save'], default='sample')
    parser.add_argument('--slow-tool-seconds', type=float, default=20.0)
    args = parser.parse_args(argv)
    if args.max_groups < 2 or args.active_groups < 1 or args.eval_repeats < 1:
        parser.error('max-groups must be >=2 and active-groups/eval-repeats must be positive')
    if args.slow_tool_seconds <= 0:
        parser.error('slow-tool-seconds must be positive')
    if args.environment_backend == 'agentenv' and not args.agentenv_template:
        parser.error('AgentENV requires --agentenv-template')
    if args.environment_backend == 'openenv' and not args.openenv_url:
        parser.error('OpenEnv requires --openenv-url')
    return ValidationConfig(
        backend=args.backend,
        phase=args.phase,
        server_url=args.server_url,
        model_id=args.model_id,
        template_model_id=args.template_model_id or args.model_id,
        agentenv_template=args.agentenv_template or '',
        result_dir=args.result_dir,
        max_groups=args.max_groups,
        active_groups=args.active_groups,
        seed=args.seed,
        eval_repeats=args.eval_repeats,
        min_success_improvement=args.min_success_improvement,
        cancel_stage=args.cancel_stage,
        slow_tool_seconds=args.slow_tool_seconds,
        environment_backend=args.environment_backend,
        openenv_url=args.openenv_url,
        openenv_name=args.openenv_name,
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
