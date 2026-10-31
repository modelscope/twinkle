"""Client-orchestrated multi-turn GRPO with independent AgentENV workspaces."""
from __future__ import annotations

import asyncio
import json
import math
import os
import posixpath
import uuid
from dataclasses import dataclass
from typing import Any

import dotenv
from peft import LoraConfig

from twinkle import get_logger, init_twinkle_client
from twinkle.advantage import GRPOAdvantage
from twinkle.data_format import SamplingParams
from twinkle.template import Qwen3_5Template
from twinkle_agentic.envs import AgentEnv, EnvTool
from twinkle_agentic.envs.base import StepResult
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_client.rollout import ClientMultiTurnRollout

dotenv.load_dotenv('.env')
logger = get_logger()

BASE_MODEL = os.environ.get('TWINKLE_MODEL_ID', 'Qwen/Qwen3.5-4B')
MODEL_ID = f'ms://{BASE_MODEL}'
ADAPTER_NAME = 'client-multiturn-grpo'
AGENTENV_TEMPLATE = os.environ.get('AGENTENV_TEMPLATE', 'python:3.11')
NUM_GENERATIONS = 4
ACTIVE_GROUPS = 2
MAX_GROUPS = 4
MAX_TURNS = 6

_INITIAL_SOLUTION = '''def smallest_positive(values):
    positives = [value for value in values if value > 0]
    return max(positives) if positives else None
'''
_SMOKE_TEST = '''from solution import smallest_positive
assert smallest_positive([3, 1, 2]) == 1
assert smallest_positive([-1, 0]) is None
'''
_HIDDEN_CASES = [([3, 1, 2], 1), ([-2, 4, 1], 1), ([0, -1], None), ([7], 7), ([5, 3, 9], 3)]
_VERIFIER = '''import importlib.util
import json

call_id = __CALL_ID__
cases = __CASES__
payload = {"call_id": call_id, "completed": True, "passed": 0, "total": len(cases)}
try:
    spec = importlib.util.spec_from_file_location("candidate_solution", "solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for values, expected in cases:
        try:
            actual = module.smallest_positive(values)
        except Exception:
            continue
        if actual == expected:
            payload["passed"] += 1
except Exception as error:
    payload["candidate_error"] = type(error).__name__
print("TWINKLE_VERIFY=" + json.dumps(payload, sort_keys=True))
'''
_TASK_PROMPT = '''A small Python workspace contains solution.py and smoke_test.py.
Read the files, run the smoke test, fix the boundary-condition bug, and rerun the test.
Use the file and command tools, then give a short final answer. Do not edit the test to make it pass.
'''


class _TaskInfrastructureError(RuntimeError):

    def __init__(self, stage: str, sandbox_id: str | None, detail: str = ''):
        self.stage = stage
        self.sandbox_id = sandbox_id
        message = f'AgentENV infrastructure failure at {stage} (sandbox={sandbox_id})'
        if detail:
            message = f'{message}: {detail}'
        super().__init__(message)


class _TaskAgentEnv(AgentEnv):
    """Keep provider/verification failures distinct from candidate test failures."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._task_infrastructure_error: _TaskInfrastructureError | None = None

    def _local_refusal(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        if tool_name not in {'run_command', 'read_file', 'write_file'}:
            return f'Error: unsupported task tool {tool_name!r}.'
        if tool_name == 'run_command' and not arguments.get('command'):
            return "Error: 'command' argument is required."
        if tool_name in {'read_file', 'write_file'}:
            path = arguments.get('path')
            if not path:
                return "Error: 'path' argument is required."
            resolved = path if str(path).startswith('/') else posixpath.join(self.workspace, str(path))
            normalized = posixpath.normpath(resolved)
            if normalized != self.workspace and not normalized.startswith(self.workspace.rstrip('/') + '/'):
                return 'Error: task files must stay inside the workspace.'
        return None

    def _record_infrastructure_error(self, stage: str, detail: str = '') -> None:
        if self._task_infrastructure_error is None:
            self._task_infrastructure_error = _TaskInfrastructureError(stage, self.sandbox_id, detail)

    def step(self, tool_name: str, arguments: dict[str, Any] | None = None) -> StepResult:
        arguments = arguments or {}
        refusal = self._local_refusal(tool_name, arguments)
        if refusal is not None:
            return StepResult(observation=refusal, done=False)
        result = super().step(tool_name, arguments)
        if result.info.get('error'):
            self._record_infrastructure_error(f'tool:{tool_name}', str(result.info['error']))
        return result

    def ensure_idle(self) -> None:
        super().ensure_idle()
        if self._task_infrastructure_error is not None:
            raise self._task_infrastructure_error

    def verify(self) -> float:
        """Run the trusted checker in the same workspace and require its own result marker."""
        call_id = uuid.uuid4().hex
        source = _VERIFIER.replace('__CALL_ID__', repr(call_id)).replace('__CASES__', repr(_HIDDEN_CASES))
        path = f'/tmp/twinkle_scripts/verify-{call_id}.py'
        try:
            self.sandbox.files.write(path, source)
        except Exception as error:
            self._record_infrastructure_error('verify_upload', str(error))
            raise self._task_infrastructure_error from error
        exit_code, output = self._execute(f'python3 {path}', self.workspace, self._command_timeout)
        try:
            super().ensure_idle()
        except Exception as error:
            self._record_infrastructure_error('verify_command', str(error))
            raise self._task_infrastructure_error from error
        if exit_code != 0:
            self._record_infrastructure_error('verify_command', output)
            raise self._task_infrastructure_error
        marker = 'TWINKLE_VERIFY='
        rows = [row[len(marker):] for row in output.splitlines() if row.startswith(marker)]
        try:
            if len(rows) != 1:
                raise ValueError(f'expected one result marker, got {len(rows)}')
            result = json.loads(rows[0])
            if result.get('call_id') != call_id or result.get('completed') is not True:
                raise ValueError('verification result does not match this call')
            passed, total = result['passed'], result['total']
            if total != len(_HIDDEN_CASES) or not isinstance(passed, int) or not 0 <= passed <= total:
                raise ValueError('verification case statistics are invalid')
            return passed / total
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._record_infrastructure_error('verify_result', str(error))
            raise self._task_infrastructure_error from error


@dataclass
class _Attempt:
    env: _TaskAgentEnv
    manager: ToolManager
    trajectory: dict[str, Any]


def _checkpoint_path(saved: Any) -> str:
    return str(saved['twinkle_path'] if isinstance(saved, dict) else saved.twinkle_path)


async def _await_operation(operation, *args, stop_event: asyncio.Event, on_result=None, **kwargs):
    """Wait for one blocking operation without losing it when the caller is cancelled."""
    work = asyncio.create_task(asyncio.to_thread(operation, *args, **kwargs))
    task = asyncio.current_task()
    cancel_error: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(work)
        except asyncio.CancelledError as error:
            externally_cancelled = task is not None and task.cancelling() > 0
            if cancel_error is None and externally_cancelled:
                cancel_error = error
                stop_event.set()
            if work.done():
                if cancel_error is None:
                    raise
                try:
                    result = work.result()
                    if on_result is not None:
                        on_result(result)
                except BaseException as drain_error:
                    raise cancel_error from drain_error
                raise cancel_error
            if cancel_error is None:
                cancel_error = error
                stop_event.set()
        except BaseException as error:
            if cancel_error is not None:
                raise cancel_error from error
            raise
        else:
            if on_result is not None:
                on_result(result)
            if cancel_error is not None:
                raise cancel_error
            return result


def _ensure_running(stop_event: asyncio.Event) -> None:
    if stop_event.is_set():
        raise asyncio.CancelledError()


def _create_attempt(on_created) -> _Attempt:
    env = _TaskAgentEnv(template=AGENTENV_TEMPLATE, api_url=os.environ.get('AGENTENV_API_URL'))
    on_created(env)
    env.reset()
    for path, content in [('solution.py', _INITIAL_SOLUTION), ('smoke_test.py', _SMOKE_TEST)]:
        result = env.step('write_file', {'path': path, 'content': content})
        env.ensure_idle()
        if result.info.get('error'):
            raise _TaskInfrastructureError('setup', env.sandbox_id, str(result.info['error']))
    manager = ToolManager([EnvTool(env, name) for name in ('run_command', 'read_file', 'write_file')])
    trajectory = {
        'messages': [{'role': 'user', 'content': _TASK_PROMPT}],
        'tools': manager.tool_infos(),
    }
    return _Attempt(env, manager, trajectory)


def _validate_training_rows(rolled: list[dict[str, Any]]) -> None:
    for index, trajectory in enumerate(rolled):
        if trajectory.get('stop_reason') in {'error', 'abort'} or trajectory.get('truncated'):
            raise RuntimeError(f'attempt {index} is not a complete valid rollout')
        tokens = list(trajectory.get('response_token_ids') or [])
        logprobs = list(trajectory.get('rollout_logprobs') or [])
        if not tokens or len(tokens) != len(logprobs) or not all(math.isfinite(value) for value in logprobs):
            raise RuntimeError(f'attempt {index} has missing or misaligned rollout logprobs')


async def _rollout_group(rollout, policy_uri: str, environments: list, stop_event: asyncio.Event,
                         sampling_params: SamplingParams | None = None):
    attempts = []
    for _ in range(NUM_GENERATIONS):
        _ensure_running(stop_event)
        attempt = await _await_operation(
            _create_attempt,
            environments.append,
            stop_event=stop_event,
            on_result=attempts.append,
        )
    call_kwargs = {}
    if sampling_params is not None:
        call_kwargs['sampling_params'] = sampling_params
    rolled = await rollout.acall(
        [attempt.trajectory for attempt in attempts],
        tool_manager=[attempt.manager for attempt in attempts],
        adapter_name=ADAPTER_NAME,
        adapter_uri=policy_uri,
        **call_kwargs,
    )
    _ensure_running(stop_event)
    _validate_training_rows(rolled)
    rewards = []
    for attempt in attempts:
        _ensure_running(stop_event)
        rewards.append(await _await_operation(attempt.env.verify, stop_event=stop_event))
    return rolled, rewards


async def _train_group(model, rolled, rewards, advantage_fn, stop_event: asyncio.Event, on_metrics=None):
    advantages = await _await_operation(
        lambda: advantage_fn(rewards, num_generations=NUM_GENERATIONS, scale='group').tolist(),
        stop_event=stop_event,
    )
    if all(abs(value) < 1e-8 for value in advantages):
        logger.info(f'all advantages are zero; skipping update rewards={rewards}')
        return False
    _ensure_running(stop_event)
    await _await_operation(
        model.forward_backward,
        inputs=rolled,
        advantages=advantages,
        old_logps=[trajectory['rollout_logprobs'] for trajectory in rolled],
        stop_event=stop_event,
    )
    _ensure_running(stop_event)
    await _await_operation(model.clip_grad_and_step, max_grad_norm=1.0, stop_event=stop_event)
    metric_response = await _await_operation(model.calculate_metric, is_training=True, stop_event=stop_event)
    metrics = metric_response['result'] if isinstance(metric_response, dict) else metric_response.result
    if on_metrics is not None:
        on_metrics(metrics)
    logger.info(f'rewards={rewards} advantages={advantages} metrics={metrics}')
    return True


async def _cleanup_resources(client, environments, active_tasks, stop_event) -> list[str]:
    issues = []
    stop_event.set()
    for task in active_tasks:
        task.cancel()
    if active_tasks:
        await asyncio.gather(*active_tasks, return_exceptions=True)
    for env in environments:
        try:
            await _await_operation(env.ensure_idle, stop_event=stop_event)
        except BaseException as error:
            issues.append(f'quarantined environment {env.sandbox_id}: {error}')
            logger.warning(issues[-1])
            continue
        try:
            await _await_operation(env.close, stop_event=stop_event)
        except BaseException as error:
            issues.append(f'could not confirm environment close for {env.sandbox_id}: {error}')
            logger.warning(issues[-1])
    try:
        await _await_operation(client.close, stop_event=stop_event)
    except BaseException as error:
        issues.append(f'could not close client: {error}')
        logger.warning(issues[-1])
    return issues


async def _drain_cleanup(client, environments, active_tasks, stop_event, *, propagate_cancel: bool = True) -> list[str]:
    work = asyncio.create_task(_cleanup_resources(client, environments, active_tasks, stop_event))
    cancel_error = None
    while True:
        try:
            issues = await asyncio.shield(work)
            break
        except asyncio.CancelledError as error:
            if cancel_error is None:
                cancel_error = error
        except BaseException as error:
            logger.warning(f'cleanup failed: {error}')
            issues = [f'cleanup failed: {error}']
            break
    if propagate_cancel and cancel_error is not None:
        raise cancel_error
    return issues


async def train() -> None:
    client = init_twinkle_client(
        base_url=os.environ.get('TWINKLE_SERVER_URL', 'http://localhost:8000'),
        api_key=os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY_TOKEN'),
    )
    stop_event = asyncio.Event()
    environments = []
    snapshots = []
    active_tasks = set()
    primary_error = None
    try:
        model = client.model(MODEL_ID)
        sampler = client.sampler(MODEL_ID)
        await _await_operation(
            model.add_adapter_to_model,
            ADAPTER_NAME,
            LoraConfig(target_modules='all-linear', r=8, lora_alpha=32),
            stop_event=stop_event,
        )
        await _await_operation(model.set_loss, 'GRPOLoss', epsilon=0.2, beta=0.0, stop_event=stop_event)
        await _await_operation(model.set_optimizer, 'AdamW', lr=2e-5, stop_event=stop_event)
        await _await_operation(model.set_processor, 'InputProcessor', stop_event=stop_event)
        await _await_operation(model.set_template, 'Qwen3_5Template', model_id=MODEL_ID, stop_event=stop_event)
        await _await_operation(sampler.set_template, 'Qwen3_5Template', model_id=MODEL_ID, stop_event=stop_event)
        template = Qwen3_5Template(model_id=MODEL_ID, max_length=8192, enable_thinking=False)
        template.truncation_strategy = 'delete'
        rollout = ClientMultiTurnRollout(
            sampler,
            template,
            sampling_params=SamplingParams(max_tokens=512, temperature=1.0, top_p=0.95, logprobs=1),
            max_turns=MAX_TURNS,
        )
        initial = await _await_operation(
            model.save,
            'policy-0',
            save_optimizer=False,
            stop_event=stop_event,
            on_result=snapshots.append,
        )
        policy_uri = _checkpoint_path(initial)
        advantage_fn = GRPOAdvantage()
        next_group = 0
        version = 0
        semaphore = asyncio.Semaphore(ACTIVE_GROUPS)

        async def run_group(uri):
            async with semaphore:
                return await _rollout_group(rollout, uri, environments, stop_event)

        while next_group < MAX_GROUPS or active_tasks:
            while next_group < MAX_GROUPS and len(active_tasks) < ACTIVE_GROUPS and not stop_event.is_set():
                task = asyncio.create_task(run_group(policy_uri))
                active_tasks.add(task)
                next_group += 1
            if not active_tasks:
                break
            done, _ = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                active_tasks.remove(task)
                rolled, rewards = task.result()
                _ensure_running(stop_event)
                updated = await _train_group(model, rolled, rewards, advantage_fn, stop_event)
                if updated:
                    _ensure_running(stop_event)
                    version += 1
                    saved = await _await_operation(
                        model.save,
                        f'policy-{version}',
                        save_optimizer=False,
                        stop_event=stop_event,
                        on_result=snapshots.append,
                    )
                    policy_uri = _checkpoint_path(saved)
    except BaseException as error:
        primary_error = error
        stop_event.set()
        raise
    finally:
        await _drain_cleanup(
            client,
            environments,
            active_tasks,
            stop_event,
            propagate_cancel=primary_error is None,
        )


if __name__ == '__main__':
    asyncio.run(train())
