# Copyright (c) ModelScope Contributors. All rights reserved.
"""Episode construction and scoring, shared by collection and eval.

Three things must not differ between the run that invents a task and the run that
measures it: how an episode is built (a sandbox with ms-agent's tools plus a local
harness that only shapes messages), how the tool contract is advertised (schemas
read off the executor that will honour them), and how a trajectory is scored (the
task's own checks, run against the state the episode left behind).

``challenge.py`` takes ``solver_harness`` from here, and ``eval.py`` takes the
whole boot/score path, so a task kept at n_pass=4 during collection is a task the
eval measures the same way. A second copy of these lines would drift.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from twinkle import get_logger
from twinkle_agentic.envs import EnvTool
from twinkle_agentic.harness import MsAgentHarness
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_agentic.verifier.result_check import (CheckContext, checks_from_dicts,
                                                  run_checks)

from remote_tool_env import RemoteMsAgentToolEnv  # noqa: I100,I202

logger = get_logger()


@dataclass(frozen=True)
class SandboxConfig:
    """Everything about where episodes run, in one object.

    Read from the environment so training and eval cannot be pointed at
    different sandboxes by accident.
    """

    agent_config: str = 'cookbook/rsi/agentic/rsi_agent.yaml'
    template: str = 'twinkle-rsi-msagent'
    api_url: str = 'http://127.0.0.1:8000'
    # Must outlast a whole episode plus the checks that run after it.
    timeout: int = 900
    # Booting and scoring are network-bound, so they are done on threads. This
    # caps how many sandboxes are talked to at once, not how many exist.
    concurrency: int = 16
    # 'fraction' gives partial credit per check; 'all_or_nothing' is stricter.
    # Applies to structured checks only -- a check script has no partial credit.
    score_mode: str = 'fraction'

    @classmethod
    def from_env(cls) -> 'SandboxConfig':
        return cls(
            agent_config=os.environ.get('RSI_AGENT_CONFIG', cls.agent_config),
            template=os.environ.get('AENV_TEMPLATE', cls.template),
            api_url=os.environ.get('AENV_API_URL', cls.api_url),
            timeout=int(os.environ.get('RSI_SANDBOX_TIMEOUT', cls.timeout)),
            concurrency=int(os.environ.get('RSI_ENV_CONCURRENCY', cls.concurrency)),
            score_mode=os.environ.get('RSI_SCORE_MODE', cls.score_mode),
        )


def load_tasks(path: str) -> List[Dict[str, Any]]:
    """Read the task file and fail loudly on a task that can never be scored.

    Supports both formats:
      - ``check_script``: a python script, scored by exit status (challenge.py).
      - ``checks``: structured Check dicts (see tasks.example.jsonl).
    """
    tasks = []
    with open(path, encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            task = json.loads(line)
            if not task.get('query'):
                raise ValueError(f'{path}:{lineno} has no query')
            if task.get('check_script'):
                task['_checks'] = None
            elif task.get('checks'):
                task['_checks'] = checks_from_dicts(task['checks'])
            else:
                raise ValueError(f'{path}:{lineno} ({task.get("id")}) declares no checks '
                                 f'and no check_script')
            tasks.append(task)
    if not tasks:
        raise ValueError(f'{path} contains no tasks')
    return tasks


def solver_harness(agent_config: str):
    """A harness that only shapes messages: no llm, no tools.

    Popping ``tools`` matters as much as popping ``llm``, and for the same reason
    omitting the section from the yaml is not enough: ms-agent merges its own
    agent.yaml underneath, which declares file_system and code_executor, so a live
    shell executor would otherwise be constructed on the training host with access
    to the whole machine. Popping them after the merge leaves the harness with zero
    tools -- and the system prompt byte-identical, because ms-agent does not fold
    the tool list into it.

    Shared with challenge.py's difficulty stage on purpose: the opening a task is
    measured against there has to be the opening it is evaluated against here, and
    a second copy of these four lines would drift.
    """
    from omegaconf import OmegaConf, open_dict

    agent_cfg = OmegaConf.load(agent_config)
    harness = MsAgentHarness(config=agent_cfg)
    with open_dict(harness.agent.config):
        harness.agent.config.pop('llm', None)
        harness.agent.config.pop('tools', None)
    harness.prepare()
    return harness


def build_episode(task: Dict[str, Any], cfg: SandboxConfig) -> Tuple[Any, Any, Any, Dict]:
    """Create one episode: a sandbox with ms-agent's tools, plus a local harness."""
    harness = solver_harness(cfg.agent_config)

    env = RemoteMsAgentToolEnv(
        template=cfg.template,
        config_path=cfg.agent_config,
        api_url=cfg.api_url,
        sandbox_timeout=cfg.timeout,
    )
    env.reset()

    # A task may carry a setup_script that writes its input files instead of asking
    # the solver to. Nothing produces one now, but a task file from an older run
    # can still hold one. Loudly, not on a best-effort basis: a
    # statement that says the inputs are on disk, run against a workspace where
    # they are not, scores 0 for a reason that has nothing to do with the task.
    setup = task.get('setup_script')
    if setup:
        exit_code, output = env.runner()(setup, 'python')
        if exit_code != 0:
            raise RuntimeError(f'[{task.get("id")}] setup_script failed '
                               f'(exit {exit_code}): {output[-400:]}')

    trajectory = harness.start(task['query'])
    # The executor's own schemas, not the harness's (which are now empty by
    # construction). Advertising what will run is the whole point of sourcing
    # them from the sandbox.
    schemas = env.tool_schemas()
    trajectory['tools'] = schemas
    tool_manager = ToolManager(EnvTool.from_schemas(env, schemas))
    return harness, env, tool_manager, trajectory


def boot_episodes(tasks: List[Dict[str, Any]],
                  cfg: SandboxConfig) -> List[Tuple[Any, Any, Any, Dict]]:
    """Bring up every rollout's sandbox at once, all-or-nothing.

    Serial boot would dominate the step: a microVM plus ms-agent's import runs
    to seconds, multiplied by ``batch_size x num_generations``.

    All-or-nothing because GRPO groups are positional -- advantages are taken
    over consecutive runs of ``num_generations`` -- so dropping one episode would
    not shrink its group, it would shift every later group onto the wrong task.
    """
    episodes: List[Optional[Tuple[Any, Any, Any, Dict]]] = [None] * len(tasks)
    error: Optional[BaseException] = None
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
        futures = {pool.submit(build_episode, task, cfg): slot
                   for slot, task in enumerate(tasks)}
        for future in as_completed(futures):
            try:
                episodes[futures[future]] = future.result()
            except Exception as e:  # noqa
                error = error or e
    if error is not None:
        for episode in episodes:
            if episode is not None:
                episode[1].close()
        raise RuntimeError(f'sandbox boot failed: {error}') from error
    return episodes  # type: ignore[return-value]


def score_episode(task: Dict[str, Any], env: RemoteMsAgentToolEnv,
                  trajectory: Dict[str, Any], snapshot_dir: str,
                  cfg: SandboxConfig) -> float:
    """Run the task's checks against the state this episode left behind.

    A ``check_script`` is the whole verdict by exit status: no partial credit, no
    judge model, no drift between the run that invented the task and the run
    being scored. Structured checks go through ``run_checks`` instead.
    """
    check_script = task.get('check_script')
    if check_script:
        exit_code, output = env.runner()(check_script, 'python')
        if exit_code != 0:
            logger.debug(f'[{task.get("id")}] check_script failed (exit {exit_code}): '
                         f'{output[-200:]}')
        return 1.0 if exit_code == 0 else 0.0

    final_answer = ''
    for msg in reversed(trajectory.get('messages') or []):
        if msg.get('role') == 'assistant' and (msg.get('content') or '').strip():
            final_answer = msg['content']
            break

    ctx = CheckContext(
        workspace=env.download_workspace(snapshot_dir),
        final_answer=final_answer,
        runner=env.runner(),
    )
    report = run_checks(task['_checks'], ctx, mode=cfg.score_mode)
    if not report.all_passed:
        logger.debug(f'[{task.get("id")}] {report.n_passed}/{report.n_total} checks: '
                     f'{report.failures()}')
    return report.score


def score_episodes(tasks: List[Dict[str, Any]], envs: List[RemoteMsAgentToolEnv],
                   outs: List[Dict[str, Any]], snapshot_root: str,
                   cfg: SandboxConfig) -> List[float]:
    """Score every episode in parallel; a scoring crash costs one reward, not the step.

    Each check is a sandbox round trip, so scoring serially would idle the GPUs
    for as long as booting did. An episode whose sandbox died mid-check scores
    zero, which is also what it would have scored had the checks simply failed.
    """

    def _score(slot: int) -> float:
        snapshot = os.path.join(snapshot_root, f'slot{slot:03d}')
        try:
            return score_episode(tasks[slot], envs[slot], outs[slot], snapshot, cfg)
        except Exception as e:  # noqa
            logger.warning(f'[{snapshot_root} slot {slot}] scoring failed: {e}')
            return 0.0

    with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
        return list(pool.map(_score, range(len(outs))))
