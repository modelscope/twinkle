# Copyright (c) ModelScope Contributors. All rights reserved.
"""OpenEnv adapter and distributed EnvPool.

- ``OpenEnv``: embedded adapter that replaces the HTTP server, implements ``Env``
- ``EnvPool``: distributed pool with ``@remote_class`` (Ray worker pool)
- ``EnvPoolAdapter``: wraps a single pool slot as a standard ``Env``
"""
import importlib
import importlib.util
import inspect
import json
import math
import os
from typing import Any, Callable, Dict, List, Optional, Type, Union

from twinkle import DeviceMesh, remote_class, remote_function
from twinkle.utils import get_logger
from .base import Env, StepResult

logger = get_logger()

# ==========================================================================
# Utilities
# ==========================================================================


def _import_env_class(path: str):
    """Import a class from 'module:ClassName' or 'module.ClassName'."""
    if ':' in path:
        module_path, class_name = path.rsplit(':', 1)
    elif '.' in path:
        module_path, class_name = path.rsplit('.', 1)
    else:
        raise ValueError(f"env_cls must be 'module.ClassName' or 'module:ClassName', got {path!r}")
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        missing = getattr(e, 'name', None) or ''
        if missing == module_path or missing == module_path.split('.')[0]:
            raise ModuleNotFoundError(f'Cannot import module {module_path!r}. '
                                      f'Make sure it is installed or on PYTHONPATH.') from e
        raise
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f'Cannot find class {class_name!r} in module {module_path!r}')
    return cls


def _format_observation(obs) -> str:
    """Normalize observation to string."""
    if obs is None:
        return ''
    if isinstance(obs, str):
        return obs
    if isinstance(obs, dict):
        for key in ('result', 'output', 'content', 'text', 'message'):
            if key in obs:
                return str(obs[key])
        try:
            return json.dumps(obs, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(obs)
    if hasattr(obs, 'model_dump'):
        try:
            return json.dumps(obs.model_dump(), ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            pass
    return str(obs)


def _normalize_result(result) -> Dict[str, Any]:
    """Normalize an environment result to a standard dict."""
    if isinstance(result, dict):
        return {
            'observation': _format_observation(result.get('observation', '')),
            'reward': float(result.get('reward', 0.0)),
            'done': bool(result.get('done', False)),
        }
    if hasattr(result, 'observation'):
        return {
            'observation': _format_observation(result.observation),
            'reward': float(getattr(result, 'reward', 0.0) or 0.0),
            'done': bool(getattr(result, 'done', False)),
        }
    return {
        'observation': _format_observation(result),
        'reward': float(getattr(result, 'reward', 0.0) or 0.0),
        'done': bool(getattr(result, 'done', False)),
    }


def _collect_single(results, device_mesh=None):
    for r in results:
        if r is not None:
            return r
    return None


def _collect_batch(results, device_mesh=None):
    merged = []
    for worker_results in results:
        if worker_results:
            merged.extend(worker_results)
    merged.sort(key=lambda x: x[0])
    return [r[1] for r in merged]


# ==========================================================================
# OpenEnv: embedded adapter (replaces HTTP server)
# ==========================================================================


def _discover_openenv_classes(env_name: str):
    """Auto-discover Environment and Action classes from an OpenEnv package."""
    import pkgutil
    try:
        pkg = importlib.import_module(env_name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Cannot import '{env_name}'. Install the environment package:\n"
                                  f"  pip install openenv-{env_name.replace('_', '-')}\n"
                                  f'Or from source:\n'
                                  f'  pip install -e /path/to/OpenEnv/envs/{env_name}')

    action_cls = None
    for name in getattr(pkg, '__all__', dir(pkg)):
        obj = getattr(pkg, name, None)
        if isinstance(obj, type) and name.endswith('Action') and name != 'Action':
            action_cls = obj
            break

    env_cls = None
    try:
        server_pkg = importlib.import_module(f'{env_name}.server')
    except ImportError:
        raise ImportError(f"Cannot import '{env_name}.server'. "
                          f'Make sure the package is correctly installed.')
    for _importer, modname, _ispkg in pkgutil.iter_modules(server_pkg.__path__):
        if modname.endswith('_environment') or modname.endswith('_env'):
            mod = importlib.import_module(f'{env_name}.server.{modname}')
            for attr_name in dir(mod):
                obj = getattr(mod, attr_name)
                if (isinstance(obj, type) and attr_name.endswith('Environment') and attr_name != 'Environment'
                        and attr_name != 'MCPEnvironment'):
                    env_cls = obj
                    break
            if env_cls:
                break

    if env_cls is None:
        raise ImportError(f"Cannot discover Environment class in '{env_name}.server'. "
                          f"Expected a '*Environment' class in a '*_environment.py' module.")
    return env_cls, action_cls


class OpenEnv(Env):
    """Embedded OpenEnv adapter — implements the standard Env interface.

    Usage::

        env = OpenEnv(env_name='openspiel_env', env_kwargs={'game_name': 'blackjack'})
        result = env.reset()
        result = env.step('play', {'action': 'hit'})
    """

    def __init__(self, env_name=None, env_cls=None, env_kwargs=None, action_cls=None, action_mapper=None, **kwargs):
        if env_cls is None and env_name is None:
            raise ValueError("Either 'env_name' or 'env_cls' is required. "
                             "Example: OpenEnv(env_name='openspiel_env', env_kwargs={'game_name': 'blackjack'})")
        if env_name and env_cls is None:
            resolved_cls, discovered_action_cls = _discover_openenv_classes(env_name)
            if action_cls is None:
                action_cls = discovered_action_cls
        else:
            resolved_cls = _import_env_class(env_cls) if isinstance(env_cls, str) else env_cls

        self._env = resolved_cls(**(env_kwargs or {}))
        self._action_cls = _import_env_class(action_cls) if isinstance(action_cls, str) else action_cls
        self._action_mapper = action_mapper

    def reset(self, trajectory=None) -> StepResult:
        result = self._env.reset()
        normalized = _normalize_result(result)
        return StepResult(
            observation=normalized['observation'],
            reward=normalized['reward'],
            done=normalized['done'],
        )

    def step(self, tool_name: str, arguments: Dict[str, Any] = None) -> StepResult:
        if self._action_mapper is not None:
            action = self._action_mapper(tool_name, arguments or {})
        else:
            action = arguments if arguments is not None else {}
        if self._action_cls is not None and isinstance(action, dict):
            action = self._action_cls(**action)
        result = self._env.step(action)
        normalized = _normalize_result(result)
        return StepResult(
            observation=normalized['observation'],
            reward=normalized['reward'],
            done=normalized['done'],
        )

    def close(self) -> None:
        if hasattr(self._env, 'close'):
            self._env.close()


# ==========================================================================
# EnvPool: distributed environment pool (OpenEnv-specific)
# ==========================================================================


@remote_class(execute='all')
class EnvPool:
    """Distributed pool of OpenEnv instances managed as a Twinkle remote_class.

    Environments are sharded across workers. Each worker manages pool_size // num_workers slots.
    """

    def __init__(self,
                 pool_size: int = 32,
                 device_mesh: Optional[DeviceMesh] = None,
                 env_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs):
        self._pool_size = pool_size
        self._env_kwargs = env_kwargs or {}

        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        shard_size = math.ceil(pool_size / world_size)
        self._start = min(rank * shard_size, pool_size)
        self._end = min(self._start + shard_size, pool_size)
        local_size = self._end - self._start

        self._episode_rewards: List[float] = [0.0] * local_size
        self._envs: List[OpenEnv] = [OpenEnv(**self._env_kwargs) for _ in range(local_size)]
        logger.info(f'EnvPool initialized: pool_size={pool_size}, '
                    f'shard=[{self._start}, {self._end}), rank={rank}/{world_size}')

    def _owns(self, idx: int) -> bool:
        return self._start <= idx < self._end

    def _do_reset(self, idx: int) -> Dict[str, Any]:
        local_idx = idx - self._start
        env = self._envs[local_idx]
        self._episode_rewards[local_idx] = 0.0
        result = env.reset()
        return {
            'observation': result.observation,
            'reward': result.reward,
            'done': result.done,
            'episode_reward': 0.0,
        }

    def _do_step(self, idx: int, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        local_idx = idx - self._start
        env = self._envs[local_idx]
        result = env.step(tool_name, arguments)
        self._episode_rewards[local_idx] += result.reward
        return {
            'observation': result.observation,
            'reward': result.reward,
            'done': result.done,
            'episode_reward': self._episode_rewards[local_idx],
        }

    @remote_function(dispatch='all', collect=_collect_single)
    def reset(self, idx: int) -> Dict[str, Any]:
        if not self._owns(idx):
            return None
        return self._do_reset(idx)

    @remote_function(dispatch='all', collect=_collect_single)
    def step(self, idx: int, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not self._owns(idx):
            return None
        return self._do_step(idx, tool_name, arguments)

    @remote_function(dispatch='all', collect=_collect_batch)
    def reset_batch(self, indices: List[int]) -> List:
        results = []
        for i in indices:
            if self._owns(i):
                results.append((i, self._do_reset(i)))
        return results

    @remote_function(dispatch='all', collect=_collect_batch)
    def step_batch(self, indices: List[int], tool_names: List[str], arguments_list: List[Dict[str, Any]]) -> List:
        results = []
        for i, tn, args in zip(indices, tool_names, arguments_list):
            if self._owns(i):
                results.append((i, self._do_step(i, tn, args)))
        return results

    @remote_function(dispatch='all', collect='first')
    def close(self) -> None:
        for env in self._envs:
            env.close()
        self._envs.clear()
        logger.info('EnvPool closed.')

    @property
    def pool_size(self) -> int:
        return self._pool_size

    def get_adapters(self, n: int) -> List['EnvPoolAdapter']:
        """Create n EnvPoolAdapter instances wrapping pool slots [0, n).

        Must be called from the driver side. Each adapter proxies
        reset/step calls to the correct worker via remote_function dispatch.
        """
        if n > self._pool_size:
            raise ValueError(f'Requested {n} adapters but pool only has {self._pool_size} slots.')
        return [EnvPoolAdapter(pool=self, idx=i) for i in range(n)]


# ==========================================================================
# EnvPoolAdapter: wraps a single pool slot as standard Env
# ==========================================================================


class EnvPoolAdapter(Env):
    """Wraps a single slot in an EnvPool as a standard Env."""

    def __init__(self, pool: EnvPool, idx: int):
        self._pool = pool
        self._idx = idx
        self._episode_reward: float = 0.0

    def reset(self, trajectory=None) -> StepResult:
        self._episode_reward = 0.0
        result = self._pool.reset(self._idx)
        return StepResult(observation=result.get('observation', ''), reward=0.0, done=False, info=result)

    def step(self, tool_name: str, arguments: Dict[str, Any] = None) -> StepResult:
        try:
            result = self._pool.step(self._idx, tool_name, arguments or {})
            self._episode_reward = float(result.get('episode_reward', 0.0))
            return StepResult(
                observation=result.get('observation', ''),
                reward=float(result.get('reward', 0.0)),
                done=bool(result.get('done', False)),
                info={'episode_reward': self._episode_reward},
            )
        except Exception as e:
            logger.warning(f'EnvPoolAdapter step error (idx={self._idx}): {e}')
            return StepResult(observation=f'Error: {e}', reward=0.0, done=True, info={'error': str(e)})

    @property
    def episode_reward(self) -> float:
        return self._episode_reward

    def close(self) -> None:
        pass
