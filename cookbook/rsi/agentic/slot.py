# Copyright (c) ModelScope Contributors. All rights reserved.
"""Build optional world-backed, profiled RSI slots.

TWINKLE_WORLD_ROOT selects a loaded host world. TWINKLE_RSI_PROFILES is a
comma-separated rotation; otherwise TWINKLE_TOPIC_AFFINITIES enables the
host-supported rotation. Per-card switching is not wired here.
"""
import os
from typing import Any, List, Optional, Sequence

from twinkle_agentic.envs import (AgentEnv, Env, HostWorld, LocalEnv, ProfiledEnvMixin, TopicProfiles,
                                  WorldToolsMixin, default_profile_registry)

__all__ = ['build_slot_envs', 'configured_profiles', 'configured_world']


def configured_world() -> Optional[HostWorld]:
    """Load and validate the configured world, or return None."""
    root = os.environ.get('TWINKLE_WORLD_ROOT', '')
    if not root:
        return None
    world = HostWorld(root)
    world.load()
    return world


def _validate_profiles(profile_ids):
    registry = default_profile_registry()
    for profile_id in profile_ids:
        profile = registry.get(profile_id)
        if not profile.available or not profile.enabled_for_new_tasks:
            raise ValueError(f'profile {profile_id!r} is unavailable or retired')
    return registry


def configured_profiles() -> List[str]:
    """Return validated IDs; an explicit list takes precedence over auto rotation."""
    ids = [part.strip() for part in os.environ.get('TWINKLE_RSI_PROFILES', '').split(',') if part.strip()]
    if not ids and os.environ.get('TWINKLE_TOPIC_AFFINITIES', ''):
        ids = list(TopicProfiles().rotation())
    if ids:
        _validate_profiles(ids)
    return ids


def build_slot_envs(*,
                    num_envs: int,
                    sandbox_template: str = '',
                    sandbox_api_url: Optional[str] = None,
                    sandbox_timeout: int = 300,
                    workspace_root: str = '/tmp/twinkle_rsi_slots',
                    pre_tool_call: Any = None,
                    world: Optional[HostWorld] = None,
                    profile_ids: Sequence[str] = (),
                    command_timeout: int = 120) -> List[Env]:
    """Build one environment per job, sharing data but isolating episode state."""
    registry = _validate_profiles(profile_ids) if profile_ids else None
    mixins = ()
    if registry is not None:
        mixins += (ProfiledEnvMixin,)
    if world is not None:
        mixins += (WorldToolsMixin,)
    envs = []
    for index in range(num_envs):
        profile = registry.get(profile_ids[index % len(profile_ids)]) if registry is not None else None
        kwargs = {}
        if world is not None:
            db_enabled = bool(set(profile.allowed_tools) & {'db_query', 'db_schema'}) if profile else world.db_enabled
            kwargs.update(world=world.fork(db_enabled=db_enabled), release_world_on_close=True)
        if registry is not None:
            kwargs['profiles'] = registry
        if sandbox_template:
            env_class = AgentEnv
            kwargs.update(template=sandbox_template, api_url=sandbox_api_url or None,
                          sandbox_timeout=sandbox_timeout, command_timeout=command_timeout,
                          metadata={'run': 'rsi_grpo', 'slot': str(index)})
            if pre_tool_call is not None:
                kwargs['pre_tool_call'] = pre_tool_call
        else:
            env_class = LocalEnv
            kwargs.update(workspace=os.path.join(workspace_root, f'slot_{index}'),
                          command_timeout=command_timeout)
        slot_class = type(f'RsiSlot{index}', mixins + (env_class,), {}) if mixins else env_class
        env = slot_class(**kwargs)
        try:
            if profile is not None:
                env.switch_profile(profile.profile_id)
        except Exception:
            env.close()
            for previous in envs:
                previous.close()
            raise
        envs.append(env)
    return envs
