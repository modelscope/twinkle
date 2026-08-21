# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import AgentHarness

__all__ = [
    'AgentHarness',
    'MsAgentHarness',
]


def __getattr__(name: str):
    if name == 'MsAgentHarness':
        from .ms_agent import MsAgentHarness
        return MsAgentHarness
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
