# Copyright (c) ModelScope Contributors. All rights reserved.
import sys

from twinkle._lazy_module import _LazyModule

sys.modules[__name__] = _LazyModule(
    __name__,
    __file__, {
        'backend': ['create_backend'],
        'base': ['BaseManager'],
        'cleanup_coordinator': ['ResourceCleanupCoordinator'],
        'count_publisher': ['ResourceCountPublisher'],
        'future_manager': ['FutureManager'],
        'model_manager': ['ModelManager'],
        'models': ['FutureRecord', 'ModelRecord', 'SamplingSessionRecord', 'SessionRecord'],
        'replica_registry': ['ReplicaRegistry'],
        'sampling_manager': ['SamplingSessionManager'],
        'server_state': ['ServerState', 'get_server_state', 'reset_server_state_cache'],
        'session_manager': ['SessionManager'],
    },
    module_spec=__spec__)

sys.modules[__name__].__all__ = [
    # Pydantic record models
    'SessionRecord',
    'ModelRecord',
    'SamplingSessionRecord',
    'FutureRecord',
    # Base
    'BaseManager',
    # Resource managers
    'SessionManager',
    'ModelManager',
    'SamplingSessionManager',
    'FutureManager',
    # Server state
    'ServerState',
    'ReplicaRegistry',
    'ResourceCleanupCoordinator',
    'ResourceCountPublisher',
    'get_server_state',
    # Persistence backend factory
    'create_backend',
]
