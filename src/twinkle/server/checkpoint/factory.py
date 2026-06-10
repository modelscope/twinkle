# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Factory functions for creating checkpoint and training-run manager instances.

Use these functions as the entry point rather than instantiating managers directly:

    from twinkle.server.checkpoint import (
        create_checkpoint_manager,
        create_training_run_manager,
    )

Imports of the concrete Tinker/Twinkle classes are deferred to call time
so that ``__init__.py`` can import this module without triggering the
``__init__ → factory → tinker → __init__`` circular-import chain.
"""


def create_training_run_manager(token: str, client_type: str = 'twinkle'):
    """Create a TrainingRunManager for the given token.

    Args:
        token: User authentication token.
        client_type: 'tinker' or 'twinkle' (default 'twinkle').
    """
    if client_type == 'tinker':
        from .tinker import TinkerTrainingRunManager
        return TinkerTrainingRunManager(token)
    from .twinkle import TwinkleTrainingRunManager
    return TwinkleTrainingRunManager(token)


def create_checkpoint_manager(token: str, client_type: str = 'twinkle'):
    """Create a CheckpointManager for the given token.

    Args:
        token: User authentication token.
        client_type: 'tinker' or 'twinkle' (default 'twinkle').
    """
    if client_type == 'tinker':
        from .tinker import TinkerCheckpointManager, TinkerTrainingRunManager
        run_mgr = TinkerTrainingRunManager(token)
        return TinkerCheckpointManager(token, run_mgr)
    from .twinkle import TwinkleCheckpointManager, TwinkleTrainingRunManager
    run_mgr = TwinkleTrainingRunManager(token)
    return TwinkleCheckpointManager(token, run_mgr)
