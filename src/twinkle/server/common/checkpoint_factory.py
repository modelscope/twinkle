# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Factory functions for creating checkpoint and training-run manager instances.

Use these functions as the entry point rather than instantiating managers directly:

    from twinkle.server.common.checkpoint_factory import (
        create_checkpoint_manager,
        create_training_run_manager,
    )
"""
from twinkle.server.common.tinker_checkpoint import TinkerCheckpointManager, TinkerTrainingRunManager
from twinkle.server.common.twinkle_checkpoint import TwinkleCheckpointManager, TwinkleTrainingRunManager


def create_training_run_manager(token: str, client_type: str = 'twinkle'):
    """Create a TrainingRunManager for the given token.

    Args:
        token: User authentication token.
        client_type: 'tinker' or 'twinkle' (default 'twinkle').
    """
    if client_type == 'tinker':
        return TinkerTrainingRunManager(token)
    return TwinkleTrainingRunManager(token)


def create_checkpoint_manager(token: str, client_type: str = 'twinkle'):
    """Create a CheckpointManager for the given token.

    Args:
        token: User authentication token.
        client_type: 'tinker' or 'twinkle' (default 'twinkle').
    """
    if client_type == 'tinker':
        run_mgr = TinkerTrainingRunManager(token)
        return TinkerCheckpointManager(token, run_mgr)
    run_mgr = TwinkleTrainingRunManager(token)
    return TwinkleCheckpointManager(token, run_mgr)
