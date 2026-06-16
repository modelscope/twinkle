# Copyright (c) ModelScope Contributors. All rights reserved.
"""Checkpoint subsystem (TIER 2 consolidation).

Top-level package consolidating the checkpoint base classes (split from the
former 1017-line ``utils/checkpoint_base.py``, now deleted) with the concrete
Tinker/Twinkle managers and the factory (moved out of ``common/``).

Public surface — import from here:

    from twinkle.server.checkpoint import (
        create_checkpoint_manager, create_training_run_manager,
        BaseCheckpointManager, BaseTrainingRunManager, BaseFileManager,
        validate_user_path, validate_ownership, _resolve_client_save_dir,
        TRAIN_RUN_INFO_FILENAME, TWINKLE_DEFAULT_SAVE_DIR,
    )
"""
from .checkpoint_manager import BaseCheckpointManager
from .factory import create_checkpoint_manager, create_training_run_manager
from .paths import (TRAIN_RUN_INFO_FILENAME, TWINKLE_DEFAULT_SAVE_DIR, _resolve_client_save_dir, validate_ownership,
                    validate_user_path)
from .training_run_manager import BaseFileManager, BaseTrainingRunManager

__all__ = [
    'create_checkpoint_manager',
    'create_training_run_manager',
    'TRAIN_RUN_INFO_FILENAME',
    'TWINKLE_DEFAULT_SAVE_DIR',
    'BaseCheckpointManager',
    'BaseFileManager',
    'BaseTrainingRunManager',
    'validate_user_path',
    'validate_ownership',
    '_resolve_client_save_dir',
]
