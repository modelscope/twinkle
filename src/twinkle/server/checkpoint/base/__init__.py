# Copyright (c) ModelScope Contributors. All rights reserved.
"""Base infrastructure for checkpoint and training-run persistence.

Split from the former ``utils/checkpoint_base.py`` (1017 lines, deleted) into
cohesive submodules:

- ``models``               — internal Pydantic base specs + type vars
- ``paths``                — constants, token hashing, save-dir resolution,
                             permission helpers (``validate_user_path``,
                             ``validate_ownership``, ``_resolve_client_save_dir``)
- ``training_run_manager`` — ``BaseFileManager`` + ``BaseTrainingRunManager``
- ``checkpoint_manager``   — ``BaseCheckpointManager``
"""
from .checkpoint_manager import BaseCheckpointManager
from .models import (BaseCheckpoint, BaseCreateModelRequest, BaseLoraConfig, BaseParsedCheckpointPath, BaseTrainingRun,
                     BaseWeightsInfoResponse)
from .paths import (CHECKPOINT_INFO_FILENAME, TRAIN_RUN_INFO_FILENAME, TWINKLE_DEFAULT_SAVE_DIR,
                    _resolve_client_save_dir, validate_ownership, validate_user_path)
from .training_run_manager import BaseFileManager, BaseTrainingRunManager

__all__ = [
    'BaseCheckpoint',
    'BaseTrainingRun',
    'BaseLoraConfig',
    'BaseCreateModelRequest',
    'BaseParsedCheckpointPath',
    'BaseWeightsInfoResponse',
    'BaseFileManager',
    'BaseTrainingRunManager',
    'BaseCheckpointManager',
    'validate_user_path',
    'validate_ownership',
    '_resolve_client_save_dir',
    'TRAIN_RUN_INFO_FILENAME',
    'TWINKLE_DEFAULT_SAVE_DIR',
    'CHECKPOINT_INFO_FILENAME',
]
