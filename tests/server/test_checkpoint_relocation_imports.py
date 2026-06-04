# Copyright (c) ModelScope Contributors. All rights reserved.
"""Import tests for the TIER 2 checkpoint consolidation (R13.2, R13.8).

Asserts the new ``twinkle.server.checkpoint`` public surface and its
``base/`` submodules resolve, that the deleted ``utils.checkpoint_base`` path
and the removed ``common``/``utils`` re-exports no longer resolve, and that the
slimmed ``common`` package still exposes its retained helpers. No shim is used —
every relocation rewrites importers to the new path.
"""
from __future__ import annotations

import importlib

import pytest


def test_checkpoint_public_surface_resolves() -> None:
    mod = importlib.import_module('twinkle.server.checkpoint')
    for name in (
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
    ):
        assert hasattr(mod, name), f'missing checkpoint public-API symbol: {name}'


def test_checkpoint_base_submodules_resolve() -> None:
    models = importlib.import_module('twinkle.server.checkpoint.base.models')
    paths = importlib.import_module('twinkle.server.checkpoint.base.paths')
    trm = importlib.import_module('twinkle.server.checkpoint.base.training_run_manager')
    cpm = importlib.import_module('twinkle.server.checkpoint.base.checkpoint_manager')

    assert hasattr(models, 'BaseCheckpoint')
    assert hasattr(models, 'BaseTrainingRun')
    assert hasattr(paths, 'TRAIN_RUN_INFO_FILENAME')
    assert hasattr(paths, 'validate_user_path')
    assert hasattr(paths, '_resolve_client_save_dir')
    assert hasattr(trm, 'BaseFileManager')
    assert hasattr(trm, 'BaseTrainingRunManager')
    assert hasattr(cpm, 'BaseCheckpointManager')


def test_old_checkpoint_base_path_is_gone() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('twinkle.server.utils.checkpoint_base')


def test_old_common_checkpoint_modules_are_gone() -> None:
    for old in (
            'twinkle.server.common.checkpoint_factory',
            'twinkle.server.common.tinker_checkpoint',
            'twinkle.server.common.twinkle_checkpoint',
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(old)


def test_removed_reexports_are_gone() -> None:
    utils = importlib.import_module('twinkle.server.utils')
    for name in ('BaseCheckpointManager', 'BaseTrainingRunManager', 'BaseFileManager', 'TRAIN_RUN_INFO_FILENAME',
                 'TWINKLE_DEFAULT_SAVE_DIR'):
        assert not hasattr(utils, name), f'utils still re-exports {name}'

    common = importlib.import_module('twinkle.server.common')
    assert not hasattr(common, 'create_checkpoint_manager')
    assert not hasattr(common, 'create_training_run_manager')


def test_common_retains_datum_and_router() -> None:
    common = importlib.import_module('twinkle.server.common')
    assert hasattr(common, 'input_feature_to_datum')
    assert hasattr(common, 'StickyLoraRequestRouter')
