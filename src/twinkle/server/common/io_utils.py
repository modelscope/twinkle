# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Unified IO utilities for managing training runs and checkpoints.

Merges tinker/common/io_utils.py and twinkle/common/io_utils.py.
Both client-type implementations share the same underlying base classes;
factory functions accept a ``client_type`` parameter ('tinker' or 'twinkle').

Pydantic models that need to be shared with the client live in
``twinkle_client.types.training``.
"""
from datetime import datetime
from tinker import types as tinker_types
from typing import Any, Dict, List, Optional

from twinkle.server.utils.io_utils import (CHECKPOINT_INFO_FILENAME, TRAIN_RUN_INFO_FILENAME, TWINKLE_DEFAULT_SAVE_DIR,
                                           BaseCheckpoint, BaseCheckpointManager, BaseCreateModelRequest,
                                           BaseLoraConfig, BaseParsedCheckpointPath, BaseTrainingRun,
                                           BaseTrainingRunManager, BaseWeightsInfoResponse, Cursor, ResolvedLoadPath,
                                           validate_ownership, validate_user_path)
# Re-export twinkle-native pydantic models from twinkle_client.types
from twinkle_client.types.training import Checkpoint as TwinkleCheckpoint
from twinkle_client.types.training import (CheckpointsListResponse, CreateModelRequest, LoraConfig,
                                           ParsedCheckpointTwinklePath)
from twinkle_client.types.training import TrainingRun as TwinkleTrainingRun
from twinkle_client.types.training import TrainingRunsResponse, WeightsInfoResponse

__all__ = [
    'create_checkpoint_manager',
    'create_training_run_manager',
    'validate_user_path',
    'validate_ownership',
    'ResolvedLoadPath',
    'Cursor',
    # Twinkle-native models (re-exported for convenience)
    'TwinkleCheckpoint',
    'TwinkleTrainingRun',
    'TrainingRunsResponse',
    'CheckpointsListResponse',
    'WeightsInfoResponse',
    'LoraConfig',
    'CreateModelRequest',
    'ParsedCheckpointTwinklePath',
]

# ---------------------------------------------------------------------------
# Tinker-specific managers (use tinker.types for model instances)
# ---------------------------------------------------------------------------


class TinkerTrainingRunManager(BaseTrainingRunManager):
    """Tinker-specific training run manager using tinker.types models."""

    @property
    def train_run_info_filename(self) -> str:
        return TRAIN_RUN_INFO_FILENAME

    def _create_training_run(self, model_id: str, run_config: tinker_types.CreateModelRequest) -> Dict[str, Any]:
        lora_config = run_config.lora_config
        train_run_data = tinker_types.TrainingRun(
            training_run_id=model_id,
            base_model=run_config.base_model,
            model_owner=self.token,
            is_lora=True if lora_config else False,
            corrupted=False,
            lora_rank=lora_config.rank if lora_config else None,
            last_request_time=datetime.now(),
            last_checkpoint=None,
            last_sampler_checkpoint=None,
            user_metadata=run_config.user_metadata)

        new_data = train_run_data.model_dump(mode='json')
        if lora_config:
            new_data['train_unembed'] = lora_config.train_unembed
            new_data['train_mlp'] = lora_config.train_mlp
            new_data['train_attn'] = lora_config.train_attn
        return new_data

    def _parse_training_run(self, data: Dict[str, Any]) -> tinker_types.TrainingRun:
        data = self._transform_checkpoint_fields(data)
        return tinker_types.TrainingRun(**data)

    def _transform_checkpoint_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = data.copy()
        for field in ['last_checkpoint', 'last_sampler_checkpoint']:
            if field in data and data[field] is not None:
                ckpt = data[field].copy()
                if 'twinkle_path' in ckpt and 'tinker_path' not in ckpt:
                    ckpt['tinker_path'] = ckpt.pop('twinkle_path')
                elif 'tinker_path' not in ckpt:
                    path = ckpt.get('path') or ckpt.get('twinkle_path')
                    if path:
                        ckpt['tinker_path'] = path
                    elif 'checkpoint_id' in ckpt and 'training_run_id' in data:
                        ckpt['tinker_path'] = f"twinkle://{data['training_run_id']}/{ckpt['checkpoint_id']}"
                data[field] = ckpt
        return data

    def _create_training_runs_response(self, runs: List[tinker_types.TrainingRun], limit: int, offset: int,
                                       total: int) -> tinker_types.TrainingRunsResponse:
        return tinker_types.TrainingRunsResponse(
            training_runs=runs, cursor=tinker_types.Cursor(limit=limit, offset=offset, total_count=total))


class TinkerCheckpointManager(BaseCheckpointManager):
    """Tinker-specific checkpoint manager using tinker.types models."""

    @property
    def path_prefix(self) -> str:
        return 'twinkle://'

    @property
    def path_field_name(self) -> str:
        return 'tinker_path'

    def _create_checkpoint(self,
                           checkpoint_id,
                           checkpoint_type,
                           path,
                           size_bytes,
                           public,
                           base_model=None,
                           is_lora=False,
                           lora_rank=None,
                           train_unembed=None,
                           train_mlp=None,
                           train_attn=None,
                           user_metadata=None) -> Dict[str, Any]:
        checkpoint = tinker_types.Checkpoint(
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            time=datetime.now(),
            tinker_path=path,
            size_bytes=size_bytes,
            public=public)
        result = checkpoint.model_dump(mode='json')
        result['base_model'] = base_model
        result['is_lora'] = is_lora
        result['lora_rank'] = lora_rank
        result['train_unembed'] = train_unembed
        result['train_mlp'] = train_mlp
        result['train_attn'] = train_attn
        result['user_metadata'] = user_metadata
        return result

    def _parse_checkpoint(self, data: Dict[str, Any]) -> tinker_types.Checkpoint:
        data = data.copy()
        if 'twinkle_path' in data and 'tinker_path' not in data:
            data['tinker_path'] = data.pop('twinkle_path')
        elif 'tinker_path' not in data and 'path' in data:
            data['tinker_path'] = data.pop('path')
        return tinker_types.Checkpoint(**data)

    def _create_checkpoints_response(
            self, checkpoints: List[tinker_types.Checkpoint]) -> tinker_types.CheckpointsListResponse:
        return tinker_types.CheckpointsListResponse(checkpoints=checkpoints, cursor=None)

    def _create_parsed_path(self, path, training_run_id, checkpoint_type,
                            checkpoint_id) -> tinker_types.ParsedCheckpointTinkerPath:
        return tinker_types.ParsedCheckpointTinkerPath(
            tinker_path=path,
            training_run_id=training_run_id,
            checkpoint_type=checkpoint_type,
            checkpoint_id=checkpoint_id,
        )

    def _create_weights_info(self, run_info: Dict[str, Any]) -> tinker_types.WeightsInfoResponse:
        return tinker_types.WeightsInfoResponse(**run_info)

    def parse_tinker_path(self, tinker_path: str) -> Optional[tinker_types.ParsedCheckpointTinkerPath]:
        return self.parse_path(tinker_path)


# ---------------------------------------------------------------------------
# Twinkle-specific managers (use twinkle_client.types.training models)
# ---------------------------------------------------------------------------


class TwinkleTrainingRunManager(BaseTrainingRunManager):
    """Twinkle-specific training run manager."""

    @property
    def train_run_info_filename(self) -> str:
        return TRAIN_RUN_INFO_FILENAME

    def _create_training_run(self, model_id: str, run_config: CreateModelRequest) -> Dict[str, Any]:
        lora_config = run_config.lora_config
        train_run_data = TwinkleTrainingRun(
            training_run_id=model_id,
            base_model=run_config.base_model,
            model_owner=self.token,
            is_lora=True if lora_config else False,
            corrupted=False,
            lora_rank=lora_config.rank if lora_config else None,
            last_request_time=datetime.now(),
            last_checkpoint=None,
            last_sampler_checkpoint=None,
            user_metadata=run_config.user_metadata)

        new_data = train_run_data.model_dump(mode='json')
        if lora_config:
            new_data['train_unembed'] = lora_config.train_unembed
            new_data['train_mlp'] = lora_config.train_mlp
            new_data['train_attn'] = lora_config.train_attn
        return new_data

    def _parse_training_run(self, data: Dict[str, Any]) -> TwinkleTrainingRun:
        return TwinkleTrainingRun(**data)

    def _create_training_runs_response(self, runs: List[TwinkleTrainingRun], limit: int, offset: int,
                                       total: int) -> TrainingRunsResponse:
        return TrainingRunsResponse(training_runs=runs, cursor=Cursor(limit=limit, offset=offset, total_count=total))

    def get_with_permission(self, model_id: str) -> Optional[TwinkleTrainingRun]:
        run = self.get(model_id)
        if run and validate_ownership(self.token, run.model_owner):
            return run
        return None


class TwinkleCheckpointManager(BaseCheckpointManager):
    """Twinkle-specific checkpoint manager."""

    @property
    def path_prefix(self) -> str:
        return 'twinkle://'

    @property
    def path_field_name(self) -> str:
        return 'twinkle_path'

    def _create_checkpoint(self,
                           checkpoint_id,
                           checkpoint_type,
                           path,
                           size_bytes,
                           public,
                           base_model=None,
                           is_lora=False,
                           lora_rank=None,
                           train_unembed=None,
                           train_mlp=None,
                           train_attn=None,
                           user_metadata=None) -> Dict[str, Any]:
        checkpoint = TwinkleCheckpoint(
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            time=datetime.now(),
            twinkle_path=path,
            size_bytes=size_bytes,
            public=public,
            base_model=base_model,
            is_lora=is_lora,
            lora_rank=lora_rank,
            train_unembed=train_unembed,
            train_mlp=train_mlp,
            train_attn=train_attn,
            user_metadata=user_metadata)
        return checkpoint.model_dump(mode='json')

    def _parse_checkpoint(self, data: Dict[str, Any]) -> TwinkleCheckpoint:
        data = data.copy()
        if 'tinker_path' in data and 'twinkle_path' not in data:
            data['twinkle_path'] = data.pop('tinker_path')
        elif 'twinkle_path' not in data and 'path' in data:
            data['twinkle_path'] = data.pop('path')
        return TwinkleCheckpoint(**data)

    def get(self, model_id: str, checkpoint_id: str) -> Optional[TwinkleCheckpoint]:
        data = self._read_ckpt_info(model_id, checkpoint_id)
        if not data:
            return None
        if 'twinkle_path' not in data and 'tinker_path' not in data and 'path' not in data:
            if 'checkpoint_id' in data:
                data = data.copy()
                data['twinkle_path'] = f"{self.path_prefix}{model_id}/{data['checkpoint_id']}"
        return self._parse_checkpoint(data)

    def _create_checkpoints_response(self, checkpoints: List[TwinkleCheckpoint]) -> CheckpointsListResponse:
        return CheckpointsListResponse(checkpoints=checkpoints, cursor=None)

    def _create_parsed_path(self, path, training_run_id, checkpoint_type, checkpoint_id) -> ParsedCheckpointTwinklePath:
        return ParsedCheckpointTwinklePath(
            path=path,
            twinkle_path=path,
            training_run_id=training_run_id,
            checkpoint_type=checkpoint_type,
            checkpoint_id=checkpoint_id,
        )

    def _create_weights_info(self, run_info: Dict[str, Any]) -> WeightsInfoResponse:
        return WeightsInfoResponse(
            training_run_id=run_info.get('training_run_id', ''),
            base_model=run_info.get('base_model', ''),
            model_owner=run_info.get('model_owner', ''),
            is_lora=run_info.get('is_lora', False),
            lora_rank=run_info.get('lora_rank'),
        )

    def parse_twinkle_path(self, twinkle_path: str) -> Optional[ParsedCheckpointTwinklePath]:
        return self.parse_path(twinkle_path)


# ---------------------------------------------------------------------------
# Unified factory functions
# ---------------------------------------------------------------------------


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
