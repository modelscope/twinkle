# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Tinker-specific checkpoint and training-run managers.

Uses ``tinker.types`` models for all serialization and response construction.
"""
from datetime import datetime
from tinker import types as tinker_types
from typing import Any, Dict, List, Optional

from twinkle.server.utils.checkpoint_base import TRAIN_RUN_INFO_FILENAME, BaseCheckpointManager, BaseTrainingRunManager


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
