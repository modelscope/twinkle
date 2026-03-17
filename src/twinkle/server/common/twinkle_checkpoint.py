# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Twinkle-specific checkpoint and training-run managers.

Uses ``twinkle_client.types.training`` models for all serialization and response construction.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from twinkle.server.utils.checkpoint_base import (TRAIN_RUN_INFO_FILENAME, BaseCheckpointManager,
                                                  BaseTrainingRunManager, validate_ownership)
from twinkle_client.types.training import (Checkpoint, CheckpointsListResponse, CreateModelRequest, Cursor,
                                           ParsedCheckpointTwinklePath, TrainingRun, TrainingRunsResponse,
                                           WeightsInfoResponse)


class TwinkleTrainingRunManager(BaseTrainingRunManager):
    """Twinkle-specific training run manager."""

    @property
    def train_run_info_filename(self) -> str:
        return TRAIN_RUN_INFO_FILENAME

    def _create_training_run(self, model_id: str, run_config: CreateModelRequest) -> Dict[str, Any]:
        lora_config = run_config.lora_config
        train_run_data = TrainingRun(
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

    def _parse_training_run(self, data: Dict[str, Any]) -> TrainingRun:
        return TrainingRun(**data)

    def _create_training_runs_response(self, runs: List[TrainingRun], limit: int, offset: int,
                                       total: int) -> TrainingRunsResponse:
        return TrainingRunsResponse(training_runs=runs, cursor=Cursor(limit=limit, offset=offset, total_count=total))

    def get_with_permission(self, model_id: str) -> Optional[TrainingRun]:
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
        checkpoint = Checkpoint(
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

    def _parse_checkpoint(self, data: Dict[str, Any]) -> Checkpoint:
        data = data.copy()
        if 'tinker_path' in data and 'twinkle_path' not in data:
            data['twinkle_path'] = data.pop('tinker_path')
        elif 'twinkle_path' not in data and 'path' in data:
            data['twinkle_path'] = data.pop('path')
        return Checkpoint(**data)

    def get(self, model_id: str, checkpoint_id: str) -> Optional[Checkpoint]:
        data = self._read_ckpt_info(model_id, checkpoint_id)
        if not data:
            return None
        if 'twinkle_path' not in data and 'tinker_path' not in data and 'path' not in data:
            if 'checkpoint_id' in data:
                data = data.copy()
                data['twinkle_path'] = f"{self.path_prefix}{model_id}/{data['checkpoint_id']}"
        return self._parse_checkpoint(data)

    def _create_checkpoints_response(self, checkpoints: List[Checkpoint]) -> CheckpointsListResponse:
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
