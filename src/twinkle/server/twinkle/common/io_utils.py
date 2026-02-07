# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Twinkle-specific IO utilities for managing training runs and checkpoints.

This module extends the base IO utilities with Twinkle-specific implementations.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from twinkle.server.utils.io_utils import (
    TWINKLE_DEFAULT_SAVE_DIR,
    TRAIN_RUN_INFO_FILENAME,
    CHECKPOINT_INFO_FILENAME,
    Cursor,
    BaseCheckpoint,
    BaseTrainingRun,
    BaseLoraConfig,
    BaseCreateModelRequest,
    BaseParsedCheckpointPath,
    BaseWeightsInfoResponse,
    BaseTrainingRunManager,
    BaseCheckpointManager,
    ResolvedLoadPath,
    validate_user_path,
    validate_ownership,
)

# ----- Twinkle-specific Pydantic Models -----

class Checkpoint(BaseCheckpoint):
    """Twinkle checkpoint model."""
    twinkle_path: str


class TrainingRun(BaseTrainingRun):
    """Twinkle training run model."""
    pass


class TrainingRunsResponse(BaseModel):
    training_runs: List[TrainingRun]
    cursor: Cursor


class CheckpointsListResponse(BaseModel):
    checkpoints: List[Checkpoint]
    cursor: Optional[Cursor] = None


class ParsedCheckpointTwinklePath(BaseParsedCheckpointPath):
    """Twinkle-specific parsed path model."""
    twinkle_path: str


class WeightsInfoResponse(BaseWeightsInfoResponse):
    """Twinkle weights info response."""
    pass


class LoraConfig(BaseLoraConfig):
    """Twinkle LoRA configuration."""
    pass


class CreateModelRequest(BaseCreateModelRequest):
    """Twinkle create model request."""
    lora_config: Optional[LoraConfig] = None


# ----- Twinkle Training Run Manager -----

class TrainingRunManager(BaseTrainingRunManager):
    """Twinkle-specific training run manager."""
    
    @property
    def train_run_info_filename(self) -> str:
        return TRAIN_RUN_INFO_FILENAME
    
    def _create_training_run(self, model_id: str, run_config: CreateModelRequest) -> Dict[str, Any]:
        """Create training run data from model_id and run_config."""
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
            user_metadata=run_config.user_metadata
        )

        new_data = train_run_data.model_dump(mode='json')
        # Store lora config details separately if needed
        if lora_config:
            new_data['train_unembed'] = lora_config.train_unembed
            new_data['train_mlp'] = lora_config.train_mlp
            new_data['train_attn'] = lora_config.train_attn

        return new_data
    
    def _parse_training_run(self, data: Dict[str, Any]) -> TrainingRun:
        """Parse training run data into TrainingRun model."""
        return TrainingRun(**data)
    
    def _create_training_runs_response(
        self, runs: List[TrainingRun], limit: int, offset: int, total: int
    ) -> TrainingRunsResponse:
        """Create a training runs response."""
        return TrainingRunsResponse(
            training_runs=runs,
            cursor=Cursor(limit=limit, offset=offset, total_count=total)
        )
    
    def get_with_permission(self, model_id: str) -> Optional[TrainingRun]:
        """
        Get training run with ownership validation.
        
        Args:
            model_id: The model identifier
            
        Returns:
            TrainingRun if found and owned by user, None otherwise
        """
        run = self.get(model_id)
        if run and validate_ownership(self.token, run.model_owner):
            return run
        return None


# ----- Twinkle Checkpoint Manager -----

class CheckpointManager(BaseCheckpointManager):
    """Twinkle-specific checkpoint manager."""
    
    @property
    def path_prefix(self) -> str:
        return "twinkle://"
    
    @property
    def path_field_name(self) -> str:
        return "twinkle_path"
    
    def _create_checkpoint(
        self, checkpoint_id: str, checkpoint_type: str, 
        path: str, size_bytes: int, public: bool,
        base_model: Optional[str] = None,
        is_lora: bool = False,
        lora_rank: Optional[int] = None,
        train_unembed: Optional[bool] = None,
        train_mlp: Optional[bool] = None,
        train_attn: Optional[bool] = None,
        user_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create checkpoint data."""
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
            user_metadata=user_metadata
        )
        return checkpoint.model_dump(mode='json')
    
    def _parse_checkpoint(self, data: Dict[str, Any]) -> Checkpoint:
        """Parse checkpoint data into Checkpoint model."""
        data = data.copy()
        # Transform tinker_path to twinkle_path if needed
        if 'tinker_path' in data and 'twinkle_path' not in data:
            data['twinkle_path'] = data.pop('tinker_path')
        elif 'twinkle_path' not in data and 'path' in data:
            data['twinkle_path'] = data.pop('path')
        return Checkpoint(**data)
    
    def get(self, model_id: str, checkpoint_id: str) -> Optional[Checkpoint]:
        """
        Get checkpoint metadata with backwards compatibility.
        
        Args:
            model_id: The model identifier
            checkpoint_id: The checkpoint identifier
            
        Returns:
            Checkpoint object or None if not found
        """
        data = self._read_ckpt_info(model_id, checkpoint_id)
        if not data:
            return None
        # Handle backwards compatibility: construct twinkle_path if missing
        if 'twinkle_path' not in data and 'tinker_path' not in data and 'path' not in data:
            if 'checkpoint_id' in data:
                data = data.copy()
                data['twinkle_path'] = f"{self.path_prefix}{model_id}/{data['checkpoint_id']}"
        return self._parse_checkpoint(data)
    
    def _create_checkpoints_response(self, checkpoints: List[Checkpoint]) -> CheckpointsListResponse:
        """Create a checkpoints list response."""
        return CheckpointsListResponse(checkpoints=checkpoints, cursor=None)
    
    def _create_parsed_path(
        self, path: str, training_run_id: str, 
        checkpoint_type: str, checkpoint_id: str
    ) -> ParsedCheckpointTwinklePath:
        """Create a parsed path model."""
        return ParsedCheckpointTwinklePath(
            path=path,
            twinkle_path=path,
            training_run_id=training_run_id,
            checkpoint_type=checkpoint_type,
            checkpoint_id=checkpoint_id,
        )
    
    def _create_weights_info(self, run_info: Dict[str, Any]) -> WeightsInfoResponse:
        """Create weights info from run info."""
        return WeightsInfoResponse(
            training_run_id=run_info.get('training_run_id', ''),
            base_model=run_info.get('base_model', ''),
            model_owner=run_info.get('model_owner', ''),
            is_lora=run_info.get('is_lora', False),
            lora_rank=run_info.get('lora_rank'),
        )
    
    def parse_twinkle_path(self, twinkle_path: str) -> Optional[ParsedCheckpointTwinklePath]:
        """Parse a twinkle:// path into its components (alias for parse_path)."""
        return self.parse_path(twinkle_path)


# ----- Factory Functions -----

def create_training_run_manager(token: str) -> TrainingRunManager:
    """Create a TrainingRunManager for the given token."""
    return TrainingRunManager(token)


def create_checkpoint_manager(token: str) -> CheckpointManager:
    """Create a CheckpointManager for the given token."""
    training_run_manager = TrainingRunManager(token)
    return CheckpointManager(token, training_run_manager)


