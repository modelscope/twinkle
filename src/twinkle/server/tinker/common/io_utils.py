# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tinker-specific IO utilities for managing training runs and checkpoints.

This module extends the base IO utilities with Tinker-specific implementations.
It uses types from the tinker package for compatibility with the Tinker API.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from tinker import types

from twinkle.server.utils.io_utils import (
    TWINKLE_DEFAULT_SAVE_DIR,
    CHECKPOINT_INFO_FILENAME,
    TRAIN_RUN_INFO_FILENAME,
    BaseTrainingRunManager,
    BaseCheckpointManager,
    ResolvedLoadPath,
    validate_user_path,
    validate_ownership,
)


# ----- Tinker Training Run Manager -----

class TrainingRunManager(BaseTrainingRunManager):
    """Tinker-specific training run manager using tinker.types models."""
    
    @property
    def train_run_info_filename(self) -> str:
        return TRAIN_RUN_INFO_FILENAME
    
    def _create_training_run(self, model_id: str, run_config: types.CreateModelRequest) -> Dict[str, Any]:
        """Create training run data from model_id and run_config."""
        lora_config = run_config.lora_config
        train_run_data = types.TrainingRun(
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
    
    def _parse_training_run(self, data: Dict[str, Any]) -> types.TrainingRun:
        """Parse training run data into TrainingRun model."""
        # Transform checkpoint data to ensure tinker_path field exists
        data = self._transform_checkpoint_fields(data)
        return types.TrainingRun(**data)
    
    def _transform_checkpoint_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform checkpoint data to ensure compatibility with tinker types.
        
        Handles cases where:
        - last_checkpoint/last_sampler_checkpoint might have twinkle_path instead of tinker_path
        - Missing path field that needs to be constructed from other data
        """
        data = data.copy()
        for field in ['last_checkpoint', 'last_sampler_checkpoint']:
            if field in data and data[field] is not None:
                ckpt = data[field].copy()
                # If twinkle_path exists but tinker_path doesn't, use twinkle_path
                if 'twinkle_path' in ckpt and 'tinker_path' not in ckpt:
                    ckpt['tinker_path'] = ckpt.pop('twinkle_path')
                # If neither exists, try to construct from checkpoint_id
                elif 'tinker_path' not in ckpt:
                    # Try to get path from any available path field
                    path = ckpt.get('path') or ckpt.get('twinkle_path')
                    if path:
                        ckpt['tinker_path'] = path
                    elif 'checkpoint_id' in ckpt and 'training_run_id' in data:
                        # Construct path from components
                        ckpt['tinker_path'] = f"twinkle://{data['training_run_id']}/{ckpt['checkpoint_id']}"
                data[field] = ckpt
        return data
    
    def _create_training_runs_response(
        self, runs: List[types.TrainingRun], limit: int, offset: int, total: int
    ) -> types.TrainingRunsResponse:
        """Create a training runs response."""
        return types.TrainingRunsResponse(
            training_runs=runs,
            cursor=types.Cursor(limit=limit, offset=offset, total_count=total)
        )


# ----- Tinker Checkpoint Manager -----

class CheckpointManager(BaseCheckpointManager):
    """Tinker-specific checkpoint manager using tinker.types models."""
    
    @property
    def path_prefix(self) -> str:
        return "twinkle://"
    
    @property
    def path_field_name(self) -> str:
        return "tinker_path"
    
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
        # Create base checkpoint using tinker types
        checkpoint = types.Checkpoint(
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            time=datetime.now(),
            tinker_path=path,
            size_bytes=size_bytes,
            public=public
        )
        result = checkpoint.model_dump(mode='json')
        
        # Add training run info fields (may not be supported by external types.Checkpoint)
        result['base_model'] = base_model
        result['is_lora'] = is_lora
        result['lora_rank'] = lora_rank
        result['train_unembed'] = train_unembed
        result['train_mlp'] = train_mlp
        result['train_attn'] = train_attn
        result['user_metadata'] = user_metadata
        
        return result
    
    def _parse_checkpoint(self, data: Dict[str, Any]) -> types.Checkpoint:
        """Parse checkpoint data into Checkpoint model."""
        data = data.copy()
        # Transform twinkle_path to tinker_path if needed
        if 'twinkle_path' in data and 'tinker_path' not in data:
            data['tinker_path'] = data.pop('twinkle_path')
        elif 'tinker_path' not in data and 'path' in data:
            data['tinker_path'] = data.pop('path')
        return types.Checkpoint(**data)
    
    def _create_checkpoints_response(self, checkpoints: List[types.Checkpoint]) -> types.CheckpointsListResponse:
        """Create a checkpoints list response."""
        return types.CheckpointsListResponse(checkpoints=checkpoints, cursor=None)
    
    def _create_parsed_path(
        self, path: str, training_run_id: str, 
        checkpoint_type: str, checkpoint_id: str
    ) -> types.ParsedCheckpointTinkerPath:
        """Create a parsed path model."""
        return types.ParsedCheckpointTinkerPath(
            tinker_path=path,
            training_run_id=training_run_id,
            checkpoint_type=checkpoint_type,
            checkpoint_id=checkpoint_id,
        )
    
    def _create_weights_info(self, run_info: Dict[str, Any]) -> types.WeightsInfoResponse:
        """Create weights info from run info."""
        return types.WeightsInfoResponse(**run_info)
    
    def parse_tinker_path(self, tinker_path: str) -> Optional[types.ParsedCheckpointTinkerPath]:
        """Parse a twinkle:// path into its components (alias for parse_path)."""
        return self.parse_path(tinker_path)


# ----- Factory Functions -----

def create_training_run_manager(token: str) -> TrainingRunManager:
    """Create a TrainingRunManager for the given token."""
    return TrainingRunManager(token)


def create_checkpoint_manager(token: str) -> CheckpointManager:
    """Create a CheckpointManager for the given token."""
    training_run_manager = TrainingRunManager(token)
    return CheckpointManager(token, training_run_manager)

