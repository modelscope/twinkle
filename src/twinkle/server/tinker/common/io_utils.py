from datetime import datetime
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from tinker import types

TWINKLE_DEFAULT_SAVE_DIR = os.environ.get('TWINKLE_DEFAULT_SAVE_DIR',
                                          './outputs')

TRAIN_RUN_INFO_FILENAME = 'tinker_metadata.json'
CHECKPOINT_INFO_FILENAME = 'checkpoint_metadata.json'


class FileManager:
    @staticmethod
    def get_dir_size(path: Path) -> int:
        total = 0
        if path.exists():
            for p in path.rglob('*'):
                if p.is_file():
                    total += p.stat().st_size
        return total


class TrainingRunManager(FileManager):

    @staticmethod
    def get_base_dir(token: Optional[str] = None) -> Path:
        """
        Get base directory with optional token-based isolation.
        
        Args:
            token: User's authentication token for directory isolation
            
        Returns:
            Path to base directory, optionally scoped by token
        """
        base_path = Path(TWINKLE_DEFAULT_SAVE_DIR)
        if token:
            # Create token-based subdirectory for user isolation
            # Sanitize token to avoid filesystem issues
            sanitized_token = re.sub(r'[^\w\-]', '_', token)
            return base_path / sanitized_token
        return base_path

    @staticmethod
    def get_model_dir(model_id: str, token: Optional[str] = None) -> Path:
        """
        Get model directory with optional token-based isolation.
        
        Args:
            model_id: The model identifier
            token: User's authentication token for directory isolation
            
        Returns:
            Path to model directory
        """
        return TrainingRunManager.get_base_dir(token) / model_id

    @staticmethod
    def _read_info(model_id: str, token: Optional[str] = None) -> Dict[str, Any]:
        """
        Read training run metadata from disk.
        
        Args:
            model_id: The model identifier
            token: User's authentication token for directory isolation
            
        Returns:
            Dictionary with metadata or empty dict if not found
        """
        metadata_path = TrainingRunManager.get_model_dir(
            model_id, token) / TRAIN_RUN_INFO_FILENAME
        if not metadata_path.exists():
            return {}
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    @staticmethod
    def _write_info(model_id: str, data: Dict[str, Any], token: Optional[str] = None):
        """
        Write training run metadata to disk.
        
        Args:
            model_id: The model identifier
            data: Metadata to write
            token: User's authentication token for directory isolation
        """
        model_dir = TrainingRunManager.get_model_dir(model_id, token)
        model_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = model_dir / TRAIN_RUN_INFO_FILENAME
        with open(metadata_path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def save(cls, model_id: str, run_config: types.CreateModelRequest,
             token: str):
        """
        Save training run metadata with token-based isolation.
        
        Args:
            model_id: The model identifier
            run_config: Training run configuration
            token: User's authentication token (becomes model owner)
        """
        lora_config = run_config.lora_config
        train_run_data = types.TrainingRun(
            training_run_id=model_id,
            base_model=run_config.base_model,
            model_owner=token,
            is_lora=True if lora_config else False,
            corrupted=False,
            lora_rank=lora_config.rank if lora_config else None,
            last_request_time=datetime.now(),
            last_checkpoint=None,  # Will be updated by checkpoint manager if needed
            last_sampler_checkpoint=None,
            user_metadata=run_config.user_metadata)

        new_data = train_run_data.model_dump(mode='json')
        # Store lora config details separately if needed, though they aren't in types.TrainingRun
        if lora_config:
            new_data['train_unembed'] = lora_config.train_unembed
            new_data['train_mlp'] = lora_config.train_mlp
            new_data['train_attn'] = lora_config.train_attn

        cls._write_info(model_id, new_data, token)

    @classmethod
    def get(cls, model_id: str, token: Optional[str] = None) -> Optional[types.TrainingRun]:
        """
        Get training run metadata with optional token-based isolation.
        
        Args:
            model_id: The model identifier
            token: User's authentication token for directory isolation
            
        Returns:
            TrainingRun object or None if not found
        """
        data = cls._read_info(model_id, token)
        if not data:
            return None
        # Clean up fields that might be stored
        # but not in the pydantic model if strict parsing issues arise
        return types.TrainingRun(**data)

    @classmethod
    def update(cls, model_id: str, updates: Dict[str, Any], token: Optional[str] = None):
        """
        Update training run metadata with optional token-based isolation.
        
        Args:
            model_id: The model identifier
            updates: Dictionary of fields to update
            token: User's authentication token for directory isolation
        """
        info = cls._read_info(model_id, token)
        if info:
            info.update(updates)
            cls._write_info(model_id, info, token)

    @classmethod
    def list_runs(cls, limit: int = 20,
                  offset: int = 0,
                  token: Optional[str] = None) -> types.TrainingRunsResponse:
        base_dir = cls.get_base_dir(token)
        if not base_dir.exists():
            return types.TrainingRunsResponse(training_runs=[],
                                              cursor=types.Cursor(
                                                  limit=limit,
                                                  offset=offset,
                                                  total_count=0))

        candidates = []
        for d in base_dir.iterdir():
            if d.is_dir() and (d / TRAIN_RUN_INFO_FILENAME).exists():
                candidates.append(d)

        candidates.sort(
            key=lambda d: (d / TRAIN_RUN_INFO_FILENAME).stat().st_mtime,
            reverse=True)

        total = len(candidates)
        selected = candidates[offset:offset + limit]

        runs = []
        for d in selected:
            run = cls.get(d.name, token)
            if run:
                runs.append(run)

        return types.TrainingRunsResponse(training_runs=runs,
                                          cursor=types.Cursor(
                                              limit=limit,
                                              offset=offset,
                                              total_count=total))


class CheckpointManager(FileManager):

    @staticmethod
    def get_ckpt_dir(model_id: str, checkpoint_id: str, token: Optional[str] = None) -> Path:
        """
        Get checkpoint directory with optional token-based isolation.
        
        Args:
            model_id: The model identifier
            checkpoint_id: The checkpoint identifier
            token: User's authentication token for directory isolation
            
        Returns:
            Path to checkpoint directory
        """
        return TrainingRunManager.get_model_dir(model_id, token) / checkpoint_id

    @staticmethod
    def get_save_dir(model_id: str, token: str, is_sampler=False) -> str:
        """
        Get save directory with token-based isolation.
        
        Args:
            model_id: The model identifier
            token: User's authentication token (required for isolation)
            is_sampler: Whether this is for sampler weights
            
        Returns:
            String path to save directory
        """
        weights_type = 'sampler_weights' if is_sampler else 'weights'
        checkpoint_id = Path(model_id) / weights_type
        save_path = TrainingRunManager.get_base_dir(token) / checkpoint_id
        return save_path.as_posix()

    @staticmethod
    def get_ckpt_name(name: Optional[str]) -> str:
        if name:
            # Normalize name to avoid issues with filesystem
            name = re.sub(r'[^\w\-]', '_', name)
            return name
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    @classmethod
    def _read_ckpt_info(cls, model_id: str,
                        checkpoint_id: str, token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Read checkpoint metadata with optional token-based isolation.
        
        Args:
            model_id: The model identifier
            checkpoint_id: The checkpoint identifier
            token: User's authentication token for directory isolation
            
        Returns:
            Dictionary with checkpoint metadata or None if not found
        """
        meta_path = cls.get_ckpt_dir(model_id,
                                     checkpoint_id, token) / CHECKPOINT_INFO_FILENAME
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    @classmethod
    def _write_ckpt_info(cls, model_id: str, checkpoint_id: str, data: Dict[str, Any], token: Optional[str] = None):
        """
        Write checkpoint metadata with token-based isolation.
        
        Args:
            model_id: The model identifier
            checkpoint_id: The checkpoint identifier
            data: Checkpoint metadata to write
            token: User's authentication token for directory isolation
        """
        ckpt_dir = cls.get_ckpt_dir(model_id, checkpoint_id, token)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        meta_path = ckpt_dir / CHECKPOINT_INFO_FILENAME
        with open(meta_path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def save(cls, model_id: str, name: str, token: str, is_sampler=False, public=False) -> str:
        """
        Save checkpoint metadata with token-based isolation.
        
        Args:
            model_id: The model identifier
            name: Checkpoint name
            token: User's authentication token (required for isolation)
            is_sampler: Whether this is a sampler checkpoint
            public: Whether the checkpoint is public
            
        Returns:
            The twinkle path for the checkpoint
        """
        weights_type = 'sampler_weights' if is_sampler else 'weights'
        checkpoint_type = 'sampler' if is_sampler else 'training'
        checkpoint_id = f'{weights_type}/{name}'
        tinker_path = f"twinkle://{model_id}/{checkpoint_id}"
        checkpoint_path = cls.get_ckpt_dir(model_id, checkpoint_id, token)
        checkpoint = types.Checkpoint(
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            time=datetime.now(),
            tinker_path=tinker_path,
            size_bytes=cls.get_dir_size(checkpoint_path),
            public=public
        )
        ckpt_data = checkpoint.model_dump(mode='json')
        cls._write_ckpt_info(model_id, checkpoint.checkpoint_id, ckpt_data, token)

        # Optionally update last_checkpoint in run info
        TrainingRunManager.update(model_id, {'last_checkpoint': ckpt_data}, token)
        return tinker_path

    @classmethod
    def get(cls, model_id: str,
            checkpoint_id: str, token: Optional[str] = None) -> Optional[types.Checkpoint]:
        """
        Get checkpoint metadata with optional token-based isolation.
        
        Args:
            model_id: The model identifier
            checkpoint_id: The checkpoint identifier
            token: User's authentication token for directory isolation
            
        Returns:
            Checkpoint object or None if not found
        """
        data = cls._read_ckpt_info(model_id, checkpoint_id, token)
        if not data:
            return None
        return types.Checkpoint(**data)

    @classmethod
    def list_checkpoints(
            cls, model_id: str, token: Optional[str] = None) -> Optional[types.CheckpointsListResponse]:
        """
        List checkpoints for a model with optional token-based isolation.
        
        Args:
            model_id: The model identifier
            token: User's authentication token for directory isolation
            
        Returns:
            CheckpointsListResponse or None if model directory not found
        """
        run_dir = TrainingRunManager.get_model_dir(model_id, token)
        if not run_dir.exists():
            return None

        checkpoints: List[types.Checkpoint] = []
        # Iterate over weights and sampler_weights directories
        for weights_type in ["weights", "sampler_weights"]:
            type_dir = run_dir / weights_type
            if not type_dir.exists() or not type_dir.is_dir():
                continue
            for d in type_dir.iterdir():
                if d.is_dir() and (d / CHECKPOINT_INFO_FILENAME).exists():
                    checkpoint_id = f"{weights_type}/{d.name}"
                    ckpt = cls.get(model_id, checkpoint_id, token)
                    if ckpt:
                        checkpoints.append(ckpt)

        # Sort by creation time
        checkpoints.sort(key=lambda x: x.time)

        return types.CheckpointsListResponse(checkpoints=checkpoints,
                                             cursor=None)

    @classmethod
    def delete(cls, model_id: str, checkpoint_id: str, token: Optional[str] = None) -> bool:
        """
        Delete a checkpoint with optional token-based isolation.
        
        Args:
            model_id: The model identifier
            checkpoint_id: The checkpoint identifier
            token: User's authentication token for directory isolation
            
        Returns:
            True if deleted successfully, False otherwise
        """
        # Basic safety check to prevent directory traversal
        if '..' in checkpoint_id:
            return False

        ckpt_dir = cls.get_ckpt_dir(model_id, checkpoint_id, token)

        if ckpt_dir.exists():
            if ckpt_dir.is_dir():
                shutil.rmtree(ckpt_dir)
            else:
                ckpt_dir.unlink()  # Should likely act on dir, but keeping safety

            # If we deleted the "last_checkpoint", we should strictly re-evaluate it
            # But finding true last requires listing all.
            # For simplicity, we can just clear it or leave it stale if acceptable.
            # To be correct:
            all_ckpts = cls.list_checkpoints(model_id, token)
            last_ckpt = all_ckpts.checkpoints[-1] if all_ckpts and all_ckpts.checkpoints else None
            TrainingRunManager.update(
                model_id, {
                    'last_checkpoint':
                    last_ckpt.model_dump(mode='json') if last_ckpt else None
                }, token)
            return True
        return False

    @classmethod
    def parse_tinker_path(cls, tinker_path: str) -> Optional[types.ParsedCheckpointTinkerPath]:
        """
        Parse a twinkle:// path into components.
        
        Args:
            tinker_path: The twinkle:// path string
            
        Returns:
            ParsedCheckpointTinkerPath or None if invalid format
        """
        if not tinker_path.startswith("twinkle://"):
            return None
        parts = tinker_path[10:].split("/")
        if len(parts) != 3:
            return None
        if parts[1] not in ["weights", "sampler_weights"]:
            return None
        checkpoint_type = "training" if parts[1] == "weights" else "sampler"
        return types.ParsedCheckpointTinkerPath(
            tinker_path=tinker_path,
            training_run_id=parts[0],
            checkpoint_type=checkpoint_type,
            checkpoint_id="/".join(parts[1:]),
        )

    @classmethod
    def get_weights_info(cls,
                         checkpoint_path: str, token: Optional[str] = None) -> Optional[types.WeightsInfoResponse]:
        """
        Get weights info with optional token-based isolation.
        
        Args:
            checkpoint_path: The twinkle:// path
            token: User's authentication token for directory isolation
            
        Returns:
            WeightsInfoResponse or None if not found
        """
        tinker_path = cls.parse_tinker_path(checkpoint_path)
        if not tinker_path:
            return None
        ckpt_info = cls.get(tinker_path.training_run_id,
                            tinker_path.checkpoint_id, token)
        if not ckpt_info:
            return None
        # weight info is stored in the training run info
        run_info = TrainingRunManager._read_info(tinker_path.training_run_id, token)
        if not run_info:
            return None
        return types.WeightsInfoResponse(**run_info)
