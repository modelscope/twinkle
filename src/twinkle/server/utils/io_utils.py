# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Base IO utilities for managing training runs and checkpoints.

This module provides abstract base classes that encapsulate common logic for
file-based storage of training run metadata and checkpoint information.
Both tinker and twinkle servers inherit from these classes.
"""
from abc import ABC, abstractmethod
from datetime import datetime
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Generic

from pydantic import BaseModel

from twinkle.hub import HubOperation


TWINKLE_DEFAULT_SAVE_DIR = os.environ.get('TWINKLE_DEFAULT_SAVE_DIR', './outputs')
CHECKPOINT_INFO_FILENAME = 'checkpoint_metadata.json'
TRAIN_RUN_INFO_FILENAME = 'twinkle_metadata.json'

# ----- Common Pydantic Models -----

class Cursor(BaseModel):
    limit: int
    offset: int
    total_count: int


class BaseCheckpoint(BaseModel):
    """Base checkpoint model that can be extended."""
    checkpoint_id: str
    checkpoint_type: str
    time: datetime
    size_bytes: int
    public: bool = False
    # Training run info (stored for hub downloads)
    base_model: Optional[str] = None
    is_lora: bool = False
    lora_rank: Optional[int] = None
    train_unembed: Optional[bool] = None
    train_mlp: Optional[bool] = None
    train_attn: Optional[bool] = None
    user_metadata: Optional[Dict[str, Any]] = None


class BaseTrainingRun(BaseModel):
    """Base training run model that can be extended."""
    training_run_id: str
    base_model: str
    model_owner: str
    is_lora: bool = False
    corrupted: bool = False
    lora_rank: Optional[int] = None
    last_request_time: Optional[datetime] = None
    last_checkpoint: Optional[Dict[str, Any]] = None
    last_sampler_checkpoint: Optional[Dict[str, Any]] = None
    user_metadata: Optional[Dict[str, Any]] = None


class BaseLoraConfig(BaseModel):
    """Base LoRA configuration model."""
    rank: int = 8
    train_unembed: bool = False
    train_mlp: bool = True
    train_attn: bool = True


class BaseCreateModelRequest(BaseModel):
    """Base request model for creating a model."""
    base_model: str
    lora_config: Optional[BaseLoraConfig] = None
    user_metadata: Optional[Dict[str, Any]] = None


class BaseParsedCheckpointPath(BaseModel):
    """Base model for parsed checkpoint paths."""
    path: str
    training_run_id: str
    checkpoint_type: str
    checkpoint_id: str


class ResolvedLoadPath(BaseModel):
    """Result of resolving a load path.
    
    Attributes:
        checkpoint_name: The name of the checkpoint (e.g., 'step-8' or hub model id)
        checkpoint_dir: The directory containing the checkpoint, or None if loading from hub
        is_twinkle_path: Whether the path was a twinkle:// path
        training_run_id: The training run ID (only set for twinkle:// paths)
        checkpoint_id: The checkpoint ID (only set for twinkle:// paths)
    """
    checkpoint_name: str
    checkpoint_dir: Optional[str] = None
    is_twinkle_path: bool = False
    training_run_id: Optional[str] = None
    checkpoint_id: Optional[str] = None


class BaseWeightsInfoResponse(BaseModel):
    """Base model for weights info response."""
    training_run_id: str
    base_model: str
    model_owner: str
    is_lora: bool = False
    lora_rank: Optional[int] = None


# Type variables for generic types
TCheckpoint = TypeVar('TCheckpoint', bound=BaseCheckpoint)
TTrainingRun = TypeVar('TTrainingRun', bound=BaseTrainingRun)
TCreateModelRequest = TypeVar('TCreateModelRequest', bound=BaseCreateModelRequest)
TParsedPath = TypeVar('TParsedPath', bound=BaseParsedCheckpointPath)
TWeightsInfo = TypeVar('TWeightsInfo', bound=BaseWeightsInfoResponse)


# ----- Permission Control Utilities -----

def validate_user_path(token: str, path: str) -> bool:
    """
    Validate that the path is safe and belongs to the user.
    
    This function checks:
    1. Path doesn't contain '..' (directory traversal attack prevention)
    2. Path doesn't start with '/' (absolute path prevention)
    3. Path doesn't contain null bytes
    4. Path components are reasonable
    
    Args:
        token: User's authentication token (used to identify ownership)
        path: The path to validate
        
    Returns:
        True if path is safe, False otherwise
    """
    if not path:
        return False
    
    # Check for directory traversal attempts
    if '..' in path:
        return False
    
    # Check for null bytes (security vulnerability)
    if '\x00' in path:
        return False
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'\.\./',      # Directory traversal
        r'/\.\.', 
        r'^/',         # Absolute path
        r'^\.\.',      # Starts with ..
        r'~',          # Home directory expansion
    ]
    for pattern in suspicious_patterns:
        if re.search(pattern, path):
            return False
    
    return True


def validate_ownership(token: str, model_owner: str) -> bool:
    """
    Validate that the user owns the resource.
    
    Args:
        token: User's authentication token
        model_owner: The owner of the model/checkpoint
        
    Returns:
        True if user owns the resource, False otherwise
    """
    if not token or not model_owner:
        return False
    return token == model_owner


# ----- Base File Manager -----

class BaseFileManager:
    """Base file manager with common utilities."""
    
    @staticmethod
    def get_dir_size(path: Path) -> int:
        """Calculate total size of files in a directory."""
        total = 0
        if path.exists():
            for p in path.rglob('*'):
                if p.is_file():
                    total += p.stat().st_size
        return total


# ----- Base Training Run Manager -----

class BaseTrainingRunManager(BaseFileManager, ABC):
    """
    Abstract base class for managing training run metadata.
    
    Subclasses must implement:
    - train_run_info_filename property
    - _create_training_run method
    - _training_runs_response_cls property
    """
    
    def __init__(self, token: str):
        """
        Initialize the manager with a user token.
        
        Args:
            token: User's authentication token for directory isolation
        """
        self.token = token
    
    @property
    @abstractmethod
    def train_run_info_filename(self) -> str:
        """Return the filename for training run metadata."""
        pass
    
    @abstractmethod
    def _create_training_run(self, model_id: str, run_config: Any) -> Dict[str, Any]:
        """
        Create training run data from model_id and run_config.
        
        Args:
            model_id: The model identifier
            run_config: The run configuration
            
        Returns:
            Dictionary with training run data
        """
        pass
    
    @abstractmethod
    def _parse_training_run(self, data: Dict[str, Any]) -> Any:
        """
        Parse training run data into the appropriate model.
        
        Args:
            data: Raw training run data
            
        Returns:
            TrainingRun model instance
        """
        pass
    
    @abstractmethod
    def _create_training_runs_response(
        self, runs: List[Any], limit: int, offset: int, total: int
    ) -> Any:
        """
        Create a training runs response.
        
        Args:
            runs: List of training runs
            limit: Page limit
            offset: Page offset
            total: Total count
            
        Returns:
            TrainingRunsResponse model instance
        """
        pass
    
    def get_base_dir(self) -> Path:
        """
        Get base directory with token-based isolation.
        
        Returns:
            Path to token-specific base directory
        """
        base_path = Path(TWINKLE_DEFAULT_SAVE_DIR)
        # Sanitize token to avoid filesystem issues
        sanitized_token = re.sub(r'[^\w\-]', '_', self.token)
        return base_path / sanitized_token

    def get_model_dir(self, model_id: str) -> Path:
        """
        Get model directory with token-based isolation.
        
        Args:
            model_id: The model identifier
            
        Returns:
            Path to model directory
        """
        return self.get_base_dir() / model_id

    def _read_info(self, model_id: str) -> Dict[str, Any]:
        """
        Read training run metadata from disk.
        
        Args:
            model_id: The model identifier
            
        Returns:
            Dictionary with metadata or empty dict if not found
        """
        metadata_path = self.get_model_dir(model_id) / self.train_run_info_filename
        if not metadata_path.exists():
            return {}
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_info(self, model_id: str, data: Dict[str, Any]):
        """
        Write training run metadata to disk.
        
        Args:
            model_id: The model identifier
            data: Metadata to write
        """
        model_dir = self.get_model_dir(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = model_dir / self.train_run_info_filename
        with open(metadata_path, 'w') as f:
            json.dump(data, f, indent=2)

    def save(self, model_id: str, run_config: Any):
        """
        Save training run metadata with token-based isolation.
        
        Args:
            model_id: Unique identifier for the model
            run_config: Configuration for the training run
        """
        new_data = self._create_training_run(model_id, run_config)
        self._write_info(model_id, new_data)

    def get(self, model_id: str) -> Optional[Any]:
        """
        Get training run metadata.
        
        Args:
            model_id: The model identifier
            
        Returns:
            TrainingRun object or None if not found
        """
        data = self._read_info(model_id)
        if not data:
            return None
        return self._parse_training_run(data)

    def update(self, model_id: str, updates: Dict[str, Any]):
        """
        Update training run metadata.
        
        Args:
            model_id: The model identifier
            updates: Dictionary of fields to update
        """
        info = self._read_info(model_id)
        if info:
            info.update(updates)
            self._write_info(model_id, info)

    def list_runs(self, limit: int = 20, offset: int = 0) -> Any:
        """
        List training runs for the current user.
        
        Args:
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            TrainingRunsResponse with list of training runs
        """
        base_dir = self.get_base_dir()
        if not base_dir.exists():
            return self._create_training_runs_response([], limit, offset, 0)

        candidates = []
        for d in base_dir.iterdir():
            if d.is_dir() and (d / self.train_run_info_filename).exists():
                candidates.append(d)

        candidates.sort(
            key=lambda d: (d / self.train_run_info_filename).stat().st_mtime,
            reverse=True
        )

        # All runs in the token directory belong to this user
        runs = []
        for d in candidates:
            run = self.get(d.name)
            if run:
                runs.append(run)

        total = len(runs)
        selected = runs[offset:offset + limit]

        return self._create_training_runs_response(selected, limit, offset, total)


# ----- Base Checkpoint Manager -----

class BaseCheckpointManager(BaseFileManager, ABC):
    """
    Abstract base class for managing checkpoint metadata.
    
    Subclasses must implement:
    - path_prefix property
    - path_field_name property
    - _create_checkpoint method
    - _parse_checkpoint method
    - _create_checkpoints_response method
    - _create_parsed_path method
    - _create_weights_info method
    """
    
    def __init__(self, token: str, training_run_manager: BaseTrainingRunManager):
        """
        Initialize the manager with a user token.
        
        Args:
            token: User's authentication token for directory isolation
            training_run_manager: Associated training run manager
        """
        self.token = token
        self.training_run_manager = training_run_manager
    
    @property
    @abstractmethod
    def path_prefix(self) -> str:
        """Return the path prefix (e.g., 'twinkle://')."""
        pass
    
    @property
    @abstractmethod
    def path_field_name(self) -> str:
        """Return the field name for the path (e.g., 'twinkle_path' or 'tinker_path')."""
        pass
    
    @abstractmethod
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
        """
        Create checkpoint data.
        
        Args:
            checkpoint_id: The checkpoint identifier
            checkpoint_type: Type of checkpoint ('training' or 'sampler')
            path: The twinkle:// path to the checkpoint
            size_bytes: Size of the checkpoint in bytes
            public: Whether the checkpoint is public
            base_model: The base model name/path
            is_lora: Whether this is a LoRA checkpoint
            lora_rank: The LoRA rank if applicable
            train_unembed: Whether unembed layers are trained
            train_mlp: Whether MLP layers are trained
            train_attn: Whether attention layers are trained
            user_metadata: User-provided metadata
            
        Returns:
            Dictionary with checkpoint data
        """
        pass
    
    @abstractmethod
    def _parse_checkpoint(self, data: Dict[str, Any]) -> Any:
        """
        Parse checkpoint data into the appropriate model.
        
        Args:
            data: Raw checkpoint data
            
        Returns:
            Checkpoint model instance
        """
        pass
    
    @abstractmethod
    def _create_checkpoints_response(self, checkpoints: List[Any]) -> Any:
        """
        Create a checkpoints list response.
        
        Args:
            checkpoints: List of checkpoints
            
        Returns:
            CheckpointsListResponse model instance
        """
        pass
    
    @abstractmethod
    def _create_parsed_path(
        self, path: str, training_run_id: str, 
        checkpoint_type: str, checkpoint_id: str
    ) -> Any:
        """
        Create a parsed path model.
        
        Returns:
            ParsedCheckpointPath model instance
        """
        pass
    
    @abstractmethod
    def _create_weights_info(self, run_info: Dict[str, Any]) -> Any:
        """
        Create weights info from run info.
        
        Args:
            run_info: Training run info
            
        Returns:
            WeightsInfoResponse model instance
        """
        pass

    def get_ckpt_dir(self, model_id: str, checkpoint_id: str) -> Path:
        """
        Get checkpoint directory with token-based isolation.
        
        Args:
            model_id: The model identifier
            checkpoint_id: The checkpoint identifier
            
        Returns:
            Path to checkpoint directory
        """
        return self.training_run_manager.get_model_dir(model_id) / checkpoint_id

    def get_save_dir(self, model_id: str, is_sampler: bool = False) -> str:
        """
        Get save directory with token-based isolation.
        
        Args:
            model_id: The model identifier
            is_sampler: Whether this is for sampler weights
            
        Returns:
            String path to save directory
        """
        weights_type = 'sampler_weights' if is_sampler else 'weights'
        checkpoint_id = Path(model_id) / weights_type
        save_path = self.training_run_manager.get_base_dir() / checkpoint_id
        return save_path.as_posix()

    @staticmethod
    def get_ckpt_name(name: Optional[str]) -> str:
        """Generate or normalize checkpoint name."""
        if name:
            # Normalize name to avoid issues with filesystem
            name = re.sub(r'[^\w\-]', '_', name)
            return name
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    def _read_ckpt_info(self, model_id: str, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Read checkpoint metadata from disk.
        
        Args:
            model_id: The model identifier
            checkpoint_id: The checkpoint identifier
            
        Returns:
            Dictionary with checkpoint metadata or None if not found
        """
        meta_path = self.get_ckpt_dir(model_id, checkpoint_id) / CHECKPOINT_INFO_FILENAME
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def _write_ckpt_info(self, model_id: str, checkpoint_id: str, data: Dict[str, Any]):
        """
        Write checkpoint metadata to disk.
        
        Args:
            model_id: The model identifier
            checkpoint_id: The checkpoint identifier
            data: Checkpoint metadata to write
        """
        ckpt_dir = self.get_ckpt_dir(model_id, checkpoint_id)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        meta_path = ckpt_dir / CHECKPOINT_INFO_FILENAME
        with open(meta_path, 'w') as f:
            json.dump(data, f, indent=2)

    def save(self, model_id: str, name: str, 
             is_sampler: bool = False, public: bool = False) -> str:
        """
        Save checkpoint metadata.
        
        Args:
            model_id: The model identifier
            name: Checkpoint name
            is_sampler: Whether this is a sampler checkpoint
            public: Whether the checkpoint is public
            
        Returns:
            The path for the checkpoint
        """
        # Validate path safety
        if not validate_user_path(self.token, name):
            raise ValueError(f"Invalid checkpoint name: {name}")
        
        weights_type = 'sampler_weights' if is_sampler else 'weights'
        checkpoint_type = 'sampler' if is_sampler else 'training'
        checkpoint_id = f'{weights_type}/{name}'
        path = f"{self.path_prefix}{model_id}/{checkpoint_id}"
        checkpoint_path = self.get_ckpt_dir(model_id, checkpoint_id)
        
        # Read training run info to include in checkpoint metadata
        run_info = self.training_run_manager._read_info(model_id)
        
        ckpt_data = self._create_checkpoint(
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            path=path,
            size_bytes=self.get_dir_size(checkpoint_path),
            public=public,
            base_model=run_info.get('base_model'),
            is_lora=run_info.get('is_lora', False),
            lora_rank=run_info.get('lora_rank'),
            train_unembed=run_info.get('train_unembed'),
            train_mlp=run_info.get('train_mlp'),
            train_attn=run_info.get('train_attn'),
            user_metadata=run_info.get('user_metadata')
        )
        self._write_ckpt_info(model_id, checkpoint_id, ckpt_data)

        # Update last_checkpoint in run info
        self.training_run_manager.update(model_id, {'last_checkpoint': ckpt_data})
        return path

    def get(self, model_id: str, checkpoint_id: str) -> Optional[Any]:
        """
        Get checkpoint metadata.
        
        Args:
            model_id: The model identifier
            checkpoint_id: The checkpoint identifier
            
        Returns:
            Checkpoint object or None if not found
        """
        data = self._read_ckpt_info(model_id, checkpoint_id)
        if not data:
            return None
        return self._parse_checkpoint(data)

    def list_checkpoints(self, model_id: str) -> Optional[Any]:
        """
        List checkpoints for a training run.
        
        Args:
            model_id: The model identifier
            
        Returns:
            CheckpointsListResponse or None if model directory not found
        """
        run_dir = self.training_run_manager.get_model_dir(model_id)
        if not run_dir.exists():
            return None

        checkpoints = []
        # Iterate over weights and sampler_weights directories
        for weights_type in ["weights", "sampler_weights"]:
            type_dir = run_dir / weights_type
            if not type_dir.exists() or not type_dir.is_dir():
                continue
            for d in type_dir.iterdir():
                if d.is_dir() and (d / CHECKPOINT_INFO_FILENAME).exists():
                    checkpoint_id = f"{weights_type}/{d.name}"
                    ckpt = self.get(model_id, checkpoint_id)
                    if ckpt:
                        checkpoints.append(ckpt)

        # Sort by creation time
        checkpoints.sort(key=lambda x: x.time)

        return self._create_checkpoints_response(checkpoints)

    def delete(self, model_id: str, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.
        
        Args:
            model_id: The model identifier
            checkpoint_id: The checkpoint identifier
            
        Returns:
            True if deleted successfully, False if not found
        """
        # Basic safety check to prevent directory traversal
        if '..' in checkpoint_id:
            return False

        ckpt_dir = self.get_ckpt_dir(model_id, checkpoint_id)

        if ckpt_dir.exists():
            if ckpt_dir.is_dir():
                shutil.rmtree(ckpt_dir)
            else:
                ckpt_dir.unlink()

            # Update last_checkpoint in run info
            all_ckpts = self.list_checkpoints(model_id)
            last_ckpt = all_ckpts.checkpoints[-1] if all_ckpts and all_ckpts.checkpoints else None
            self.training_run_manager.update(
                model_id, {
                    'last_checkpoint': last_ckpt.model_dump(mode='json') if last_ckpt else None
                }
            )
            return True
        return False

    def parse_path(self, path: str) -> Optional[Any]:
        """
        Parse a path into its components.
        
        Args:
            path: The path string (e.g., twinkle://model_id/weights/name)
            
        Returns:
            ParsedCheckpointPath or None if invalid format
        """
        if not path.startswith(self.path_prefix):
            return None
        parts = path[len(self.path_prefix):].split("/")
        if len(parts) != 3:
            return None
        if parts[1] not in ["weights", "sampler_weights"]:
            return None
        checkpoint_type = "training" if parts[1] == "weights" else "sampler"
        return self._create_parsed_path(
            path=path,
            training_run_id=parts[0],
            checkpoint_type=checkpoint_type,
            checkpoint_id="/".join(parts[1:]),
        )

    def get_weights_info(self, checkpoint_path: str) -> Optional[Any]:
        """
        Get weights info.
        
        Supports both twinkle:// paths (local checkpoints) and hub model IDs.
        For hub model IDs, downloads checkpoint_metadata.json from ModelScope.
        
        Args:
            checkpoint_path: The twinkle:// path or hub model ID
            
        Returns:
            WeightsInfoResponse or None if not found
        """
        # Use resolve_load_path to determine if this is a twinkle path or hub path
        try:
            resolved = self.resolve_load_path(checkpoint_path, validate_exists=False)
        except ValueError:
            return None
        
        if resolved.is_twinkle_path:
            # Local twinkle:// path - read from local checkpoint metadata
            ckpt_data = self._read_ckpt_info(resolved.training_run_id, resolved.checkpoint_id)
            if not ckpt_data or not ckpt_data.get('base_model'):
                return None
            return self._create_weights_info(ckpt_data)
        else:
            # Hub model ID - download checkpoint_metadata.json from ModelScope
            return self._get_weights_info_from_hub(checkpoint_path)
    
    def _get_weights_info_from_hub(self, hub_model_id: str) -> Optional[Any]:
        """
        Download and parse checkpoint_metadata.json from hub.
        
        Args:
            hub_model_id: The hub model ID (e.g., 'user/model-name')
            
        Returns:
            WeightsInfoResponse or None if not found or failed to download
        """
        try:
            # Download only the checkpoint_metadata.json file from hub
            local_dir = HubOperation.download_file(
                repo_id=hub_model_id,
                allow_patterns=[CHECKPOINT_INFO_FILENAME],
                token=self.token
            )
            
            # Read and parse the metadata
            metadata_path = os.path.join(local_dir, CHECKPOINT_INFO_FILENAME)
            if not os.path.exists(metadata_path):
                return None
                
            with open(metadata_path, 'r') as f:
                ckpt_data = json.load(f)
            
            if not ckpt_data.get('base_model'):
                return None
            
            return self._create_weights_info(ckpt_data)
            
        except Exception:
            return None

    def parse_adapter_uri(self, adapter_uri: str) -> tuple:
        """Parse adapter URI to extract user_id and resolved lora_path.

        Args:
            adapter_uri: The adapter URI, supports formats:
                - twinkle://{training_run_id}/weights/{checkpoint_name} or sampler_weights/{name}
                - Local filesystem path

        Returns:
            Tuple of (user_id, lora_path) where lora_path is the resolved filesystem path
        """
        if adapter_uri.startswith(self.path_prefix):
            parsed = self.parse_path(adapter_uri)
            if parsed:
                # Get the filesystem path using get_ckpt_dir
                lora_path = str(self.get_ckpt_dir(
                    parsed.training_run_id, parsed.checkpoint_id
                ))
                return parsed.training_run_id, lora_path
            else:
                # Fallback: parse manually for non-standard formats
                suffix = adapter_uri[len(self.path_prefix):]
                return 'default', suffix
        else:
            # Local path
            return 'default', adapter_uri

    def resolve_load_path(self, path: str, validate_exists: bool = True) -> ResolvedLoadPath:
        """
        Resolve a checkpoint load path.
        
        This method handles two types of paths:
        1. twinkle:// paths: Parse, validate permissions, return checkpoint_name and checkpoint_dir
        2. Hub model IDs: Return the path as checkpoint_name with checkpoint_dir=None
        
        Args:
            path: The path to resolve (either twinkle:// format or hub model ID)
            validate_exists: Whether to validate that the checkpoint exists (default: True)
            
        Returns:
            ResolvedLoadPath with checkpoint_name and checkpoint_dir
            
        Raises:
            ValueError: If the path format is invalid or checkpoint not found
        """
        # Check if path starts with twinkle:// prefix
        if path.startswith(self.path_prefix):
            # Parse the twinkle:// path
            parsed = self.parse_path(path)
            if not parsed:
                raise ValueError(f"Invalid {self.path_prefix} path format: {path}")
            
            # Extract components
            training_run_id = parsed.training_run_id
            checkpoint_id = parsed.checkpoint_id
            checkpoint_name = checkpoint_id.split('/')[-1]  # Extract name from "weights/step-8"
            
            if validate_exists:
                # Verify checkpoint exists and user has access
                checkpoint = self.get(training_run_id, checkpoint_id)
                if not checkpoint:
                    raise ValueError(
                        f"Checkpoint not found or access denied: {path}"
                    )
            
            # Get the checkpoint directory
            checkpoint_dir = str(self.get_ckpt_dir(training_run_id, checkpoint_id))
            
            if validate_exists:
                # Verify the directory exists
                from pathlib import Path as PathLib
                if not PathLib(checkpoint_dir).exists():
                    raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
            
            return ResolvedLoadPath(
                checkpoint_name=checkpoint_name,
                checkpoint_dir=checkpoint_dir,
                is_twinkle_path=True,
                training_run_id=training_run_id,
                checkpoint_id=checkpoint_id
            )
        else:
            # Not a twinkle:// path - treat as hub model ID
            # Return the path as checkpoint_name with no checkpoint_dir
            return ResolvedLoadPath(
                checkpoint_name=path,
                checkpoint_dir=None,
                is_twinkle_path=False,
                training_run_id=None,
                checkpoint_id=None
            )
