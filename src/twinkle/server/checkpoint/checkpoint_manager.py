# Copyright (c) ModelScope Contributors. All rights reserved.
"""Abstract base checkpoint manager.

Relocated from ``utils/checkpoint_base.py`` (TIER 2 consolidation). No logic change.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from twinkle import get_logger
from twinkle.hub import HubOperation
from twinkle_client.types import ResolvedLoadPath
from .paths import CHECKPOINT_INFO_FILENAME, validate_user_path
from .training_run_manager import BaseFileManager, BaseTrainingRunManager

logger = get_logger()


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
    def _create_checkpoint(self,
                           checkpoint_id: str,
                           checkpoint_type: str,
                           path: str,
                           size_bytes: int,
                           public: bool,
                           base_model: str | None = None,
                           is_lora: bool = False,
                           lora_rank: int | None = None,
                           train_unembed: bool | None = None,
                           train_mlp: bool | None = None,
                           train_attn: bool | None = None,
                           user_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
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
    def _parse_checkpoint(self, data: dict[str, Any]) -> Any:
        """
        Parse checkpoint data into the appropriate model.

        Args:
            data: Raw checkpoint data

        Returns:
            Checkpoint model instance
        """
        pass

    @abstractmethod
    def _create_checkpoints_response(self, checkpoints: list[Any]) -> Any:
        """
        Create a checkpoints list response.

        Args:
            checkpoints: List of checkpoints

        Returns:
            CheckpointsListResponse model instance
        """
        pass

    @abstractmethod
    def _create_parsed_path(self, path: str, training_run_id: str, checkpoint_type: str, checkpoint_id: str) -> Any:
        """
        Create a parsed path model.

        Returns:
            ParsedCheckpointPath model instance
        """
        pass

    @abstractmethod
    def _create_weights_info(self, run_info: dict[str, Any]) -> Any:
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
        save_path = self.training_run_manager.get_model_dir(model_id) / weights_type
        return save_path.as_posix()

    @staticmethod
    def get_ckpt_name(name: str | None) -> str:
        """Generate or normalize checkpoint name."""
        if name:
            # Normalize name to avoid issues with filesystem
            name = re.sub(r'[^\w\-]', '_', name)
            return name
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    def _read_ckpt_info(self, model_id: str, checkpoint_id: str) -> dict[str, Any] | None:
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
            with open(meta_path) as f:
                return json.load(f)
        except Exception:
            return None

    def _write_ckpt_info(self, model_id: str, checkpoint_id: str, data: dict[str, Any]):
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

    def save(self, model_id: str, name: str, is_sampler: bool = False, public: bool = False) -> str:
        """
        Save checkpoint metadata.

        Args:
            model_id: The model identifier
            name: Checkpoint name. For sampler checkpoints this is ignored; weights are
                always stored under the fixed name ``'latest'`` and a per-save timestamp
                symlink is created in the same ``sampler_weights/`` directory.
            is_sampler: Whether this is a sampler checkpoint
            public: Whether the checkpoint is public

        Returns:
            The ``twinkle://`` path for the checkpoint. For sampler checkpoints this
            points to the timestamp symlink so callers always receive a unique path
            and bypass any filesystem-path-based weight cache.
        """
        # Validate path safety
        if not validate_user_path(self.token, name):
            raise ValueError(f'Invalid checkpoint name: {name}')

        weights_type = 'sampler_weights' if is_sampler else 'weights'
        checkpoint_type = 'sampler' if is_sampler else 'training'
        # Sampler weights are always stored under the fixed name 'latest' so only one
        # version exists on disk at a time; cleanup is handled by _delete_existing_sampler_weights.
        effective_name = 'latest' if is_sampler else name
        checkpoint_id = f'{weights_type}/{effective_name}'
        path = f'{self.path_prefix}{model_id}/{checkpoint_id}'
        checkpoint_path = self.get_ckpt_dir(model_id, checkpoint_id)

        # For sampler checkpoints, delete existing sampler weights for this model_id
        if is_sampler:
            self._delete_existing_sampler_weights(model_id)

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
            user_metadata=run_info.get('user_metadata'))
        self._write_ckpt_info(model_id, checkpoint_id, ckpt_data)

        # Update last_checkpoint in run info
        self.training_run_manager.update(model_id, {'last_checkpoint': ckpt_data})

        if is_sampler:
            # Create a per-save timestamp symlink in sampler_weights/ so callers always
            # receive a unique twinkle:// path and bypass any filesystem-path-based cache.
            save_dir = self.get_save_dir(model_id, is_sampler=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fixed_path = os.path.join(save_dir, 'latest')
            symlink_path = os.path.join(save_dir, timestamp)
            if os.path.islink(symlink_path):
                os.unlink(symlink_path)
            os.symlink(fixed_path, symlink_path)
            return f'{self.path_prefix}{model_id}/sampler_weights/{timestamp}'

        return path

    def _delete_existing_sampler_weights(self, model_id: str):
        """
        Delete all existing sampler weights for a model_id.

        Args:
            model_id: The model identifier
        """
        run_dir = self.training_run_manager.get_model_dir(model_id)
        sampler_weights_dir = run_dir / 'sampler_weights'

        if sampler_weights_dir.exists() and sampler_weights_dir.is_dir():
            for item in sampler_weights_dir.iterdir():
                if item.is_symlink():
                    # Unlink symlinks explicitly; shutil.rmtree on a symlink-to-directory
                    # can follow the link and unexpectedly delete the target contents.
                    item.unlink()
                elif item.is_dir():
                    meta_path = item / CHECKPOINT_INFO_FILENAME
                    if meta_path.exists():
                        meta_path.unlink()
                    shutil.rmtree(item)
            logger.info(f'Deleted existing sampler weights for model_id: {model_id}')

    def get(self, model_id: str, checkpoint_id: str) -> Any | None:
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

    def list_checkpoints(self, model_id: str) -> Any | None:
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
        for weights_type in ['weights', 'sampler_weights']:
            type_dir = run_dir / weights_type
            if not type_dir.exists() or not type_dir.is_dir():
                continue
            for d in type_dir.iterdir():
                if d.is_dir() and (d / CHECKPOINT_INFO_FILENAME).exists():
                    checkpoint_id = f'{weights_type}/{d.name}'
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
                model_id, {'last_checkpoint': last_ckpt.model_dump(mode='json') if last_ckpt else None})
            return True
        return False

    def parse_path(self, path: str) -> Any | None:
        """
        Parse a path into its components.

        Args:
            path: The path string (e.g., twinkle://model_id/weights/name)

        Returns:
            ParsedCheckpointPath or None if invalid format
        """
        if not path.startswith(self.path_prefix):
            return None
        parts = path[len(self.path_prefix):].split('/')
        if len(parts) != 3:
            return None
        if parts[1] not in ['weights', 'sampler_weights']:
            return None
        checkpoint_type = 'training' if parts[1] == 'weights' else 'sampler'
        return self._create_parsed_path(
            path=path,
            training_run_id=parts[0],
            checkpoint_type=checkpoint_type,
            checkpoint_id='/'.join(parts[1:]),
        )

    def get_weights_info(self, checkpoint_path: str) -> Any | None:
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

    def _get_weights_info_from_hub(self, hub_model_id: str) -> Any | None:
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
                repo_id=hub_model_id, allow_patterns=[CHECKPOINT_INFO_FILENAME], token=self.token)

            # Read and parse the metadata
            metadata_path = os.path.join(local_dir, CHECKPOINT_INFO_FILENAME)
            if not os.path.exists(metadata_path):
                return None

            with open(metadata_path) as f:
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
                lora_path = str(self.get_ckpt_dir(parsed.training_run_id, parsed.checkpoint_id))
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
                raise ValueError(f'Invalid {self.path_prefix} path format: {path}')

            # Extract components
            training_run_id = parsed.training_run_id
            checkpoint_id = parsed.checkpoint_id
            checkpoint_name = checkpoint_id.split('/')[-1]  # Extract name from "weights/step-8"

            if validate_exists:
                # Verify checkpoint exists and user has access
                checkpoint = self.get(training_run_id, checkpoint_id)
                if not checkpoint:
                    raise ValueError(f'Checkpoint not found or access denied: {path}')

            # Get the checkpoint directory parent path (no checkpoint name in the path)
            checkpoint_dir = self.get_ckpt_dir(training_run_id, checkpoint_id).parent

            if validate_exists:
                if not checkpoint_dir.exists():
                    raise ValueError(f'Checkpoint directory not found: {checkpoint_dir}')

            return ResolvedLoadPath(
                checkpoint_name=checkpoint_name,
                checkpoint_dir=checkpoint_dir.as_posix(),
                is_twinkle_path=True,
                training_run_id=training_run_id,
                checkpoint_id=checkpoint_id)
        else:
            # Not a twinkle:// path - treat as hub model ID
            # Return the path as checkpoint_name with no checkpoint_dir
            return ResolvedLoadPath(
                checkpoint_name=path,
                checkpoint_dir=None,
                is_twinkle_path=False,
                training_run_id=None,
                checkpoint_id=None)
